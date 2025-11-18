import torch
import torch.nn as nn
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import matplotlib.pyplot as plt
import nvdiffrast.torch as dr
import numpy as np
import gc
import math
import time
import pdb
import cv2
import torchvision
from lib.networks.renderer.raster_utils import (
    load_obj, compute_near_far, 
    perspective_projection_opencv_to_opengl, world_to_clip_space
)
import os
import matplotlib.pyplot as plt
from .nerf_net_utils import raw2outputs

class Renderer:

    def __init__(self, net):
        self.net = net
        self.glctx = dr.RasterizeCudaContext() 


    import torch
    import torch.nn.functional as F

    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Depth fusion: input views → target view (z-buffer in target image plane)
    # ──────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fuse_input_depths_to_target(
        self,
        input_depths: torch.Tensor,   # [B, V, 1, Hs, Ws] meters (Sapiens depth per input view)
        input_masks: torch.Tensor,    # [B, V, 1, Hs, Ws] 0/1 valid mask (same views)
        input_K: torch.Tensor,        # [B, V, 3, 3]
        input_R: torch.Tensor,        # [B, V, 3, 3]   world←→cam uses X_cam = R*X_world + T
        input_T: torch.Tensor,        # [B, V, 3, 1]
        target_K: torch.Tensor,       # [B, 3, 3] or [1, 3, 3]
        target_R: torch.Tensor,       # [B, 3, 3] or [1, 3, 3]
        target_T: torch.Tensor,       # [B, 3, 1] or [1, 3, 1]
        target_hw: tuple,             # (Ht, Wt) target resolution
        chunked: bool = True,
        max_points: int = 1_000_000,  # limit per chunk for memory safety
    ):
        """
        Returns:
            target_depth : [B, 1, Ht, Wt] fused depth (meters) in target view (z-buffered)
            target_count : [B, 1, Ht, Wt] how many source pixels contributed (for diagnostics)

        Notes:
        • Back-project each valid input pixel to world, then project to target.
        • Z-buffer by taking amin (nearest) along each target pixel.
        • If scatter_reduce_('amin') isn't available, falls back to a sort-based reducer.
        """
        B, V, Hs, Ws = input_depths.shape
        Ht, Wt = target_hw
        dev = input_depths.device
        dtype = input_depths.dtype

        # Broadcast target extrinsics/intrinsics
        if target_K.dim() == 2: target_K = target_K[None].expand(B, -1, -1).to(dev, dtype)
        if target_R.dim() == 2: target_R = target_R[None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 2: target_T = target_T[None, :, None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 3 and target_T.shape[-1] != 1: target_T = target_T[..., None]

        # Init fused depth with +inf (z-buffer)
        target_depth = torch.full((B, 1, Ht, Wt), float('inf'), device=dev, dtype=dtype)
        target_count = torch.zeros((B, 1, Ht, Wt), device=dev, dtype=torch.int32)

        # Precompute input pixel grid (homogeneous) for back-projection
        ys, xs = torch.meshgrid(
            torch.arange(Hs, device=dev, dtype=dtype),
            torch.arange(Ws, device=dev, dtype=dtype),
            indexing='ij'
        )
        ones = torch.ones_like(xs)
        pix_homo = torch.stack([xs, ys, ones], dim=0).view(1, 1, 3, Hs * Ws)  # [1,1,3,M]

        for v in range(V):
            d = input_depths[:, v, 0]              # [B, Hs, Ws]
            m = input_masks[:, v, 0]               # [B, Hs, Ws]
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

            valid = (d > 0) & (m > 0)
            if not valid.any():
                continue

            # Flatten & mask
            d_flat = d.view(B, -1)                 # [B, M]
            valid_flat = valid.view(B, -1)         # [B, M]

            # K^-1 for this view
            K = input_K[:, v]                      # [B,3,3]
            R = input_R[:, v]                      # [B,3,3]
            T = input_T[:, v]                      # [B,3,1]
            Kinv = torch.inverse(K)                # [B,3,3]

            # Back-project to camera coords: Xc = z * K^-1 @ [u,v,1]
            # Prepare per-batch Kinv @ pix_homo
            # [B,3,3] @ [1,1,3,M] -> [B,1,3,M] -> [B,3,M]
            Kinv_pix = torch.matmul(Kinv[:, None, :, :], pix_homo).squeeze(1)  # [B,3,M]

            # Chunking to save memory
            M = Hs * Ws
            start = 0
            while start < M:
                end = min(start + max_points, M)
                sel = valid_flat[:, start:end]  # [B, m]
                if not sel.any():
                    start = end
                    continue

                # Gather per-batch selected indices
                # Build index tensors per-batch (masking)
                idx_list = [torch.nonzero(sel[b], as_tuple=False).squeeze(1) for b in range(B)]
                # Concatenate with batch offsets later
                max_m = max((i.numel() for i in idx_list), default=0)
                if max_m == 0:
                    start = end
                    continue

                # For each batch, gather points
                gathered_points = []
                gathered_z = []
                gathered_b = []
                for b in range(B):
                    idxb = idx_list[b]
                    if idxb.numel() == 0:
                        continue
                    # Xc_dir = Kinv @ [u,v,1]  (for those pixels)
                    Xc_dir = Kinv_pix[b, :, start:end][:, idxb]                 # [3, mb]
                    zb = d_flat[b, start:end][idxb]                              # [mb]
                    Xc = Xc_dir * zb[None, :]                                    # [3, mb]

                    # Cam→world: Xw = R^T @ (Xc - T)
                    Xw = torch.matmul(R[b].transpose(0, 1), (Xc - T[b, :, 0:1]))  # [3, mb]

                    # World→target cam: Xt = Rt * Xw + Tt
                    Xt = torch.matmul(target_R[b], Xw) + target_T[b]             # [3, mb]
                    zt = Xt[2, :]                                                # [mb]
                    front = zt > 1e-6
                    if not front.any():
                        continue

                    Xt = Xt[:, front]
                    zt = zt[front]

                    # Project to target pixels
                    x = Xt[0, :] / zt
                    y = Xt[1, :] / zt
                    fx, fy = target_K[b, 0, 0], target_K[b, 1, 1]
                    cx, cy = target_K[b, 0, 2], target_K[b, 1, 2]
                    u = fx * x + cx
                    v = fy * y + cy

                    # Nearest-neighbor raster for z-buffer (simple & fast)
                    ui = u.round().long()
                    vi = v.round().long()
                    inb = (ui >= 0) & (ui < Wt) & (vi >= 0) & (vi < Ht)
                    if not inb.any():
                        continue

                    ui = ui[inb]; vi = vi[inb]; zt = zt[inb]
                    flat = (vi * Wt + ui)  # [mb']

                    gathered_points.append(flat)
                    gathered_z.append(zt)
                    gathered_b.append(torch.full_like(flat, b))

                if len(gathered_points) == 0:
                    start = end
                    continue

                flat_idx = torch.cat(gathered_points, dim=0)             # [K]
                z_vals  = torch.cat(gathered_z, dim=0)                   # [K]
                b_idx   = torch.cat(gathered_b, dim=0)                   # [K]

                td_flat = target_depth.view(B, -1)     # [B, Ht*Wt]
                tc_flat = target_count.view(B, -1)     # [B, Ht*Wt]

                for b in range(B):
                    sel_b = (b_idx == b)
                    if not sel_b.any():
                        continue
                    fb = flat_idx[sel_b]   # [Kb] flattened pixel indices (0..Ht*Wt-1)
                    zb = z_vals[sel_b]     # [Kb]

                    # Fast path if available (PyTorch >= 1.12/2.x)
                    try:
                        td_flat[b].scatter_reduce_(0, fb, zb, reduce='amin', include_self=True)
                        tc_flat[b].scatter_add_(0, fb, torch.ones_like(fb, dtype=tc_flat.dtype))
                        continue
                    except Exception:
                        pass

                    # Fallback: sort + segment-min
                    sort_idx = torch.argsort(fb)
                    fb = fb[sort_idx]; zb = zb[sort_idx]
                    # boundaries where index changes
                    head = torch.ones_like(fb, dtype=torch.bool)
                    head[1:] = fb[1:] != fb[:-1]
                    grp_ids = head.cumsum(0) - 1                      # [Kb], 0..G-1
                    G = int(grp_ids[-1].item()) + 1 if fb.numel() else 0
                    # compute min per group
                    mins = torch.full((G,), float('inf'), device=zb.device, dtype=zb.dtype)
                    mins = mins.scatter_reduce(0, grp_ids, zb, reduce='amin', include_self=True)
                    uniq_fb = fb[head]                                 # first occurrence per group
                    # write into target
                    existing = td_flat[b, uniq_fb]
                    td_flat[b, uniq_fb] = torch.minimum(existing, mins)
                    tc_flat[b, uniq_fb] += 1
                start = end

        # Replace inf by 0 (no hit)
        target_depth = torch.where(torch.isinf(target_depth), torch.zeros_like(target_depth), target_depth)
        return target_depth, target_count


    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Rays → 5 query points using fused target depth
    # ──────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def query_points_from_fused_depth(
        self,
        ray_o: torch.Tensor,          # [B, N, 3]
        ray_d: torch.Tensor,          # [B, N, 3]
        target_depth: torch.Tensor,   # [B, 1, Ht, Wt] (from fuse_input_depths_to_target)
        target_K: torch.Tensor,       # [B,3,3] or [1,3,3]
        target_R: torch.Tensor,       # [B,3,3] or [1,3,3]
        target_T: torch.Tensor,       # [B,3,1] or [1,3,1]
        target_hw: tuple,             # (Ht, Wt)
        offsets_m: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01),
        align_corners: bool = True,
    ):
        """
        Returns:
            pts5   : [B, N, 5, 3]  5 query points along rays at (depth ± offsets)
            t5     : [B, N, 5]    corresponding meter t-values
            valid  : [B, N]       valid where projected pixel is in-FOV and depth>0
        """
        B, N, _ = ray_o.shape
        Ht, Wt = target_hw
        dev, dtype = ray_o.device, ray_o.dtype

        # Project a point on each ray to get pixel coords in target view
        Pw = ray_o + ray_d
        if target_K.dim() == 2: target_K = target_K[None].expand(B, -1, -1).to(dev, dtype)
        if target_R.dim() == 2: target_R = target_R[None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 2: target_T = target_T[None, :, None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 3 and target_T.shape[-1] != 1: target_T = target_T[..., None]

        Xc = torch.matmul(target_R[:, None, :, :], Pw.unsqueeze(-1)).squeeze(-1) + target_T[:, None, :, 0]  # [B,N,3]
        zc = Xc[..., 2]
        front = zc > 1e-6

        x = Xc[..., 0] / zc.clamp_min(1e-6)
        y = Xc[..., 1] / zc.clamp_min(1e-6)
        fx, fy = target_K[:, 0, 0][:, None], target_K[:, 1, 1][:, None]
        cx, cy = target_K[:, 0, 2][:, None], target_K[:, 1, 2][:, None]
        u = fx * x + cx
        v = fy * y + cy

        # Normalized coords for grid_sample
        if align_corners:
            xu = (2.0 * u / (Wt - 1)) - 1.0
            yv = (2.0 * v / (Ht - 1)) - 1.0
            inb = (xu >= -1.0) & (xu <= 1.0) & (yv >= -1.0) & (yv <= 1.0)
        else:
            xu = (2.0 * (u + 0.5) / Wt) - 1.0
            yv = (2.0 * (v + 0.5) / Ht) - 1.0
            inb = (xu > -1.0) & (xu < 1.0) & (yv > -1.0) & (yv < 1.0)

        grid = torch.stack([xu, yv], dim=-1).view(B, N, 1, 2)  # [B,N,1,2]
        d_samp = F.grid_sample(target_depth, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=align_corners)  # [B,1,N,1]
        d_sapiens = d_samp.view(B, N)
        depth_pos = d_sapiens > 0.0

        valid = front & inb & depth_pos
        d_sapiens = torch.where(valid, d_sapiens, torch.zeros_like(d_sapiens))

        # Build ±1 cm band
        offs = torch.tensor(offsets_m, device=dev, dtype=dtype).view(1, 1, -1)  # [1,1,5]
        t5 = (d_sapiens.unsqueeze(-1) + offs).clamp_min(1e-4)                   # [B,N,5]
        pts5 = ray_o.unsqueeze(2) + t5.unsqueeze(-1) * ray_d.unsqueeze(2)       # [B,N,5,3]
        return pts5, t5, valid


    def prepare_sp_input(self, batch):
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords
    

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map,
                                  pixel_feat_scale, batchify=False):

        image_shape = batch['input_imgs'][0].shape[-2:]
        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)


        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3)
        xyz = repeat_interleave(xyz, input_R.shape[0])
        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        uv = xyz[:, :, :2] / xyz[:, :, 2:]

        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, image_shape,
                                                  uv)

        return pixel_feat


    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):
        """
        Clark: Samples per-point features from a 2D feature map using projected 2D UV coordinates.
        uv: size[3, 6890, 2], (98.4262, 394.4973)
        """

        scale = feat_scale / image_shape
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)    # size[3, 6890, 1, 2], (-0.8015, 0.6471), normalized coordinates
        #pdb.set_trace()

        # Clark: interpolation, 
        # samples: size[3, 6890, 128, 1], (-9.82, 10.5426)
        samples = F.grid_sample(
            feat_map,   # 3, 64, 256, 256
            uv,   # 3, 6890, 1, 2
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]
    


    def render(self, batch):
        """
        Depth-guided NeRF rendering with SAPIENS:

        1) Fuse multi-view input depths into a single target-view depth map.
        2) For each ray, get 5 depth-centered samples around the fused depth
        (with a fallback to near/far sampling if no valid depth).
        3) Push these samples through:
            - xyz positional embedding
            - viewdir embedding
            - sparse 3D volume grid (sp_input + grid_coords)
            - pixel-aligned features (encoder + get_pixel_aligned_feature)
        4) Call self.net to get raw (rgb, sigma) per sample, and volume-render
        with raw2outputs to obtain rgb / acc / depth maps.

        No SMPL geometry is used in this pipeline.
        """
        ray_o = batch['ray_o']  # [B, N, 3]
        ray_d = batch['ray_d']  # [B, N, 3]
        near  = batch['near']   # [B, N] or [B, N, 1]
        far   = batch['far']    # [B, N] or [B, N, 1]

        sh = ray_o.shape            # [B, N, 3]
        B, N, _ = sh
        dev = ray_o.device
        dtype = ray_o.dtype

        # ------------------------------------------------------------------
        # 1) Camera setup
        # ------------------------------------------------------------------
        # Input cameras for SAPIENS depths
        input_K = batch['input_K'].reshape(B, -1, 3, 3)   # [B, V, 3, 3]
        input_R = batch['input_R'].reshape(B, -1, 3, 3)   # [B, V, 3, 3]
        input_T = batch['input_T'].reshape(B, -1, 3, 1)   # [B, V, 3, 1]

        # Target camera: either explicit or default to first input view
        if 'target_K' in batch:
            target_K = batch['target_K']  # [B,3,3] or [1,3,3]
            target_R = batch['target_R']
            target_T = batch['target_T']
        else:
            target_K = input_K[:, 0]      # [B,3,3]
            target_R = input_R[:, 0]      # [B,3,3]
            target_T = input_T[:, 0]      # [B,3,1]

        # Choose target resolution (Ht, Wt) for fused depth
        # If you already know the novel-view resolution, you can store it in batch['target_hw'].
        if 'target_hw' in batch:
            Ht, Wt = batch['target_hw']
        else:
            # Fallback: use resolution of first input image
            Ht, Wt = batch['input_imgs'][0].shape[-2:]

        # ------------------------------------------------------------------
        # 2) Fuse multi-view SAPIENS depths into target-view z-buffer
        # ------------------------------------------------------------------
        # Use time-step 0 depths/masks for now: [T, B, V, 1, Hs, Ws] -> [B, V, 1, Hs, Ws]
        input_depths = batch['input_depths'][0]  # [B, V, 1, Hs, Ws]
        if 'input_masks' in batch:
            input_masks = batch['input_masks'][0]  # [B, V, 1, Hs, Ws]
        else:
            # Fallback: everything with depth > 0 is valid
            input_masks = (input_depths > 0).to(input_depths.dtype)

        target_depth, target_count = self.fuse_input_depths_to_target(
            input_depths=input_depths,     # [B, V, 1, Hs, Ws]
            input_masks=input_masks,       # [B, V, 1, Hs, Ws]
            input_K=input_K,               # [B, V, 3, 3]
            input_R=input_R,               # [B, V, 3, 3]
            input_T=input_T,               # [B, V, 3, 1]
            target_K=target_K,             # [B, 3, 3] or [1,3,3]
            target_R=target_R,             # [B, 3, 3] or [1,3,3]
            target_T=target_T,             # [B, 3, 1] or [1,3,1]
            target_hw=(Ht, Wt),
            chunked=True,
            max_points=1_000_000,
        )
        # target_depth: [B,1,Ht,Wt]

        # ------------------------------------------------------------------
        # 3) Depth-centered sampling: 5 points per ray
        # ------------------------------------------------------------------
        pts5, t5, valid = self.query_points_from_fused_depth(
            ray_o=ray_o,                  # [B, N, 3]
            ray_d=ray_d,                  # [B, N, 3]
            target_depth=target_depth,    # [B, 1, Ht, Wt]
            target_K=target_K,            # [B, 3, 3] or [1,3,3]
            target_R=target_R,            # [B, 3, 3] or [1,3,3]
            target_T=target_T,            # [B, 3, 1] or [1,3,1]
            target_hw=(Ht, Wt),
            offsets_m=(-0.01, -0.005, 0.0, 0.005, 0.01),
            align_corners=True,
        )
        # pts5: [B, N, 5, 3], t5: [B, N, 5], valid: [B, N]

        # Fallback sampling (near/far) for rays without fused depth
        near = near.to(dtype=dtype).view(B, N)
        far  = far.to(dtype=dtype).view(B, N)

        t_lin = torch.linspace(0.0, 1.0, steps=5, device=dev, dtype=dtype)[None, None, :]  # [1,1,5]
        z_fallback = near[..., None] * (1.0 - t_lin) + far[..., None] * t_lin              # [B,N,5]

        valid_exp = valid.unsqueeze(-1)   # [B,N,1]

        # Combine SAPIENS depth-centered samples with near/far fallback
        z_vals = torch.where(valid_exp, t5, z_fallback)                      # [B, N, 5]
        pts_fallback = ray_o.unsqueeze(2) + z_fallback.unsqueeze(-1) * ray_d.unsqueeze(2)  # [B,N,5,3]
        pts = torch.where(valid_exp.unsqueeze(-1), pts5, pts_fallback)      # [B, N, 5, 3]

        # We'll use `pts` as world-space sample positions; no SMPL canonicalization.

        # ------------------------------------------------------------------
        # 4) Positional & view-direction embeddings
        # ------------------------------------------------------------------
        xyz = pts  # [B, N, 5, 3]

        # Position encoding (e.g. NeRF positional encoding)
        light_pts = embedder.xyz_embedder(xyz)  # [B, N, 5, xyz_dim]
        light_pts = light_pts.view(B, -1, embedder.xyz_dim)  # [B, N*5, xyz_dim]

        # View direction encoding
        viewdir = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)       # [B, N, 3]
        viewdir = embedder.view_embedder(viewdir)                       # [B, N, view_dim]
        viewdir = viewdir[:, :, None, :].expand(B, N, xyz.shape[2], -1) # [B, N, 5, view_dim]
        viewdir = viewdir.contiguous().view(B, -1, embedder.view_dim)   # [B, N*5, view_dim]

        # ------------------------------------------------------------------
        # 5) Sparse 3D volume grid (world-scene bounding box)
        # ------------------------------------------------------------------
        sp_input = self.prepare_sp_input(batch)
        grid_coords = self.get_grid_coords(xyz, sp_input, batch)     # [B, N, 5, 3]
        grid_coords = grid_coords.view(B, -1, 3)                     # [B, N*5, 3]

        # ------------------------------------------------------------------
        # 6) Encoder + pixel-aligned feature maps (unchanged)
        # ------------------------------------------------------------------
        image_list = batch['input_imgs']  # e.g. [T, B, V, C, Hs, Ws]
        # Use time-step 0 for pixel-aligned features (same as original code)
        images0 = image_list[0].reshape(-1, *image_list[0].shape[2:])  # [B*V, C, Hs, Ws]

        # Encoder returns holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale
        # We ignore SMPL-related "holder" features and keep pixel-aligned features.
        _, _, pixel_feat_map, pixel_feat_scale = self.net.encoder(images0)

        # ------------------------------------------------------------------
        # 7) Run NeRF network: either all rays at once or chunked
        # ------------------------------------------------------------------
        if ray_o.size(1) <= 2048:
            pixel_feat = self.get_pixel_aligned_feature(
                batch, xyz, pixel_feat_map, pixel_feat_scale
            )  # pixel_feat: [B * n_input, C_img, N_points]

            raw = self.net(
                pixel_feat,   # [B * n_input, C_img, N_points]
                viewdir,      # [B, N_points, view_dim]
                light_pts,    # [B, N_points, xyz_dim] (only B is used in forward)
            )
        else:
            raw = self.batchify_rays(
                sp_input=sp_input,
                grid_coords=grid_coords,
                viewdir=viewdir,
                light_pts=light_pts,
                chunk=1024 * 32,
                batch=batch,
                xyz=xyz,
                pixel_feat_map=pixel_feat_map,
                pixel_feat_scale=pixel_feat_scale,
            )

        # if ray_o.size(1) <= 2048:
        #     # xyz: [B, N, 5, 3]
        #     pixel_feat = self.get_pixel_aligned_feature(
        #         batch, xyz, pixel_feat_map, pixel_feat_scale
        #     )  # [B, N*5, C_feat]
        #     pdb.set_trace()
        #     raw = self.net(
        #         pixel_feat,   # [B, N*5, C_feat]
        #         sp_input,
        #         grid_coords,  # [B, N*5, 3]
        #         viewdir,      # [B, N*5, view_dim]
        #         light_pts,    # [B, N*5, xyz_dim]
        #     )
        # else:
        #     raw = self.batchify_rays(
        #         sp_input=sp_input,
        #         grid_coords=grid_coords,
        #         viewdir=viewdir,
        #         light_pts=light_pts,
        #         chunk=1024 * 32,
        #         batch=batch,
        #         xyz=xyz,
        #         pixel_feat_map=pixel_feat_map,
        #         pixel_feat_scale=pixel_feat_scale,
        #     )

        # ------------------------------------------------------------------
        # 8) Volume rendering with raw2outputs
        # ------------------------------------------------------------------
        # raw: [B, N*5, 4] → [B*N, 5, 4]
        raw = raw.reshape(-1, z_vals.size(2), 4)
        z_vals_flat = z_vals.view(-1, z_vals.size(2))   # [B*N, 5]
        ray_d_flat  = ray_d.view(-1, 3)                 # [B*N, 3]

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals_flat, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
        )

        # Reshape back to [B, N, ...]
        rgb_map   = rgb_map.view(*sh[:-1], -1)  # [B, N, 3]
        acc_map   = acc_map.view(*sh[:-1])      # [B, N]
        depth_map = depth_map.view(*sh[:-1])    # [B, N]

        ret = {
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            # Optionally expose fused depth for debugging:
            # 'fused_depth': target_depth,
            # 'fused_count': target_count,
        }

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()

        return ret


    # def batchify_rays(
    #     self,
    #     sp_input,
    #     grid_coords,
    #     viewdir,
    #     light_pts,
    #     chunk=1024 * 32,
    #     batch=None,
    #     xyz=None,
    #     pixel_feat_map=None,
    #     pixel_feat_scale=None,
    # ):
    #     """
    #     Render rays in smaller minibatches to avoid OOM.

    #     xyz:          [B, N, Ns, 3]
    #     grid_coords:  [B, N*Ns, 3]
    #     viewdir:      [B, N*Ns, view_dim]
    #     light_pts:    [B, N*Ns, xyz_dim]
    #     pixel_feat_*: encoder feature maps (same as in non-batchified path)

    #     No SMPL / holder features here.
    #     """
    #     all_ret = []

    #     # Flatten xyz once: [B, N*Ns, 3]
    #     xyz_shape = xyz.shape
    #     xyz_flat = xyz.reshape(xyz_shape[0], -1, 3)

    #     B, M, _ = xyz_flat.shape  # M = N * Ns

    #     for i in range(0, M, chunk):
    #         j = min(i + chunk, M)

    #         # Pixel-aligned features for this chunk
    #         pixel_feat = self.get_pixel_aligned_feature(
    #             batch,
    #             xyz_flat[:, i:j],     # [B, m, 3]
    #             pixel_feat_map,
    #             pixel_feat_scale,
    #             batchify=True,
    #         )  # [B, m, C_feat]

    #         ret = self.net(
    #             pixel_feat,                 # [B, m, C_feat]
    #             sp_input,
    #             grid_coords[:, i:j],        # [B, m, 3]
    #             viewdir[:, i:j],            # [B, m, view_dim]
    #             light_pts[:, i:j],          # [B, m, xyz_dim]
    #         )
    #         all_ret.append(ret)

    #     all_ret = torch.cat(all_ret, dim=1)  # [B, N*Ns, 4]
    #     return all_ret
    def batchify_rays(
        self,
        sp_input,
        grid_coords,
        viewdir,
        light_pts,
        chunk=1024 * 32,
        batch=None,
        xyz=None,
        pixel_feat_map=None,
        pixel_feat_scale=None,
    ):
        """
        Render rays in minibatches.

        New rules:
        - DO NOT pass sp_input or grid_coords to the network (your new Network no longer uses them)
        - Only pass: pixel_feat, viewdir_slice, light_pts_slice
        - Match the same input signature as the non-batchified path
        """
        all_ret = []

        # Flatten xyz: [B, N*Ns, 3]
        B, N, Ns, _ = xyz.shape
        xyz_flat = xyz.reshape(B, -1, 3)   # [B, M, 3]
        _, M, _ = xyz_flat.shape           # total points = N * Ns

        for i in range(0, M, chunk):
            j = min(i + chunk, M)

            # Get pixel-aligned features for this chunk
            pixel_feat = self.get_pixel_aligned_feature(
                batch,
                xyz_flat[:, i:j],      # [B, m, 3]
                pixel_feat_map,
                pixel_feat_scale,
                batchify=True,
            )  # → [B, m, C_feat]  BUT Network expects [B*n_input, C_feat, m]

            # To match Network.forward:
            #   pixel_feat must be [B*n_input, C_feat, m]
            #   viewdir_slice = [B, m, view_dim]
            #   light_pts_slice = [B, m, xyz_dim]
            ret = self.net(
                pixel_feat,            # [B*n_input, C_img, m]
                viewdir[:, i:j],       # [B, m, view_dim]
                light_pts[:, i:j],     # [B, m, xyz_dim]
            )

            all_ret.append(ret)

        # Concatenate along ray dimension
        all_ret = torch.cat(all_ret, dim=1)   # [B, N*Ns, 4]
        return all_ret
