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
        B, V, _, Hs, Ws = input_depths.shape
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

                # # Reduce by (b, flat) taking minimum z (z-buffer)
                # # Try fast scatter_reduce_ if available
                # try:
                #     td_flat = target_depth.view(B, -1)
                #     td_flat.scatter_reduce_(1, flat_idx.unsqueeze(1), z_vals.unsqueeze(1),
                #                             reduce='amin', include_self=True)
                #     # Count (not strictly necessary, but useful diagnostics)
                #     tc_flat = target_count.view(B, -1)
                #     one = torch.ones_like(z_vals, dtype=tc_flat.dtype)
                #     tc_flat.scatter_add_(1, flat_idx.unsqueeze(1), one.unsqueeze(1))
                # except Exception:
                #     # Fallback: sort-based segment min per batch
                #     for b in range(B):
                #         sel_b = (b_idx == b)
                #         if not sel_b.any(): continue
                #         fb = flat_idx[sel_b]
                #         zb = z_vals[sel_b]
                #         sort_idx = torch.argsort(fb)
                #         fb = fb[sort_idx]; zb = zb[sort_idx]
                #         # group mins
                #         diff = torch.ones_like(fb, dtype=torch.bool)
                #         diff[1:] = fb[1:] != fb[:-1]
                #         group_ids = torch.cumsum(diff, dim=0) - 1
                #         # compute mins per group
                #         # (simple loop to be robust without extra deps)
                #         unique_fb = []
                #         mins = []
                #         cur_gid = int(group_ids[0].item()) if fb.numel() > 0 else -1
                #         cur_min = float('inf')
                #         prev_gid = cur_gid
                #         for i in range(fb.numel()):
                #             gid = int(group_ids[i].item())
                #             if gid != prev_gid:
                #                 unique_fb.append(prev_flat.item())
                #                 mins.append(cur_min)
                #                 cur_min = float('inf')
                #             cur_min = zb[i].min(cur_min)
                #             prev_gid = gid
                #             prev_flat = fb[i]
                #         if fb.numel() > 0:
                #             unique_fb.append(prev_flat.item())
                #             mins.append(cur_min)
                #         if len(unique_fb) > 0:
                #             unique_fb = torch.tensor(unique_fb, device=dev, dtype=torch.long)
                #             mins = torch.tensor(mins, device=dev, dtype=dtype)
                #             td_flat = target_depth.view(B, -1)
                #             existing = td_flat[b, unique_fb]
                #             td_flat[b, unique_fb] = torch.minimum(existing, mins)
                #             target_count.view(B, -1)[b, unique_fb] += 1
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
        
        # sanity check
        if not os.path.exists("./sanity_check/uv_vis"):
            os.makedirs("./sanity_check/uv_vis")

        for i in range(min(3, input_R.shape[0])):  # visualize a few input views
            uvi = uv[i].detach().cpu().numpy()
            img = batch['input_imgs'][0][0, i].detach().cpu().permute(1,2,0).numpy()
            H, W = img.shape[:2]
            u = np.clip(uvi[:, 0], 0, W-1).astype(np.int32)
            v = np.clip(uvi[:, 1], 0, H-1).astype(np.int32)
            vis = img.copy()
            vis[v, u] = [1.0, 0, 0]  # red marks where query points project
            cv2.imwrite(f"./sanity_check/uv_vis/view{i:02d}_projections.png",
                        (vis[..., ::-1]*255).astype(np.uint8))

        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

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


    def batchify_rays(self, *, sp_input, grid_coords, viewdir, light_pts,
                  chunk, net_c, batch, xyz, pixel_feat_map, pixel_feat_scale, holder=None):
        """
        Expects:
        viewdir     : [B, N, Dv]
        light_pts   : [B, N, Dx]
        grid_coords : [B, N, 3]
        xyz         : [B, N, 3]  or [B, Nr, S, 3]  (WORLD coords of query points)
        Returns:
        raw         : [B, N, 4]
        """
        B = viewdir.shape[0]
        # --- normalize shapes ---
        if xyz.dim() == 4 and xyz.shape[-1] == 3:
            xyz = xyz.view(B, -1, 3)
        elif xyz.dim() == 3 and xyz.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"batchify_rays: expected xyz [...,3], got {tuple(xyz.shape)}")

        N = xyz.shape[1]
        assert viewdir.shape[:2] == (B, N), f"viewdir {viewdir.shape} vs N={N}"
        assert light_pts.shape[:2] == (B, N), f"light_pts {light_pts.shape} vs N={N}"
        assert grid_coords.shape == (B, N, 3), f"grid_coords {grid_coords.shape} vs {(B,N,3)}"

        outs = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xyz_c   = xyz[:, s:e, :].contiguous()
            vd_c    = viewdir[:, s:e, :].contiguous()
            lp_c    = light_pts[:, s:e, :].contiguous()
            gc_c    = grid_coords[:, s:e, :].contiguous()

            # sample pixel-aligned features for THIS chunk
            pf_c = self.get_pixel_aligned_feature(
                batch, xyz_c, pixel_feat_map, pixel_feat_scale, batchify=False
            )  # [B*n_input, C_img, e-s]

            # run the network on THIS chunk
            raw_c = self.net(
                pf_c,          # [B*n_input, C_img, e-s]
                vd_c,          # [B, e-s, Dv]
                lp_c,          # [B, e-s, Dx]
                holder=None,
            )                  # -> [B, e-s, 4]

            outs.append(raw_c)
            del xyz_c, vd_c, lp_c, gc_c, pf_c, raw_c
            torch.cuda.empty_cache()

        return torch.cat(outs, dim=1)  # [B, N, 4]

    
    def render(self, batch):
        dev = batch["ray_o"].device
        ray_o = batch["ray_o"]           # [B,Nr,3]
        ray_d = batch["ray_d"]           # [B,Nr,3]
        B, Nr = ray_o.shape[:2]
        V = batch["input_K"].shape[1]

        # --- gather inputs ---
        K = batch["input_K"].to(dev)                 # [B,V,3,3]
        R = batch["input_R"].to(dev)                 # [B,V,3,3]
        T = batch["input_T"].to(dev)                 # [B,V,3,1]
        depths = batch["input_depths"][0].to(dev)    # [1,3,H,W]
        depths = depths.unsqueeze(2)                 # [1,3,1,H,W]
        depths = torch.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
        imgs = batch["input_imgs"][0].to(dev)        # [B,V,3,H,W]
        masks = batch["input_msks"][0].to(dev)

        H, W = batch['input_imgs'][0].shape[-2:]
        
        # new
        Ht, Wt = imgs.shape[-2:]  # or your desired target resolution

        # 1) Fuse input views into the target view depth
        target_depth, _ = self.fuse_input_depths_to_target(
            input_depths=depths,
            input_masks=masks,
            input_K=K,
            input_R=R,
            input_T=T,
            target_K=batch['target_K'].to(dev),
            target_R=batch['target_R'].to(dev),
            target_T=batch['target_T'].to(dev),
            target_hw=(Ht, Wt),
        )
        #self.save_depth_as_image(target_depth, prefix="fused_depth")

        # 2) Sample 5 query points per target ray (±1 cm around fused depth)
        pts5, t5, valid = self.query_points_from_fused_depth(
            ray_o=ray_o,
            ray_d=ray_d,
            target_depth=target_depth,
            target_K=batch['target_K'].to(dev),
            target_R=batch['target_R'].to(dev),
            target_T=batch['target_T'].to(dev),
            target_hw=(Ht, Wt),
        )
        self.save_pts5_as_ply(pts5, prefix="query_pts5")
        #pdb.set_trace()
        # pts5 → feed into self.net instead of SMPL-driven samples.



                # ===== Encode input views (no SMPL holder) =====
        # Each time step produces pixel-aligned feature maps
        image_list = batch['input_imgs']
        pixel_feat_map = None
        pixel_feat_scale = None
        for t in range(getattr(cfg, "time_steps", 1)):
            imgs_t = image_list[t].reshape(-1, *image_list[t].shape[2:])  # [B*V,3,H,W]
            _, _, pfm, pfs = self.net.encoder(imgs_t)
            if pixel_feat_map is None:
                pixel_feat_map, pixel_feat_scale = pfm, pfs

        # ===== Pixel-aligned features at 5 depth-centered query points =====
        # get_pixel_aligned_feature uses batch["coord"] internally to locate per-pixel features
        pixel_feat = self.get_pixel_aligned_feature(
            batch, pts5, pixel_feat_map, pixel_feat_scale
        )  # → [B * n_input, C_img, N]  (already correct for self.net)
        #pdb.set_trace()

        # ======= shapes & embeddings on flattened N = Nr*S =======
        B, Nr, S, _ = pts5.shape
        N = Nr * S

        # world-space query points flattened
        xyz_flat = pts5.view(B, N, 3).contiguous()                  # [B,N,3]

        # viewdir: [B,N, Dv]
        vd = ray_d / (torch.norm(ray_d, dim=2, keepdim=True) + 1e-8)  # [B,Nr,3]
        vd = vd.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, N, 3)
        viewdir = embedder.view_embedder(vd)                        # [B,N,Dv]

        # light_pts (positional embed of world xyz): [B,N, Dx]
        light_pts = embedder.xyz_embedder(xyz_flat)                 # [B,N,Dx]

        # sparse grid input & canonical coords -> [B,N,3]
        sp_input   = self.prepare_sp_input(batch)
        pts_can    = self.pts_to_can_pts(pts5, batch)               # [B,Nr,S,3]
        grid_coords = self.get_grid_coords(pts_can, sp_input, batch).view(B, N, 3)

        # ======= choose chunked or direct path (reuse original batchify_rays idea) =======
        chunk_N = getattr(cfg, "net_chunk", 1024 * 32)
        #pdb.set_trace()

        if N > chunk_N:
            # batchify_rays is expected to:
            #   - slice along N by 'chunk'
            #   - call get_pixel_aligned_feature(...) on xyz_chunk
            #   - call self.net(pixel_feat_chunk, viewdir_chunk, light_pts_chunk, holder=None)
            #   - concat per-chunk raw along N
            raw = self.batchify_rays(
                sp_input=sp_input,
                grid_coords=grid_coords,     # [B,N,3]
                viewdir=viewdir,             # [B,N,Dv]
                light_pts=light_pts,         # [B,N,Dx]
                chunk=chunk_N,
                net_c=None,
                batch=batch,
                xyz=xyz_flat,                # [B,N,3]  (world-space query pts)
                pixel_feat_map=pixel_feat_map,
                pixel_feat_scale=pixel_feat_scale,
                holder=None,                 # no SMPL
            )                                # → [B,N,4]
        else:
            # direct path: sample pixel features for all N, then one net call
            pixel_feat = self.get_pixel_aligned_feature(
                batch, xyz_flat, pixel_feat_map, pixel_feat_scale, batchify=False
            )                                # [B*n_input, C_img, N]

            raw = self.net(
                pixel_feat,                  # [B*n_input, C_img, N]
                viewdir,                     # [B, N, Dv]
                light_pts,                   # [B, N, Dx]
                holder=None,
            )                                # → [B, N, 4]

        # ======= volume compositing with your depth-centered samples t5 =======
        raw   = raw.view(B * Nr, S, 4)                # [B*Nr, S, 4]
        z_vals = t5.view(B * Nr, S)                   # [B*Nr, S]
        rays_d = ray_d.view(B * Nr, 3)                # [B*Nr, 3]

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, cfg.raw_noise_std, cfg.white_bkgd
        )

        # back to [B, Nr, ...]
        rgb_map   = rgb_map.view(B, Nr, -1)
        acc_map   = acc_map.view(B, Nr)
        depth_map = depth_map.view(B, Nr)

        if getattr(cfg, "run_mode", "") == "test":
            gc.collect(); torch.cuda.empty_cache()
        #return ret
        
        return {"rgb_map": rgb_map, "acc_map": acc_map, "depth_map": depth_map}
    
    # sanity check helpers
    @torch.no_grad()
    def save_depth_as_image(self, depth_tensor, prefix="target_depth", save_dir="./sanity_check", cmap=True):
        """
        Save depth map(s) as grayscale or Jet-colored PNG images.

        Args:
            depth_tensor : torch.Tensor [B,1,H,W] or [B,H,W]
            prefix       : filename prefix, e.g. 'target_depth'
            save_dir     : output directory (created if missing)
            cmap         : if True, apply Jet colormap
        """
        os.makedirs(save_dir, exist_ok=True)
        depth = depth_tensor.detach().cpu()

        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]        # [B,H,W]
        elif depth.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected depth shape {tuple(depth.shape)}")

        B, H, W = depth.shape
        for b in range(B):
            d = depth[b].numpy()
            valid_mask = d > 1e-6
            if not np.any(valid_mask):
                print(f"[warn] no valid pixels in depth[{b}]")
                continue
            d_valid = d[valid_mask]
            d_min, d_max = np.percentile(d_valid, [1, 99])  # robust normalization
            d_norm = np.clip((d - d_min) / (d_max - d_min + 1e-8), 0, 1)
            img = (d_norm * 255).astype(np.uint8)
            if cmap:
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            out_path = os.path.join(save_dir, f"{prefix}_{b:02d}.png")
            cv2.imwrite(out_path, img)
            print(f"[saved] {out_path}")



    @torch.no_grad()
    def save_pts5_as_ply(self, pts5, prefix="query_pts5", save_dir="./sanity_check"):
        """
        Save the 5-sample query points (world-space) as a PLY point cloud.

        Args:
            pts5      : torch.Tensor [B, Nr, 5, 3] (world coords in meters)
            prefix    : filename prefix (e.g. "query_pts5")
            save_dir  : directory to save .ply files
        """
        os.makedirs(save_dir, exist_ok=True)
        pts = pts5.detach().cpu().numpy()   # [B,Nr,5,3]
        B = pts.shape[0]

        for b in range(B):
            pts_b = pts[b].reshape(-1, 3)   # flatten [Nr*5,3]
            valid = np.isfinite(pts_b).all(axis=1)
            pts_b = pts_b[valid]

            ply_path = os.path.join(save_dir, f"{prefix}_{b:02d}.ply")
            with open(ply_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {pts_b.shape[0]}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                np.savetxt(f, pts_b, fmt="%.6f %.6f %.6f")

            print(f"[saved] {ply_path}  ({pts_b.shape[0]} points)")