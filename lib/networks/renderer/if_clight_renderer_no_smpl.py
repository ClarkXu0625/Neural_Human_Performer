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
        self.use_depth_smoothing = True

        # color weight modes: angle_softmax, cos_power, uniform
        self.color_weight_mode = getattr(cfg, "color_weight_mode", "angle_soft_max")
        self.color_weight_beta = getattr(cfg, "color_weight_beta", 10.0)
        self.color_weight_gamma = getattr(cfg, "color_weight_gamma", 2.0)
        self.save_depth = getattr(cfg, "save_depth", True)

        # NEW: whether we use SMPL-X depth to fill holes in fused depth
        self.use_smpl_depth = getattr(cfg, "use_smpl_depth", True)

        # # NEW: load SMPL faces from the same OBJ you used for visibility
        # obj_path = getattr(cfg, "smpl_obj_path", "data/smplx/smpl/smpl_uv_neutral.obj")
        # # Keep faces on CPU; we’ll move to the correct device on use.
        # self.smpl_faces = load_smpl_faces(obj_path, device=torch.device('cpu'))

    # ──────────────────────────────────────────────────────────────────────────────
    # 0) (Optional) SMPL-depth at target view
    # ──────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_smpl_depth_at_target_view(
        self,
        smpl_vertices: torch.Tensor,  # [B,Nv,3]
        smpl_faces: torch.Tensor,     # [F,3]
        target_K: torch.Tensor,
        target_R: torch.Tensor,
        target_T: torch.Tensor,
        target_hw: tuple,
    ) -> torch.Tensor:
        """
        Render SMPL depth from the target camera using the SAME projection
        pipeline as compute_visibility_on_the_fly.

        Returns:
            smpl_depth: [B, 1, Ht, Wt] camera-space z (meters),
                        0 where SMPL is not visible.
        """
        #assert self.smpl_faces is not None, "smpl_faces must be set on Renderer"

        Ht, Wt = target_hw           # H, W (PyTorch image convention)
        dev = smpl_vertices.device
        dtype = smpl_vertices.dtype
        B, Nv, _ = smpl_vertices.shape

        # --- broadcast camera params to [B,...] ---
        if target_K.dim() == 2:
            target_K = target_K[None].expand(B, -1, -1)
        if target_R.dim() == 2:
            target_R = target_R[None].expand(B, -1, -1)
        if target_T.dim() == 2:
            target_T = target_T[None, :, None].expand(B, -1, -1)
        elif target_T.dim() == 3 and target_T.shape[-1] != 1:
            target_T = target_T[..., None]

        target_K = target_K.to(dev, dtype)
        target_R = target_R.to(dev, dtype)
        target_T = target_T.to(dev, dtype)

        faces = smpl_faces.to(dev)        # [F,3], same as in visibility code
        if faces.dim() == 4:
            faces = faces[0]
        if faces.dim() == 3 and faces.shape[-1] == 1:
            faces = faces[..., 0]
        smpl_depth = torch.zeros((B, 1, Ht, Wt), device=dev, dtype=dtype)

        for b in range(B):
            verts_world = smpl_vertices[b]     # [Nv,3]
            Rb = target_R[b]                   # [3,3]
            Tb = target_T[b, :, 0]             # [3]
            Kb = target_K[b]                   # [3,3]

            # ---- 4x4 extrinsic, same as compute_visibility_on_the_fly ----
            extr = torch.eye(4, device=dev, dtype=dtype)
            extr[:3, :3] = Rb
            extr[:3, 3] = Tb

            # ---- near/far based on world verts + extrinsics ----
            near, far = compute_near_far(verts_world, extr[:3])

            fx, fy = Kb[0, 0], Kb[1, 1]
            cx, cy = Kb[0, 2], Kb[1, 2]

            P_clip, translate = perspective_projection_opencv_to_opengl(
                fx, fy, cx, cy,
                near=near,
                far=far,
                width=Wt,
                height=Ht,
                device=dev
            )

            # ---- world → clip (same helper as visibility) ----
            pos_clip = world_to_clip_space(
                verts_world,   # [Nv,3]
                extr[:3],      # [3,4]
                P_clip,        # [4,4]
                translate      # [4]
            )                   # [Nv,4]
            pos_clip = pos_clip.unsqueeze(0).contiguous()  # [1,Nv,4]

            # ---- camera-space z as attribute ----
            verts_cam = (Rb @ verts_world.T).T + Tb        # [Nv,3]
            z_cam = verts_cam[:, 2:3]                      # [Nv,1]
            z_attr = z_cam.unsqueeze(0)                    # [1,Nv,1]

            # ---- rasterize (same style as visibility) ----
            # nvdiffrast uses (height, width) here; we want Ht, Wt
            rast, _ = dr.rasterize(self.glctx, pos_clip, faces, (Ht, Wt))  # [1,Ht,Wt,4]

            # interpolate depth
            z_img, _ = dr.interpolate(z_attr, rast, faces)  # [1,Ht,Wt,1]
            z_img = z_img[0, ..., 0]                        # [Ht,Wt]

            hit = rast[0, ..., 3] > 0                       # [Ht,Wt]
            depth_b = torch.where(hit, z_img, torch.zeros_like(z_img))

            # ---- fix upside-down (OpenGL y-up → image y-down) ----
            depth_b = torch.flip(depth_b, dims=[0])         # flip vertically

            smpl_depth[b, 0] = depth_b

        return smpl_depth



    

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

        #self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth_before_smoothing.png")
        # if self.use_depth_smoothing:
        #     target_depth = self.gaussian_smooth_depth(
        #         target_depth,
        #         target_count,
        #         ksize=5,
        #         sigma=1.0,
        #     )
        #self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth_after_smoothing.png")

        return target_depth, target_count


    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Rays → 5 query points using fused target depth
    # ──────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def query_points_from_fused_depth(
        self,
        ray_o: torch.Tensor,          # [B, N, 3] world
        ray_d: torch.Tensor,          # [B, N, 3] world (assumed normalized or close)
        target_depth: torch.Tensor,   # [B, 1, Ht, Wt] (from fuse_input_depths_to_target)
        target_K: torch.Tensor,       # [B,3,3] or [1,3,3]
        target_R: torch.Tensor,       # [B,3,3] or [1,3,3]  (world -> cam)
        target_T: torch.Tensor,       # [B,3,1] or [1,3,1]
        target_hw: tuple,             # (Ht, Wt)
        offsets_m: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01),
        align_corners: bool = True,
    ):
        """
        Geometry-consistent version:

        - target_depth encodes camera-space z at the surface: z_c^*
        - We compute per-ray t_hit such that:
            z_c^* = z0 + t_hit * dz
        where z0 is the origin's z in camera frame, dz is ray_dir_cam.z
        - Then we sample along the ray in world space:
            pts5 = ray_o + (t_hit + offsets) * ray_d

        Returns:
            pts5   : [B, N, 5, 3]  query points along rays
            t5     : [B, N, 5]    corresponding t-values
            valid  : [B, N]       depth>0, ray in front, in-FOV
        """
        B, N, _ = ray_o.shape
        Ht, Wt = target_hw
        dev, dtype = ray_o.device, ray_o.dtype

        # ─────────────────────────────────────────────────────────
        # 1) Broadcast target intrinsics/extrinsics to [B,...]
        # ─────────────────────────────────────────────────────────
        if target_K.dim() == 2:
            target_K = target_K[None].expand(B, -1, -1).to(dev, dtype)
        if target_R.dim() == 2:
            target_R = target_R[None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 2:
            target_T = target_T[None, :, None].expand(B, -1, -1).to(dev, dtype)
        if target_T.dim() == 3 and target_T.shape[-1] != 1:
            target_T = target_T[..., None]

        # ─────────────────────────────────────────────────────────
        # 2) Compute pixel coords (u,v) for each ray in target view
        #    We can use any point along the ray; use Pw = ray_o + ray_d.
        # ─────────────────────────────────────────────────────────
        Pw = ray_o + ray_d                                     # [B,N,3] world
        Xc = torch.matmul(target_R[:, None, :, :],
                        Pw.unsqueeze(-1)).squeeze(-1)        # [B,N,3] cam
        Xc = Xc + target_T[:, None, :, 0]                      # add T
        zc = Xc[..., 2]                                        # [B,N]
        front_dir = zc > 1e-6

        x = Xc[..., 0] / zc.clamp_min(1e-6)
        y = Xc[..., 1] / zc.clamp_min(1e-6)
        fx = target_K[:, 0, 0][:, None]
        fy = target_K[:, 1, 1][:, None]
        cx = target_K[:, 0, 2][:, None]
        cy = target_K[:, 1, 2][:, None]
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

        # ─────────────────────────────────────────────────────────
        # 3) Sample fused depth (camera z) at those (u,v)
        # ─────────────────────────────────────────────────────────
        d_samp = F.grid_sample(
            target_depth, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners
        )  # [B,1,N,1]
        d_sapiens = d_samp.view(B, N)                          # camera z
        depth_pos = d_sapiens > 0.0

        # ─────────────────────────────────────────────────────────
        # 4) Convert camera z → ray parameter t_hit
        #    Use per-ray origin and direction in camera coordinates.
        # ─────────────────────────────────────────────────────────
        # ray origin in camera frame
        ray_o_cam = torch.matmul(
            target_R[:, None, :, :],
            ray_o.unsqueeze(-1)
        ).squeeze(-1) + target_T[:, None, :, 0]               # [B,N,3]

        # ray direction in camera frame
        ray_d_cam = torch.matmul(
            target_R[:, None, :, :],
            ray_d.unsqueeze(-1)
        ).squeeze(-1)                                         # [B,N,3]

        z0 = ray_o_cam[..., 2]                                # [B,N]
        dz = ray_d_cam[..., 2]                                # [B,N]

        # rays that actually go forward in camera z
        forward = dz > 1e-6

        # Solve: d_sapiens = z0 + t_hit * dz  → t_hit = (d_sapiens - z0) / dz
        eps = 1e-6
        t_hit = (d_sapiens - z0) / dz.clamp_min(eps)          # [B,N]

        # valid if:
        #  - ray points forward,
        #  - depth > 0,
        #  - pixel is inside view,
        #  - t_hit > 0
        valid = forward & depth_pos & inb & (t_hit > 0)

        # For invalid rays, set t_hit to a tiny positive fallback to avoid NaNs
        t_hit = torch.where(valid, t_hit,
                            torch.full_like(t_hit, 1e-4))

        # ─────────────────────────────────────────────────────────
        # 5) Build ±offset band around t_hit and sample points in world
        # ─────────────────────────────────────────────────────────
        offs = torch.tensor(offsets_m, device=dev, dtype=dtype).view(1, 1, -1)  # [1,1,5]
        t5 = (t_hit.unsqueeze(-1) + offs).clamp_min(1e-4)                        # [B,N,5]

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
        # humanid = batch.get('obj_id').item() or -1
        # print(f"rendering human object {humanid}")
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
        input_depths = torch.nan_to_num(input_depths, nan=0.0, posinf=0.0, neginf=0.0)
        input_depths = input_depths.unsqueeze(2) 

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
        if self.save_depth:
            self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth_before.png")

        # post process depth by (1) combine with smpl-x depth (2) smooth by gaussian filter
        if self.use_smpl_depth:
            smpl_vertices = batch['smpl_vertice']  # [B, Nv, 3]
            smpl_faces = batch['smpl_faces']

            smpl_depth = self.get_smpl_depth_at_target_view(
                smpl_vertices=smpl_vertices[0],
                smpl_faces = smpl_faces,
                target_K=target_K,
                target_R=target_R,
                target_T=target_T,
                target_hw=(Ht, Wt),
            )  # [B,1,Ht,Wt]

            # Holes = pixels with no fused SAPIENS depth
            hole_mask = (target_depth <= 0.0) | (target_count == 0)

            # Fill only the holes with SMPL depth
            target_depth = torch.where(hole_mask, smpl_depth, target_depth)


            # Build a combined count: SAPIENS count, plus 1 where SMPL filled a hole
            combined_count = target_count.clone()
            smpl_valid = (smpl_depth > 0)
            combined_count[hole_mask & smpl_valid] = 1

            k_size = 15
            sigma=2.0
            target_depth_smoothed = self.gaussian_smooth_depth(
                target_depth,
                combined_count, #target_count,
                ksize=k_size,
                sigma=sigma,
            )

            if self.save_depth: # save smple_depth
                self.viz_depth_map(smpl_depth, fname="sanity_check/smpl_depth.png")
                self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth_after_combine.png")
                filename=f"sanity_check/smoothing+combine_k{k_size}_sigma{sigma}.png"
                self.viz_depth_map(target_depth_smoothed, fname=filename)
            target_depth = target_depth_smoothed

        # if self.use_depth_smoothing:
        #     target_depth = self.gaussian_smooth_depth(
        #         target_depth,
        #         target_count,
        #         ksize=5,
        #         sigma=1.0,
        #     )

        # save depth after combining with smpl-x and smoothing
        # if self.save_depth:
        #     self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth_after_smoothing_and_combine.png")

        #pdb.set_trace()
        # Sanity check: visualize fused depth for batch 0
        #if cfg.run_mode == "test" and getattr(cfg, "debug_depth_vis", False):
        # self.viz_depth_map(target_depth, fname="sanity_check/fused_target_depth.png")
        # ------------------------------------------------------------------
        # DEBUG: save fused SAPIENS→target depth map (eval only)
        # ------------------------------------------------------------------
        if cfg.run_mode == "test":
            # Same result root layout as Evaluator
            result_root = os.path.join(
                cfg.result_dir,
                "epoch_" + str(cfg.test.epoch),
                cfg.exp_folder_name,
            )
            debug_dir = os.path.join(result_root, "debug_fused_depth")
            os.makedirs(debug_dir, exist_ok=True)

            # Use the same naming convention: human, frame, view
            human_idx = int(batch.get("human_idx", torch.tensor(-1)).item())
            frame_idx = int(batch.get("frame_index", torch.tensor(-1)).item())
            view_idx  = int(batch.get("cam_ind", torch.tensor(-1)).item())

            base = f"human{human_idx}_frame{frame_idx}_view{view_idx}_fused_depth"
            png_path = os.path.join(debug_dir, base + ".png")
            npy_path = os.path.join(debug_dir, base + ".npy")

            # Use your viz_depth_map for the PNG
            # target_depth: [B,1,Ht,Wt], we pass it directly
            self.viz_depth_map(target_depth, fname=png_path)

            # Save raw fused depth (first batch element) as numpy
            depth_np = target_depth[0, 0].detach().cpu().numpy().astype("float32")
            np.save(npy_path, depth_np)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 3) Depth-centered sampling: 5 points per ray
        # ------------------------------------------------------------------
        query_distances = np.linspace(-0.075, 0.075, 31)
        #query_distances = np.linspace(-0.035, 0.035, 15)
        pts5, t5, valid = self.query_points_from_fused_depth(
            ray_o=ray_o,                  # [B, N, 3]
            ray_d=ray_d,                  # [B, N, 3]
            target_depth=target_depth,    # [B, 1, Ht, Wt]
            target_K=target_K,            # [B, 3, 3] or [1,3,3]
            target_R=target_R,            # [B, 3, 3] or [1,3,3]
            target_T=target_T,            # [B, 3, 1] or [1,3,1]
            target_hw=(Ht, Wt),
            #offsets_m=(-0.01, -0.005, 0.0, 0.005, 0.01),
            offsets_m=query_distances,
            align_corners=True,
        )
        # pts5: [B, N, 5, 3], t5: [B, N, 5], valid: [B, N]
        if self.save_depth:
            self.save_pts5_as_ply(pts5, prefix="query_pts5")

        # # Fallback sampling (near/far) for rays without fused depth
        # near = near.to(dtype=dtype).view(B, N)
        # far  = far.to(dtype=dtype).view(B, N)

        # t_lin = torch.linspace(0.0, 1.0, steps=5, device=dev, dtype=dtype)[None, None, :]  # [1,1,5]
        # z_fallback = near[..., None] * (1.0 - t_lin) + far[..., None] * t_lin              # [B,N,5]

        # valid_exp = valid.unsqueeze(-1)   # [B,N,1]

        # # Combine SAPIENS depth-centered samples with near/far fallback
        # z_vals = torch.where(valid_exp, t5, z_fallback)                      # [B, N, 5]
        # pts_fallback = ray_o.unsqueeze(2) + z_fallback.unsqueeze(-1) * ray_d.unsqueeze(2)  # [B,N,5,3]
        # pts = torch.where(valid_exp.unsqueeze(-1), pts5, pts_fallback)      # [B, N, 5, 3]
        # Fallback sampling (near/far) for rays without fused depth)
        near = near.to(dtype=dtype).view(B, N)
        far  = far.to(dtype=dtype).view(B, N)
        t_lin = torch.linspace(0.0, 1.0, steps=5, device=dev, dtype=dtype)[None, None, :]
        z_fallback = near[..., None] * (1.0 - t_lin) + far[..., None] * t_lin

        valid_exp = valid.unsqueeze(-1)   # [B,N,1]

        # If you really don't want fallback samples in the geometry:
        z_vals = t5.clone()          # [B,N,5]
        pts    = pts5.clone()        # [B,N,5,3]


        # We'll use `pts` as world-space sample positions; no SMPL canonicalization.

        # ------------------------------------------------------------------
        # 4) Positional & view-direction embeddings
        # ------------------------------------------------------------------
        xyz = pts  # [B, N, 5, 3]
        #self.save_pts5_as_ply(xyz, prefix="query_pts")

        # Position encoding (e.g. NeRF positional encoding)
        light_pts = embedder.xyz_embedder(xyz)  # [B, N, 5, xyz_dim]
        light_pts = light_pts.view(B, -1, embedder.xyz_dim)  # [B, N*5, xyz_dim]

        # View direction encoding
        viewdir = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)       # [B, N, 3]
        viewdir = embedder.view_embedder(viewdir)                       # [B, N, view_dim]
        viewdir = viewdir[:, :, None, :].expand(B, N, xyz.shape[2], -1) # [B, N, 5, view_dim]
        viewdir = viewdir.contiguous().view(B, -1, embedder.view_dim)   # [B, N*5, view_dim]

        # ------------------------------------------------------------------
        # 5) Sparse 3D volume grid (world-scene bounding box, unused)
        # ------------------------------------------------------------------
        # sp_input = self.prepare_sp_input(batch)
        # grid_coords = self.get_grid_coords(xyz, sp_input, batch)     # [B, N, 5, 3]
        # grid_coords = grid_coords.view(B, -1, 3)                     # [B, N*5, 3]

        # ------------------------------------------------------------------
        # 6) Encoder + pixel-aligned feature maps (unchanged)
        # ------------------------------------------------------------------
        image_list = batch['input_imgs']  # e.g. [T, B, V, C, Hs, Ws]
        # Use time-step 0 for pixel-aligned features (same as original code)
        images0 = image_list[0].reshape(-1, *image_list[0].shape[2:])  # [B*V, C, Hs, Ws]

        # Encoder returns holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale
        # We ignore SMPL-related "holder" features and keep pixel-aligned features.
        _, _, pixel_feat_map, pixel_feat_scale = self.net.encoder(images0)

        # pixel_feat_map shape is [B*V, C, Hs, Ws]


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
                #grid_coords=grid_coords,
                viewdir=viewdir,
                light_pts=light_pts,
                chunk=1024 * 32,
                batch=batch,
                xyz=xyz,
                pixel_feat_map=pixel_feat_map,
                pixel_feat_scale=pixel_feat_scale,
            )


        # # ------------------------------------------------------------------
        # # 6) Encoder + pixel-aligned feature maps (unchanged)
        # # ------------------------------------------------------------------
        # image_list = batch['input_imgs']  # [T, B, V, C, Hs, Ws]
        # #pdb.set_trace()
        # B_img, V, C_img, Hs, Ws = image_list[0].shape
        # assert B_img == B, "Batch size mismatch between rays and input_imgs"

        # # time-step 0 images for both encoder and IBR
        # images0_flat = image_list[0].reshape(-1, C_img, Hs, Ws)  # [B*V, C, Hs, Ws]
        # _, _, pixel_feat_map, pixel_feat_scale = self.net.encoder(images0_flat)
        # # pixel_feat_map: [B*V, C_feat, Hs, Ws]

        # # Also keep a [B, V, C, Hs, Ws] view for IBR
        # input_imgs0 = image_list[0].view(B, V, C_img, Hs, Ws)

        # # ------------------------------------------------------------------
        # # 6.1 IBR color for each sample point (no gradients needed)
        # # ------------------------------------------------------------------
        # rgb_ibr_flat = self.compute_ibr_colors(
        #     xyz=xyz,                    # [B, N, 5, 3]
        #     ray_d=ray_d,                # [B, N, 3]
        #     input_imgs0=input_imgs0,    # [B, V, C, Hs, Ws]
        #     input_K=input_K,            # [B, V, 3, 3]
        #     input_R=input_R,            # [B, V, 3, 3]
        #     input_T=input_T,            # [B, V, 3, 1]
        # )                                # -> [B, N*5, 3]

        # # ------------------------------------------------------------------
        # # 7) Run NeRF network: predict density (and ignore its RGB)
        # # ------------------------------------------------------------------
        # if ray_o.size(1) <= 2048:
        #     pixel_feat = self.get_pixel_aligned_feature(
        #         batch, xyz, pixel_feat_map, pixel_feat_scale
        #     )  # pixel_feat: [B * n_input, C_img, N_points]

        #     raw = self.net(
        #         pixel_feat,   # [B * n_input, C_img, N_points]
        #         viewdir,      # [B, N_points, view_dim]
        #         light_pts,    # [B, N_points, xyz_dim]
        #     )               # [B, N_points, 4]
        # else:
        #     raw = self.batchify_rays(
        #         viewdir=viewdir,
        #         light_pts=light_pts,
        #         chunk=1024 * 32,
        #         batch=batch,
        #         xyz=xyz,
        #         pixel_feat_map=pixel_feat_map,
        #         pixel_feat_scale=pixel_feat_scale,
        #     )               # [B, N_points, 4]

        # # raw: [B, N*5, 4]  (rgb_pred, sigma)
        # # Replace RGB with IBR color but keep sigma from the network
        # sigma = raw[..., 3:4]                 # [B, N*5, 1]
        # raw = torch.cat([rgb_ibr_flat, sigma], dim=-1)  # [B, N*5, 4]

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
        nerf_sigma = torch.relu(raw[..., 3])      # [N_rays, N_samples]
        nerf_z_vals = z_vals.clone()              # [N_rays, N_samples]

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
            B, Nrays = sh[0], sh[1]
            N_samples = nerf_sigma.shape[1]
            ret['nerf_sigma'] = nerf_sigma.view(B, Nrays, N_samples)
            ret['nerf_z_vals'] = nerf_z_vals.view(B, Nrays, N_samples)

        return ret


    def batchify_rays(
        self,
        #grid_coords,
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

    def viz_depth_map(self, depth, fname="debug_depth.png", vmin=None, vmax=None):
        """
        depth: torch.Tensor
            - Either [B, 1, H, W] (image-like) 
            - Or [B, N] (per-ray depth; we will reshape if needed).
        Only uses batch index 0.
        """
        depth = depth.detach().float().cpu()

        if depth.ndim == 4:
            # [B, 1, H, W]
            d0 = depth[0, 0]  # [H, W]
        elif depth.ndim == 3 and depth.size(1) == 1:
            # [B, 1, N] -> not common, but handle
            # try to guess square-ish shape
            B, _, N = depth.shape
            side = int(N ** 0.5)
            d0 = depth[0, 0, :side * side].view(side, side)
        elif depth.ndim == 2:
            # [B, N] → try to reshape into (H, W) using target_hw if you have it
            B, N = depth.shape
            # naive guess: square
            side = int(N ** 0.5)
            d0 = depth[0, :side * side].view(side, side)
        else:
            raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

        d0 = d0.clone()
        valid = d0 > 0

        if valid.any():
            d_valid = d0[valid]
            lo = d_valid.min().item()
            hi = d_valid.max().item()
            # normalize valid region to [0,1]
            # if hi > lo:
            #     d0[valid] = (d0[valid] - lo) / (hi - lo)
        else:
            # nothing valid; just zeros
            d0[:] = 0.0

        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)

        plt.figure(figsize=(5, 5))
        plt.imshow(d0.numpy(), cmap="magma", vmin=vmin, vmax=vmax)
        plt.colorbar(label="depth (meters)")
        plt.title(os.path.basename(fname))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    @torch.no_grad()
    def gaussian_smooth_depth(
        self,
        depth: torch.Tensor,        # [B, 1, H, W], meters
        count: torch.Tensor = None, # [B, 1, H, W], optional hit-count
        ksize: int = 5,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to depth while preventing bleeding into invalid regions.

        Strategy:
        - Work in numpy on CPU (OpenCV).
        - Use a binary valid mask (depth>0 and optionally count>0).
        - Blur both depth and mask.
        - Renormalize: depth_blur = depth_blur / mask_blur where mask_blur>eps.
        - Keep invalid pixels at 0.
        """
        assert depth.dim() == 4 and depth.shape[1] == 1, "expected [B,1,H,W] depth"

        B, _, H, W = depth.shape
        dev = depth.device
        dtype = depth.dtype

        depth_np = depth.detach().cpu().numpy()  # [B,1,H,W]
        if count is not None:
            count_np = count.detach().cpu().numpy()
        else:
            count_np = None

        out_np = np.zeros_like(depth_np, dtype=np.float32)

        # OpenCV kernel size must be odd
        if ksize % 2 == 0:
            ksize = ksize + 1
        k = (ksize, ksize)
        eps = 1e-6

        for b in range(B):
            d = depth_np[b, 0]  # [H,W]
            if count_np is not None:
                c = count_np[b, 0]
                valid = (d > 0) & (c > 0)
            else:
                valid = (d > 0)

            if not np.any(valid):
                # no valid depth; leave zeros
                continue

            d_valid = d.copy()
            d_valid[~valid] = 0.0

            mask = valid.astype(np.float32)

            # Blur depth and mask
            d_blur   = cv2.GaussianBlur(d_valid, k, sigmaX=sigma, sigmaY=sigma)
            m_blur   = cv2.GaussianBlur(mask,   k, sigmaX=sigma, sigmaY=sigma)

            # Renormalize to avoid fading near boundaries
            # where m_blur is very small, keep depth 0
            safe = m_blur > eps
            d_norm = np.zeros_like(d_blur, dtype=np.float32)
            d_norm[safe] = d_blur[safe] / (m_blur[safe] + eps)

            # Only keep depth in originally valid regions (optional but safer)
            d_norm[~valid] = 0.0

            out_np[b, 0] = d_norm

        out = torch.from_numpy(out_np).to(dev).to(dtype)
        return out
    

    # ----------------------------------------------------------
    # IBR weighting strategies
    # ----------------------------------------------------------
    @torch.no_grad()
    def _weights_angle_softmax(self, cos_theta, valid_mask):
        """
        cos_theta:  [B, V, M]
        valid_mask: [B, V, M] bool
        returns:    [B, V, M] weights sum_v=1
        """
        beta = getattr(self, "color_weight_beta", 10.0)
        logits = beta * cos_theta

        # mask invalid views / pixels
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        w = torch.softmax(logits, dim=1)  # along view dimension
        return w

    @torch.no_grad()
    def _weights_cos_power(self, cos_theta, valid_mask):
        """
        cos^gamma with renormalization.
        """
        gamma = getattr(self, "color_weight_gamma", 2.0)
        w = cos_theta.clamp(min=0.0).pow(gamma)
        if valid_mask is not None:
            w = w * valid_mask.float()

        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w = w / w_sum
        return w

    @torch.no_grad()
    def _weights_uniform(self, cos_theta, valid_mask):
        """
        All valid views equally weighted, invalid = 0.
        """
        if valid_mask is not None:
            w = valid_mask.float()
        else:
            w = torch.ones_like(cos_theta)

        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w = w / w_sum
        return w

    @torch.no_grad()
    def _compute_color_weights(self, cos_theta, valid_mask):
        """
        Dispatch to the selected weighting strategy.
        cos_theta:  [B, V, M]
        valid_mask: [B, V, M] bool
        """
        mode = getattr(self, "color_weight_mode", "angle_softmax")
        if mode == "angle_softmax":
            return self._weights_angle_softmax(cos_theta, valid_mask)
        elif mode == "cos_power":
            return self._weights_cos_power(cos_theta, valid_mask)
        elif mode == "uniform":
            return self._weights_uniform(cos_theta, valid_mask)
        else:
            raise ValueError(f"Unknown color_weight_mode: {mode}")

    @torch.no_grad()
    def compute_ibr_colors(
        self,
        xyz,            # [B, N, S, 3] sample points (S = #samples per ray, e.g. 5)
        ray_d,          # [B, N, 3] novel-view ray directions
        input_imgs0,    # [B, V, C, Hs, Ws] RGB input images (time=0)
        input_K,        # [B, V, 3, 3]
        input_R,        # [B, V, 3, 3] world -> cam
        input_T,        # [B, V, 3, 1]
    ):
        """
        Multi-view image-based color for each sample point.

        Returns:
            rgb_flat : [B, N*S, 3]  (same ordering as xyz.view(B, -1, 3))
        """
        import torch.nn.functional as F  # safe here

        B, N, S, _ = xyz.shape
        _, V, C, Hs, Ws = input_imgs0.shape
        M = N * S

        dev = xyz.device
        dtype = xyz.dtype

        # Flatten samples and ray directions to [B, M, 3]
        pts_flat = xyz.view(B, M, 3)             # [B, M, 3]
        ray_d_flat = ray_d[:, :, None, :].expand(B, N, S, 3).reshape(B, M, 3)
        d_n = F.normalize(ray_d_flat, dim=-1)    # [B, M, 3]

        # Containers over views
        colors_v = []        # list of [B, M, 3]
        cos_theta_v = []     # list of [B, M]
        valid_v = []         # list of [B, M] bool

        for v in range(V):
            Rv = input_R[:, v]          # [B, 3, 3]
            Tv = input_T[:, v]          # [B, 3, 1]
            Kv = input_K[:, v]          # [B, 3, 3]
            img_v = input_imgs0[:, v]   # [B, C, Hs, Ws]

            # ---- world -> camera ----
            Xc = torch.matmul(
                Rv.unsqueeze(1),            # [B,1,3,3]
                pts_flat.unsqueeze(-1)      # [B,M,3,1]
            ) + Tv.unsqueeze(1)             # [B,1,3,1]
            Xc = Xc.squeeze(-1)             # [B,M,3]

            z = Xc[..., 2]                  # [B,M]
            x = Xc[..., 0] / z.clamp_min(1e-6)
            y = Xc[..., 1] / z.clamp_min(1e-6)

            fx = Kv[:, 0, 0][:, None]
            fy = Kv[:, 1, 1][:, None]
            cx = Kv[:, 0, 2][:, None]
            cy = Kv[:, 1, 2][:, None]

            u = fx * x + cx                 # [B,M]
            v_img = fy * y + cy             # [B,M]

            # ---- pixel bounds mask ----
            xu = (2.0 * u / (Ws - 1.0)) - 1.0
            yv = (2.0 * v_img / (Hs - 1.0)) - 1.0

            inb = (xu >= -1.0) & (xu <= 1.0) & (yv >= -1.0) & (yv <= 1.0)
            in_front = z > 1e-6
            valid = inb & in_front          # [B,M]

            # ---- sample image color ----
            grid = torch.stack([xu, yv], dim=-1).view(B, M, 1, 2)  # [B,M,1,2]
            samples = F.grid_sample(
                img_v, grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )                       # [B,C,M,1]
            color = samples[:, :, :, 0].permute(0, 2, 1)  # [B,M,C] -> [B,M,3]

            # ---- view direction from camera center to point ----
            # camera center in world: C = -R^T T
            Cw = -torch.matmul(Rv.transpose(1, 2), Tv).squeeze(-1)  # [B,3]
            d_v = F.normalize(pts_flat - Cw[:, None, :], dim=-1)    # [B,M,3]

            # cos angle between novel view dir and this camera's dir
            cos_theta = (d_n * d_v).sum(dim=-1).clamp(min=0.0)      # [B,M]

            colors_v.append(color)           # [B,M,3]
            cos_theta_v.append(cos_theta)    # [B,M]
            valid_v.append(valid)            # [B,M]

        # Stack over views
        colors = torch.stack(colors_v, dim=1)      # [B, V, M, 3]
        cos_theta = torch.stack(cos_theta_v, dim=1)  # [B, V, M]
        valid_mask = torch.stack(valid_v, dim=1)     # [B, V, M]

        # Compute weights per view
        w = self._compute_color_weights(cos_theta, valid_mask)      # [B, V, M]
        w = w.unsqueeze(-1)                                         # [B, V, M, 1]

        # Weighted sum over views
        rgb_flat = (w * colors).sum(dim=1)                          # [B, M, 3]

        return rgb_flat    # [B, N*S, 3]
    

def get_smpl_vertice(self, human, frame=0):
    obj_path = os.path.join(cfg.virt_data_root, 'smplx_obj', f'{human}.obj')
    smpl_obj = load_obj(obj_path)
    vi = smpl_obj['vi'].astype(np.float32)
    return vi