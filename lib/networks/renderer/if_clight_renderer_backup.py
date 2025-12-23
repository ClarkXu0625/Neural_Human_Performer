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

    @torch.no_grad()
    def get_sapiens_depth_and_query_points(
        self,
        ray_o: torch.Tensor,        # [B, N, 3] world
        ray_d: torch.Tensor,        # [B, N, 3] world
        depth_map: torch.Tensor,    # [B,1,H,W] or [B,H,W]  (TARGET Sapiens depth, meters)
        K: torch.Tensor,            # [B,3,3] or [1,3,3]   (TARGET intrinsics)
        R: torch.Tensor,            # [B,3,3] or [1,3,3]   (TARGET world->cam)
        T: torch.Tensor,            # [B,3,1] or [1,3,1]   (TARGET world->cam)
        image_shape: tuple,         # (H, W)
        offsets_m: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01),
        mask: torch.Tensor = None,  # OPTIONAL [B,1,H,W] or [B,H,W]; if None, validity ignores mask
        align_corners: bool = True,
    ):
        """
        Returns:
            d_sapiens : [B, N]         depth (meters) sampled at each ray's pixel in TARGET view
            valid     : [B, N]         boolean: in front of camera, inside FOV, (mask>0 if provided), depth>0
            pts5      : [B, N, 5, 3]   5 query points at (depth ± 1cm band) along each ray
            t5        : [B, N, 5]      corresponding t values (meters) along each ray

        Notes:
        • Uses bilinear sampling on the target depth map at the projected pixel of each ray.
        • Works for sparse rays; no need for H*W == N.
        • If `mask` is None, validity is computed without it.
        """
        B, N, _ = ray_o.shape
        H, W = image_shape
        dev, dtype = ray_o.device, ray_o.dtype

        # Ensure shapes
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)   # [B,1,H,W]
        depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        use_mask = mask is not None
        if use_mask:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)         # [B,1,H,W]
            mask = mask.to(dtype=depth_map.dtype)

        # Broadcast K,R,T
        if K.dim() == 2: K = K[None].expand(B, -1, -1).to(dev, dtype)
        if R.dim() == 2: R = R[None].expand(B, -1, -1).to(dev, dtype)
        if T.dim() == 2: T = T[None, :, None].expand(B, -1, -1).to(dev, dtype)
        if T.dim() == 3 and T.shape[-1] != 1: T = T[..., None]

        # Pick a point on each ray (t=1 is fine for pixel projection)
        Pw = ray_o + ray_d  # [B,N,3]

        # World→cam
        Xc = torch.matmul(R[:, None, :, :], Pw.unsqueeze(-1)).squeeze(-1) + T[:, None, :, 0]  # [B,N,3]
        zc = Xc[..., 2]                            # [B,N]
        front = zc > 1e-6

        # Project to pixels
        x = Xc[..., 0] / zc.clamp_min(1e-6)
        y = Xc[..., 1] / zc.clamp_min(1e-6)
        fx, fy = K[:, 0, 0][:, None], K[:, 1, 1][:, None]
        cx, cy = K[:, 0, 2][:, None], K[:, 1, 2][:, None]
        u = fx * x + cx
        v = fy * y + cy

        # Normalize to [-1,1] for grid_sample
        if align_corners:
            xu = (2.0 * u / (W - 1)) - 1.0
            yv = (2.0 * v / (H - 1)) - 1.0
            x_ok = (xu >= -1.0) & (xu <= 1.0)
            y_ok = (yv >= -1.0) & (yv <= 1.0)
        else:
            xu = (2.0 * (u + 0.5) / W) - 1.0
            yv = (2.0 * (v + 0.5) / H) - 1.0
            x_ok = (xu > -1.0) & (xu < 1.0)
            y_ok = (yv > -1.0) & (yv < 1.0)

        in_bounds = x_ok & y_ok
        grid = torch.stack([xu, yv], dim=-1).view(B, N, 1, 2)  # [B,N,1,2]

        # Sample target depth (+ optional mask)
        d_samp = F.grid_sample(depth_map, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=align_corners)  # [B,1,N,1]
        d_sapiens = d_samp.view(B, N)  # [B,N]
        depth_pos = d_sapiens > 0.0

        if use_mask:
            m_samp = F.grid_sample(mask, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=align_corners)  # [B,1,N,1]
            m_val = (m_samp.view(B, N) > 0.5)
            valid = front & in_bounds & m_val & depth_pos
        else:
            valid = front & in_bounds & depth_pos

        # Zero-out invalid depths
        d_sapiens = torch.where(valid, d_sapiens, torch.zeros_like(d_sapiens))

        # Build ±1 cm band query points
        offs = torch.tensor(offsets_m, device=dev, dtype=dtype).view(1, 1, -1)  # [1,1,S=5]
        t5 = (d_sapiens.unsqueeze(-1) + offs).clamp_min(1e-4)                   # [B,N,5]
        pts5 = ray_o.unsqueeze(2) + t5.unsqueeze(-1) * ray_d.unsqueeze(2)       # [B,N,5,3]

        return d_sapiens, valid, pts5, t5


    @torch.no_grad()
    def depth_centered_ray_samples(
        self,
        ray_o, ray_d,               # [B, Nr, 3] world
        K, R, T,                    # [B, V, 3, 3], [B, V, 3, 3], [B, V, 3, 1]  (world→cam)
        depth_map,                  # [B, V, H, W] or [B, V, 1, H, W] (meters)
        mask_map=None,              # NEW: [B, V, 1, H, W] or None  (1=fg, 0=bg)
        n_samples: int = 5,
        step_cm: float = 1.0,
        fallback_near: float = 0.5,
        fallback_far: float = 6.0,
        hard_mask_out: bool = False # if True, drop samples where mask==0 instead of fallback
    ):
        """
        Returns:
            pts_world : [B, V, Nr, S, 3]
            t_vals    : [B, V, Nr, S]
            valid     : [B, V, Nr]   (per-ray validity from depth & mask)
            valid_s   : [B, V, Nr, S]  (per-sample validity; includes hard mask if enabled)
        """
        import torch
        import torch.nn.functional as F

        dev = ray_o.device
        B, Nr = ray_o.shape[:2]
        V = K.shape[1]

        # Ensure channel dims
        if depth_map.dim() == 4:           # [B,V,H,W] -> [B,V,1,H,W]
            depth_map = depth_map.unsqueeze(2)
        elif depth_map.dim() == 5 and depth_map.shape[2] != 1:
            depth_map = depth_map[:, :, :1, ...]
        depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        _, _, _, H, W = depth_map.shape

        # Masks (optional)
        if mask_map is not None:
            if mask_map.dim() == 4:        # [B,V,H,W] -> [B,V,1,H,W]
                mask_map = mask_map.unsqueeze(2)
            elif mask_map.dim() == 5 and mask_map.shape[2] != 1:
                mask_map = mask_map[:, :, :1, ...]
            mask_map = mask_map.to(depth_map.dtype)
        else:
            # all ones if not provided
            mask_map = torch.ones_like(depth_map)

        # Expand rays to [B,V,Nr,3]
        ray_o = ray_o[:, None].expand(-1, V, -1, -1).contiguous()
        ray_d = ray_d[:, None].expand(-1, V, -1, -1).contiguous()
        BV = B * V
        ray_o_flat = ray_o.view(BV, Nr, 3)
        ray_d_flat = ray_d.view(BV, Nr, 3)

        # Use extrinsics as world→camera (no flips)
        K_flat = K.view(BV, 3, 3)
        R_flat = R.view(BV, 3, 3)
        T_flat = T.view(BV, 3, 1)
        depth_flat = depth_map.view(BV, 1, H, W)
        mask_flat  = mask_map.view(BV, 1, H, W)

        # Project rays to camera space
        dir_cam = (R_flat @ ray_d_flat.transpose(1, 2)).transpose(1, 2)  # [BV,Nr,3]
        o_cam   = (R_flat @ ray_o_flat.transpose(1, 2) + T_flat).transpose(1, 2)  # [BV,Nr,3]

        # Intrinsics
        fx, fy = K_flat[:, 0, 0][:, None], K_flat[:, 1, 1][:, None]
        cx, cy = K_flat[:, 0, 2][:, None], K_flat[:, 1, 2][:, None]
        dx, dy, dz = dir_cam.unbind(-1)
        dz = dz.clamp_min(1e-6)  # for projection only

        u = fx * (dx / dz) + cx
        v = fy * (dy / dz) + cy
        u_n = (u / (W - 1)) * 2 - 1
        v_n = (v / (H - 1)) * 2 - 1
        grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(1)  # [BV,1,Nr,2]

        # Sample depth (bilinear) & mask (nearest)
        d0 = F.grid_sample(depth_flat, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
        d0 = d0.squeeze(1).squeeze(1)  # [BV,Nr]
        d0 = torch.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)

        m0 = F.grid_sample(mask_flat, grid, mode="nearest",
                        padding_mode="zeros", align_corners=True)
        m0 = m0.squeeze(1).squeeze(1)  # [BV,Nr]
        m0 = torch.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)

        # Valid if has positive depth AND inside mask
        valid = ((d0 > 0.0) & (m0 > 0.5)).view(B, V, Nr)

        # Build the depth band (meters)
        half = n_samples // 2
        offsets = (torch.arange(-half, half + 1, device=dev, dtype=d0.dtype) *
                (step_cm * 0.01))
        d_band = d0[..., None] + offsets[None, None, :]    # [BV,Nr,S]

        # Fallback near→far (meters)
        t_lin = torch.linspace(0, 1, steps=n_samples, device=dev)[None, None, :]
        d_fb = fallback_near * (1 - t_lin) + fallback_far * t_lin  # [1,1,S]

        has_d = (d0 > 0.0)[..., None]          # [BV,Nr,1]
        in_m  = (m0 > 0.5)[..., None]          # [BV,Nr,1]
        ok_ray = has_d & in_m                  # depth+mask valid

        if hard_mask_out:
            # Hard drop: invalidate samples where mask/depth invalid
            d_used = d_band.clone()
            valid_s = ok_ray.expand_as(d_band)        # [BV,Nr,S]
            # optional: keep numbers sane for invalids (won’t be used if you gate later)
            d_used[~valid_s] = 0.0
        else:
            # Soft: fallback to near→far when invalid (keeps rays alive on bg)
            d_used = torch.where(ok_ray, d_band, d_fb) # broadcast [1,1,S]
            valid_s = ok_ray.expand_as(d_band)         # [BV,Nr,S]

        d_used = d_used.clamp_min_(0.0)

        # Solve for t: o_cam.z + t * dir_cam.z = d_used
        # a, b are [BV, Nr]
        a = dir_cam[..., 2]
        b = o_cam[..., 2]
        valid_a = a.abs() > 1e-4                      # [BV, Nr]

        t_vals = torch.zeros_like(d_used)             # [BV, Nr, S]
        t_raw  = (d_used - b[..., None]) / a[..., None]  # [BV, Nr, S]

        # Good where denom is ok and depth positive
        good = (valid_a[..., None] & (d_used > 0.0))  # [BV, Nr, S]
        t_vals[good] = t_raw[good]

        # Fallback for bad 'a' (only in soft mode)
        bad_a = ~valid_a                               # [BV, Nr]
        bad_a_s = bad_a[..., None].expand_as(t_vals)   # [BV, Nr, S]

        t_fb_expanded = d_fb.expand_as(t_vals)         # [BV, Nr, S] via broadcast [1,1,S] → [BV,Nr,S]
        t_vals[bad_a_s] = t_fb_expanded[bad_a_s]

        # keep forward samples, clip outliers
        t_vals = t_vals.clamp_min_(0.0).clamp_max_(fallback_far * 1.5)


        # Back to world
        pts_world = ray_o_flat[..., None, :] + t_vals[..., None] * ray_d_flat[..., None, :]
        pts_world = pts_world.view(B, V, Nr, n_samples, 3)
        t_vals    = t_vals.view(B, V, Nr, n_samples)
        valid_s   = valid_s.view(B, V, Nr, n_samples)

        return pts_world, t_vals, valid, valid_s


    # @torch.no_grad()
    # def depth_centered_ray_samples(
    #     self,
    #     ray_o, ray_d,               # [B, Nr, 3] in world space
    #     K, R, T,                    # [B, V, 3, 3], [B, V, 3, 3], [B, V, 3, 1]
    #     depth_map,                  # [B, V, H, W] (meters)
    #     n_samples: int = 5,
    #     step_cm: float = 1.0,
    #     fallback_near: float = 0.5,
    #     fallback_far: float = 6.0,
    # ):
    #     """
    #     Multi-view version:
    #     Samples 3D query points around per-pixel depth for all V views together.
    #     All in meters.
    #     Returns:
    #         pts_world : [B, V, Nr, S, 3]
    #         t_vals    : [B, V, Nr, S]
    #         valid     : [B, V, Nr]
    #     """
    #     import torch.nn.functional as F
    #     dev = ray_o.device
    #     B, Nr = ray_o.shape[:2]
    #     V = K.shape[1]

    #     # Ensure depth_map shape [B,V,1,H,W]
    #     if depth_map.dim() == 4:
    #         depth_map = depth_map.unsqueeze(2)
    #     elif depth_map.dim() == 5 and depth_map.shape[2] != 1:
    #         depth_map = depth_map[:, :, :1, ...]
    #     depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    #     _, _, _, H, W = depth_map.shape

    #     # Expand rays to [B,V,Nr,3]
    #     ray_o = ray_o[:, None].expand(-1, V, -1, -1).contiguous()
    #     ray_d = ray_d[:, None].expand(-1, V, -1, -1).contiguous()

    #     # Flatten BV for batched matmul
    #     BV = B * V
    #     ray_o_flat = ray_o.view(BV, Nr, 3)
    #     ray_d_flat = ray_d.view(BV, Nr, 3)

    #     K_flat = K.view(BV, 3, 3)
    #     R_flat = R.view(BV, 3, 3)
    #     T_flat = T.view(BV, 3, 1)
    #     depth_flat = depth_map.view(BV, 1, H, W)
        

    #     # --- project rays to camera space ---
    #     dir_cam = (R_flat @ ray_d_flat.transpose(1, 2)).transpose(1, 2)  # [BV,Nr,3]
    #     o_cam   = (R_flat @ ray_o_flat.transpose(1, 2) + T_flat).transpose(1, 2)  # [BV,Nr,3]
    #     fx, fy = K_flat[:, 0, 0][:, None], K_flat[:, 1, 1][:, None]
    #     cx, cy = K_flat[:, 0, 2][:, None], K_flat[:, 1, 2][:, None]
    #     dx, dy, dz = dir_cam.unbind(-1)
    #     dz = dz.clamp_min(1e-6)
    #     u = fx * (dx / dz) + cx
    #     v = fy * (dy / dz) + cy
    #     u_n = (u / (W - 1)) * 2 - 1
    #     v_n = (v / (H - 1)) * 2 - 1
    #     grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(1)              # [BV,1,Nr,2]

    #     # --- sample depth map per ray ---
    #     d0 = F.grid_sample(depth_flat, grid, mode="bilinear",
    #                     padding_mode="zeros", align_corners=True)
    #     d0 = d0.squeeze(1).squeeze(1)                                    # [BV,Nr]
    #     d0 = torch.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)
    #     valid = (d0 > 0).view(B, V, Nr)

    #     # --- sample offsets around depth ---
    #     half = n_samples // 2
    #     offsets = (torch.arange(-half, half + 1, device=dev, dtype=d0.dtype)
    #             * (step_cm * 0.01))                                   # meters
    #     d_band = d0[..., None] + offsets[None, None, :]                  # [BV,Nr,S]

    #     # fallback near-far
    #     t_lin = torch.linspace(0, 1, steps=n_samples, device=dev)[None, None, :]
    #     d_fb = fallback_near * (1 - t_lin) + fallback_far * t_lin
    #     d_used = torch.where((d0 > 0)[..., None], d_band, d_fb).clamp_min_(0.0)

    #     # --- solve t and world points ---
    #     a = dir_cam[..., 2].clamp_min(1e-4)
    #     b = o_cam[..., 2]
    #     t_vals = (d_used - b[..., None]) / a[..., None]                  # [BV,Nr,S]
    #     t_vals = torch.nan_to_num(t_vals, nan=0.0, posinf=0.0, neginf=0.0)
    #     pts_world = ray_o_flat[..., None, :] + t_vals[..., None] * ray_d_flat[..., None, :]
    #     pts_world = pts_world.view(B, V, Nr, n_samples, 3)
    #     t_vals = t_vals.view(B, V, Nr, n_samples)
    #     pdb.set_trace()

    #     return pts_world, t_vals, valid

    # @torch.no_grad()
    # def depth_centered_ray_samples(
    #     self,
    #     ray_o, ray_d,            # [B, Nr, 3] in world space
    #     K, R, T,                 # [B,3,3], [B,3,3], [B,3,1], world→camera, meters
    #     depth_map,               # [B,1,H,W] or [B,H,W], meters
    #     n_samples: int = 5,
    #     step_cm: float = 1.0,
    #     fallback_near: float = 0.5,
    #     fallback_far: float = 6.0,
    # ):
    #     """
    #     Sample 3D query points along each ray, centered on per-pixel depth.
    #     All inputs and outputs are in **meters**.
    #     """
    #     import torch.nn.functional as F

    #     dev = ray_o.device
    #     B, Nr = ray_o.shape[:2]
    #     #pdb.set_trace()

    #     # ---- ensure depth shape [B,1,H,W] ----
    #     if depth_map.dim() == 3:
    #         depth_map = depth_map[:, None, ...]
    #     elif depth_map.dim() == 4 and depth_map.shape[1] != 1:
    #         depth_map = depth_map[:, :1, ...]
    #     depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    #     Bdm, _, H, W = depth_map.shape
    #     assert Bdm == B, f"Depth batch mismatch: {Bdm} vs {B}"

    #     # ---- project rays to (u,v) ----
    #     dir_cam = (R @ ray_d.transpose(1, 2)).transpose(1, 2)       # [B,Nr,3]
    #     o_cam   = (R @ ray_o.transpose(1, 2) + T).transpose(1, 2)   # [B,Nr,3]
    #     fx, fy = K[:, 0, 0][:, None], K[:, 1, 1][:, None]
    #     cx, cy = K[:, 0, 2][:, None], K[:, 1, 2][:, None]
    #     dx, dy, dz = dir_cam.unbind(-1)
    #     dz = dz.clamp_min(1e-6)

    #     u = fx * (dx / dz) + cx
    #     v = fy * (dy / dz) + cy
    #     u_n = (u / (W - 1)) * 2 - 1
    #     v_n = (v / (H - 1)) * 2 - 1
    #     grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(1)         # [B,1,Nr,2]

    #     # ---- sample depth ----
    #     d0 = F.grid_sample(depth_map, grid, mode="bilinear",
    #                     padding_mode="zeros", align_corners=True)
    #     d0 = d0.squeeze(1).squeeze(1)
    #     d0 = torch.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)
    #     valid = d0 > 0

    #     # ---- build local depth band ----
    #     half = n_samples // 2
    #     offsets = (torch.arange(-half, half + 1, device=dev, dtype=d0.dtype)
    #             * (step_cm * 0.01))                             # cm → m
    #     d_band = d0[..., None] + offsets[None, None, :]            # [B,Nr,S]

    #     # ---- fallback near→far ----
    #     t_lin = torch.linspace(0, 1, steps=n_samples, device=dev)[None, None, :]
    #     d_fb = fallback_near * (1 - t_lin) + fallback_far * t_lin
    #     d_used = torch.where(valid[..., None], d_band, d_fb).clamp_min_(0.0)

    #     # ---- solve t and build samples ----
    #     a = dir_cam[..., 2].clamp_min(1e-4)
    #     b = o_cam[..., 2]
    #     t_vals = (d_used - b[..., None]) / a[..., None]             # [B,Nr,S]
    #     t_vals = torch.nan_to_num(t_vals, nan=0.0, posinf=0.0, neginf=0.0)
    #     pts_world = ray_o[..., None, :] + t_vals[..., None] * ray_d[..., None, :]
    #     pts_world = torch.nan_to_num(pts_world, nan=0.0, posinf=0.0, neginf=0.0)

    #     return pts_world, t_vals, valid



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

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)

            # pixel_feat = self.get_pixel_aligned_feature(batch,
            #                                             xyz[:, i:i + chunk],
            #                                             pixel_feat_map,
            #                                             pixel_feat_scale,
            #                                             batchify=True)
            B, Nr, S = xyz.shape[:3]
            pixfeat = self.get_pixel_aligned_feature(
                batch,
                xyz.view(B, Nr*S, 3),
                pixel_feat_map,
                pixel_feat_scale,
                batchify=True
            )  # -> [V?, Nr*S, C] but since we pass current view, will effectively be [1, Nr*S, C]

            # Collapse the view dimension if present, then restore [B,Nr,S,C]
            if pixfeat.dim() == 4 and pixfeat.shape[0] == 1:
                pixfeat = pixfeat[0]
            pixfeat = pixfeat.view(Nr, S, -1)[None, ...].contiguous()  # [1,Nr,S,C]

            ret = self.net(pixfeat, sp_input,
                           grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk],
                           light_pts[:, i:i + chunk],
                           holder=holder)

            all_ret.append(ret)

        all_ret = torch.cat(all_ret, 1)

        return all_ret
    
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
        depths = batch["input_depths"][0].to(dev)    # [B,V,H,W]
        imgs = batch["input_imgs"][0].to(dev)        # [B,V,3,H,W]
        masks = batch["input_msks"][0].to(dev)

        H, W = batch['input_imgs'][0].shape[-2:]
        
        # new
        pdb.set_trace()
        H, W = batch['input_imgs'][0].shape[-2:]

        # If you have TARGET depth in the batch:
        target_depth = batch['target_depth']  # [B,1,H,W]
        d_sapiens, valid, pts5, t5 = self.get_sapiens_depth_and_query_points(
            batch['ray_o'], batch['ray_d'],
            depth_map=target_depth,
            K=batch['target_K'], R=batch['target_R'], T=batch['target_T'],
            image_shape=(H, W),
            mask=None,  # <- not required
        )
        pdb.set_trace()

        # --- depth-centered sampling for all views ---
        pts, t_vals_all, valid, valid_s = self.depth_centered_ray_samples(
            ray_o, ray_d, K, R, T, depths,
            mask_map=masks,
            n_samples=getattr(cfg, "n_depth_samples", 5),
            step_cm=getattr(cfg, "depth_step_cm", 1.0),
            fallback_near=getattr(cfg, "fallback_near_m", 0.5),
            fallback_far=getattr(cfg, "fallback_far_m", 6.0),
        )  # pts:[B,V,Nr,S,3], t_vals_all:[B,V,Nr,S]
        S = pts.shape[3]  # number of samples along each ray

        # --- prepare embeddings ---
        # xyz per view (we'll aggregate later)
        xyz = pts.view(B * V, Nr, S, 3)                       # [B*V, Nr, S, 3]
        light_pts = embedder.xyz_embedder(xyz)                # [B*V, Nr, S, Dxyz]

        # viewdir does NOT depend on the view; we just tile to shape-match then reduce later
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)   # [B, Nr, 3]
        viewdir = embedder.view_embedder(viewdir)                  # [B, Nr, Dv]
        viewdir = viewdir[:, :, None, :].expand(B, Nr, S, viewdir.size(-1))  # [B, Nr, S, Dv]
        viewdir = viewdir[:, None, ...].expand(B, V, Nr, S, viewdir.size(-1)) # [B,V,Nr,S,Dv]
        viewdir = viewdir.reshape(B * V, Nr, S, -1)                 # [B*V, Nr, S, Dv]

        # --- encode all images together (flatten batch × views) ---
        imgs_flat = imgs.view(B * V, 3, imgs.shape[-2], imgs.shape[-1])
        holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(imgs_flat)

        # --- pixel-aligned feature sampling (per-view; keep get_pixel_aligned_feature unchanged) ---
        BV = B * V
        pixel_feat_chunks = []

        def bv_index(b, v): return b * V + v

        for b in range(B):
            for v in range(V):
                idx = bv_index(b, v)

                # xyz for this (b,v): [1, Nr*S, 3] — get_pixel_aligned_feature expects first dim==1
                xyz_bv = pts[b, v].reshape(1, Nr * S, 3)

                # per-view feature map slice
                pfm_bv = pixel_feat_map[idx:idx+1]   # [1, C, Hf, Wf]
                pfs_bv = pixel_feat_scale

                # a tiny batch with only this view, matching callee's expectations
                batch_bv = {
                    "input_R":    batch["input_R"][b:b+1, v:v+1],   # [1,1,3,3]
                    "input_T":    batch["input_T"][b:b+1, v:v+1],   # [1,1,3,1]
                    "input_K":    batch["input_K"][b:b+1, v:v+1],   # [1,1,3,3]
                    "input_imgs": [ batch["input_imgs"][0][b:b+1, v:v+1] ],
                }

                pixel_feat_bv = self.get_pixel_aligned_feature(
                    batch_bv, xyz_bv, pfm_bv, pfs_bv, batchify=False
                )  # -> [1, C, Nr*S]
                pixel_feat_chunks.append(pixel_feat_bv)

        # stack back and aggregate across V: [B*V, C, Nr*S] -> [B, V, C, Nr*S] -> mean over V -> [B, C, Nr*S]
        pixel_feat = torch.cat(pixel_feat_chunks, dim=0)                 # [B*V, C, Nr*S]
        C = pixel_feat.shape[1]
        pixel_feat = pixel_feat.view(B, V, C, Nr * S).mean(dim=1)        # [B, C, Nr*S]

        # --- also aggregate embeddings over views (light_pts/viewdir were tiled per-view)
        # light_pts: [B*V, Nr, S, Dxyz] -> [B,V,Nr,S,Dxyz] -> mean over V -> [B,Nr,S,Dxyz]
        Dxyz = light_pts.shape[-1]
        light_pts = light_pts.view(B, V, Nr, S, Dxyz).mean(dim=1)        # [B, Nr, S, Dxyz]
        # viewdir: same logic
        Dv = viewdir.shape[-1]
        viewdir = viewdir.view(B, V, Nr, S, Dv).mean(dim=1)              # [B, Nr, S, Dv]

        # flatten M = Nr*S for the MLP
        M = Nr * S
        light_pts = light_pts.reshape(B, M, Dxyz)                        # [B, M, Dxyz]
        viewdir   = viewdir.reshape(B, M, Dv)                            # [B, M, Dv]
        pixel_feat = pixel_feat.contiguous()                              # [B, C, M]

        # --- radiance prediction (single target view) ---
        raw = self.net(pixel_feat, viewdir.contiguous(), light_pts.contiguous(), holder=None)  # [B, M, 4] typically
        raw = raw.reshape(B, Nr, S, 4).reshape(B * Nr, S, 4)

        # choose t_vals from a single target camera (rays belong to the target; inputs were sources)
        # use the first view's t-values; or t_vals_all.mean(dim=1) if you prefer symmetric
        t_vals = t_vals_all[:, 0]                                        # [B, Nr, S]
        z_vals = t_vals.reshape(B * Nr, S)

        ray_d_flat = ray_d.reshape(B * Nr, 3)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
        )

        # --- reshape back to expected shapes ---
        rgb_map   = rgb_map.reshape(B, Nr, 3)    # [B, Nr, 3]
        acc_map   = acc_map.reshape(B, Nr)       # [B, Nr]
        depth_map = depth_map.reshape(B, Nr)     # [B, Nr]
        pdb.set_trace()

        return {"rgb_map": rgb_map, "acc_map": acc_map, "depth_map": depth_map}

    # def render(self, batch):
    #     dev = batch["ray_o"].device
    #     ray_o = batch["ray_o"]           # [B,Nr,3]
    #     ray_d = batch["ray_d"]           # [B,Nr,3]
    #     B, Nr = ray_o.shape[:2]
    #     V = batch["input_K"].shape[1]

    #     # --- gather inputs ---
    #     K = batch["input_K"].to(dev)                 # [B,V,3,3]
    #     R = batch["input_R"].to(dev)                 # [B,V,3,3]
    #     T = batch["input_T"].to(dev)                 # [B,V,3,1]
    #     depths = batch["input_depths"][0].to(dev)    # [B,V,H,W]
    #     imgs = batch["input_imgs"][0].to(dev)        # [B,V,3,H,W]
    #     masks = batch["input_msks"][0].to(dev)

    #     # --- depth-centered sampling for all views ---
    #     pts, t_vals, valid, valid_s = self.depth_centered_ray_samples(
    #         ray_o, ray_d, K, R, T, depths,
    #         mask_map=masks,
    #         n_samples=getattr(cfg, "n_depth_samples", 5),
    #         step_cm=getattr(cfg, "depth_step_cm", 1.0),
    #         fallback_near=getattr(cfg, "fallback_near_m", 0.5),
    #         fallback_far=getattr(cfg, "fallback_far_m", 6.0),
    #     )  # pts:[B,V,Nr,S,3], t_vals:[B,V,Nr,S]

    #     S = pts.shape[3]

    #     # --- prepare embeddings ---
    #     xyz = pts.view(B * V, Nr, S, 3)
    #     light_pts = embedder.xyz_embedder(xyz)
    #     viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    #     viewdir = embedder.view_embedder(viewdir)
    #     viewdir = viewdir[:, None, :, :].expand(B, V, Nr, viewdir.size(-1))
    #     viewdir = viewdir.reshape(B * V, Nr, viewdir.size(-1))
    #     viewdir = viewdir[:, :, None].expand(-1, -1, S, -1).contiguous()

    #     # --- encode all images together (flatten batch × views) ---
    #     imgs_flat = imgs.view(B * V, 3, imgs.shape[-2], imgs.shape[-1])
    #     holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(imgs_flat)
    #     #pdb.set_trace()

    #     # --- pixel-aligned feature sampling (per-view, keep get_pixel_aligned_feature unchanged) ---
    #     BV = B * V
    #     pixel_feat_chunks = []

    #     # helper to map (b,v) -> flat index in encoder outputs
    #     def bv_index(b, v): return b * V + v

    #     for b in range(B):
    #         for v in range(V):
    #             idx = bv_index(b, v)

    #             # xyz for this (b,v): [1, Nr*S, 3]  (get_pixel_aligned_feature expects first dim==1)
    #             xyz_bv = pts[b, v].reshape(1, Nr * S, 3)

    #             # slice per-view feature map to keep shapes aligned internally
    #             pfm_bv = pixel_feat_map[idx:idx+1]   # [1, C, Hf, Wf]
    #             pfs_bv = pixel_feat_scale            # pass through (scalar/tuple as your sampler expects)

    #             # build a tiny batch with the same nested structure that the function reads:
    #             # it uses batch['input_imgs'][0].shape[-2:], and reshapes input_{R,T,K}
    #             batch_bv = {
    #                 # only keys used by get_pixel_aligned_feature need to be provided
    #                 "input_R":   batch["input_R"][b:b+1, v:v+1],   # [1,1,3,3] -> reshaped inside
    #                 "input_T":   batch["input_T"][b:b+1, v:v+1],   # [1,1,3,1]
    #                 "input_K":   batch["input_K"][b:b+1, v:v+1],   # [1,1,3,3]
    #                 "input_imgs": [ batch["input_imgs"][0][b:b+1, v:v+1] ]  # keep the [0] indexing contract
    #             }

    #             # call the original function (unchanged)
    #             pixel_feat_bv = self.get_pixel_aligned_feature(
    #                 batch_bv, xyz_bv, pfm_bv, pfs_bv, batchify=False
    #             )  # -> [1, C, Nr*S]

    #             pixel_feat_chunks.append(pixel_feat_bv)

    #     # stack back to [B*V, C, Nr*S] for the downstream network
    #     pixel_feat = torch.cat(pixel_feat_chunks, dim=0)

    #     # --- radiance prediction ---
    #     light_pts = light_pts.view(B * V, -1, embedder.xyz_dim)
    #     viewdir = viewdir.view(B * V, -1, embedder.view_dim)

    #     # new start here---
    #     pixel_feat = pixel_feat.contiguous()
    #     viewdir    = viewdir.contiguous()
    #     light_pts  = light_pts.contiguous()
    #     #new end here ---


    #     raw = self.net(pixel_feat, viewdir, light_pts, holder=None)
    #     # raw = raw.view(B * V * Nr, S, 4)
    #     # z_vals = t_vals.view(B * V * Nr, S)

    #     # new start here ---
    #     raw    = raw.reshape(B * V, Nr, S, 4).reshape(B * V * Nr, S, 4)
    #     z_vals = t_vals.reshape(B * V * Nr, S)

    #     ray_d_flat = ray_d[:, None, :, :].expand(B, V, Nr, 3).reshape(B * V * Nr, 3)

    #     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
    #         raw, z_vals, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
    #     )
    #     #new end here---

    #     ray_d_flat = ray_d[:, None, :, :].expand(B, V, Nr, 3).reshape(B * V * Nr, 3)

    #     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
    #         raw, z_vals, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
    #     )

    #     # --- reshape back ---
    #     rgb_map = rgb_map.view(B, V, Nr, 3)
    #     acc_map = acc_map.view(B, V, Nr)
    #     depth_map = depth_map.view(B, V, Nr)
    #     pdb.set_trace()

    #     return {"rgb_map": rgb_map, "acc_map": acc_map, "depth_map": depth_map}

    # def render(self, batch):
    #     dev = batch["ray_o"].device
    #     ray_o = batch["ray_o"]           # [B,Nr,3]
    #     ray_d = batch["ray_d"]           # [B,Nr,3]
    #     B, Nr = ray_o.shape[:2]
    #     V = batch["input_K"].shape[1]

    #     # --- gather inputs ---
    #     K = batch["input_K"].to(dev)                 # [B,V,3,3]
    #     R = batch["input_R"].to(dev)                 # [B,V,3,3]
    #     T = batch["input_T"].to(dev)                 # [B,V,3,1]
    #     depths = batch["input_depths"][0].to(dev)    # [B,V,H,W]
    #     imgs = batch["input_imgs"][0].to(dev)        # [B,V,3,H,W]
    #     masks = batch["input_msks"][0].to(dev)
    #     #pdb.set_trace()

    #     # --- depth-centered sampling for all views ---
    #     pts, t_vals, valid, valid_s = self.depth_centered_ray_samples(
    #         ray_o, ray_d, K, R, T, depths,
    #         mask_map=masks,
    #         n_samples=getattr(cfg, "n_depth_samples", 5),
    #         step_cm=getattr(cfg, "depth_step_cm", 1.0),
    #         fallback_near=getattr(cfg, "fallback_near_m", 0.5),
    #         fallback_far=getattr(cfg, "fallback_far_m", 6.0),
    #     )  # pts:[B,V,Nr,S,3], t_vals:[B,V,Nr,S]

    #     S = pts.shape[3]

    #     # --- prepare embeddings ---
    #     xyz = pts.view(B * V, Nr, S, 3)
    #     light_pts = embedder.xyz_embedder(xyz)
    #     viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    #     viewdir = embedder.view_embedder(viewdir)
    #     viewdir = viewdir[:, None, :, :].expand(B, V, Nr, viewdir.size(-1))
    #     viewdir = viewdir.reshape(B * V, Nr, viewdir.size(-1))
    #     viewdir = viewdir[:, :, None].expand(-1, -1, S, -1).contiguous()

    #     # --- encode all images together (flatten batch × views) ---
    #     imgs_flat = imgs.view(B * V, 3, imgs.shape[-2], imgs.shape[-1])
    #     holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(imgs_flat)
    #     pdb.set_trace()

    #     # --- pixel-aligned feature sampling (all views at once) ---
    #     pixel_feat = self.get_pixel_aligned_feature(
    #         batch, xyz, pixel_feat_map, pixel_feat_scale
    #     )  # [B*V, C, Nr*S]

    #     # --- radiance prediction ---
    #     light_pts = light_pts.view(B * V, -1, embedder.xyz_dim)
    #     viewdir = viewdir.view(B * V, -1, embedder.view_dim)
    #     raw = self.net(pixel_feat, viewdir, light_pts, holder=None)
    #     raw = raw.view(B * V * Nr, S, 4)
    #     z_vals = t_vals.view(B * V * Nr, S)
    #     ray_d_flat = ray_d[:, None, :, :].expand(B, V, Nr, 3).reshape(B * V * Nr, 3)

    #     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
    #         raw, z_vals, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
    #     )

    #     # --- reshape back ---
    #     rgb_map = rgb_map.view(B, V, Nr, 3)
    #     acc_map = acc_map.view(B, V, Nr)
    #     depth_map = depth_map.view(B, V, Nr)
    #     pdb.set_trace()

    #     return {"rgb_map": rgb_map, "acc_map": acc_map, "depth_map": depth_map}

    # def render(self, batch):
    #     dev = batch["ray_o"].device
    #     ray_o = batch["ray_o"]               # [B, Nr, 3]
    #     ray_d = batch["ray_d"]               # [B, Nr, 3]
    #     B, Nr = ray_o.shape[:2]
    #     n_views = batch["input_K"].shape[1]

    #     # unpack fixed lists once
    #     imgs_all  = batch["input_imgs"][0]   # [B, 13, 3, H, W]
    #     depths_all = batch["input_depths"][0]# [B, 13, H, W]
    #     msks_all  = batch["input_msks"][0]   # [B, 13, 1, H, W]

    #     rgb_out, acc_out, depth_out = [], [], []

    #     for v in range(n_views):
    #         # ----- 1. Select matching camera params & inputs -----
    #         K_v = batch["input_K"][:, v].to(dev)        # [B,3,3]
    #         R_v = batch["input_R"][:, v].to(dev)        # [B,3,3]
    #         T_v = batch["input_T"][:, v].to(dev)        # [B,3,1]
    #         depth_v = depths_all[:, v:v+1].to(dev)      # [B,1,H,W]
    #         img_v   = imgs_all[:, v:v+1].to(dev)        # [B,1,3,H,W]
    #         msk_v   = msks_all[:, v:v+1].to(dev)        # [B,1,1,H,W]

    #         # ----- 2. Sample points around Sapiens depth -----
    #         pts, t_vals, valid = self.depth_centered_ray_samples(
    #             ray_o, ray_d, K_v, R_v, T_v, depth_v,
    #             n_samples=getattr(cfg, "n_depth_samples", 5),
    #             step_cm=getattr(cfg, "depth_step_cm", 1.0),
    #             fallback_near=getattr(cfg, "fallback_near_m", 0.5),
    #             fallback_far=getattr(cfg, "fallback_far_m", 6.0),
    #         )  # pts:[B,Nr,S,3], t_vals:[B,Nr,S]
    #         S = pts.shape[2]
            

    #         # ----- 3. Embed coordinates & viewdirs -----
    #         xyz = pts
    #         light_pts = embedder.xyz_embedder(xyz)
    #         viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    #         viewdir = embedder.view_embedder(viewdir)
    #         viewdir = viewdir[:, :, None].expand(-1, -1, S, -1).contiguous()

    #         light_pts = light_pts.view(B, -1, embedder.xyz_dim)
    #         viewdir   = viewdir.view(B, -1, embedder.view_dim)

    #         # ----- 4. Encode image features for this view -----
    #         # collapse [B,1,3,H,W] → [B,3,H,W]
    #         images_v = img_v.squeeze(1)
    #         holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = \
    #             self.net.encoder(images_v)

    #         # ----- 5. Pixel-aligned features -----
    #         pixel_feat = self.get_pixel_aligned_feature(
    #             batch, xyz, pixel_feat_map, pixel_feat_scale
    #         )  # [V(=n_input), C, Nr*S]
    #         pdb.set_trace()


    #         # ----- 6. Predict RGB + density -----
    #         raw = self.net(pixel_feat, viewdir, light_pts, holder=None)
    #         raw = raw.view(B * Nr, S, 4)
    #         z_vals = t_vals.view(B * Nr, S)
    #         ray_d_flat = ray_d.view(B * Nr, 3)

    #         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
    #             raw, z_vals, ray_d_flat, cfg.raw_noise_std, cfg.white_bkgd
    #         )

    #         rgb_out.append(rgb_map.view(B, Nr, 3))
    #         acc_out.append(acc_map.view(B, Nr))
    #         depth_out.append(depth_map.view(B, Nr))

    #     # ----- 7. Stack results from all views -----
    #     rgb_all   = torch.stack(rgb_out, dim=1)   # [B,13,Nr,3]
    #     acc_all   = torch.stack(acc_out, dim=1)   # [B,13,Nr]
    #     depth_all = torch.stack(depth_out, dim=1) # [B,13,Nr]
    #     pdb.set_trace()


    #     return {"rgb_map": rgb_all, "acc_map": acc_all, "depth_map": depth_all}

