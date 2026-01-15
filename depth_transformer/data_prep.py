# depth_transformer/data_prep.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import pdb


@torch.no_grad()
def fuse_depths_to_target(
    input_depths: torch.Tensor,  # [B,V,H,W] or [B,V,1,H,W]
    input_masks: torch.Tensor | None,  # [B,V,H,W] or [B,V,1,H,W]
    input_K: torch.Tensor,  # [B,V,3,3]
    input_R: torch.Tensor,  # [B,V,3,3]
    input_T: torch.Tensor,  # [B,V,3,1]
    target_K: torch.Tensor, # [B,3,3]
    target_R: torch.Tensor, # [B,3,3]
    target_T: torch.Tensor, # [B,3,1]
    target_hw: tuple[int,int],
):
    """
    Simple z-buffer fusion:
      backproject input pixels -> world -> project to target -> scatter amin into target depth.

    Vectorized across batch (removes Python loop over b).
    Returns:
      target_depth: [B,1,Ht,Wt]
      target_count: [B,1,Ht,Wt] int
    """
    if input_depths.dim() == 5:
        input_depths = input_depths.squeeze(2)  # [B,V,H,W]
    if input_masks is not None and input_masks.dim() == 5:
        input_masks = input_masks.squeeze(2)

    B, V, Hs, Ws = input_depths.shape
    Ht, Wt = target_hw
    dev = input_depths.device
    dtype = input_depths.dtype

    Hw = Ht * Wt

    # flattened buffers for global scatter
    td = torch.full((B * Hw,), float("inf"), device=dev, dtype=dtype)
    tc = torch.zeros((B * Hw,), device=dev, dtype=torch.int32)

    # batch offsets so each batch writes to its own slice in [0, B*Hw)
    b_off = (torch.arange(B, device=dev, dtype=torch.long) * Hw)[:, None]  # [B,1]

    # pixel grid (compute once)
    # Use float32 for projection math stability (optional but usually safer)
    ys, xs = torch.meshgrid(
        torch.arange(Hs, device=dev, dtype=torch.float32),
        torch.arange(Ws, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).view(1, 3, Hs * Ws)  # [1,3,M]
    M = Hs * Ws

    # target intrinsics scalars
    fx_t = target_K[:, 0, 0].to(torch.float32)[:, None]
    fy_t = target_K[:, 1, 1].to(torch.float32)[:, None]
    cx_t = target_K[:, 0, 2].to(torch.float32)[:, None]
    cy_t = target_K[:, 1, 2].to(torch.float32)[:, None]

    # target extrinsics
    Rt = target_R.to(torch.float32)
    Tt = target_T.to(torch.float32)[:, :, 0]  # [B,3]

    for v in range(V):
        d = input_depths[:, v].reshape(B, M)  # [B,M]
        if input_masks is None:
            m = (d > 0)
        else:
            mv = input_masks[:, v].reshape(B, M) > 0
            m = mv & (d > 0)

        if not m.any():
            continue

        K = input_K[:, v].to(torch.float32)
        R = input_R[:, v].to(torch.float32)
        T = input_T[:, v].to(torch.float32)[:, :, 0]  # [B,3]

        # dirs = Kinv @ pix  -> [B,3,M] -> transpose -> [B,M,3]
        Kinv = torch.inverse(K)  # [B,3,3]
        dirs = torch.matmul(Kinv, pix.expand(B, -1, -1)).transpose(1, 2)  # [B,M,3]

        # camera points
        Xc = dirs * d.to(torch.float32).unsqueeze(-1)  # [B,M,3]

        # world: Xw = R^T (Xc - T)
        Xw = torch.matmul(R.transpose(1, 2), (Xc - T[:, None, :]).transpose(1, 2)).transpose(1, 2)  # [B,M,3]

        # target cam: Xt = Rt Xw + Tt
        Xt = torch.matmul(Rt[:, None, :, :], Xw.unsqueeze(-1)).squeeze(-1) + Tt[:, None, :]  # [B,M,3]
        zt = Xt[..., 2]
        front = zt > 1e-6
        keep = m & front
        if not keep.any():
            continue

        x = Xt[..., 0] / zt.clamp_min(1e-6)
        y = Xt[..., 1] / zt.clamp_min(1e-6)

        u = (fx_t * x + cx_t).round().long()
        v_img = (fy_t * y + cy_t).round().long()

        inb = (u >= 0) & (u < Wt) & (v_img >= 0) & (v_img < Ht)
        keep = keep & inb
        if not keep.any():
            continue

        flat = (v_img * Wt + u)  # [B,M] long in [0, Hw)
        flat_global = flat + b_off  # [B,M] long in [0, B*Hw)

        idx = flat_global[keep]               # [K] long
        val = zt[keep].to(dtype)              # [K] depth dtype

        # one scatter per view, no Python loop over b
        td.scatter_reduce_(0, idx, val, reduce="amin", include_self=True)
        tc.scatter_add_(0, idx, torch.ones_like(idx, dtype=tc.dtype))

    td = td.view(B, 1, Ht, Wt)
    tc = tc.view(B, 1, Ht, Wt)
    td = torch.where(torch.isinf(td), torch.zeros_like(td), td)
    return td, tc

# @torch.no_grad()
# def fuse_depths_to_target(
#     input_depths: torch.Tensor,  # [B,V,H,W] or [B,V,1,H,W]
#     input_masks: torch.Tensor | None,  # [B,V,H,W] or [B,V,1,H,W]
#     input_K: torch.Tensor,  # [B,V,3,3]
#     input_R: torch.Tensor,  # [B,V,3,3]
#     input_T: torch.Tensor,  # [B,V,3,1]
#     target_K: torch.Tensor, # [B,3,3]
#     target_R: torch.Tensor, # [B,3,3]
#     target_T: torch.Tensor, # [B,3,1]
#     target_hw: tuple[int,int],
# ):
#     """
#     Simple z-buffer fusion:
#       backproject input pixels -> world -> project to target -> scatter amin into target depth.
#     Returns:
#       target_depth: [B,1,Ht,Wt]
#       target_count: [B,1,Ht,Wt] int
#     """
#     if input_depths.dim() == 5:
#         input_depths = input_depths.squeeze(2)  # [B,V,H,W]
#     if input_masks is not None and input_masks.dim() == 5:
#         input_masks = input_masks.squeeze(2)

#     B, V, Hs, Ws = input_depths.shape
#     Ht, Wt = target_hw
#     dev = input_depths.device
#     dtype = input_depths.dtype

#     td = torch.full((B, Ht * Wt), float("inf"), device=dev, dtype=dtype)
#     tc = torch.zeros((B, Ht * Wt), device=dev, dtype=torch.int32)

#     ys, xs = torch.meshgrid(
#         torch.arange(Hs, device=dev, dtype=dtype),
#         torch.arange(Ws, device=dev, dtype=dtype),
#         indexing="ij",
#     )
#     ones = torch.ones_like(xs)
#     pix = torch.stack([xs, ys, ones], dim=0).view(1, 3, Hs * Ws)  # [1,3,M]
#     M = Hs * Ws

#     for v in range(V):
#         d = input_depths[:, v].reshape(B, M)  # [B,M]
#         if input_masks is None:
#             m = (d > 0)
#         else:
#             m = input_masks[:, v].reshape(B, M) > 0
#             m = m & (d > 0)

#         if not m.any():
#             continue

#         K = input_K[:, v]
#         R = input_R[:, v]
#         T = input_T[:, v]

#         Kinv = torch.inverse(K)                       # [B,3,3]
#         dirs = (Kinv @ pix).transpose(1, 2)          # [B,M,3] (broadcast pix)
#         # dirs = Kinv @ [u,v,1] -> [B,3,M] then transpose to [B,M,3]
#         # correct broadcasting:
#         dirs = torch.matmul(Kinv, pix.expand(B, -1, -1)).transpose(1, 2)  # [B,M,3]

#         # Xc = dirs * z
#         Xc = dirs * d.unsqueeze(-1)                   # [B,M,3]

#         # world: Xw = R^T (Xc - T)
#         Xw = torch.matmul(R.transpose(1, 2), (Xc - T.squeeze(-1)[:, None, :]).transpose(1, 2)).transpose(1, 2)  # [B,M,3]

#         # target cam: Xt = Rt Xw + Tt
#         Xt = torch.matmul(target_R[:, None, :, :], Xw.unsqueeze(-1)).squeeze(-1) + target_T[:, None, :, 0]  # [B,M,3]
#         zt = Xt[..., 2]
#         front = zt > 1e-6
#         keep = m & front
#         if not keep.any():
#             continue

#         x = Xt[..., 0] / zt.clamp_min(1e-6)
#         y = Xt[..., 1] / zt.clamp_min(1e-6)

#         fx = target_K[:, 0, 0][:, None]
#         fy = target_K[:, 1, 1][:, None]
#         cx = target_K[:, 0, 2][:, None]
#         cy = target_K[:, 1, 2][:, None]
#         u = (fx * x + cx).round().long()
#         v_img = (fy * y + cy).round().long()

#         inb = (u >= 0) & (u < Wt) & (v_img >= 0) & (v_img < Ht)
#         keep = keep & inb
#         if not keep.any():
#             continue

#         flat = (v_img * Wt + u)  # [B,M]
#         # scatter per batch
#         for b in range(B):
#             kb = keep[b].nonzero(as_tuple=False).squeeze(1)
#             if kb.numel() == 0:
#                 continue
#             fb = flat[b, kb]
#             zb = zt[b, kb]
#             td[b].scatter_reduce_(0, fb, zb, reduce="amin", include_self=True)
#             tc[b].scatter_add_(0, fb, torch.ones_like(fb, dtype=tc.dtype))

#     td = td.view(B, 1, Ht, Wt)
#     tc = tc.view(B, 1, Ht, Wt)
#     td = torch.where(torch.isinf(td), torch.zeros_like(td), td)
#     return td, tc


@torch.no_grad()
def query_points_from_fused_depth(
    ray_o: torch.Tensor,        # [B,N,3]
    ray_d: torch.Tensor,        # [B,N,3]
    target_depth: torch.Tensor, # [B,1,Ht,Wt] camera-z
    target_K: torch.Tensor,     # [B,3,3]
    target_R: torch.Tensor,     # [B,3,3] world->cam
    target_T: torch.Tensor,     # [B,3,1]
    target_hw: tuple[int,int],
    offsets_m: torch.Tensor,    # [Q] in meters along ray t
    align_corners: bool = True,
    return_grid: bool = True,
):
    B, N, _ = ray_o.shape
    Ht, Wt = target_hw
    dev, dtype = ray_o.device, ray_o.dtype

    # project a point along ray to get (u,v) per ray (same trick you used)
    Pw = ray_o + ray_d  # [B,N,3]
    Xc = torch.matmul(target_R[:, None, :, :], Pw.unsqueeze(-1)).squeeze(-1) + target_T[:, None, :, 0]
    zc = Xc[..., 2]
    x = Xc[..., 0] / zc.clamp_min(1e-6)
    y = Xc[..., 1] / zc.clamp_min(1e-6)
    fx = target_K[:, 0, 0][:, None]
    fy = target_K[:, 1, 1][:, None]
    cx = target_K[:, 0, 2][:, None]
    cy = target_K[:, 1, 2][:, None]
    u = fx * x + cx
    v_img = fy * y + cy

    if align_corners:
        xu = (2.0 * u / (Wt - 1.0)) - 1.0
        yv = (2.0 * v_img / (Ht - 1.0)) - 1.0
        inb = (xu >= -1.0) & (xu <= 1.0) & (yv >= -1.0) & (yv <= 1.0)
    else:
        xu = (2.0 * (u + 0.5) / Wt) - 1.0
        yv = (2.0 * (v_img + 0.5) / Ht) - 1.0
        inb = (xu > -1.0) & (xu < 1.0) & (yv > -1.0) & (yv < 1.0)

    grid = torch.stack([xu, yv], dim=-1).view(B, N, 1, 2)
    d_samp = F.grid_sample(target_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    z_surf = d_samp.view(B, N)  # camera-z at surface
    depth_pos = z_surf > 0

    # ray origin/direction in target camera
    ray_o_cam = torch.matmul(target_R[:, None, :, :], ray_o.unsqueeze(-1)).squeeze(-1) + target_T[:, None, :, 0]
    ray_d_cam = torch.matmul(target_R[:, None, :, :], ray_d.unsqueeze(-1)).squeeze(-1)

    z0 = ray_o_cam[..., 2]
    dz = ray_d_cam[..., 2]
    forward = dz > 1e-6

    t_hit = (z_surf - z0) / dz.clamp_min(1e-6)
    valid = forward & depth_pos & inb & (t_hit > 0)
    t_hit = torch.where(valid, t_hit, torch.full_like(t_hit, 1e-4))

    # t samples
    offs = offsets_m.to(device=ray_o.device, dtype=dtype).view(1, 1, -1)  # [1,1,Q]
    tQ = (t_hit.unsqueeze(-1) + offs).clamp_min(1e-4)            # [B,N,Q]
    ptsQ = ray_o.unsqueeze(2) + tQ.unsqueeze(-1) * ray_d.unsqueeze(2)  # [B,N,Q,3]

    # dump_query_points_to_ply(
    #     ptsQ=ptsQ,
    #     #valid_q=valid,  # if you want coloring
    #     ply_path="sanity_check/depth_transformer_query_points.ply",
    #     center_only=False,   # all Q samples per ray
    #     stride_n=8,          # downsample rays to keep file manageable
    #     max_points=1_000_000
    # )
    #pdb.set_trace()

    if return_grid:
        return ptsQ, tQ, valid, grid
    return ptsQ, tQ, valid
    #return ptsQ, tQ, valid, grid  # grid reused for gt sampling

@torch.no_grad()
def sample_depth_map(depth_b1hw: torch.Tensor, grid_bnhw2: torch.Tensor, align_corners=True):
    """
    depth_b1hw: [B,1,H,W]
    grid_bnhw2: [B,N,1,2] normalized grid coords
    returns:    [B,N] sampled depth
    """
    d = F.grid_sample(depth_b1hw, grid_bnhw2, mode="bilinear",
                      padding_mode="zeros", align_corners=align_corners)  # [B,1,N,1]
    return d.view(depth_b1hw.shape[0], -1)  # [B,N]

@torch.no_grad()
def project_world_to_view_grid(
    pts_world: torch.Tensor,   # [B,N,Q,3]
    K: torch.Tensor,           # [B,V,3,3]
    R: torch.Tensor,           # [B,V,3,3]
    T: torch.Tensor,           # [B,V,3,1]
    H: int, W: int,
    align_corners=True,
):
    """
    Returns:
      grid:  [B,V,N,Q,1,2] normalized coords for grid_sample
      zcam:  [B,V,N,Q] camera-space z of pts in that view
      valid_front: [B,V,N,Q] zcam > 0
      valid_inb:   [B,V,N,Q] inside image bounds (normalized)
    """
    B, N, Q, _ = pts_world.shape
    _, V, _, _ = K.shape
    dev = pts_world.device
    dtype = pts_world.dtype

    # world -> cam
    pts = pts_world[:, None]  # [B,1,N,Q,3]
    Rv = R[:, :, None, None]  # [B,V,1,1,3,3]
    Tv = T[:, :, None, None]  # [B,V,1,1,3,1]
    Xc = torch.matmul(Rv, pts.unsqueeze(-1)).squeeze(-1) + Tv.squeeze(-1)  # [B,V,N,Q,3]

    z = Xc[..., 2]  # [B,V,N,Q]
    valid_front = z > 1e-6

    x = Xc[..., 0] / z.clamp_min(1e-6)
    y = Xc[..., 1] / z.clamp_min(1e-6)

    fx = K[:, :, 0, 0][:, :, None, None]
    fy = K[:, :, 1, 1][:, :, None, None]
    cx = K[:, :, 0, 2][:, :, None, None]
    cy = K[:, :, 1, 2][:, :, None, None]

    u = fx * x + cx
    v = fy * y + cy

    if align_corners:
        xu = (2.0 * u / (W - 1)) - 1.0
        yv = (2.0 * v / (H - 1)) - 1.0
        valid_inb = (xu >= -1.0) & (xu <= 1.0) & (yv >= -1.0) & (yv <= 1.0)
    else:
        xu = (2.0 * (u + 0.5) / W) - 1.0
        yv = (2.0 * (v + 0.5) / H) - 1.0
        valid_inb = (xu > -1.0) & (xu < 1.0) & (yv > -1.0) & (yv < 1.0)

    grid = torch.stack([xu, yv], dim=-1).unsqueeze(-2)  # [B,V,N,Q,1,2]
    return grid, z, valid_front, valid_inb

@torch.no_grad()
def build_dt_batch(
    batch: dict,
    half_window: int,
    step_size: float,
    tau: float = 0.01,          # inlier threshold (meters) - tune later
    min_valid_views: int = 2,
    align_corners: bool = True,
):
    """
    Depth-only DT batch builder.

    Expects batch contains (from your dataloader):
      - ray_o: [B,N,3]
      - ray_d: [B,N,3]
      - input_depths: [T=1, V, H, W] OR [T=1, V, 1, H, W] depending on your collator
      - input_K/input_R/input_T: [V,3,3]/[V,3,3]/[V,3,1] (or batched)
      - target_K/target_R/target_T for target view
      - gt_depth: [H,W] or [B,H,W]  (train only)

    Returns dict with:
      x:        [B*N, Q, F]
      z_samples:[B*N, Q]
      valid_q:  [B*N, Q]
      k_gt:     [B*N] (argmin query index to GT depth)
      z_gt:     [B*N]
    """
    ray_o = batch["ray_o"]   # [B,N,3]
    ray_d = batch["ray_d"]   # [B,N,3]
    B, N, _ = ray_o.shape
    dev = ray_o.device
    dtype = ray_o.dtype

    # ---- cameras ----
    # reshape to [B,V,*,*]
    input_K = batch["input_K"].reshape(B, -1, 3, 3)
    input_R = batch["input_R"].reshape(B, -1, 3, 3)
    input_T = batch["input_T"].reshape(B, -1, 3, 1)

    target_K = batch["target_K"]
    target_R = batch["target_R"]
    target_T = batch["target_T"]

    if target_K.ndim == 2: target_K = target_K[None].expand(B, -1, -1)
    if target_R.ndim == 2: target_R = target_R[None].expand(B, -1, -1)
    if target_T.ndim == 2: target_T = target_T[None, :, None].expand(B, -1, -1)
    if target_T.ndim == 3 and target_T.shape[-1] != 1: target_T = target_T[..., None]

    target_K = target_K.to(dev, dtype)
    target_R = target_R.to(dev, dtype)
    target_T = target_T.to(dev, dtype)

    # ---- input depths at time 0 ----
    d0 = batch["input_depths"][0]  # could be [V,H,W] or [B,V,H,W] depending collator
    if d0.ndim == 3:               # [V,H,W] -> [B,V,1,H,W]
        d0 = d0.unsqueeze(0).expand(B, -1, -1, -1)
    # now [B,V,H,W]
    if d0.ndim == 4:
        d0 = d0.unsqueeze(2)       # [B,V,1,H,W]
    input_depths = torch.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- optional masks ----
    if "input_msks" in batch:
        m0 = batch["input_msks"][0]   # [V,1,H,W] or [B,V,1,H,W]
        if m0.ndim == 4:
            m0 = m0.unsqueeze(0).expand(B, -1, -1, -1, -1)
        input_masks = m0.to(dtype=input_depths.dtype)
    else:
        input_masks = (input_depths > 0).to(input_depths.dtype)

    # infer H/W from input depth
    _, V, _, Hs, Ws = input_depths.shape

    # ---- (A) fuse depths to target ----
    target_depth, target_count = fuse_depths_to_target(
        input_depths=input_depths,
        input_masks=input_masks,
        input_K=input_K, input_R=input_R, input_T=input_T,
        target_K=target_K, target_R=target_R, target_T=target_T,
        target_hw=(Hs, Ws),   # you resized all views to same H/W
        #chunked=True,
        #max_points=1_000_000,
    )  # [B,1,H,W]

    

    # ---- (B) query points around fused depth ----
    Q = 2 * half_window + 1
    offsets = torch.linspace(-half_window * step_size, half_window * step_size, Q, device=dev, dtype=dtype)

    ptsQ, tQ, valid_ray, grid = query_points_from_fused_depth(
        ray_o=ray_o, ray_d=ray_d,
        target_depth=target_depth,
        target_K=target_K, target_R=target_R, target_T=target_T,
        target_hw=(Hs, Ws),
        offsets_m=offsets,  # tensor
        align_corners=align_corners,
    )  # ptsQ: [B,N,Q,3]

    # camera-z in target view for each query
    Xt = torch.matmul(target_R[:, None, None], ptsQ.unsqueeze(-1)).squeeze(-1) + target_T[:, None, None, :, 0]
    z_samples = Xt[..., 2]  # [B,N,Q]

    # ---- (C) depth-consistency across input views ----
    grid_v, z_v, front_v, inb_v = project_world_to_view_grid(
        pts_world=ptsQ,
        K=input_K, R=input_R, T=input_T,
        H=Hs, W=Ws,
        align_corners=align_corners,
    )  # grid [B,V,N,Q,1,2], z_v [B,V,N,Q]

    # sample input depths at those reprojected pixels
    # need [B*V,1,H,W] and [B*V, N*Q, 1,2]
    grid_flat = grid_v.view(B*V, N*Q, 1, 2)
    depth_maps = input_depths[:, :, 0].contiguous().view(B*V, 1, Hs, Ws)
    d_samp = F.grid_sample(depth_maps, grid_flat, mode="bilinear",
                           padding_mode="zeros", align_corners=align_corners)  # [B*V,1,N*Q,1]
    d_samp = d_samp.view(B, V, N, Q)  # [B,V,N,Q]

    valid_v = front_v & inb_v & (d_samp > 0)

    # residual in meters
    resid = (d_samp - z_v).abs()  # [B,V,N,Q]
    resid = resid.masked_fill(~valid_v, 0.0)

    # aggregates over views
    num_valid = valid_v.sum(dim=1).clamp_min(1)  # [B,N,Q]
    mean_res  = resid.sum(dim=1) / num_valid
    # variance: E[r^2] - (E[r])^2
    mean_r2   = (resid * resid).sum(dim=1) / num_valid
    var_res   = (mean_r2 - mean_res * mean_res).clamp_min(0.0)

    inlier = ((resid < tau) & valid_v).sum(dim=1).float() / valid_v.sum(dim=1).clamp_min(1).float()
    num_valid_f = valid_v.sum(dim=1).float()  # [B,N,Q]

    # relative offset feature
    z_center = z_samples[..., Q // 2]  # [B,N]
    z_rel = z_samples - z_center.unsqueeze(-1)  # [B,N,Q]

    # valid query if enough supporting views AND ray valid
    valid_q = (valid_v.sum(dim=1) >= min_valid_views) & valid_ray.unsqueeze(-1)  # [B,N,Q]

    # feature tensor [B,N,Q,F]
    # F=5
    x = torch.stack(
        [
            mean_res,
            var_res,
            inlier,
            (num_valid_f / float(V)).clamp(0, 1),
            z_rel,
        ],
        dim=-1,
    )  # [B,N,Q,5]

    # ---- (D) GT depth supervision per ray ----
    gt = batch["gt_depth"]
    if gt is None:
        raise RuntimeError("gt_depth is required to train depth transformer")

    if isinstance(gt, torch.Tensor) and gt.ndim == 2:
        gt = gt.unsqueeze(0)  # [1,H,W]
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)  # [B,1,H,W]
    gt = gt.to(dev, dtype)

    # ---- DEBUG: save fused vs GT depth ----
    # save_depth_debug(
    #     fused_depth=target_depth,
    #     gt_depth=gt,
    #     out_dir="sanity_check/depth_transformer",
    #     prefix=f"epoch0_batch0",
    # )
    # pdb.set_trace()

    # sample GT depth at the same ray pixels we used in query_points_from_fused_depth
    # easiest: reuse the same grid we already computed inside query_points... IF your function returns it.
    # If not, recompute pixel grid from ray_o+ray_d (same as renderer did).
    Pw = ray_o + ray_d
    Xc = torch.matmul(target_R[:, None], Pw.unsqueeze(-1)).squeeze(-1) + target_T[:, None, :, 0]
    zc = Xc[..., 2].clamp_min(1e-6)
    xh = Xc[..., 0] / zc
    yh = Xc[..., 1] / zc
    fx = target_K[:, 0, 0][:, None]
    fy = target_K[:, 1, 1][:, None]
    cx = target_K[:, 0, 2][:, None]
    cy = target_K[:, 1, 2][:, None]
    u = fx * xh + cx
    v = fy * yh + cy
    xu = (2.0 * u / (Ws - 1)) - 1.0
    yv = (2.0 * v / (Hs - 1)) - 1.0
    grid_r = torch.stack([xu, yv], dim=-1).view(B, N, 1, 2)

    z_gt = sample_depth_map(gt, grid_r, align_corners=align_corners)  # [B,N]
    z_gt = z_gt.clamp_min(0.0)

    # pick GT index for classification: closest query z to GT z
    diff = (z_samples - z_gt.unsqueeze(-1)).abs()  # [B,N,Q]
    diff = diff.masked_fill(~valid_q, float("inf"))
    k_gt = diff.argmin(dim=-1)                     # [B,N]

    bad = (~valid_q).gather(2, k_gt.unsqueeze(-1)).squeeze(-1)
    #pdb.set_trace()

    # flatten to [B*N,...]
    BN = B * N
    out = {
        "x": x.view(BN, Q, -1),
        "z_samples": z_samples.view(BN, Q),
        "valid_q": valid_q.view(BN, Q),
        "k_gt": k_gt.view(BN),
        "z_gt": z_gt.view(BN),
    }
    return out



def save_depth_debug(
    fused_depth: torch.Tensor,   # [B,1,H,W]
    gt_depth: torch.Tensor,      # [B,1,H,W]
    out_dir: str,
    prefix: str,
):
    os.makedirs(out_dir, exist_ok=True)

    fused = fused_depth[0, 0].detach().cpu().numpy()
    gt = gt_depth[0, 0].detach().cpu().numpy()

    valid = (gt > 0) & (fused > 0)

    # --- save raw numpy ---
    np.save(os.path.join(out_dir, f"{prefix}_fused.npy"), fused)
    np.save(os.path.join(out_dir, f"{prefix}_gt.npy"), gt)

    # --- visualization ---
    vmin = np.percentile(gt[gt > 0], 2) if (gt > 0).any() else None
    vmax = np.percentile(gt[gt > 0], 98) if (gt > 0).any() else None

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fused, cmap="magma", vmin=vmin, vmax=vmax)
    plt.title("Fused depth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="magma", vmin=vmin, vmax=vmax)
    plt.title("GT depth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    diff = np.zeros_like(gt)
    diff[valid] = np.abs(fused[valid] - gt[valid])
    plt.imshow(diff, cmap="viridis")
    plt.title("|Fused − GT|")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_compare.png"), dpi=200)
    plt.close()



def write_ply_points(
    ply_path: str,
    xyz: np.ndarray,            # [P,3] float32/float64
    rgb: np.ndarray | None = None,  # [P,3] uint8
):
    """
    Minimal ASCII PLY writer for point clouds.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    if rgb is not None:
        assert rgb.shape == (xyz.shape[0], 3)
        assert rgb.dtype == np.uint8

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    n = xyz.shape[0]
    has_color = rgb is not None

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if has_color:
            for p in range(n):
                x, y, z = xyz[p]
                r, g, b = rgb[p]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for p in range(n):
                x, y, z = xyz[p]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


@torch.no_grad()
def dump_query_points_to_ply(
    ptsQ: torch.Tensor,                 # [B,N,Q,3] world coords
    ply_path: str,
    valid_q: torch.Tensor | None = None,  # [B,N,Q] bool (optional)
    center_only: bool = False,
    stride_n: int = 1,                  # downsample rays (every stride_n ray)
    max_points: int = 2_000_000,        # cap for huge dumps
):
    """
    Save query points to PLY for visualization in MeshLab / CloudCompare / Blender.

    Coloring scheme (if valid_q provided):
      - invalid -> red
      - valid   -> green
    If center_only=True, saves only Q//2 points (one per ray).

    Notes:
      - This function moves data to CPU, so keep stride/max_points reasonable.
      - ptsQ is expected in WORLD coordinates.
    """
    assert ptsQ.ndim == 4 and ptsQ.shape[-1] == 3
    B, N, Q, _ = ptsQ.shape

    if center_only:
        c = Q // 2
        pts = ptsQ[:, ::stride_n, c:c+1, :]  # [B, N', 1, 3]
        if valid_q is not None:
            vq = valid_q[:, ::stride_n, c:c+1]
        else:
            vq = None
    else:
        pts = ptsQ[:, ::stride_n, :, :]      # [B, N', Q, 3]
        vq = valid_q[:, ::stride_n, :, :] if valid_q is not None else None

    pts = pts.reshape(-1, 3)  # [P,3]
    P = pts.shape[0]

    # cap points if too large
    if P > max_points:
        idx = torch.randperm(P, device=pts.device)[:max_points]
        pts = pts[idx]
        if vq is not None:
            vq = vq.reshape(-1)[idx]
        P = max_points
    else:
        if vq is not None:
            vq = vq.reshape(-1)

    xyz = pts.detach().cpu().float().numpy()

    rgb = None
    if vq is not None:
        v = vq.detach().cpu().numpy().astype(np.bool_)
        rgb = np.zeros((P, 3), dtype=np.uint8)
        rgb[ v] = np.array([0, 255, 0], dtype=np.uint8)   # valid: green
        rgb[~v] = np.array([255, 0, 0], dtype=np.uint8)   # invalid: red

    write_ply_points(ply_path, xyz, rgb=rgb)
    print(f"[ply] wrote {P} points -> {ply_path}")