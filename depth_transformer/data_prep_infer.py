# depth_transformer/data_prep_infer.py
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from depth_transformer.data_prep import fuse_depths_to_target, query_points_from_fused_depth


@torch.no_grad()
def build_dt_batch_infer(
    batch: dict,
    half_window: int,
    step_size: float,
    align_corners: bool = True,
):
    """
    Build DT inputs for inference only (NO GT usage).
    Returns dict with:
      x        : [BN, Q, d_in]
      z_samples: [BN, Q]
      valid_q  : [BN, Q]
      meta     : some debug tensors
    """
    device = batch["ray_o"].device
    dtype  = batch["ray_o"].dtype

    ray_o = batch["ray_o"]  # [B,N,3]
    ray_d = batch["ray_d"]  # [B,N,3]
    B, N, _ = ray_o.shape

    # cameras
    input_K = batch["input_K"].reshape(B, -1, 3, 3)
    input_R = batch["input_R"].reshape(B, -1, 3, 3)
    input_T = batch["input_T"].reshape(B, -1, 3, 1)

    target_K = batch["target_K"]
    target_R = batch["target_R"]
    target_T = batch["target_T"]

    # target resolution
    if "target_hw" in batch:
        Ht, Wt = batch["target_hw"]
    else:
        # input_imgs is [T, V, 3, H, W] stored as singleton list in your loader
        Ht, Wt = batch["input_imgs"][0].shape[-2:]

    # depths/masks (time=0)
    # your loader returns: input_depths: [1, V, H, W]  (no channel)
    # renderer expects: [B, V, 1, H, W]
    input_depths = batch["input_depths"][0]  # [V,H,W] or [B,V,H,W] depending on collate
    if input_depths.dim() == 3:
        input_depths = input_depths.unsqueeze(0)  # [B,V,H,W]
    input_depths = input_depths.unsqueeze(2)      # [B,V,1,H,W]
    input_depths = torch.nan_to_num(input_depths, nan=0.0, posinf=0.0, neginf=0.0)

    if "input_msks" in batch:
        input_masks = batch["input_msks"][0]  # [V,1,H,W] or [B,V,1,H,W]
        if input_masks.dim() == 4:
            input_masks = input_masks.unsqueeze(0)  # [B,V,1,H,W]
        input_masks = input_masks.to(dtype=input_depths.dtype)
    else:
        input_masks = (input_depths > 0).to(input_depths.dtype)

    # fuse
    target_depth, target_count = fuse_depths_to_target(
        input_depths=input_depths,   # [B,V,1,H,W]
        input_masks=input_masks,     # [B,V,1,H,W]
        input_K=input_K,
        input_R=input_R,
        input_T=input_T,
        target_K=target_K,
        target_R=target_R,
        target_T=target_T,
        target_hw=(Ht, Wt),
        # chunked=True,
        # max_points=1_000_000,
    )  # [B,1,Ht,Wt], [B,1,Ht,Wt]

    # query points
    Q = 2 * half_window + 1
    offsets = torch.linspace(
        -half_window * step_size,
        half_window * step_size,
        steps=Q,
        device=device,
        dtype=dtype,
    )

    ptsQ, tQ, valid_ray, grid = query_points_from_fused_depth(
        ray_o=ray_o,
        ray_d=ray_d,
        target_depth=target_depth,
        target_K=target_K,
        target_R=target_R,
        target_T=target_T,
        target_hw=(Ht, Wt),
        offsets_m=offsets,
        align_corners=align_corners,
        return_grid=True,   # <-- you will add this option (see note below)
    )
    # ptsQ: [B,N,Q,3]  tQ: [B,N,Q]  valid_ray:[B,N]  grid:[B,N,1,2]

    # camera-z for each query point
    # World -> target cam
    Xt = torch.matmul(target_R[:, None, None, :, :], ptsQ.unsqueeze(-1)).squeeze(-1)  # [B,N,Q,3]
    Xt = Xt + target_T[:, None, None, :, 0]
    z_samples = Xt[..., 2]  # [B,N,Q]

    # center depth at this pixel (sample fused depth at the grid)
    d_center = F.grid_sample(
        target_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    ).view(B, N)  # [B,N]

    # count at pixel (optional but useful feature)
    c_center = F.grid_sample(
        target_count.float(), grid, mode="nearest", padding_mode="zeros", align_corners=align_corners
    ).view(B, N)  # [B,N]

    # ray_d in camera for dz feature
    ray_d_cam = torch.matmul(target_R[:, None, :, :], ray_d.unsqueeze(-1)).squeeze(-1)  # [B,N,3]
    dz = ray_d_cam[..., 2].clamp(min=1e-6)  # [B,N]

    # build 5-dim feature to match your checkpoint (d_in=5)
    # 1) z_rel  2) |z_rel|  3) dz  4) count_norm  5) bias(1)
    z_rel = z_samples - d_center[:, :, None]               # [B,N,Q]
    abs_z_rel = z_rel.abs()
    dz_q = dz[:, :, None].expand_as(z_rel)
    count_norm = (c_center / (c_center.max().clamp_min(1.0)))[:, :, None].expand_as(z_rel)
    bias = torch.ones_like(z_rel)

    x = torch.stack([z_rel, abs_z_rel, dz_q, count_norm, bias], dim=-1)  # [B,N,Q,5]

    # valid per query: ray valid AND z_samples positive
    valid_q = valid_ray[:, :, None] & (z_samples > 1e-6)  # [B,N,Q]

    # flatten rays
    BN = B * N
    x = x.view(BN, Q, 5)
    z_samples = z_samples.view(BN, Q)
    valid_q = valid_q.view(BN, Q)

    return {
        "x": x,
        "z_samples": z_samples,
        "valid_q": valid_q,
        "meta": {
            "target_depth": target_depth,   # [B,1,Ht,Wt]
            "target_count": target_count,   # [B,1,Ht,Wt]
            "d_center": d_center,           # [B,N]
        },
        "valid_ray": valid_ray,     # [B,N]
        "grid": grid,              # [B,N,1,2]  <-- ADD THIS
        "B": B,
        "N": N,
    }
