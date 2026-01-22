#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-object variance maps from multi-view pixel-aligned features.

Output:
  outputs/variance_map/<split>/<target_map_source>/<obj_id>_<cam>_<subview>.npy   (float32, HxW)

Determinism:
  - fixed seeds
  - cudnn deterministic
  - dataloader: shuffle disabled (as configured) + num_workers=0
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_network
from lib.networks import make_network
from lib.networks.renderer.if_clight_renderer import Renderer
import pdb


def set_deterministic(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_cuda_minimal(batch: dict, device: torch.device) -> dict:
    for k, v in batch.items():
        if k == "meta":
            continue
        if isinstance(v, list):
            batch[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


@torch.no_grad()
def backproject_depth_to_world(
    depth_hw: torch.Tensor,          # [H, W], depth in *camera* z units
    K_33: torch.Tensor,              # [3,3]
    R_33: torch.Tensor,              # [3,3] world->cam
    T_31: torch.Tensor,              # [3,1] world->cam translation
    use_pixel_center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      xyz_world: [H*W, 3]
      valid_mask: [H*W] bool (depth > 0)
    """
    assert depth_hw.dim() == 2
    H, W = depth_hw.shape

    device = depth_hw.device
    dtype = torch.float32

    # pixels (u, v)
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    v, u = torch.meshgrid(ys, xs, indexing="ij")  # [H,W], [H,W]

    if use_pixel_center:
        u = u + 0.5
        v = v + 0.5

    z = depth_hw.to(dtype=dtype)  # [H,W]
    valid = z > 0

    # homogeneous pixel rays
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=-1).reshape(-1, 3)  # [HW,3]
    z_flat = z.reshape(-1)  # [HW]
    valid_flat = valid.reshape(-1)

    # cam coords: X_cam = inv(K) @ (pix * z)
    Kinv = torch.inverse(K_33.to(dtype=dtype))
    cam = (pix @ Kinv.T) * z_flat[:, None]  # [HW,3]

    # world coords: X_world = R^{-1} @ (X_cam - T)
    # R,T are world->cam, so invert:
    Rinv = torch.inverse(R_33.to(dtype=dtype))
    T = T_31.to(dtype=dtype).reshape(1, 3)  # [1,3]
    world = (cam - T) @ Rinv.T  # [HW,3]

    return world, valid_flat


def avg_pairwise_variance(pixel_feat_nvc: torch.Tensor) -> torch.Tensor:
    """
    pixel_feat_nvc: [N, V, C]
    Return:
      var_v: [V]  average over all pairs i<j of mean_c (feat_i - feat_j)^2
    """
    assert pixel_feat_nvc.dim() == 3
    N, V, C = pixel_feat_nvc.shape
    if N < 2:
        # no variance across views
        return torch.zeros((V,), device=pixel_feat_nvc.device, dtype=pixel_feat_nvc.dtype)

    # Pairwise accumulation
    acc = torch.zeros((V,), device=pixel_feat_nvc.device, dtype=pixel_feat_nvc.dtype)
    cnt = 0
    for i in range(N):
        fi = pixel_feat_nvc[i]  # [V,C]
        for j in range(i + 1, N):
            fj = pixel_feat_nvc[j]
            d2 = (fi - fj).pow(2).mean(dim=-1)  # [V]
            acc += d2
            cnt += 1
    return acc / float(cnt)


def load_target_map(batch, split: str, H: int, W: int, device):
    src = cfg.variance.target_map_source

    if src == "gt_depth":
        d = batch["gt_depth"]
        if d.dim() == 3:
            d = d[0]
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return d.to(device)

    if src == "nerf_density":
        obj_id = int(batch["obj_id"][0].item()) if torch.is_tensor(batch["obj_id"]) else int(batch["obj_id"])
        cam = int(batch["cam_ind"][0].item()) if torch.is_tensor(batch["cam_ind"]) else int(batch["cam_ind"])
        subview = int(batch["i"][0].item()) if torch.is_tensor(batch["i"]) else int(batch["i"])

        pattern = cfg.variance.nerf_density_pattern
        path = pattern.format(
            root=cfg.variance.nerf_density_root,
            split=split,
            obj_id=obj_id,
            cam=cam,
            subview=subview,
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"NeRF density map not found: {path}")

        arr = np.load(path).astype(np.float32)
        if arr.shape != (H, W):
            # if stored at a different resolution, resize deterministically
            arr = center_pad_to_shape(arr, H, W, fill=0.0)

        return torch.from_numpy(arr).to(device)

    raise ValueError(f"Unknown variance.target_map_source: {src}")


def center_pad_to_shape(arr: np.ndarray, out_h: int, out_w: int, fill: float = 0.0) -> np.ndarray:
    """
    Center-pad (or center-crop) a 2D array to (out_h, out_w) without interpolation.
    Deterministic.

    - If arr is smaller: pads with `fill`.
    - If arr is larger: center-crops.
    """
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"
    in_h, in_w = arr.shape

    out = np.full((out_h, out_w), fill, dtype=arr.dtype)

    # Compute source crop (if needed)
    src_y0 = max(0, (in_h - out_h) // 2)
    src_x0 = max(0, (in_w - out_w) // 2)
    src_y1 = min(in_h, src_y0 + out_h)
    src_x1 = min(in_w, src_x0 + out_w)

    # Compute destination placement
    dst_y0 = max(0, (out_h - in_h) // 2)
    dst_x0 = max(0, (out_w - in_w) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out



def main():
    # ---------- deterministic ----------
    seed = int(getattr(cfg, "seed", 1234))
    set_deterministic(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- build loader (reuse train_depth_transformer.py style) ----------
    split = getattr(cfg, "split", None)
    if split is None:
        split = "train"
    is_train = (split == "train")

    # Force deterministic data loading (no multiprocessing jitter)
    # If your make_data_loader reads cfg.workers, set it to 0 here.
    # if hasattr(cfg, "workers"):
    #     cfg.workers = 0

    loader = make_data_loader(
        cfg,
        is_train=is_train,
        is_distributed=False,
        max_iter=getattr(cfg, "ep_iter", -1),
    )

    # ---------- build net + renderer ----------
    try:
        from lib.networks import make_network
        net = make_network(cfg).to(device)
    except Exception as e:
        raise RuntimeError(
            "Could not import/build net via `from lib.networks import make_network`.\n"
            "Please replace this block with your project’s actual network constructor."
        ) from e

    #net = make_network(cfg).cuda()
    #load_network(net, args.ckpt)

    net.eval()
    renderer = Renderer(net)

    # ---------- output dir ----------
    target_map_source = cfg.variance.target_map_source
    out_root = os.path.join("outputs", "variance_map", split, target_map_source)
    os.makedirs(out_root, exist_ok=True)

    # Chunking to control memory
    chunk = int(getattr(args, "chunk", 65536))  # points per chunk

    # ---------- iterate ----------
    for it, batch in enumerate(loader):
        batch = to_cuda_minimal(batch, device)

        obj_id = int(batch["obj_id"][0].item()) if isinstance(batch["obj_id"], torch.Tensor) else int(batch["obj_id"])
        cam = int(batch["cam_ind"][0].item()) if isinstance(batch["cam_ind"], torch.Tensor) else int(batch["cam_ind"])
        subview = int(batch["i"][0].item()) if isinstance(batch["i"], torch.Tensor) else int(batch["i"])

        # Filter (empty list means "no filtering")
        if len(cfg.variance.obj_ids) > 0 and obj_id not in cfg.variance.obj_ids:
            continue
        if len(cfg.variance.cam_inds) > 0 and cam not in cfg.variance.cam_inds:
            continue
        if hasattr(cfg.variance, "subviews") and len(cfg.variance.subviews) > 0 and subview not in cfg.variance.subviews:
            continue

        # target depth: prefer gt_depth (per your ret construction)
        if "gt_depth" not in batch:
            # If you truly always have it, you can hard error instead.
            print(f"[skip] missing gt_depth for obj={obj_id} cam={cam} subview={subview}")
            continue

        # depth = batch["gt_depth"]  # could be [H,W] or [1,H,W]
        # if depth.dim() == 3:
        #     depth = depth[0]
        # depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # H, W = depth.shape
        if "gt_depth" in batch:
            tmp = batch["gt_depth"]
            if tmp.dim() == 3: tmp = tmp[0]
            H, W = tmp.shape
        else:
            # fallback: use input image resolution (last resort)
            H, W = batch["input_imgs"][0].shape[-2:]

        target_map = load_target_map(batch, split, H, W, device=device)
        # treat target_map as "depth-like" for backprojection
        depth = target_map

        
        # target camera params: may be batched
        target_K = batch["target_K"]
        target_R = batch["target_R"]
        target_T = batch["target_T"]

        if target_K.dim() == 3:
            target_K = target_K[0]
        if target_R.dim() == 3:
            target_R = target_R[0]
        if target_T.dim() == 3:
            target_T = target_T[0]

        # Backproject all pixels to world
        xyz_world, valid_flat = backproject_depth_to_world(
            depth_hw=depth,
            K_33=target_K,
            R_33=target_R,
            T_31=target_T,
            use_pixel_center=True,
        )
        # xyz_world: [HW,3], valid_flat: [HW]

        # Build pixel feature maps from encoder
        image_list = batch["input_imgs"]  # list with singleton time dim: [input_imgs]
        images = image_list[0].reshape(-1, *image_list[0].shape[2:])  # [N,3,H,W]
        _, _, pixel_feat_map, pixel_feat_scale = net.encoder(images)

        # Query pixel-aligned feature in chunks
        var_flat = torch.zeros((H * W,), device=device, dtype=torch.float32)

        # Only compute for valid pixels; others remain 0
        valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            var_map = var_flat.view(H, W).detach().cpu().numpy().astype(np.float32)
            out_path = os.path.join(out_root, f"human{obj_id:04d}_cam{cam:03d}_subview{subview:03d}.npy")
            np.save(out_path, var_map)
            print(f"[saved] {out_path} (all invalid depth)")
            continue

        # Process valid points in chunks
        for s in range(0, valid_idx.numel(), chunk):
            ids = valid_idx[s : s + chunk]
            xyz_chunk = xyz_world[ids].unsqueeze(0)  # [1, M, 3]

            pixel_feat = renderer.get_pixel_aligned_feature(
                batch=batch,
                xyz=xyz_chunk,
                pixel_feat_map=pixel_feat_map,
                pixel_feat_scale=pixel_feat_scale,
                batchify=True,
            )

            # transpose to [N, M, C].
            if pixel_feat.dim() == 3:
                pixel_feat_nmc = pixel_feat.permute(0, 2, 1).contiguous()
            else:
                raise RuntimeError(f"Unexpected pixel_feat shape: {tuple(pixel_feat.shape)}")

            var_m = avg_pairwise_variance(pixel_feat_nmc).float()  # [M]
            var_flat[ids] = var_m

        var_map = var_flat.view(H, W).detach().cpu().numpy().astype(np.float32)

        out_path = os.path.join(out_root, f"{obj_id:04d}_{cam:03d}_{subview:03d}.npy")
        np.save(out_path, var_map)
        avg = var_map.mean()
        print(f"[saved] {out_path}  shape={var_map.shape}  it={it} avg_var={avg:.6f}")

        seen = set()
        key = (target_map_source, split, obj_id, cam, subview)  # include source so gt/nerf don't collide
        if key in seen:
            continue
        seen.add(key)


if __name__ == "__main__":
    main()
