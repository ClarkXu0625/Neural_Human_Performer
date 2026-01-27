# variance_map/depth_fusion.py
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


def _ensure_bv1hw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize depth/mask tensor to [B, V, 1, H, W].
    Accepts:
      - [B,V,1,H,W] (ok)
      - [B,V,H,W]   -> add channel dim
      - [V,1,H,W]   -> add batch dim
      - [V,H,W]     -> add batch+channel dim
      - [1,H,W] / [H,W] -> treat as one view
    """
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"Expected list/tuple of length 1, got len={len(x)}")
        x = x[0]

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.dim() == 5:
        return x
    if x.dim() == 4:
        # [B,V,H,W] or [V,1,H,W]
        if x.shape[1] in (1, 3) and x.shape[0] > 1 and x.shape[-1] > 8:
            # ambiguous; assume [V,1,H,W] (rare)
            return x.unsqueeze(0)
        return x.unsqueeze(2)  # [B,V,1,H,W]
    if x.dim() == 3:
        # [V,H,W] or [1,H,W]
        if x.shape[0] == 1:
            return x.unsqueeze(0).unsqueeze(0)  # [1,1,1,H,W]
        return x.unsqueeze(0).unsqueeze(2)      # [1,V,1,H,W]
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,H,W]
    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


@torch.no_grad()
def fuse_input_depths_to_target(
    input_depths: torch.Tensor,   # [B, V, 1, Hs, Ws] meters
    input_masks: torch.Tensor,    # [B, V, 1, Hs, Ws] 0/1
    input_K: torch.Tensor,        # [B, V, 3, 3]
    input_R: torch.Tensor,        # [B, V, 3, 3] world->cam
    input_T: torch.Tensor,        # [B, V, 3, 1] world->cam
    target_K: torch.Tensor,       # [B, 3, 3] or [1,3,3] or [3,3]
    target_R: torch.Tensor,       # [B, 3, 3] or [1,3,3] or [3,3]
    target_T: torch.Tensor,       # [B, 3, 1] or [1,3,1] or [3,1] or [3]
    target_hw: Tuple[int, int],   # (Ht, Wt)
    chunked: bool = True,
    max_points: int = 1_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      target_depth : [B, 1, Ht, Wt] fused depth (meters)
      target_count : [B, 1, Ht, Wt] contributing count (diagnostic)
    """
    # Normalize shapes
    input_depths = _ensure_bv1hw(input_depths)
    input_masks  = _ensure_bv1hw(input_masks)

    B, V, _, Hs, Ws = input_depths.shape
    Ht, Wt = target_hw
    dev = input_depths.device
    dtype = input_depths.dtype

    # Normalize target K/R/T to [B,...]
    if target_K.dim() == 2:
        target_K = target_K[None].expand(B, -1, -1).to(dev, dtype)
    elif target_K.dim() == 3 and target_K.shape[0] == 1:
        target_K = target_K.expand(B, -1, -1).to(dev, dtype)
    else:
        target_K = target_K.to(dev, dtype)

    if target_R.dim() == 2:
        target_R = target_R[None].expand(B, -1, -1).to(dev, dtype)
    elif target_R.dim() == 3 and target_R.shape[0] == 1:
        target_R = target_R.expand(B, -1, -1).to(dev, dtype)
    else:
        target_R = target_R.to(dev, dtype)

    # target_T can be [3], [3,1], [1,3,1], [B,3,1]
    if target_T.dim() == 1:
        target_T = target_T.view(1, 3, 1).expand(B, 3, 1).to(dev, dtype)
    elif target_T.dim() == 2:
        if target_T.shape == (3, 1):
            target_T = target_T[None].expand(B, 3, 1).to(dev, dtype)
        elif target_T.shape == (1, 3):
            target_T = target_T.view(1, 3, 1).expand(B, 3, 1).to(dev, dtype)
        else:
            raise ValueError(f"Unexpected target_T shape: {tuple(target_T.shape)}")
    elif target_T.dim() == 3 and target_T.shape[0] == 1:
        target_T = target_T.expand(B, -1, -1).to(dev, dtype)
    else:
        target_T = target_T.to(dev, dtype)
    if target_T.shape[-1] != 1:
        target_T = target_T[..., None]

    # Z-buffer init
    target_depth = torch.full((B, 1, Ht, Wt), float("inf"), device=dev, dtype=dtype)
    target_count = torch.zeros((B, 1, Ht, Wt), device=dev, dtype=torch.int32)

    # Precompute pixel grid (u,v,1) in input image coordinates
    ys, xs = torch.meshgrid(
        torch.arange(Hs, device=dev, dtype=dtype),
        torch.arange(Ws, device=dev, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix_homo = torch.stack([xs, ys, ones], dim=0).view(1, 1, 3, Hs * Ws)  # [1,1,3,M]
    M = Hs * Ws

    td_flat = target_depth.view(B, -1)  # [B, Ht*Wt]
    tc_flat = target_count.view(B, -1)  # [B, Ht*Wt]

    for v in range(V):
        d = input_depths[:, v, 0]  # [B,Hs,Ws]
        m = input_masks[:, v, 0]   # [B,Hs,Ws]
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        valid = (d > 0) & (m > 0)
        if not valid.any():
            continue

        d_flat_in = d.view(B, -1)        # [B,M]
        valid_in  = valid.view(B, -1)    # [B,M]

        K = input_K[:, v].to(dev, dtype)  # [B,3,3]
        R = input_R[:, v].to(dev, dtype)  # [B,3,3] world->cam
        T = input_T[:, v].to(dev, dtype)  # [B,3,1]
        if T.shape[-1] != 1:
            T = T[..., None]
        Kinv = torch.inverse(K)           # [B,3,3]

        # Kinv @ pix_homo -> [B,3,M]
        Kinv_pix = torch.matmul(Kinv[:, None, :, :], pix_homo).squeeze(1)  # [B,3,M]

        # Iterate chunks over the *flattened* M pixels
        step = max_points if chunked else M
        start = 0
        while start < M:
            end = min(start + step, M)
            sel = valid_in[:, start:end]  # [B, m]
            if not sel.any():
                start = end
                continue

            gathered_flat = []
            gathered_z    = []
            gathered_b    = []

            for b in range(B):
                idxb = torch.nonzero(sel[b], as_tuple=False).squeeze(1)
                if idxb.numel() == 0:
                    continue

                # back-project in camera coords
                Xc_dir = Kinv_pix[b, :, start:end][:, idxb]      # [3,mb]
                zb     = d_flat_in[b, start:end][idxb]           # [mb]
                Xc     = Xc_dir * zb[None, :]                    # [3,mb]

                # cam->world: Xw = R^T (Xc - T)
                Xw = torch.matmul(R[b].transpose(0, 1), (Xc - T[b, :, 0:1]))  # [3,mb]

                # world->target cam: Xt = Rt Xw + Tt
                Xt = torch.matmul(target_R[b], Xw) + target_T[b]  # [3,mb]
                zt = Xt[2, :]

                front = zt > 1e-6
                if not front.any():
                    continue
                Xt = Xt[:, front]
                zt = zt[front]

                # project to target pixels
                x = Xt[0, :] / zt
                y = Xt[1, :] / zt
                fx, fy = target_K[b, 0, 0], target_K[b, 1, 1]
                cx, cy = target_K[b, 0, 2], target_K[b, 1, 2]
                u = fx * x + cx
                v_ = fy * y + cy

                ui = u.round().long()
                vi = v_.round().long()

                inb = (ui >= 0) & (ui < Wt) & (vi >= 0) & (vi < Ht)
                if not inb.any():
                    continue

                ui = ui[inb]
                vi = vi[inb]
                zt = zt[inb]

                flat = vi * Wt + ui  # [k]
                gathered_flat.append(flat)
                gathered_z.append(zt)
                gathered_b.append(torch.full_like(flat, b))

            if len(gathered_flat) == 0:
                start = end
                continue

            flat_idx = torch.cat(gathered_flat, dim=0)
            z_vals   = torch.cat(gathered_z, dim=0)
            b_idx    = torch.cat(gathered_b, dim=0)

            # z-buffer reduce per batch
            for b in range(B):
                sb = (b_idx == b)
                if not sb.any():
                    continue
                fb = flat_idx[sb]
                zb = z_vals[sb]

                # Prefer scatter_reduce_ if available
                try:
                    td_flat[b].scatter_reduce_(0, fb, zb, reduce="amin", include_self=True)
                    tc_flat[b].scatter_add_(0, fb, torch.ones_like(fb, dtype=tc_flat.dtype))
                    continue
                except Exception:
                    pass

                # Fallback: sort + group-min
                order = torch.argsort(fb)
                fb = fb[order]
                zb = zb[order]
                head = torch.ones_like(fb, dtype=torch.bool)
                head[1:] = fb[1:] != fb[:-1]
                grp = head.cumsum(0) - 1
                G = int(grp[-1].item()) + 1 if fb.numel() else 0
                mins = torch.full((G,), float("inf"), device=zb.device, dtype=zb.dtype)
                mins = mins.scatter_reduce(0, grp, zb, reduce="amin", include_self=True)
                uniq_fb = fb[head]
                td_flat[b, uniq_fb] = torch.minimum(td_flat[b, uniq_fb], mins)
                tc_flat[b, uniq_fb] += 1

            start = end

    target_depth = torch.where(torch.isinf(target_depth), torch.zeros_like(target_depth), target_depth)
    return target_depth, target_count


def save_depth_npy_png(
    depth_hw: np.ndarray,
    npy_path: str,
    png_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Lightweight saver (no matplotlib dependency here).
    - Writes .npy always
    - Writes .png via imageio if available; otherwise falls back to OpenCV if available.
    """
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, depth_hw.astype(np.float32))

    # Normalize to 0..255 for PNG
    d = depth_hw.astype(np.float32)
    if vmin is None:
        vmin = float(np.percentile(d[d > 0], 1)) if np.any(d > 0) else 0.0
    if vmax is None:
        vmax = float(np.percentile(d[d > 0], 99)) if np.any(d > 0) else 1.0
    denom = max(1e-8, vmax - vmin)
    vis = np.clip((d - vmin) / denom, 0.0, 1.0)
    vis_u8 = (vis * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(png_path, vis_u8)
    except Exception:
        try:
            import cv2
            cv2.imwrite(png_path, vis_u8)
        except Exception:
            # If neither backend exists, silently skip PNG
            pass
