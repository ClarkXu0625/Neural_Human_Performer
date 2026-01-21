#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Uses your repo cfg system (supports --cfg_file)
from lib.config import cfg, args
from lib.datasets import make_data_loader

from depth_transformer.model import DepthTransformer
from depth_transformer.data_prep_infer import build_dt_batch_infer
import pdb


def _to_device(batch: dict, device: torch.device) -> dict:
    """Minimal version of Trainer.to_cuda."""
    for k, v in list(batch.items()):
        if k == "meta":
            continue
        if isinstance(v, (list, tuple)):
            batch[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
        elif torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def _save_depth_png(depth_2d: np.ndarray, out_png: str, title: str = "", vmin=None, vmax=None):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_2d, cmap="magma", vmin=vmin, vmax=vmax)
    plt.colorbar(label="depth (m)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_conf_png(conf_2d: np.ndarray, out_png: str, title: str = ""):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_2d, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="max prob")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_depth_compare_png(
    fused_2d: np.ndarray,
    refined_2d: np.ndarray,
    out_png: str,
    title: str = "",
    robust: bool = True,
):
    """
    Saves a side-by-side panel: fused | refined | refined-fused
    Uses shared vmin/vmax for the two depth maps for fair visual comparison.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    fused = fused_2d.astype(np.float32)
    refined = refined_2d.astype(np.float32)

    # Treat 0 as invalid (matches your scatter fill). Adjust if you use another convention.
    valid = (fused > 0) & (refined > 0)

    if valid.any():
        vals = np.concatenate([fused[valid], refined[valid]], axis=0)
        if robust:
            vmin = np.percentile(vals, 1.0)
            vmax = np.percentile(vals, 99.0)
        else:
            vmin = float(vals.min())
            vmax = float(vals.max())
    else:
        vmin, vmax = None, None

    diff = np.zeros_like(refined, dtype=np.float32)
    diff[valid] = refined[valid] - fused[valid]

    # symmetric limits for diff
    if valid.any():
        if robust:
            d = np.abs(diff[valid])
            dmax = np.percentile(d, 99.0)
        else:
            dmax = float(np.max(np.abs(diff[valid])))
        dmax = max(dmax, 1e-6)
    else:
        dmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(fused, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("fused depth")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="depth (m)")

    im1 = axes[1].imshow(refined, cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("refined depth (DT)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="depth (m)")

    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-dmax, vmax=dmax)
    axes[2].set_title("difference (refined - fused)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Δ depth (m)")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _grid_to_pix(grid_norm: torch.Tensor, H: int, W: int, align_corners: bool = True) -> torch.Tensor:
    """
    grid_norm: [B, N, 1, 2] in [-1,1] (x,y)
    returns:   [B, N, 2] pixel coords (u,v) float
    """
    x = grid_norm[..., 0].squeeze(2)  # [B,N]
    y = grid_norm[..., 1].squeeze(2)

    if align_corners:
        u = (x + 1.0) * (W - 1) / 2.0
        v = (y + 1.0) * (H - 1) / 2.0
    else:
        # matches grid_sample align_corners=False mapping
        u = ((x + 1.0) * W - 1.0) / 2.0
        v = ((y + 1.0) * H - 1.0) / 2.0

    return torch.stack([u, v], dim=-1)  # [B,N,2]


def _scatter_depth_min(
    uv_pix: torch.Tensor,
    z: torch.Tensor,
    H: int,
    W: int,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    uv_pix: [B,N,2] float pixel coords (u,v)
    z:      [B,N] depth values
    valid:  [B,N] bool optional
    returns [B,1,H,W] depth map, z-buffered by min (nearest)
    """
    B, N, _ = uv_pix.shape
    device = uv_pix.device

    u = uv_pix[..., 0].round().long()
    v = uv_pix[..., 1].round().long()

    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if valid is not None:
        inb = inb & valid

    out = torch.full((B, H * W), float("inf"), device=device, dtype=z.dtype)

    for b in range(B):
        sel = inb[b]
        if sel.sum() == 0:
            continue
        flat = v[b, sel] * W + u[b, sel]         # [K]
        zb = z[b, sel]                           # [K]

        # z-buffer: take min per pixel
        try:
            out[b].scatter_reduce_(0, flat, zb, reduce="amin", include_self=True)
        except Exception:
            # fallback: sort + segment min
            idx = torch.argsort(flat)
            flat_s = flat[idx]
            z_s = zb[idx]
            head = torch.ones_like(flat_s, dtype=torch.bool)
            head[1:] = flat_s[1:] != flat_s[:-1]
            grp = head.cumsum(0) - 1
            G = int(grp[-1].item()) + 1
            mins = torch.full((G,), float("inf"), device=device, dtype=z.dtype)
            mins = mins.scatter_reduce(0, grp, z_s, reduce="amin", include_self=True)
            uniq = flat_s[head]
            out[b, uniq] = torch.minimum(out[b, uniq], mins)

    out = out.view(B, 1, H, W)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


@torch.no_grad()
def run_one_batch(
    batch: dict,
    model: DepthTransformer,
    half_window: int,
    step_size: float,
    out_dir: str,
    tag: str,
):
    """
    Saves: fused_depth, refined_depth_map, gt_depth (if present), confidence map.
    """
    # build_dt_batch is expected to do fusion + query point prep internally
    dtb = build_dt_batch_infer(
        batch=batch,
        #encoder=None,               # keep simple: no NeRF encoder
        half_window=half_window,
        step_size=step_size,
        #training=False,
    )

    # --- required DT inputs ---
    x = dtb["x"].to(next(model.parameters()).device)                  # [BN,Q,2]
    z_samples = dtb["z_samples"].to(next(model.parameters()).device)  # [BN,Q]
    valid_q = dtb["valid_q"].to(next(model.parameters()).device)      # [BN,Q]

    # out = model(x=x, z_samples=z_samples, valid_q=valid_q, temperature=1.0)

    # # refined per-ray depth
    # z_refined = out["z_soft"]          # [BN]
    # prob = out.get("prob", None)       # [BN,Q] if your model returns it
    # logits = out.get("logits", None)   # [BN,Q]

    # ------------------------------
    # DT forward in BN-chunks (OOM-safe)
    # ------------------------------
    device = next(model.parameters()).device
    BN = x.shape[0]

    # put these in yaml if you want; otherwise defaults are fine
    chunk = int(getattr(cfg.depth_transformer, "infer_chunk", 8192))
    use_amp = bool(getattr(cfg.depth_transformer, "use_amp", True))

    z_refined_chunks = []
    prob_chunks = []   # optional
    logits_chunks = [] # optional

    model.eval()
    for s in range(0, BN, chunk):
        e = min(s + chunk, BN)

        x_i = x[s:e]
        z_i = z_samples[s:e]
        v_i = valid_q[s:e]

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out_i = model(x=x_i, z_samples=z_i, valid_q=v_i, temperature=1.0)
        else:
            out_i = model(x=x_i, z_samples=z_i, valid_q=v_i, temperature=1.0)

        # refined per-ray depth (SOFT, as you requested)
        z_refined_chunks.append(out_i["z_soft"].detach())

        # keep these only if needed (they can be large)
        if "prob" in out_i and out_i["prob"] is not None:
            prob_chunks.append(out_i["prob"].detach())
        if "logits" in out_i and out_i["logits"] is not None:
            logits_chunks.append(out_i["logits"].detach())

    # concatenate back
    z_refined = torch.cat(z_refined_chunks, dim=0)                  # [BN]
    prob = torch.cat(prob_chunks, dim=0) if prob_chunks else None   # [BN,Q] or None
    logits = torch.cat(logits_chunks, dim=0) if logits_chunks else None  # [BN,Q] or None


    print("x mean/std:", x.mean().item(), x.std().item())
    print("x per-dim std:", x.view(-1, x.shape[-1]).std(dim=0))
    print("z_samples std:", z_samples.std().item())
    idx = logits.argmax(dim=-1)
    print(torch.bincount(idx.cpu(), minlength=z_samples.shape[1]))
    # pdb.set_trace()
    # ------------------------------
    # DT forward in BN-chunks ends here
    # ------------------------------

    # --- recover B,N,H,W for reshaping ---
    # We prefer dtb-provided metadata if present.
    B = int(dtb.get("B", None) or batch["ray_o"].shape[0])
    N = int(dtb.get("N", None) or batch["ray_o"].shape[1])

    # fused depth map (if dtb provides it)
    fused = dtb.get("fused_depth", None)  # expected [B,1,H,W]
    if fused is None:
        fused = dtb.get("target_depth", None)

    if fused is not None:
        H, W = fused.shape[-2], fused.shape[-1]
    else:
        # fallback: use input image resolution
        H, W = batch["input_imgs"][0].shape[-2:]

    # --- get per-ray pixel coords ---
    # Try dtb keys first.
    uv_pix = None
    if "uv_pix" in dtb:
        # expected [B,N,2]
        uv_pix = dtb["uv_pix"].to(z_refined.device)
    elif "grid" in dtb:
        # expected [B,N,1,2] normalized coords
        uv_pix = _grid_to_pix(dtb["grid"].to(z_refined.device), H, W, align_corners=True)
    elif "grid_norm" in dtb:
        uv_pix = _grid_to_pix(dtb["grid_norm"].to(z_refined.device), H, W, align_corners=True)
    else:
        raise KeyError(
            "build_dt_batch did not return ray pixel coords. "
            "Please make it return either dtb['grid'] (normalized) or dtb['uv_pix']."
        )

    # --- valid ray mask ---
    valid_ray = dtb.get("valid_ray", None)  # expected [B,N] bool
    if valid_ray is None:
        # fallback: valid if any query valid
        valid_ray = valid_q.view(B, N, -1).any(dim=-1)

    # reshape per-ray refined depth to [B,N]
    z_ref_bn = z_refined.view(B, N)

    print("valid_ray ratio:", valid_ray.float().mean().item())
    print("valid_q ratio:", valid_q.float().mean().item())

    num_valid_q = valid_q.view(B, N, -1).sum(dim=-1)  # [B,N]
    print("num_valid_q min/mean/max:",
        num_valid_q.min().item(), num_valid_q.float().mean().item(), num_valid_q.max().item())


    # build refined depth map via z-buffer min scatter (nearest)
    refined_map = _scatter_depth_min(
        uv_pix=uv_pix,
        z=z_ref_bn,
        H=H,
        W=W,
        valid=valid_ray,
    )  # [B,1,H,W]
    if fused is not None:
        refined_map = torch.where(fused > 1e-6, refined_map, fused)

    # confidence map (max prob per ray)
    conf_map = None
    if prob is not None:
        conf = prob.max(dim=-1).values  # [BN]
        conf_map = conf.view(B, N)
        Q = logits.shape[1]
    elif logits is not None:
        conf = torch.softmax(logits, dim=-1).max(dim=-1).values
        conf_map = conf.view(B, N)
        Q = logits.shape[1]

    ray_valid = valid_q.any(dim=-1)  # [BN]
    print("ray_valid:", ray_valid.float().mean().item(), ray_valid.sum().item(), "/", ray_valid.numel())

    idx_v = idx[ray_valid]
    print("hist(valid rays only):", torch.bincount(idx_v.cpu(), minlength=Q))

    # --- save outputs (batch 0 only for sanity) ---
    os.makedirs(out_dir, exist_ok=True)

    if fused is not None:
        fused0 = fused[0, 0].detach().cpu().numpy().astype(np.float32)
        _save_depth_png(fused0, os.path.join(out_dir, f"{tag}_fused.png"), title="fused depth")
        np.save(os.path.join(out_dir, f"{tag}_fused.npy"), fused0)

    refined0 = refined_map[0, 0].detach().cpu().numpy().astype(np.float32)
    #pdb.set_trace()
    if fused is not None:
        _save_depth_compare_png(
            fused_2d=fused0,
            refined_2d=refined0,
            out_png=os.path.join(out_dir, f"{tag}_fused_refined_diff.png"),
            title=tag,
            robust=True,   # uses 1–99% for stable visualization
        )
    
    _save_depth_png(refined0, os.path.join(out_dir, f"{tag}_refined.png"), title="refined depth (DT)")
    np.save(os.path.join(out_dir, f"{tag}_refined.npy"), refined0)

    # optional GT
    if "gt_depth" in batch and batch["gt_depth"] is not None:
        gt = batch["gt_depth"]
        if torch.is_tensor(gt):
            gt0 = gt.detach().cpu().numpy().astype(np.float32)
        else:
            gt0 = np.asarray(gt, dtype=np.float32)
        _save_depth_png(gt0, os.path.join(out_dir, f"{tag}_gt.png"), title="GT depth")
        np.save(os.path.join(out_dir, f"{tag}_gt.npy"), gt0)

    # confidence visualization (scatter to image optional; here we just save per-ray reshaped to N->image is not valid)
    # We'll save a "ray confidence" image by scattering max confidence to pixels similarly.
    if conf_map is not None:
        conf_img = _scatter_depth_min(
            uv_pix=uv_pix,
            z=conf_map,          # treat confidence like "z", but min-scatter isn't right.
            H=H,
            W=W,
            valid=valid_ray,
        )
        # Above min-scatter is not semantically perfect for confidence.
        # Better: just overwrite (last write). For quick sanity this is fine; you can improve later.
        conf0 = conf_img[0, 0].detach().cpu().numpy().astype(np.float32)
        _save_conf_png(conf0, os.path.join(out_dir, f"{tag}_conf.png"), title="DT confidence (rough)")
        np.save(os.path.join(out_dir, f"{tag}_conf.npy"), conf0)

    print(f"[saved] {out_dir}/{tag}_*.png/.npy")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Read everything from cfg (YAML), NOT args
    # ------------------------------------------------------------
    ckpt = getattr(cfg, "ckpt", None)
    if ckpt is None:
        # optional alternative nesting if you prefer
        ckpt = getattr(getattr(cfg, "depth_transformer", None), "ckpt", None)
    if not ckpt:
        raise ValueError(
            "Missing checkpoint path in YAML. Add one of:\n"
            "  ckpt: checkpoints/depth_transformer/dt_epoch_0007.pth\n"
            "or\n"
            "  depth_transformer:\n"
            "    ckpt: checkpoints/depth_transformer/dt_epoch_0007.pth"
        )

    # Optional outputs and run controls
    out_dir = getattr(cfg, "out_dir", None)
    if out_dir is None:
        out_dir = getattr(getattr(cfg, "depth_transformer", None), "out_dir", None)
    if not out_dir:
        out_dir = "outputs/dt_infer"

    split = getattr(cfg, "split", None)
    if split is None:
        split = getattr(getattr(cfg, "depth_transformer", None), "split", None)
    if not split:
        split = "test"

    num_batches = getattr(cfg, "num_batches", None)
    if num_batches is None:
        num_batches = getattr(getattr(cfg, "depth_transformer", None), "num_batches", None)
    if not num_batches:
        num_batches = 1
    num_batches = int(num_batches)

    # DT hyperparams (prefer nested depth_transformer section)
    dt_cfg = getattr(cfg, "depth_transformer", None)
    if dt_cfg is None:
        raise ValueError("Expected cfg.depth_transformer section in YAML.")

    half_window = int(getattr(dt_cfg, "half_window", 15))
    step_size   = float(getattr(dt_cfg, "step_size", 0.005))

    os.makedirs(out_dir, exist_ok=True)


    state = torch.load(ckpt, map_location="cpu")

    # infer d_in from checkpoint
    d_in_ckpt = state["model"]["proj.weight"].shape[1]
    print(f"[dt] inferred d_in from ckpt = {d_in_ckpt}")
    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = DepthTransformer(
        d_in=d_in_ckpt,
        d_model=int(getattr(dt_cfg, "d_model", 128)),
        nhead=int(getattr(dt_cfg, "nhead", 4)),
        num_layers=int(getattr(dt_cfg, "num_layers", 4)),
    ).to(device)

    model.load_state_dict(state["model"], strict=True)
    model.eval()

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    is_train = (split == "train")
    loader = make_data_loader(cfg, is_train=is_train, is_distributed=False)

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------
    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break

        batch = _to_device(batch, device)

        # keep your normalization convention
        if "input_imgs" in batch:
            batch["input_imgs"] = [img / img.max() for img in batch["input_imgs"]]

        # stable tag
        human_idx = int(batch.get("human_idx", torch.tensor(-1)).item()) if torch.is_tensor(batch.get("human_idx", None)) else int(batch.get("human_idx", -1))
        frame_idx = int(batch.get("frame_index", torch.tensor(-1)).item()) if torch.is_tensor(batch.get("frame_index", None)) else int(batch.get("frame_index", -1))
        view_idx  = int(batch.get("cam_ind", torch.tensor(-1)).item()) if torch.is_tensor(batch.get("cam_ind", None)) else int(batch.get("cam_ind", -1))
        tag = f"b{bi:03d}_human{human_idx}_frame{frame_idx}_view{view_idx}"

        run_one_batch(
            batch=batch,
            model=model,
            half_window=half_window,
            step_size=step_size,
            out_dir=out_dir,
            tag=tag,
        )

    print("[done]")


if __name__ == "__main__":
    main()
