# train_depth_transformer.py
from __future__ import annotations
import os
import time
import math

import torch
import torch.nn.functional as F

from lib.config import cfg, args
from lib.datasets import make_data_loader

from depth_transformer.model import DepthTransformer
from depth_transformer.data_prep import build_dt_batch


def to_cuda_minimal(batch, device):
    for k, v in batch.items():
        if k == "meta":
            continue
        if isinstance(v, list):
            batch[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def _fmt_hhmmss(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "??:??:??"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


class EMA:
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * float(x)
        return self.value

def cuda_sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- read DT params from cfg (put these in your depth_transformer.yaml) ---
    dt_cfg = cfg.depth_transformer
    out_dir = getattr(dt_cfg, "out_dir", "checkpoints/depth_transformer")
    epochs = int(getattr(dt_cfg, "epochs", 10))
    lr = float(getattr(dt_cfg, "lr", 1e-4))
    half_window = int(getattr(dt_cfg, "half_window", 15))
    step_size = float(getattr(dt_cfg, "step_size", 0.005))
    lambda_reg = float(getattr(dt_cfg, "lambda_reg", 0.1))
    tau = float(getattr(dt_cfg, "tau", 0.01))
    min_valid_views = int(getattr(dt_cfg, "min_valid_views", 2))
    resume = getattr(dt_cfg, "resume", "")

    # logging knobs (optional)
    log_every = int(getattr(dt_cfg, "log_every", 1))  # print every N loader iters (default: every iter)

    os.makedirs(out_dir, exist_ok=True)

    train_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        max_iter=cfg.ep_iter,
    )

    # robust epoch length for ETA
    try:
        iters_per_epoch = len(train_loader)
    except TypeError:
        iters_per_epoch = int(getattr(cfg, "ep_iter", 0)) or None  # fallback if available

    d_in = 5
    model = DepthTransformer(d_in=d_in, d_model=128, nhead=4, num_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    start_epoch = 0
    global_step = 0  # counts *optimizer updates*, not raw loader iters

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # optional: restore global_step if you saved it before; otherwise keep 0
        global_step = int(ckpt.get("global_step", 0))
        print(f"[resume] epoch={start_epoch} from {resume}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        nsteps = 0  # optimizer updates this epoch

        ema_iter = EMA(alpha=0.1)
        epoch_t0 = time.perf_counter()

        # NOTE: we use enumerate to get "loader iteration" index for ETA.
        for it, batch in enumerate(train_loader):
            iter_t0 = time.perf_counter()

            batch = to_cuda_minimal(batch, device)
            cuda_sync_if_needed(device)

            if "input_imgs" in batch:
                batch["input_imgs"] = [img / img.max() for img in batch["input_imgs"]]

            with torch.no_grad():
                dtb = build_dt_batch(
                    batch=batch,
                    half_window=half_window,
                    step_size=step_size,
                    tau=tau,
                    min_valid_views=min_valid_views,
                )

            x = dtb["x"]                  # [BN,Q,5]
            z_samples = dtb["z_samples"]  # [BN,Q]
            valid_q = dtb["valid_q"]      # [BN,Q]
            k_gt = dtb["k_gt"]            # [BN]
            z_gt = dtb["z_gt"]            # [BN]

            keep = valid_q.any(dim=-1) & (z_gt > 0)

            # prepare ETA using iteration timing (even if we SKIP)
            cuda_sync_if_needed(device)
            iter_dt = time.perf_counter() - iter_t0
            avg_iter_dt = ema_iter.update(iter_dt)

            # compute remaining iters in this epoch if we know epoch length
            if iters_per_epoch is not None and iters_per_epoch > 0:
                remain = max(iters_per_epoch - (it + 1), 0)
                eta_epoch = remain * avg_iter_dt
                iter_str = f"{it+1:05d}/{iters_per_epoch:05d}"
            else:
                eta_epoch = float("nan")
                iter_str = f"{it+1:05d}/?????"

            # too few valid rays -> skip backward/step, but still log
            if keep.sum() < 16:
                if (it % log_every) == 0:
                    cur_lr = opt.param_groups[0].get("lr", float("nan"))
                    elapsed = time.perf_counter() - epoch_t0
                    print(
                        f"[e{epoch:03d}] it {iter_str} | upd {nsteps:05d} | "
                        f"SKIP(keep={int(keep.sum())}) | lr {cur_lr:.2e} | "
                        f"eta { _fmt_hhmmss(eta_epoch) } | elap { _fmt_hhmmss(elapsed) }"
                    )
                continue

            x = x[keep]
            z_samples = z_samples[keep]
            valid_q = valid_q[keep]
            k_gt = k_gt[keep]
            z_gt = z_gt[keep]

            out = model(x=x, z_samples=z_samples, valid_q=valid_q, temperature=1.0)
            logits = out["logits"]   # [BN,Q]
            z_soft = out["z_soft"]   # [BN]

            loss_cls = F.cross_entropy(logits, k_gt)
            loss_reg = F.smooth_l1_loss(z_soft, z_gt)
            loss = loss_cls + lambda_reg * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach().cpu())
            nsteps += 1
            global_step += 1

            if (it % log_every) == 0:
                cur_lr = opt.param_groups[0].get("lr", float("nan"))
                elapsed = time.perf_counter() - epoch_t0
                print(
                    f"[e{epoch:03d}] it {iter_str} | upd {nsteps:05d} (g{global_step:07d}) | "
                    f"loss {float(loss):.6f} (cls {float(loss_cls):.6f}, reg {float(loss_reg):.6f}) | "
                    f"lr {cur_lr:.2e} | eta { _fmt_hhmmss(eta_epoch) } | elap { _fmt_hhmmss(elapsed) }"
                )

        sched.step()
        avg = total / max(nsteps, 1)
        print(f"[epoch {epoch}] loss={avg:.6f} steps={nsteps}")

        save_path = os.path.join(out_dir, f"dt_epoch_{epoch:04d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "cfg_file": getattr(args, "cfg_file", None),
            },
            save_path,
        )
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
