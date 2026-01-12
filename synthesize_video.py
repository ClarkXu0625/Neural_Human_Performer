import argparse
import os
import re
import cv2
import numpy as np
from collections import defaultdict

"""
Examples:

# NERF (debug_nerf)
python synthesize_video.py \
  --root data/result/if_nerf \
  --exp thuman_depth_no_smpl_full_31 \
  --epoch 250 \
  --task nerf \
  --output_dir data/result/if_nerf/thuman_depth_no_smpl_full_31/epoch_250/perceptual/debug_depth_video \
  --human_id 1 \
  --fps 1 \
  --keep_rgb

# FUSE (debug_fused_depth)
python synthesize_video.py \
  --root data/result/if_nerf \
  --exp thuman_depth_no_smpl6 \
  --epoch -1 \
  --task fuse \
  --output_dir data/result/if_nerf/thuman_depth_no_smpl6/epoch_-1/perceptual/debug_depth_video \
  --fps 1 \
  --keep_rgb

# GT (gt) - note different filename pattern
python synthesize_video.py \
  --root data/result/if_nerf \
  --exp thuman_depth_no_smpl6 \
  --epoch -1 \
  --task gt \
  --output_dir data/result/if_nerf/thuman_depth_no_smpl6/epoch_-1/perceptual/debug_depth_video \
  --fps 1 \
  --keep_rgb
"""

def parse_args():
    p = argparse.ArgumentParser("Create rotation videos from depth maps per human object")

    p.add_argument("--root", required=True, type=str,
                   help="Root folder (e.g. if_nerf)")
    p.add_argument("--exp", required=True, type=str,
                   help="Experiment folder name (e.g. thuman_depth_no_smpl6)")
    p.add_argument("--epoch", required=True, type=int,
                   help='Epoch number (e.g. -1 -> folder "epoch_-1")')

    p.add_argument("--task", required=True, choices=["nerf", "fuse", "gt"],
                   help="Which task folder/pattern to use")

    p.add_argument("--output_dir", required=True, type=str,
                   help="Where to save videos")

    p.add_argument("--human_id", default=None, type=int,
                   help="Optional: only process this human id")
    p.add_argument("--fps", default=1, type=int,
                   help="1 fps = 1 second per view")
    p.add_argument("--invalid_leq", default=0.0, type=float,
                   help="Treat depth <= this as invalid (only used for raw depth normalization)")

    # If your PNGs are already colored visualizations (like your screenshot), keep them as-is.
    p.add_argument("--keep_rgb", action="store_true",
                   help="Write images as-is (recommended for debug PNGs that already have colormap/text)")

    # If processing raw depth and you want colored output
    p.add_argument("--use_colormap", action="store_true",
                   help="If set (and not keep_rgb), apply a colormap to normalized depth instead of grayscale")

    return p.parse_args()

def build_input_dir(root, exp, epoch, task):
    base = os.path.join(root, exp, f"epoch_{epoch}", "perceptual")
    if task == "nerf":
        return os.path.join(base, "debug_nerf")
    if task == "fuse":
        return os.path.join(base, "debug_fused_depth")
    if task == "gt":
        return os.path.join(base, "gt")
    raise ValueError(task)

def get_filename_re(task):
    if task == "nerf":
        # human<id>_frame0_view<view_id>_depth.png
        return re.compile(r"human(?P<human_id>\d+)_frame0_view(?P<view_id>\d+)_depth\.png")
    if task == "fuse":
        return re.compile(r"human(?P<human_id>\d+)_frame0_view(?P<view_id>\d+)_fused_depth\.png")
    if task == "gt":
        # <human_id>_frame0_view<view_id>.png
        return re.compile(r"(?P<human_id>\d+)_frame0_view(?P<view_id>\d+)\.png")
    raise ValueError(task)

def collect_images(input_dir, task):
    filename_re = get_filename_re(task)
    humans = defaultdict(list)

    for fname in os.listdir(input_dir):
        m = filename_re.match(fname)
        if not m:
            continue
        hid = int(m.group("human_id"))
        vid = int(m.group("view_id"))
        humans[hid].append((vid, os.path.join(input_dir, fname)))

    return humans

def compute_global_minmax(frames, invalid_leq):
    dmin = np.inf
    dmax = -np.inf

    for _, path in frames:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # If it loads as HxWxC, reduce to single channel
        if img.ndim == 3:
            img = img[..., 0]

        arr = img.astype(np.float32)
        valid = arr > invalid_leq
        if not np.any(valid):
            continue

        dmin = min(dmin, float(arr[valid].min()))
        dmax = max(dmax, float(arr[valid].max()))

    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        return None, None
    return dmin, dmax

def depth_to_uint8_frame(depth, dmin, dmax, invalid_leq, use_colormap, target_wh=None):
    """
    Converts raw depth (single-channel 16-bit/float/etc.) to uint8 BGR frame.
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    arr = depth.astype(np.float32)

    out = np.zeros_like(arr, dtype=np.uint8)
    valid = arr > invalid_leq
    if np.any(valid):
        norm = (arr[valid] - dmin) / (dmax - dmin)
        norm = np.clip(norm, 0.0, 1.0)
        out[valid] = (norm * 255.0).astype(np.uint8)

    if use_colormap:
        bgr = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
        bgr[~valid] = (0, 0, 0)
    else:
        bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if target_wh is not None:
        tw, th = target_wh
        if (bgr.shape[1], bgr.shape[0]) != (tw, th):
            bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_NEAREST)

    return bgr

def make_video(human_id, frames, output_dir, fps, invalid_leq, task, keep_rgb, use_colormap):
    frames = sorted(frames, key=lambda x: x[0])
    if len(frames) < 1:
        return

    # Find first readable frame to set output size
    first = None
    for _, p in frames:
        first = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if first is not None:
            break
    if first is None:
        print(f"[SKIP] Human {human_id}: no readable images.")
        return

    h, w = first.shape[:2]

    # For raw depth normalization, compute global min/max per human
    dmin, dmax = (None, None)
    if not keep_rgb:
        dmin, dmax = compute_global_minmax(frames, invalid_leq)
        if dmin is None:
            print(f"[SKIP] Human {human_id}: cannot find valid depth pixels (all <= {invalid_leq}?)")
            return

    out_path = os.path.join(output_dir, f"human{human_id}_{task}_rotation_depth.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {out_path}. "
              f"Try a different codec/container (e.g., .avi + XVID).")
        return

    wrote = 0
    for view_id, img_path in frames:
        if keep_rgb:
            # Force 8-bit BGR
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Human {human_id}: unreadable {img_path}")
                continue
            # Resize if needed
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Human {human_id}: unreadable {img_path}")
                continue
            frame = depth_to_uint8_frame(img, dmin, dmax, invalid_leq, use_colormap, target_wh=(w, h))

        writer.write(frame)
        wrote += 1

    writer.release()

    if wrote == 0:
        print(f"[ERROR] Human {human_id}: wrote 0 frames -> {out_path}")
    else:
        if keep_rgb:
            print(f"[OK] Saved video: {out_path} (frames={wrote}, fps={fps}, keep_rgb=True)")
        else:
            print(f"[OK] Saved video: {out_path} (frames={wrote}, fps={fps}, depth_range=[{dmin:.4g},{dmax:.4g}])")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = build_input_dir(args.root, args.exp, args.epoch, args.task)
    if not os.path.isdir(input_dir):
        print(f"[ERROR] input_dir not found: {input_dir}")
        return

    humans = collect_images(input_dir, args.task)

    if args.human_id is not None:
        if args.human_id not in humans:
            print(f"[ERROR] Human id {args.human_id} not found in {input_dir}")
            return
        humans = {args.human_id: humans[args.human_id]}

    for hid, frames in humans.items():
        if len(frames) < 2:
            print(f"[SKIP] Human {hid} has <2 views")
            continue
        make_video(
            hid, frames,
            args.output_dir,
            args.fps,
            args.invalid_leq,
            task=args.task,
            keep_rgb=args.keep_rgb,
            use_colormap=args.use_colormap
        )

if __name__ == "__main__":
    main()
