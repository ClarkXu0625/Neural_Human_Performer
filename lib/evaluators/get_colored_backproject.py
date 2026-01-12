# get_colored_backproject.py
import os
import numpy as np
from typing import Optional, Tuple


def save_ply_xyzrgb(path: str, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
    """
    Save point cloud as ASCII PLY.
    xyz: (N,3) float32/float64
    rgb_uint8: (N,3) uint8 in RGB order
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb N mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]) + "\n"

    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def backproject_rgbd_crop_to_points(
    rgb_crop: np.ndarray,
    depth_crop: np.ndarray,
    K_full: np.ndarray,
    crop_xywh: Tuple[int, int, int, int],
    *,
    stride: int = 1,
    depth_min: float = 1e-6,
    depth_max: Optional[float] = None,
    return_cam: bool = True,
    to_world: bool = False,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
):
    """
    Backproject a *cropped* rgb+depth to a colored point cloud.

    Returns:
      xyz_cam, rgb_u8                 if return_cam=True and to_world=False
      xyz_world, rgb_u8               if return_cam=False and to_world=True
      (xyz_cam, xyz_world, rgb_u8)    if return_cam=True and to_world=True
    """
    x0, y0, w, h = crop_xywh
    if rgb_crop.shape[0] != h or rgb_crop.shape[1] != w:
        raise ValueError(f"rgb_crop shape {rgb_crop.shape} does not match crop h,w=({h},{w})")
    if depth_crop.shape[0] != h or depth_crop.shape[1] != w:
        raise ValueError(f"depth_crop shape {depth_crop.shape} does not match crop h,w=({h},{w})")

    if stride < 1:
        raise ValueError("stride must be >= 1")

    depth_crop = np.asarray(depth_crop, dtype=np.float32)

    # RGB -> uint8
    if rgb_crop.dtype == np.uint8:
        rgb_u8_full = rgb_crop
    else:
        rgb_u8_full = (np.clip(rgb_crop, 0.0, 1.0) * 255.0).astype(np.uint8)

    fx = float(K_full[0, 0]); fy = float(K_full[1, 1])
    cx = float(K_full[0, 2]); cy = float(K_full[1, 2])

    # Pixel grid in FULL-image coordinates (important!)
    ys = np.arange(y0, y0 + h, stride, dtype=np.float32)
    xs = np.arange(x0, x0 + w, stride, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)  # (h', w')

    z = depth_crop[::stride, ::stride]  # (h', w')

    valid = np.isfinite(z) & (z > depth_min)
    if depth_max is not None:
        valid &= (z < float(depth_max))

    if not np.any(valid):
        empty_xyz = np.zeros((0, 3), np.float32)
        empty_rgb = np.zeros((0, 3), np.uint8)
        if to_world and return_cam:
            return empty_xyz, empty_xyz, empty_rgb
        return empty_xyz, empty_rgb

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    xyz_cam = np.stack([x, y, z_valid], axis=-1).astype(np.float32)

    rgb_u8 = rgb_u8_full[::stride, ::stride, :][valid]

    if not to_world:
        return xyz_cam, rgb_u8

    if R is None or T is None:
        raise ValueError("to_world=True requires R and T")

    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    T = np.asarray(T, dtype=np.float32).reshape(-1)
    if T.size != 3:
        raise ValueError(f"T must have 3 elements, got {T.size}")
    T = T.reshape(3, 1)

    # X_world = R^T (X_cam - T)
    xyz_world = (R.T @ (xyz_cam.T - T)).T.astype(np.float32)

    if return_cam:
        return xyz_cam, xyz_world, rgb_u8
    return xyz_world, rgb_u8


def get_crop_bbox_from_mask(mask_at_box_full: np.ndarray) -> tuple:
    """
    mask_at_box_full: (H,W) bool or 0/1 uint8
    returns (x,y,w,h) in full-image coords.

    Pure numpy implementation (no cv2 dependency).
    """
    m = mask_at_box_full.astype(bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return (0, 0, mask_at_box_full.shape[1], mask_at_box_full.shape[0])  # fallback: full image
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
