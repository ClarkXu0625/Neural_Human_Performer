import os
import numpy as np
import matplotlib.pyplot as plt

'''
cam=008 | input_views=[6, 7, 9, 10]
GT  avg(valid)=3.091444e-02, avg(all)=2.910022e-03 || 
FUSED avg(valid)=3.388586e-02, avg(all)=3.523616e-03
'''

GT_DIR   = "outputs/variance_map1/val/gt_depth"
NERF_DIR = "outputs/variance_map1/val/fused_depth"  #"outputs/variance_map1/val/nerf_density"


def avg_variance(arr, valid_only=True):
    """
    Compute average variance.
    - valid_only=True: average over non-zero pixels only
    - valid_only=False: average over entire map (including background)
    """
    if valid_only:
        vals = arr[arr > 0]
        return float(vals.mean()) if vals.size > 0 else 0.0
    else:
        return float(arr.mean())


# your three views
cams = [8]      #[3, 8, 13]
obj_id = 0
subview = 0

def load(path):
    x = np.load(path).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def robust_vmax(arrs, q=99.0):
    vals = np.concatenate([a.reshape(-1) for a in arrs])
    vals = vals[vals > 0]  # ignore background zeros when setting scale
    if vals.size == 0:
        return 1.0
    return np.percentile(vals, q)

# Load all maps first
gt_maps = []
nerf_maps = []
for cam in cams:
    gt_path = os.path.join(GT_DIR,   f"{obj_id:04d}_{cam:03d}_{subview:03d}.npy")
    nf_path = os.path.join(NERF_DIR, f"{obj_id:04d}_{cam:03d}_{subview:03d}.npy")
    assert os.path.exists(gt_path), f"Missing {gt_path}"
    assert os.path.exists(nf_path), f"Missing {nf_path}"
    gt_maps.append(load(gt_path))
    nerf_maps.append(load(nf_path))

# Use a shared color scale for fair comparison
vmax = robust_vmax(gt_maps + nerf_maps, q=99.0)

fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(9, 12),
    dpi=150,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05}
)

# Plot images
for r, cam in enumerate(cams):
    axes[r, 0].imshow(gt_maps[r], vmin=0, vmax=vmax)
    axes[r, 0].set_title(f"GT variance cam={cam:03d}")
    axes[r, 0].axis("off")

    im = axes[r, 1].imshow(nerf_maps[r], vmin=0, vmax=vmax)
    axes[r, 1].set_title(f"NeRF variance cam={cam:03d}")
    axes[r, 1].axis("off")

# --- ADD COLORBAR AXIS ON THE RIGHT ---
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
# [left, bottom, width, height] in figure fraction

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("avg pairwise feature variance")

plt.show()


for i, cam in enumerate(cams):
    gt_avg_valid   = avg_variance(gt_maps[i],   valid_only=True)
    gt_avg_all     = avg_variance(gt_maps[i],   valid_only=False)
    nerf_avg_valid = avg_variance(nerf_maps[i], valid_only=True)
    nerf_avg_all   = avg_variance(nerf_maps[i], valid_only=False)

    print(
        f"cam={cam:03d} | "
        f"GT  avg(valid)={gt_avg_valid:.6e}, avg(all)={gt_avg_all:.6e} || "
        f"FUSED avg(valid)={nerf_avg_valid:.6e}, avg(all)={nerf_avg_all:.6e}"
    )

# Optional: save
os.makedirs("outputs/variance_map1/val/_viz", exist_ok=True)
out_png = f"outputs/variance_map1/val/_viz/obj{obj_id:04d}_sub{subview:03d}_gt_vs_nerf_valid.png"
fig.savefig(out_png, bbox_inches="tight")
print("Saved:", out_png)
