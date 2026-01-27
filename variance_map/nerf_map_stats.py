import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    x = np.load(args.path).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    flat = x.reshape(-1)
    nz = flat[flat > 0]

    print("shape:", x.shape)
    print("zero fraction:", (flat == 0).mean())
    if nz.size > 0:
        print("nonzero min/max:", nz.min(), nz.max())
        print("nonzero percentiles:", np.percentile(nz, [1,5,50,95,99]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)

    # 1) raw map
    im0 = axes[0].imshow(x)
    axes[0].set_title("NeRF map (raw)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) nonzero mask
    mask = (x > 0).astype(np.float32)
    im1 = axes[1].imshow(mask, vmin=0, vmax=1)
    axes[1].set_title("Nonzero mask (x>0)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) histogram
    if nz.size == 0:
        axes[2].text(0.1, 0.5, "No nonzero values", transform=axes[2].transAxes)
        axes[2].set_axis_off()
    else:
        vals = np.log(nz + 1e-8) if args.log else nz
        axes[2].hist(vals, bins=args.bins)
        axes[2].set_title("Histogram (log)" if args.log else "Histogram (nonzero)")
        axes[2].set_xlabel("log(x)" if args.log else "x")
        axes[2].set_ylabel("count")

    plt.tight_layout()
    plt.show()

    PATH = args.path
    x = np.load(PATH).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    nz = x[x>0]
    baseline = np.median(nz)
    print("baseline:", baseline)
    print("frac within 1e-7 of baseline:", np.mean(np.abs(nz-baseline) < 1e-7))
    print("frac within 1e-6 of baseline:", np.mean(np.abs(nz-baseline) < 1e-6))

if __name__ == "__main__":
    main()
