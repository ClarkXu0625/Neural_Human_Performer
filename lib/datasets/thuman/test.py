import numpy as np
import os
from utils import readobj
import imageio.v2 as imageio
import cv2
import pdb

path = '/home/clark/Documents/GitHub/Neural_Human_Performer/data/THuman/train'
smplx_dir = os.path.join(path, 'smplx_obj')
parm_dir = os.path.join(path, 'parm')
img_dir = os.path.join(path, 'img')
out_dir = os.path.join(path, 'output')
os.makedirs(out_dir, exist_ok=True)


# params = np.load(params_path, allow_pickle=True).item()
#pdb.set_trace()

# Loop through all .obj files
for obj_file in sorted(os.listdir(smplx_dir)):
    if not obj_file.endswith('.obj'):
        continue

    obj_id = os.path.splitext(obj_file)[0]  # e.g., "0004"
    obj_path = os.path.join(smplx_dir, obj_file)

    # Read SMPLX obj
    smpl_obj = readobj(obj_path)
    vi = np.transpose(smpl_obj['vi'])
    f = np.squeeze(smpl_obj['f'])  # ensure 0-based indexing

    #pdb.set_trace()
    # Load matching extrinsic / intrinsic (0004 -> 0004_000)
    cam_folder = f"{obj_id}_000"
    extr_path = os.path.join(parm_dir, cam_folder, '0_extrinsic.npy')
    intr_path = os.path.join(parm_dir, cam_folder, '0_intrinsic.npy')

    if not os.path.exists(extr_path) or not os.path.exists(intr_path):
        print(f"Skipping {obj_id}: missing camera files.")
        continue

    extr = np.load(extr_path)
    intr = np.load(intr_path)

    R = extr[:3, :3]
    T = extr[:3, 3:]
    K = intr

    # Project vertices
    uv = np.dot(K, np.dot(R, vi) + T)

    #pdb.set_trace()

    uv = uv[:2] / uv[2:]
    uv = uv.T.astype(np.int32)


    # Load matching image
    img_path = os.path.join(img_dir, cam_folder, '0.jpg')
    if not os.path.exists(img_path):
        print(f"Skipping {obj_id}: missing image file.")
        continue

    img = imageio.imread(img_path)
    overlay = img.copy()

    # Draw mesh edges
    for tri in f:
        pts = uv[tri]
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    # Blend with original
    alpha = 0.6
    vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Save result
    out_path = os.path.join(out_dir, f"{obj_id}_overlay.jpg")
    imageio.imwrite(out_path, vis)
    print(f"Saved: {out_path}")
