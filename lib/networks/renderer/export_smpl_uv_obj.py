import pickle
import numpy as np
import trimesh
import torch
from smplx import SMPL

# === CONFIG ===
pkl_path = 'data/smplx/smpl/SMPL_NEUTRAL.pkl'  # or 'smpl_male.pkl'
output_obj_path = 'data/smplx/smpl/smpl_uv.obj'


# === LOAD SMPL MODEL ===
# Optional: SMPLX model manager
model = SMPL(model_path=pkl_path, gender='neutral', batch_size=1)

# Set to T-pose (all zeros)
betas = torch.zeros(1, 10)             # shape params
body_pose = torch.zeros(1, 69)         # 23 joints × 3 axis-angle
global_orient = torch.zeros(1, 3)      # no rotation
transl = torch.zeros(1, 3)             # no translation

# Forward pass
output = model(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    transl=transl,
    return_verts=True
)

vertices = output.vertices[0].cpu().numpy()  # [6890, 3]
faces = model.faces                          # [13776, 3]

# === EXPORT TO OBJ ===
mesh = trimesh.Trimesh(vertices, faces, process=False)
mesh.export(output_obj_path)

print(f'Exported SMPL mesh to: {output_obj_path}')