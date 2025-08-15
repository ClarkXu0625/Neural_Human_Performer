import torch
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.datasets import get_human_info
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import pdb
import random
import time
from lib.datasets.thuman.utils import readobj


class Dataset(data.Dataset):
    def __init__(self, data_root, split='train'):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.img_dir = os.path.join(data_root, 'img')
        self.obj_dir = os.path.join(data_root, 'smplx_obj')
        self.parm_dir = os.path.join(data_root, 'parm')
        self.mask_dir = os.path.join(data_root, 'mask')

        self.samples = []

        for fname in sorted(os.listdir(self.obj_dir)):
            if not fname.endswith('.obj'):
                continue

            obj_id = os.path.splitext(fname)[0]
            obj_path = os.path.join(self.obj_dir, fname)

            views = []
            frame_id = '0'
            for cam_idx in range(16):  # camera ids: 000–015
                cam_folder = f"{obj_id}_{cam_idx:03d}"
                img_path = os.path.join(self.img_dir, cam_folder, f'{frame_id}.jpg')
                extr_path = os.path.join(self.parm_dir, cam_folder, f'{frame_id}_extrinsic.npy')
                intr_path = os.path.join(self.parm_dir, cam_folder, f'{frame_id}_intrinsic.npy')
                mask_path = os.path.join(self.mask_dir, cam_folder, f'{frame_id}.png')

                if os.path.exists(img_path) and os.path.exists(extr_path) and os.path.exists(intr_path) and os.path.exists(mask_path):
                    views.append({
                        'img_path': img_path,
                        'extr_path': extr_path,
                        'intr_path': intr_path,
                        'mask_path': mask_path,
                        'cam_idx': cam_idx
                    })

            if views:
                self.samples.append({
                    'obj_id': obj_id,
                    'obj_path': obj_path,
                    'views': views,
                    'frame': int(frame_id)
                })
            else:
                print(f"Skipping {obj_id}: no complete views found.")

        self.nrays = cfg.N_rand
        self.num_humans = len(self.samples)

        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()
        else:
            self.jitter = lambda x: x  # no-op if jitter is off


    def image_to_tensor(self):

        ops = []

        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def color_jitter(self):
        ops = []

        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )

        return transforms.Compose(ops)

    def get_mask(self, index):
        """
        Load and process the foreground mask for a given index.
        Mask path: {data_root}/mask/{obj_id}_000/0.png
        """
        sample = self.samples[index]
        obj_id = sample['obj_id']
        mask_path = os.path.join(self.data_root, 'mask', f'{obj_id}_000', '0.png')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {obj_id}: {mask_path}")

        msk = imageio.imread(mask_path)
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = (msk != 0).astype(np.uint8)

        # Add edge smoothing (same as original)
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk


    def get_input_mask(self, obj_id):
        """
        Load binary mask for a given object ID from THuman.
        """
        mask_path = os.path.join(self.data_root, 'mask', f'{obj_id}_000', '0.png')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Input mask not found for {obj_id}: {mask_path}")

        msk = imageio.imread(mask_path)
        msk = (msk != 0).astype(np.uint8)
        return msk


    def get_smpl_vertice(self, index):
        """
        Load SMPLX vertices from .obj file.
        """
        sample = self.samples[index]
        obj_path = sample['obj_path']
        smpl_obj = readobj(obj_path)

        #vi = np.transpose(smpl_obj['vi'])  # [3, N] → consistent with original logic
        vi = smpl_obj['vi'].astype(np.float32)
        #return vi.astype(np.float32)
        return vi
    

    def prepare_input(self, human, cam_idx):
        """
        - human: str, like '0004'
        - cam_idx: int, 0–15
        - frame: int, e.g., 0
        """

        # === Load SMPL vertices ===
        vertices_path = os.path.join(self.data_root, 'smplx_obj', f'{human}.obj')
        xyz = readobj(vertices_path)  # Assuming readobj() returns Nx3 np.array
        xyz = xyz.astype(np.float32)
        smpl_vertices = xyz.copy() if cfg.time_steps == 1 else None

        nxyz = np.zeros_like(xyz).astype(np.float32)

        # === Original bounds (before canonical transform) ===
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # === Load extrinsic matrix ===
        cam_folder = f'{human}_{cam_idx:03d}'
        extr_path = os.path.join(self.data_root, 'parm', cam_folder, f'0_extrinsic.npy')
        extr = np.load(extr_path).astype(np.float32)
        R = extr[:3, :3]
        Th = extr[:3, 3:]

        # Convert R to rotation vector (Rh) to match original logic
        Rh, _ = cv2.Rodrigues(R)

        # === Canonical transform ===
        xyz = np.dot(xyz - Th, R)
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # === Canonical space bounds ===
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # === Feature and coordinate construction ===
        cxyz = xyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        out_sh = (out_sh | (32 - 1)) + 1  # align to 32

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vertices

    # def prepare_input(self, index):
    #     """
    #     Prepares canonical input representation for a single frame in THuman.
    #     """
    #     sample = self.samples[index]
    #     pdb.set_trace()
    #     obj_id = sample['obj_id']
    #     obj_path = sample['obj_path']
    #     extr_path = sample['extr_path']

    #     # Load SMPLX vertices
    #     smpl_obj = readobj(obj_path)
    #     xyz = smpl_obj['vi'].astype(np.float32)  # [N, 3]

    #     smpl_vertices = xyz.copy() if cfg.time_steps == 1 else None
    #     nxyz = np.zeros_like(xyz).astype(np.float32)

    #     # Original bounding box (world space)
    #     min_xyz = np.min(xyz, axis=0)
    #     max_xyz = np.max(xyz, axis=0)

    #     if cfg.big_box:
    #         min_xyz -= 0.05
    #         max_xyz += 0.05
    #     else:
    #         min_xyz[2] -= 0.05
    #         max_xyz[2] += 0.05

    #     can_bounds = np.stack([min_xyz, max_xyz], axis=0)

    #     # Load extrinsic matrix
    #     extr = np.load(extr_path)
    #     R = extr[:3, :3].astype(np.float32)
    #     T = extr[:3, 3:].astype(np.float32)

    #     # Transform to SMPL coordinate (R and T from camera to world → inverse)
    #     xyz = np.dot(xyz - T.T, R.T)

    #     # Optional canonical transformation
    #     xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

    #     # Canonical bounding box
    #     min_xyz = np.min(xyz, axis=0)
    #     max_xyz = np.max(xyz, axis=0)
    #     if cfg.big_box:
    #         min_xyz -= 0.05
    #         max_xyz += 0.05
    #     else:
    #         min_xyz[2] -= 0.05
    #         max_xyz[2] += 0.05

    #     bounds = np.stack([min_xyz, max_xyz], axis=0)

    #     cxyz = xyz.astype(np.float32)
    #     nxyz = nxyz.astype(np.float32)
    #     feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

    #     # Construct coordinates
    #     dhw = xyz[:, [2, 1, 0]]
    #     min_dhw = min_xyz[[2, 1, 0]]
    #     max_dhw = max_xyz[[2, 1, 0]]
    #     voxel_size = np.array(cfg.voxel_size)
    #     coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

    #     out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    #     x = 32
    #     out_sh = (out_sh | (x - 1)) + 1

    #     # Return everything needed downstream
    #     return feature, coord, out_sh, can_bounds, bounds, R, T.reshape(3), center, rot, trans, smpl_vertices

    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):

        prob = np.random.randint(9000000)
        
        sample = self.samples[index]
        obj_id = sample['obj_id']
        cam_views = sample['views']  # list of dicts: each with img_path, intr_path, extr_path

        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        pdb.set_trace()

        for cam_ind, view in enumerate(cam_views):
            img_path = view['img_path']
            intr_path = view['intr_path']
            extr_path = view['extr_path']

            # Load image
            img = imageio.imread(img_path)
            if self.split == 'train' and cfg.jitter:
                img = Image.fromarray(img)
                img = self.jitter(img)
                img = np.array(img)
            img = img.astype(np.float32) / 255.

            # Load mask
            msk = self.get_mask(index)

            # Load intrinsics/extrinsics
            K = np.load(intr_path).astype(np.float32)
            extr = np.load(extr_path).astype(np.float32)
            R = extr[:3, :3]
            T = extr[:3, 3:].reshape(3, 1)

            # Resize
            H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

            # Mask background
            if cfg.mask_bkgd:
                img[msk == 0] = 1 if cfg.white_bkgd else 0
            K[:2] *= cfg.ratio

            # To tensor
            input_imgs.append(self.im2tensor(img))
            input_msks.append(self.im2tensor(msk).bool())
            input_K.append(torch.tensor(K))
            input_R.append(torch.tensor(R))
            input_T.append(torch.tensor(T))

        # Canonical transformation (use first view's index as base)
        feature, coord, out_sh, can_bounds, bounds, R_orig, T_orig, center, rot, trans, smpl_vert = self.prepare_input(index, cam_views[0]['cam_idx'])

        # Use first view for target
        target_img = input_imgs[0].permute(1, 2, 0).cpu().numpy()
        target_msk = input_msks[0][0].cpu().numpy().astype(np.uint8)
        K0 = input_K[0].numpy()
        R0 = input_R[0].numpy()
        T0 = input_T[0].numpy()

        # Ray sampling
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            target_img, target_msk, K0, R0, T0, can_bounds, self.nrays, self.split
        )
        acc = if_nerf_dutils.get_acc(coord_, target_msk)

        # Package results
        ret = {
            'smpl_vertice': [smpl_vert],
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc,
            'mask_at_box': mask_at_box,
            'bounds': bounds,
            'R': R_orig,
            'Th': T_orig,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': int(obj_id),
            'cam_ind': 0,
            'frame_index': int(obj_id),
            'human_idx': index,

            # Multi-view conditioning
            'input_imgs': torch.stack(input_imgs),    # [V, C, H, W]
            'input_msks': torch.stack(input_msks),    # [V, 1, H, W]
            'input_K': torch.stack(input_K),          # [V, 3, 3]
            'input_R': torch.stack(input_R),          # [V, 3, 3]
            'input_T': torch.stack(input_T),          # [V, 3, 1]

            # Target (first view)
            'target_K': input_K[0],
            'target_R': input_R[0],
            'target_T': input_T[0]
        }

        return ret


    def get_length(self):
        return self.__len__()

    # def __len__(self):
    #     return len(self.ims)
    def __len__(self):
        return len(self.samples)



# import os
# import glob
# import torch
# import numpy as np
# import imageio
# import cv2
# from PIL import Image
# from torchvision import transforms
# import torch.utils.data as data
# from lib.config import cfg
# from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils

# class Dataset(data.Dataset):
#     def __init__(self, data_root, human, ann_file, split):
#         super().__init__()
#         self.split = split
#         self.data_root = os.path.join(cfg.virt_data_root, split)
#         self.image_dir = os.path.join(self.data_root, 'img')
#         self.mask_dir = os.path.join(self.data_root, 'mask')
#         self.transform_dir = os.path.join(self.data_root, 'transform')
#         self.vertex_dir = os.path.join(self.data_root, 'position_map_uv_space')

#         self.frame_paths = sorted(glob.glob(os.path.join(self.image_dir, '*/*.jpg')))
#         self.nrays = cfg.N_rand

#         self.im2tensor = transforms.Compose([transforms.ToTensor()])
#         self.H = cfg.H
#         self.W = cfg.W
#         self.ratio = cfg.ratio
#         self.time_steps = cfg.time_steps
#         self.time_mult = cfg.time_mult if 'time_mult' in cfg else [0]

#     def __len__(self):
#         return len(self.frame_paths)

#     def __getitem__(self, idx):
#         img_path = self.frame_paths[idx]
#         rel_path = os.path.relpath(img_path, self.image_dir)
#         cam_id, fname = rel_path.split('/')
#         frame_id = int(os.path.splitext(fname)[0])

#         img = imageio.imread(img_path).astype(np.float32) / 255.0
#         mask_path = os.path.join(self.mask_dir, cam_id, fname.replace('.jpg', '.png'))
#         msk = imageio.imread(mask_path)
#         msk = (msk != 0).astype(np.uint8)

#         transform_path = os.path.join(self.transform_dir, cam_id, fname.replace('.jpg', '.npy'))
#         transform = np.load(transform_path, allow_pickle=True).item()
#         K = transform['K'].astype(np.float32)
#         R = transform['R'].astype(np.float32)
#         T = transform['T'].astype(np.float32)

#         H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
#         img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

#         if cfg.mask_bkgd:
#             img[msk == 0] = 1 if cfg.white_bkgd else 0

#         K[:2] *= self.ratio

#         rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
#             img, msk, K, R, T, np.array([[-1, -1, -1], [1, 1, 1]]), self.nrays, self.split)
#         acc = if_nerf_dutils.get_acc(coord_, msk)

#         smpl_vertices = []
#         input_imgs, input_msks, input_K, input_R, input_T = [], [], [], [], []

#         for t in range(self.time_steps):
#             cur_frame = frame_id + self.time_mult[t] if t < len(self.time_mult) else frame_id
#             cur_fname = f"{cur_frame:04d}.jpg"
#             cur_img_path = os.path.join(self.image_dir, cam_id, cur_fname)
#             cur_img = imageio.imread(cur_img_path).astype(np.float32) / 255.
#             cur_mask_path = os.path.join(self.mask_dir, cam_id, cur_fname.replace('.jpg', '.png'))
#             cur_msk = imageio.imread(cur_mask_path)
#             cur_msk = (cur_msk != 0).astype(np.uint8)

#             cur_img = cv2.resize(cur_img, (W, H), interpolation=cv2.INTER_AREA)
#             cur_msk = cv2.resize(cur_msk, (W, H), interpolation=cv2.INTER_NEAREST)
#             if cfg.mask_bkgd:
#                 cur_img[cur_msk == 0] = 1 if cfg.white_bkgd else 0

#             cur_img = self.im2tensor(cur_img)
#             cur_msk = self.im2tensor(cur_msk).bool()

#             input_imgs.append(torch.stack([cur_img]))
#             input_msks.append(torch.stack([cur_msk]))

#             if t == 0:
#                 input_K.append(torch.from_numpy(K))
#                 input_R.append(torch.from_numpy(R))
#                 input_T.append(torch.from_numpy(T))

#             vert_path = os.path.join(self.vertex_dir, f"{cur_frame:04d}.npy")
#             if os.path.exists(vert_path):
#                 verts = np.load(vert_path).reshape(-1, 3).astype(np.float32)
#             else:
#                 verts = np.zeros((6890, 3), dtype=np.float32)
#             smpl_vertices.append(verts)

#         input_K = torch.stack(input_K)
#         input_R = torch.stack(input_R)
#         input_T = torch.stack(input_T)

#         feature = np.concatenate([smpl_vertices[0], np.zeros_like(smpl_vertices[0])], axis=1).astype(np.float32)
#         coord = np.zeros((6890, 3), dtype=np.int32)
#         out_sh = np.array([64, 64, 64], dtype=np.int32)
#         bounds = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)

#         ret = {
#             'smpl_vertice': [torch.from_numpy(x) for x in smpl_vertices],
#             'feature': torch.from_numpy(feature),
#             'coord': torch.from_numpy(coord),
#             'out_sh': torch.from_numpy(out_sh),
#             'rgb': torch.from_numpy(rgb),
#             'ray_o': torch.from_numpy(ray_o),
#             'ray_d': torch.from_numpy(ray_d),
#             'near': torch.from_numpy(near),
#             'far': torch.from_numpy(far),
#             'acc': torch.from_numpy(acc),
#             'mask_at_box': torch.from_numpy(mask_at_box),
#             'R': torch.from_numpy(R),
#             'Th': torch.from_numpy(T),
#             'center': torch.zeros(3),
#             'rot': torch.eye(3),
#             'trans': torch.zeros(3),
#             'i': frame_id,
#             'cam_ind': int(cam_id),
#             'frame_index': frame_id,
#             'human_idx': 0,
#             'input_imgs': input_imgs,
#             'input_msks': input_msks,
#             'input_K': input_K,
#             'input_R': input_R,
#             'input_T': input_T,
#             'target_K': K,
#             'target_R': R,
#             'target_T': T
#         }
#         return ret
