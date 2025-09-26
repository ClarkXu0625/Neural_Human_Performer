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
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()
        self.split = split
        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}

        data_name = cfg.virt_data_root.split('/')[-1]
        human_info = get_human_info.get_thuman_info(self.split)
        human_list = list(human_info.keys())

        self.depth_path_root = os.path.join(cfg.depth_map_root, data_root.split('/')[-1])

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]

            self.cams[human] = {'K': [], 'R': [], 'T': [], 'D': []}  # D will be empty

            ims = []
            cam_inds = []

            for cam_id in range(16):  # cameras 000 to 015
                cam_folder = f"{human}_{str(cam_id).zfill(3)}"
                img_path = os.path.join(data_root, 'img', cam_folder, '0.jpg')
                intr_path = os.path.join(data_root, 'parm', cam_folder, '0_intrinsic.npy')
                extr_path = os.path.join(data_root, 'parm', cam_folder, '0_extrinsic.npy')
                depth_path = os.path.join(self.depth_path_root, 'depth_aligned', 'sapiens_0.3b',  cam_folder, '0_aligned.npy')

                if not (os.path.exists(img_path) and os.path.exists(intr_path) and os.path.exists(extr_path)):
                    continue

                if not os.path.exists(depth_path):
                    raise FileNotFoundError(f"[Depth] Missing file: {depth_path}")

                #ims.append(f"img/{cam_folder}/0.jpg")
                ims.append(os.path.join(data_root, 'img', cam_folder, '0.jpg'))

                cam_inds.append(cam_id)

                K = np.load(intr_path).astype(np.float32)
                extr = np.load(extr_path).astype(np.float32)
                R = extr[:3, :3].astype(np.float32)
                T = extr[:3, 3:].astype(np.float32)

                self.cams[human]['K'].append(K)
                self.cams[human]['R'].append(R)
                self.cams[human]['T'].append(T)
                self.cams[human]['D'].append(np.zeros(5))  # placeholder

            test_view = list(range(len(cam_inds)))

            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)

            self.start_end[human] = {
                'start': 0,
                'end': 0,
                'length': 1,
                'intv': 1
            }

        self.nrays = cfg.N_rand
        self.num_humans = len(human_list)


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

    def get_mask(self, msk_path):
        #data_info = self.ims[index].split('/')
        #human = data_info[-3]  # e.g., '0004_000'
        #frame = data_info[-1]  # e.g., '0.jpg'

        #msk_path = os.path.join(cfg.virt_data_root, 'mask', human, '0.png')#frame.replace('.jpg', '.png'))

        if os.path.exists(msk_path):
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)
        else:
            raise FileNotFoundError(f"Mask not found: {msk_path}")

        # Add boundary feathering as in original
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        if msk.ndim == 3:
            msk = msk[:, :, 0]
            msk = msk.squeeze()
        
        return msk


    def get_input_mask(self, human, index):
        cam_folder = f"{int(human):04d}_{int(index):03d}"
        msk_path = os.path.join(cfg.virt_data_root, 'mask', cam_folder, '0.png')

        if os.path.exists(msk_path):
            msk = imageio.imread(msk_path)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
                #msk = msk.unsqueeze(0)
            msk = (msk != 0).astype(np.uint8)
        else:
            raise FileNotFoundError(f"Input mask not found: {msk_path}")

        return msk


    def get_smpl_vertice(self, human, frame=0):
        obj_path = os.path.join(cfg.virt_data_root, 'smplx_obj', f'{human}.obj')
        #obj_path = sample['obj_path']
        smpl_obj = readobj(obj_path)

        #vi = np.transpose(smpl_obj['vi'])  # [3, N] → consistent with original logic
        vi = smpl_obj['vi'].astype(np.float32)
        #return vi.astype(np.float32)
        return vi


    def prepare_input(self, human, i):
        human_id = f"{int(human):04d}"
        frame_id = f"{i:03d}"
        cam_folder = f"{human_id}_{frame_id}"

        # Load SMPL vertices
        #vertices_path = os.path.join(cfg.virt_data_root, 'vertices', f"{human_id}.npy")
        #xyz = np.load(vertices_path).astype(np.float32)
        xyz = self.get_smpl_vertice(human)
        smpl_vertices = np.array(xyz) if cfg.time_steps == 1 else None

        nxyz = np.zeros_like(xyz).astype(np.float32)

        # Get original bounds
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # Load extrinsics and intrinsics
        extr_path = os.path.join(cfg.virt_data_root, 'parm', cam_folder, f"{i}_extrinsic.npy")
        intr_path = os.path.join(cfg.virt_data_root, 'parm', cam_folder, f"{i}_intrinsic.npy")

        if not os.path.exists(extr_path) or not os.path.exists(intr_path):
            raise FileNotFoundError(f"Missing camera files: {extr_path} or {intr_path}")

        extr = np.load(extr_path).astype(np.float32)
        intr = np.load(intr_path).astype(np.float32)
        R = extr[:3, :3]
        Th = extr[:3, 3]
        K = intr

        # Compute Rh from R (inverse Rodrigues)
        Rh, _ = cv2.Rodrigues(R)

        # Transform to canonical space
        xyz = np.dot(xyz - Th, R)
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # Bounds in canonical space
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # Feature construction
        cxyz = xyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # Discretize coordinates
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        out_sh = (out_sh | (32 - 1)) + 1  # Align to 32

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vertices
        
        
    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        prob = np.random.randint(9000000)
        img_path = self.ims[index]

        # Extract human and camera info
        data_info = img_path.split('/')
        # if len(data_info) < 4:
        #     raise ValueError(f"[THuman] Unexpected path format: {img_path}")

        human = data_info[-2].split('_')[0]     # '0004_000' → '0004'
        camera = data_info[-2].split('_')[1]    # '0004_000' → '000'
        frame = data_info[-1].split('.')[0]      # '0.jpg' → '0'
        
        # human = data_info[-3].split('_')[0]      # e.g., '0004_000' → '0004'
        # camera = data_info[-3].split('_')[1]     # e.g., '0004_000' → '000'
        # frame = data_info[-1].split('.')[0]      # e.g., '0.jpg' → '0'

        # Load image
        img = imageio.imread(img_path)
        if self.split == 'train' and cfg.jitter:
            img = Image.fromarray(img)
            torch.manual_seed(prob)
            img = self.jitter(img)
            img = np.array(img)
        img = img.astype(np.float32) / 255.

        # Load mask
        msk_path = os.path.join(cfg.virt_data_root, 'mask', f"{human}_{camera}", f"{frame}.png")
        # msk = imageio.imread(msk_path)

        # msk = (msk != 0).astype(np.uint8)
        msk=self.get_mask(msk_path)

        # Load intrinsics and extrinsics
        intr_path = os.path.join(cfg.virt_data_root, 'parm', f"{human}_{camera}", f"{frame}_intrinsic.npy")
        extr_path = os.path.join(cfg.virt_data_root, 'parm', f"{human}_{camera}", f"{frame}_extrinsic.npy")
        K = np.load(intr_path).astype(np.float32)
        extr = np.load(extr_path).astype(np.float32)
        R = extr[:3, :3]
        T = extr[:3, 3:]

        # Resize
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        # Apply mask
        if cfg.mask_bkgd:
            img[msk == 0] = 1. if cfg.white_bkgd else 0.
        K[:2] = K[:2] * cfg.ratio

        # Canonical feature prep
        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vert = self.prepare_input(human, int(frame))

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split
        )
        acc = if_nerf_dutils.get_acc(coord_, msk)

        # Input views — THuman has 0~15, randomly pick
        num_inputs = len(cfg['test_input_view'])
        if cfg.run_mode == 'train':
            all_views = list(range(16))
            random.shuffle(all_views)
            input_view = all_views[:num_inputs]
        else:
            input_view = cfg.test_input_view

        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        input_depths = []
        smpl_vertices = []

        for t in range(cfg.time_steps):
            tmp_imgs = []
            tmp_msks = []
            tmp_K = []
            tmp_R = []
            tmp_T = []
            tmp_depths = []

            for cam_idx in input_view:
                cam_str = f"{human}_{cam_idx:03d}"
                frame_str = f"{frame}.jpg"
                img_path = os.path.join(cfg.virt_data_root, 'img', cam_str, frame_str)
                msk_path = os.path.join(cfg.virt_data_root, 'mask', cam_str, f"{frame}.png")
                intr_path = os.path.join(cfg.virt_data_root, 'parm', cam_str, f"{frame}_intrinsic.npy")
                extr_path = os.path.join(cfg.virt_data_root, 'parm', cam_str, f"{frame}_extrinsic.npy")
                depth_p = os.path.join(self.depth_path_root, 'depth_aligned', 'sapiens_0.3b', cam_str, f"{frame}_aligned.npy")

                input_img = imageio.imread(img_path)
                input_msk = self.get_input_mask(human, cam_idx)

                # input_msk = imageio.imread(msk_path)
                # input_msk = (input_msk != 0).astype(np.uint8)

                if self.split == 'train' and cfg.jitter:
                    input_img = Image.fromarray(input_img)
                    torch.manual_seed(prob)
                    input_img = self.jitter(input_img)
                    input_img = np.array(input_img)
                input_img = input_img.astype(np.float32) / 255.

                in_K = np.load(intr_path).astype(np.float32)
                extr = np.load(extr_path).astype(np.float32)
                in_R = extr[:3, :3]
                in_T = extr[:3, 3:]

                input_img = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA)
                input_msk = cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
                # cv2.imwrite("output_img.png", input_img*255)
                # cv2.imwrite("output_msk.png", input_msk*255)
                # pdb.set_trace()

                if cfg.mask_bkgd:
                    input_img[input_msk == 0] = 1. if cfg.white_bkgd else 0.
                
                if cfg.use_viz_test and cfg.use_fg_masking:
                    if cfg.ratio == 0.5:
                        border = 5


                    kernel = np.ones((border, border), np.uint8)

                    input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                          kernel)
                
                in_K[:2] *= cfg.ratio

                # tmp_imgs.append(self.im2tensor(input_img))         # (C,H,W)
                # tmp_msks.append(self.im2tensor(input_msk).bool())  # (1,H,W)
                tmp_imgs.append(self.im2tensor(input_img).squeeze(0))         # from [1, 3, H, W] → [3, H, W]
                tmp_msks.append(self.im2tensor(input_msk).bool())  # from [1, 1, H, W] → [1, H, W]


                tmp_K.append(torch.tensor(in_K))
                tmp_R.append(torch.tensor(in_R))
                tmp_T.append(torch.tensor(in_T))


                if not os.path.exists(depth_p):
                    raise FileNotFoundError(f"[Depth] Missing input view depth: {depth_p}")
                inp_depth_np = np.load(depth_p).astype(np.float32)
                inp_depth_np = cv2.resize(inp_depth_np, (W, H), interpolation=cv2.INTER_NEAREST)
                tmp_depths.append(torch.from_numpy(inp_depth_np)) 
                #tmp_depths.append(torch.from_numpy(inp_depth_np).unsqueeze(0))

            # Stack per-view → (V, C, H, W) and append per-timestep
            # input_imgs.append(torch.stack(tmp_imgs))
            # input_msks.append(torch.stack(tmp_msks))
            img_stack = torch.stack(tmp_imgs)       # (V, 3, H, W)
            msk_stack = torch.stack(tmp_msks)       # (V, 1, H, W)
            depth_stack = torch.stack(tmp_depths)

            # Append per-timestep
            input_imgs.append(img_stack)            # List of (V, 3, H, W)
            input_msks.append(msk_stack)            # List of (V, 1, H, W)
            input_depths.append(depth_stack)


            if t == 0:
                input_K = torch.stack(tmp_K)
                input_R = torch.stack(tmp_R)
                input_T = torch.stack(tmp_T)

            if cfg.time_steps > 1:
                smpl_vertices.append(self.get_smpl_vertice(human, int(frame)))

        if cfg.time_steps == 1:
            smpl_vertices.append(smpl_vert)

        # Stack per-time-step → (T, V, C, H, W)
        input_imgs = torch.stack(input_imgs).squeeze(0)
        input_msks = torch.stack(input_msks).squeeze(0)
        input_depths = torch.stack(input_depths).squeeze(0)


        ret = {
            'smpl_vertice': smpl_vertices,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc,
            'mask_at_box': mask_at_box
        }

        meta = {
            'bounds': bounds,
            'R': cv2.Rodrigues(Rh)[0].astype(np.float32),
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': int(frame),
            'cam_ind': int(camera),
            'frame_index': int(frame),
            'human_idx': 0 if self.split == 'train' else self.human_idx_name[human],
            'input_imgs': [input_imgs],
            'input_msks': [input_msks],
            'input_depths': [input_depths],
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'target_K': K,
            'target_R': R,
            'target_T': T
        }

        ret.update(meta)
        
        return ret


    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)
