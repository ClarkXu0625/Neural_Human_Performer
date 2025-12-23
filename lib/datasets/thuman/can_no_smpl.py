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
        self.ims = []        # full image paths (each item = one sample)
        self.cam_inds = []
        self.start_end = {}

        human_info = get_human_info.get_thuman_info(self.split)
        human_list = sorted(list(human_info.keys()))

        # (4) optional object id filter
        # obj_ids = _cfg_opt_list('object_ids')   # e.g., [4]
        # if obj_ids is not None:
        #     # ids may be ints; THuman names are '0004', etc.
        #     obj_ids = [f"{int(x):04d}" for x in obj_ids]
        #     human_list = [h for h in human_list if h in obj_ids]
        ids = _normalize_ids(_cfg_ids_for_split(self.split))
        if ids is not None:
            human_list = [h for h in human_list if h in ids]

        self.depth_path_root = os.path.join(cfg.depth_map_root, data_root.split('/')[-1])

        # pick subviews by split (1)
        if self.split == 'train':
            subviews = _cfg_list('thuman_subviews_train', [0,1,2,3,4])
        else:
            subviews = _cfg_list('thuman_subviews_test',  [0,1,2,3,4])
        print(subviews)
        
        print(self.split)

        # keep a reverse map for test indexing (unchanged idea)
        if self.split == 'test':
            self.human_idx_name = {h:i for i,h in enumerate(human_list)}

        for human in human_list:
            self.cams[human] = {'K': [], 'R': [], 'T': [], 'D': []}

            ims = []
            cam_inds = []

            for cam_id in range(16):  # camera ids 000..015
                for sub in subviews:  # subview files 0.jpg..4.jpg
                    cam_folder = f"{human}_{str(cam_id).zfill(3)}"
                    img_path   = os.path.join(data_root, 'img',  cam_folder, f'{sub}.jpg')
                    intr_path  = os.path.join(data_root, 'parm', cam_folder, f'{sub}_intrinsic.npy')
                    extr_path  = os.path.join(data_root, 'parm', cam_folder, f'{sub}_extrinsic.npy')
                    if self.split == "train":
                        depth_path = os.path.join(self.depth_path_root, 'depth_aligned_clothed', 'sapiens_1b',
                                            cam_folder, f'{sub}_aligned.npy')
                    else:
                        depth_path = os.path.join(self.depth_path_root, 'depth_aligned_clothed', 'sapiens_1b', cam_folder, f'{sub}_aligned.npy')

                    if not (os.path.exists(img_path) and os.path.exists(intr_path) and os.path.exists(extr_path)):
                        continue
                    if not os.path.exists(depth_path):
                        raise FileNotFoundError(f"[Depth] Missing file: {depth_path}")

                    ims.append(img_path)
                    cam_inds.append(cam_id)

                    # cache K/R/T (one per (human,cam,sub); storing duplicates is fine)
                    K = np.load(intr_path).astype(np.float32)
                    extr = np.load(extr_path).astype(np.float32)
                    R = extr[:3, :3].astype(np.float32)
                    T = extr[:3, 3:].astype(np.float32)
                    self.cams[human]['K'].append(K)
                    self.cams[human]['R'].append(R)
                    self.cams[human]['T'].append(T)
                    self.cams[human]['D'].append(np.zeros(5, dtype=np.float32))  # placeholder

            # record and extend global lists
            start_idx = len(self.ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)
            self.start_end[human] = {'start': start_idx, 'end': len(self.ims), 'length': len(ims), 'intv': 1}

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


    def get_input_mask(self, human, index, subview):
        cam_folder = f"{int(human):04d}_{int(index):03d}"
        msk_path = os.path.join(cfg.virt_data_root, 'mask', cam_folder, f'{int(subview)}.png')
        if os.path.exists(msk_path):
            msk = imageio.imread(msk_path)
            if msk.ndim == 3:
                msk = msk[:, :, 0]
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
        faces = smpl_obj['f'].astype(np.int32)
        #pdb.set_trace()
        return vi, faces


    def prepare_input(self, human, i):
        human_id = f"{int(human):04d}"
        frame_id = f"{i:03d}"
        cam_folder = f"{human_id}_{frame_id}"

        # Load SMPL vertices
        #vertices_path = os.path.join(cfg.virt_data_root, 'vertices', f"{human_id}.npy")
        #xyz = np.load(vertices_path).astype(np.float32)
        xyz, faces = self.get_smpl_vertice(human)
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

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vertices, faces
        
        
    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        prob = np.random.randint(9000000)
        img_path = self.ims[index]

        # Parse human/camera/subview from path
        data_info = img_path.split('/')
        human  = data_info[-2].split('_')[0]      # '0004'
        camera = data_info[-2].split('_')[1]      # '000'
        subview = data_info[-1].split('.')[0]     # '0'..'4'  (← no time dim; this IS the frame id)

        # Load image
        img = imageio.imread(img_path)
        if self.split == 'train' and cfg.jitter:
            img = Image.fromarray(img); torch.manual_seed(prob); img = np.array(self.jitter(img))
        img = img.astype(np.float32) / 255.

        # Load mask (per subview)
        msk_path = os.path.join(cfg.virt_data_root, 'mask', f"{human}_{camera}", f"{subview}.png")
        msk = self.get_mask(msk_path)

        # Load intr/extr for this subview
        intr_path = os.path.join(cfg.virt_data_root, 'parm', f"{human}_{camera}", f"{subview}_intrinsic.npy")
        extr_path = os.path.join(cfg.virt_data_root, 'parm', f"{human}_{camera}", f"{subview}_extrinsic.npy")
        K = np.load(intr_path).astype(np.float32)
        extr = np.load(extr_path).astype(np.float32)
        R = extr[:3, :3]; T = extr[:3, 3:]

        # Resize
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 1. if cfg.white_bkgd else 0.
        K[:2] *= cfg.ratio

        # Canonical features (unchanged; pass subview index in place of "frame")
        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vert, faces = self.prepare_input(human, int(subview))

        # Sample rays on target (unchanged)
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split
        )
        acc = if_nerf_dutils.get_acc(coord_, msk)

        # (2) choose input views by split (train/test) from YAML
        if self.split == 'train':
            input_view = _cfg_list('input_views_train', list(range(16)))
        else:
            input_view = _cfg_list('input_views_test',  list(range(16)))

        # Build per-view inputs using the SAME subview id
        input_imgs, input_msks, input_depths = [], [], []
        tmp_K, tmp_R, tmp_T = [], [], []

        for cam_idx in input_view:
            cam_str  = f"{human}_{cam_idx:03d}"
            frame_fn = f"{subview}"  # subview filename stem

            img_p  = os.path.join(cfg.virt_data_root, 'img',  cam_str, f"{frame_fn}.jpg")
            msk_p  = os.path.join(cfg.virt_data_root, 'mask', cam_str, f"{frame_fn}.png")
            intr_p = os.path.join(cfg.virt_data_root, 'parm', cam_str, f"{frame_fn}_intrinsic.npy")
            extr_p = os.path.join(cfg.virt_data_root, 'parm', cam_str, f"{frame_fn}_extrinsic.npy")
            if self.split == 'train':
                depth_p= os.path.join(self.depth_path_root, 'depth_aligned_clothed', 'sapiens_1b', cam_str, f"{frame_fn}_aligned.npy")
            else:
                depth_p= os.path.join(self.depth_path_root, 'depth_aligned_clothed', 'sapiens_1b', cam_str, f"{frame_fn}_aligned.npy")

            # RGB
            in_img = imageio.imread(img_p)
            if self.split == 'train' and cfg.jitter:
                in_img = Image.fromarray(in_img); torch.manual_seed(prob); in_img = np.array(self.jitter(in_img))
            in_img = in_img.astype(np.float32) / 255.
            in_img = cv2.resize(in_img, (W, H), interpolation=cv2.INTER_AREA)

            # Mask (use helper with subview)
            in_msk = self.get_input_mask(human, cam_idx, subview)
            in_msk = cv2.resize(in_msk, (W, H), interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                in_img[in_msk == 0] = 1. if cfg.white_bkgd else 0.

            # K/R/T
            K_i = np.load(intr_p).astype(np.float32)
            E_i = np.load(extr_p).astype(np.float32)
            R_i = E_i[:3, :3]; T_i = E_i[:3, 3:]
            K_i[:2] *= cfg.ratio

            # Depth (3) assumed pre-rendered
            if not os.path.exists(depth_p):
                raise FileNotFoundError(f"[Depth] Missing input view depth: {depth_p}")
            d_i = np.load(depth_p).astype(np.float32)
            d_i = cv2.resize(d_i, (W, H), interpolation=cv2.INTER_NEAREST)

            # Stack
            input_imgs.append(self.im2tensor(in_img))              # [3,H,W]
            input_msks.append(self.im2tensor(in_msk).bool())       # [1,H,W]
            input_depths.append(torch.from_numpy(d_i))             # [H,W]
            tmp_K.append(torch.tensor(K_i)); tmp_R.append(torch.tensor(R_i)); tmp_T.append(torch.tensor(T_i))

        # To tensors
        input_imgs   = torch.stack(input_imgs)      # [V,3,H,W]
        input_msks   = torch.stack(input_msks)      # [V,1,H,W]
        input_depths = torch.stack(input_depths)    # [V,H,W]
        input_K      = torch.stack(tmp_K)           # [V,3,3]
        input_R      = torch.stack(tmp_R)           # [V,3,3]
        input_T      = torch.stack(tmp_T)           # [V,3,1]
        #pdb.set_trace()

        smpl_vertices = [smpl_vert]  # keep shape compatible

        ret = {
            'smpl_vertice': smpl_vertices,
            'smpl_faces': faces,
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
            # meta
            'bounds': bounds,
            'R': cv2.Rodrigues(Rh)[0].astype(np.float32),
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': int(subview),                 # subview id
            'cam_ind': int(camera),
            'frame_index': int(subview),
            'human_idx': 0 if self.split == 'train' else self.human_idx_name[human],
            # keep time-dim as a singleton list for compatibility with your renderer
            'input_imgs':   [input_imgs],      # [1, V, 3, H, W]
            'input_msks':   [input_msks],      # [1, V, 1, H, W]
            'input_depths': [input_depths],    # [1, V, H, W]
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'target_K': K,
            'target_R': R,
            'target_T': T,
            'obj_id': int(human),
            # 'ray_flat_idx': ray_flat_idx,  # [1, N_rays]
            # 'mask_at_box_full': mask_at_box_full,          # [1, H*W]
        }
        #pdb.set_trace()
        return ret



    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)


def _cfg_list(name, default):
    v = getattr(cfg, name, None)
    return list(v) if v is not None else list(default)

def _cfg_opt_list(name):
    v = getattr(cfg, name, None)
    if v in (None, [], ()):
        return None
    return list(v)


def _cfg_ids_for_split(split):
    # prefer split-specific, fallback to global object_ids
    if split == 'train':
        ids = getattr(cfg, 'train_object_ids', None)
    else:
        ids = getattr(cfg, 'test_object_ids', None)
    if ids in (None, [], ()):
        ids = getattr(cfg, 'object_ids', None)
    return ids

def _normalize_ids(ids):
    if ids is None:
        return None
    if not isinstance(ids, (list, tuple)):
        ids = [ids]
    # THuman folder names are zero-padded strings like "0004"
    return [f"{int(x):04d}" for x in ids]
