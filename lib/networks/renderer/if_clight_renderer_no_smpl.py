import torch
import torch.nn as nn
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import matplotlib.pyplot as plt
import nvdiffrast.torch as dr
import numpy as np
import gc
import math
import time
import pdb
import cv2
import torchvision
from lib.networks.renderer.raster_utils import (
    load_obj, compute_near_far, 
    perspective_projection_opencv_to_opengl, world_to_clip_space
)
import os
import matplotlib.pyplot as plt

class Renderer:

    def __init__(self, net):
        self.net = net
        self.glctx = dr.RasterizeCudaContext() 

    @torch.no_grad()
    def depth_centered_ray_samples(
        self,
        ray_o, ray_d,
        K, R, T,
        depth_map,
        n_samples: int = 5,
        step_cm: float = 1.0,
        fallback_near: float = 0.5,
        fallback_far: float = 6.0,
    ):
        """
        Build n_samples query points per ray centered around Sapiens depth (meters) for the *current* camera.

        Inputs:
            ray_o: [B, Nrays, 3] world-space ray origins (should be camera centers)
            ray_d: [B, Nrays, 3] world-space ray directions (normalized)
            K:     [B, 3, 3] or [1, 3, 3]
            R:     [B, 3, 3] or [1, 3, 3]  (world→cam)
            T:     [B, 3, 1] or [1, 3, 1]  (world→cam)
            depth_map: [B, H, W] Sapiens depth in meters for this *same* camera/view
        Returns:
            pts_world: [B, Nrays, n_samples, 3]
            t_vals:    [B, Nrays, n_samples]   (distance along the world ray)
            valid:     [B, Nrays]              (True where a center depth exists)
        """
        dev = ray_o.device
        B, Nr = ray_o.shape[:2]
        H, W = depth_map.shape[-2:]

        # make K,R,T shape [B,3,3]/[B,3,1]
        if K.shape[0] == 1 and B > 1: K = K.expand(B, -1, -1)
        if R.shape[0] == 1 and B > 1: R = R.expand(B, -1, -1)
        if T.shape[0] == 1 and B > 1: T = T.expand(B, -1, -1)

        # ray_d in camera space to get pixel coords uv = K*(dir_cam) projected
        # dir_cam = R * dir_world   (T does not affect directions)
        dir_cam = torch.matmul(R, ray_d.transpose(1,2)).transpose(1,2)              # [B,Nr,3]
        dx, dy, dz = dir_cam.unbind(-1)
        dz = dz.clamp_min(1e-6)
        # project direction to pixel coordinate
        fx, fy = K[:,0,0], K[:,1,1]
        cx, cy = K[:,0,2], K[:,1,2]
        # broadcast per-batch scalars to Nr
        fx = fx[:,None]; fy = fy[:,None]; cx = cx[:,None]; cy = cy[:,None]

        u = fx*(dx/dz) + cx
        v = fy*(dy/dz) + cy
        u_idx = u.round().long().clamp(0, W-1)
        v_idx = v.round().long().clamp(0, H-1)

        # sample center depth d0 (meters) from Sapiens
        b_idx = torch.arange(B, device=dev)[:, None].expand(B, Nr)
        d0 = depth_map[b_idx, v_idx, u_idx]                                         # [B,Nr]
        valid = d0 > 0

        # build depth samples around d0: d0 + {-2,-1,0,1,2} * 1cm
        half = (n_samples // 2)
        offsets = torch.arange(-half, half+1, device=dev, dtype=d0.dtype) * (step_cm * 0.01)
        d_all = d0[..., None] + offsets[None, None, :]                               # [B,Nr,S]

        # For rays with no valid depth, linearly span [fallback_near,fallback_far] in meters (camera Z)
        # Solve for t per sample: Pc_z = (R*(o + t d) + T)_z = b + t*a
        o_cam = (torch.matmul(R, ray_o.transpose(1,2)) + T).transpose(1,2)          # [B,Nr,3]
        a = dir_cam[..., 2].clamp_min(1e-6)                                         # [B,Nr]
        b = o_cam[..., 2]                                                           # [B,Nr]

        # fallback depths (in camera Z)
        t_lin = torch.linspace(0, 1, steps=n_samples, device=dev)[None, None, :]    # [1,1,S]
        d_fb = fallback_near*(1 - t_lin) + fallback_far*t_lin                        # [1,1,S]

        d_used = torch.where(valid[...,None], d_all, d_fb)                           # [B,Nr,S]
        t_vals = (d_used - b[...,None]) / a[...,None]                                # [B,Nr,S]

        # world points
        pts_world = ray_o[..., None, :] + t_vals[..., None] * ray_d[..., None, :]    # [B,Nr,S,3]
        return pts_world, t_vals, valid


    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):
        """
        Clark: Samples per-point features from a 2D feature map using projected 2D UV coordinates.
        uv: size[3, 6890, 2], (98.4262, 394.4973)
        """

        scale = feat_scale / image_shape
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)    # size[3, 6890, 1, 2], (-0.8015, 0.6471), normalized coordinates
        #pdb.set_trace()

        # Clark: interpolation, 
        # samples: size[3, 6890, 128, 1], (-9.82, 10.5426)
        samples = F.grid_sample(
            feat_map,   # 3, 64, 256, 256
            uv,   # 3, 6890, 1, 2
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map,
                                  pixel_feat_scale, batchify=False):

        image_shape = batch['input_imgs'][0].shape[-2:]
        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)


        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3)
        xyz = repeat_interleave(xyz, input_R.shape[0])
        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        uv = xyz[:, :, :2] / xyz[:, :, 2:]

        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, image_shape,
                                                  uv)

        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

    def prepare_sp_input(self, batch):
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)

            # pixel_feat = self.get_pixel_aligned_feature(batch,
            #                                             xyz[:, i:i + chunk],
            #                                             pixel_feat_map,
            #                                             pixel_feat_scale,
            #                                             batchify=True)
            B, Nr, S = xyz.shape[:3]
            pixfeat = self.get_pixel_aligned_feature(
                batch,
                xyz.view(B, Nr*S, 3),
                pixel_feat_map,
                pixel_feat_scale,
                batchify=True
            )  # -> [V?, Nr*S, C] but since we pass current view, will effectively be [1, Nr*S, C]

            # Collapse the view dimension if present, then restore [B,Nr,S,C]
            if pixfeat.dim() == 4 and pixfeat.shape[0] == 1:
                pixfeat = pixfeat[0]
            pixfeat = pixfeat.view(Nr, S, -1)[None, ...].contiguous()  # [1,Nr,S,C]

            ret = self.net(pixfeat, sp_input,
                           grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk],
                           light_pts[:, i:i + chunk],
                           holder=holder)

            all_ret.append(ret)

        all_ret = torch.cat(all_ret, 1)

        return all_ret


    def render(self, batch):
        dev  = batch['ray_o'].device
        ray_o = batch['ray_o']             # [B, Nr, 3]
        ray_d = batch['ray_d']             # [B, Nr, 3]
        sh    = ray_o.shape
        B, Nr = sh[0], sh[1]

        # --- keep your previous 5-point sampling around Sapiens depth ---
        cur = 0
        K_cur = batch['input_K'].reshape(-1,3,3)[cur:cur+1].to(dev)
        R_cur = batch['input_R'].reshape(-1,3,3)[cur:cur+1].to(dev)
        T_cur = batch['input_T'].reshape(-1,3,1)[cur:cur+1].to(dev)
        depth_map_cur = batch['input_depths'][0][cur:cur+1].to(dev)  # [1,H,W]

        pts, t_vals, valid_mask = self.depth_centered_ray_samples(
            ray_o, ray_d, K_cur, R_cur, T_cur, depth_map_cur,
            n_samples=getattr(cfg, 'n_depth_samples', 5),
            step_cm=getattr(cfg, 'depth_step_cm', 1.0),
            fallback_near=getattr(cfg, 'fallback_near_m', 0.5),
            fallback_far=getattr(cfg, 'fallback_far_m', 6.0),
        )  # pts: [B,Nr,S,3]; t_vals: [B,Nr,S]
        S = pts.shape[2]

        # --- embeddings as before ---
        xyz = pts.clone()
        light_pts = embedder.xyz_embedder(xyz)
        pts = self.pts_to_can_pts(pts, batch)

        ray_d0  = batch['ray_d']
        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)
        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()

        light_pts = light_pts.view(sh[0], -1, embedder.xyz_dim)
        viewdir   = viewdir.view(sh[0], -1, embedder.view_dim)

        # ====== IMPORTANT: drop sp_input / grid_coords / paint_neural_human ======
        # (they are not used anymore)
        # sp_input = self.prepare_sp_input(batch)
        # grid_coords = self.get_grid_coords(pts, sp_input, batch)
        # grid_coords = grid_coords.view(sh[0], -1, 3)

        # --- encoder per time-step (unchanged) ---
        image_list = batch['input_imgs']
        weight = None
        holder = None
        temporal_holders = []
        temporal_weights = []

        for t in range(cfg.time_steps):
            images = image_list[t].reshape(-1, *image_list[t].shape[2:])
            if t == 0:
                holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(images)
            else:
                holder_feat_map, holder_feat_scale, _, _ = self.net.encoder(images)

            # NO paint_neural_human anymore
            # weight, holder = self.paint_neural_human(...)

            if cfg.weight == 'cross_transformer' and cfg.cross_att_mode == 'cross_att':
                temporal_holders.append(holder)
                temporal_weights.append(weight)

        if cfg.time_steps == 1:
            holder = temporal_holders[0] if temporal_holders else None

        # --- pixel-aligned features for all samples (same call as before) ---
        pixel_feat = self.get_pixel_aligned_feature(
            batch, xyz, pixel_feat_map, pixel_feat_scale
        )   # shape: [V (=n_input), C, Nr*S]

        # ====== network call now takes ONLY (pixel_feat, viewdir, light_pts, holder) ======
        raw = self.net(pixel_feat, viewdir, light_pts, holder=holder)  # <— signature changed

        # --- reshape for raw2outputs exactly like before, but with our S=t_vals.size(2) ---
        raw   = raw.view(B * Nr, S, 4)
        z_vals = t_vals.view(B * Nr, S)       # use depth-centered t as z_vals
        ray_d  = ray_d.view(B * Nr, 3)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd
        )

        rgb_map   = rgb_map.view(*sh[:-1], -1)
        acc_map   = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])
        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()
        return ret
