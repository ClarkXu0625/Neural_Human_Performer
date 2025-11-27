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

    def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale,
                           prev_weight=None, prev_holder=None):
        """
        Args:
        holder_feat_map: CNN feature map from the input image (per time step), [3, 64, 256, 256]
        holder_feat_scale: [2,]
        batch: contains SMPL vertices, camera intrinsics/extrinsics, masks, etc., Size([3, 64, 256, 256])
        t: time index

        uv = (K @ (R @ X + T)) → normalize to pixel space


        Output:
        final_result: (3, 6890)	Visibility mask: True where vertex is visible in view
        big_holder: 
            (3, 6890, 64)  Values range from -14.82 to 17.49, 
            3 is the number of views
            64 is the feature dimension per vertex.
            They are image-aligned latent features, which are used for cross-attention.

        """

        smpl_vertice = batch['smpl_vertice'][t]
        dev = holder_feat_map.device

        # if cfg.rasterize:
        #     vizmap = batch['input_vizmaps'][t]  # torch.Size([1, 3, 6890])
        

        image_shape = batch['input_imgs'][t].shape[-2:]  # p\[512, 512]

        input_R = batch['input_R']  # torch.Size([1, 3, 3, 3]); (-0.9349, 0.6108)
        input_T = batch['input_T']  # torch.Size([1, 3, 3, 1]); (0.0288, 3.8127)
        input_K = batch['input_K']  # torch.Size([1, 3, 3, 3]); (0, 576.8790)

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)

        # load depth map
        depth_maps = batch['input_depths'][t]
        depth_maps = depth_maps.squeeze(0)
        depth_maps = torch.nan_to_num(depth_maps, nan=0.0, posinf=0.0, neginf=0.0)

        
        plt.imsave("debug/depth_view0.png", depth_maps[0].cpu().numpy(), cmap="jet")

        # # squeeze batch dim if present
        # if depth_maps.dim() == 5 and depth_maps.shape[0] == 1:
        #     depth_maps = depth_maps.squeeze(0)  # -> (V,1,H,W) or (V,H,W)

        # # squeeze channel dim if present
        # if depth_maps.dim() == 4 and depth_maps.shape[1] == 1:
        #     depth_maps = depth_maps.squeeze(1)  # -> (V,H,W)

        # # final guard
        # V_expected = batch['input_K'].reshape(-1,3,3).shape[0]
        # assert depth_maps.dim() == 3 and depth_maps.shape[0] == V_expected, \
        #     f"depth_maps must be [V,H,W], got {tuple(depth_maps.shape)} vs V={V_expected}"


        if cfg.use_depth_vis:
            result, z_cam, uv, uv_int = self.compute_visibility_from_depth(
                smpl_vertice.to(dev), input_K, input_R, input_T,
                depth_maps, image_shape,
                z_tol_m=getattr(cfg, 'depth_visibility_tol_m', 0.01)
            )
        else:

            if cfg.rasterize:
                result = self.compute_visibility_on_the_fly(
                    smpl_vertice, input_K, input_R, input_T, image_shape
            )
                
        # sanity check ------------------------------
        vis = result.clone().float()  # (V, Nverts) 0/1

        

        # if cfg.rasterize:
        #     gt_visibility = vizmap[0]  # torch.Size([3, 6890])
        # pdb.set_trace()

        # uv - 2D projected pixel coordinates of 3D points on the image plane 
        # Clark: convert SMPL mesh vertices from canonical space into the camera's coordinate frame
        # Clark: rotation
        vertice_rot = \
        torch.matmul(input_R[:, None], smpl_vertice.unsqueeze(-1))[..., 0]  # [3, 6890, 3], (-1.4, 1.2)

        # Clark: translation
        vertice = vertice_rot + input_T[:, None, :3, 0]

        # Clark: projection
        vertice = torch.matmul(input_K[:, None], vertice.unsqueeze(-1))[..., 0] # [3, 6890, 3], (2.37, 1434.8195)

        # Clark: normalize, convert from homogeneous to 2D image coordinates
        uv = vertice[:, :, :2] / vertice[:, :, 2:]

        # save one view as an image (project vis back to pixel space)
        for v in range(vis.shape[0]):
            # project vertices to pixels (already have uv for this view)
            u = uv[v, :, 0].round().long()
            vcoord = uv[v, :, 1].round().long()
            H, W = image_shape
            mask_img = torch.zeros((H, W), dtype=torch.uint8, device=vis.device)

            # mark visible vertices
            in_bounds = (u >= 0) & (u < W) & (vcoord >= 0) & (vcoord < H)
            mask_img[vcoord[in_bounds], u[in_bounds]] = (vis[v, in_bounds] * 255).byte()

            # move to CPU + save
            out_path = f"debug/vismap_view{v}_t{t}.png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            import imageio
            imageio.imwrite(out_path, mask_img.cpu().numpy())
        # end sanity check ------------------------------


        latent = self.sample_from_feature_map(holder_feat_map,
                                              holder_feat_scale, image_shape,
                                              uv)

        latent = latent.permute(0, 2, 1)    # torch.Size([3, 6890, 128]); (-23.2543, 28.2827)

        num_input = latent.shape[0] # 3

        # sanity check ------------------------------
        vis_rate = float(result.float().mean())
        #pdb.set_trace()

        if cfg.use_viz_test:

            final_result = result   # [3, 6890], True/False

            # big_holder: size[3, 6890, 64], (-7.2, 7.4)
            big_holder = torch.zeros((latent.shape[0], latent.shape[1],
                                      cfg.embed_size)).cuda()
            big_holder[final_result == True, :] = latent[final_result == True,
                                                  :]
            
            if cfg.weight == 'cross_transformer':
                return final_result, big_holder

        else:

            holder = latent.sum(0)
            holder = holder / num_input
            return holder
        
    @torch.no_grad()
    def compute_visibility_from_depth(self, smpl_vertice, K, R, T, depth_maps, image_shape,
                                    z_tol_m: float = 0.01):
        """
        Args:
            smpl_vertice: [N,3] or [1,N,3] world verts (N can be 6890 or 10475 etc.)
            K,R,T: [V,3,3], [V,3,3], [V,3,1]
            depth_maps: [V,H,W] float32 meters (already resized to target image size)
            image_shape: (H, W)
        Returns:
            visible: [V,N] bool
            z_cam : [V,N] float
            uv    : [V,N,2] float
            uv_int: [V,N,2] long (clamped)
        """
        # ---- Sanity + shape normalization ----
        dev = K.device
        K = K.to(dev); R = R.to(dev); T = T.to(dev)
        depth_maps = depth_maps.to(dev).contiguous()

        if smpl_vertice.dim() == 3:
            smpl_vertice = smpl_vertice.squeeze(0)   # [N,3]
        sv = smpl_vertice.to(dev).contiguous()
        V = K.shape[0]
        N = sv.shape[0]

        # Prefer true depth map size, not passed-in image_shape (avoid mismatch)
        Hdm, Wdm = depth_maps.shape[-2], depth_maps.shape[-1]
        H, W = Hdm, Wdm

        # ---- Project verts to each view ----
        # Xc = R Xw + T
        Xw = sv[None].expand(V, N, 3)                                  # [V,N,3]
        Xc = torch.matmul(R[:, None], Xw.unsqueeze(-1)).squeeze(-1)    # [V,N,3]
        Xc = Xc + T[:, None, :3, 0]                                    # [V,N,3]
        z_cam = Xc[..., 2]                                             # [V,N]
        z_pos = z_cam > 1e-6                                           # [V,N]

        # uv = (K Xc) / z
        Xc_h = torch.matmul(K[:, None], Xc.unsqueeze(-1)).squeeze(-1)  # [V,N,3]
        denom = Xc_h[..., 2].clone()
        denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
        uv = Xc_h[..., :2] / denom[..., None]                          # [V,N,2]

        # ---- Build safe integer indices ----
        u = uv[..., 0]; v = uv[..., 1]                                 # [V,N]
        # Eliminate NaN/Inf
        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))

        u_idx = u.round().long()
        v_idx = v.round().long()
        # if z <= 0, do not use these indices (set to 0 to stay in-bounds)
        u_idx = torch.where(z_pos, u_idx, torch.zeros_like(u_idx))
        v_idx = torch.where(z_pos, v_idx, torch.zeros_like(v_idx))
        u_idx = u_idx.clamp(0, W - 1)
        v_idx = v_idx.clamp(0, H - 1)

        # ---- Advanced indexing into depth map ----
        # Shapes MUST be [V,N]; depth_maps is [V,H,W].
        assert u_idx.shape == v_idx.shape == (V, N), f"idx shapes {u_idx.shape},{v_idx.shape}"
        assert depth_maps.dim() == 3 and depth_maps.shape[0] == V, f"depth_maps shape {depth_maps.shape}"

        view_ids = torch.arange(V, device=dev)[:, None].expand(V, N)   # [V,N]
        # Safe fetch (meters)
        d_samp = depth_maps[view_ids, v_idx, u_idx]                    # [V,N]
        assert d_samp.shape == (V, N), f"d_samp shape {d_samp.shape}"

        # ---- Visibility test ----
        valid_depth = d_samp > 0                                       # [V,N]
        # All operands are [V,N] 2D; no hidden 3rd dim allowed
        assert z_pos.dim() == valid_depth.dim() == z_cam.dim() == d_samp.dim() == 2
        visible = z_pos & valid_depth & (z_cam <= d_samp + float(z_tol_m))

        uv_int = torch.stack([u_idx, v_idx], dim=-1)                   # [V,N,2]
        return visible, z_cam, uv, uv_int



    @torch.no_grad()
    def depth_centered_samples(self, uv_int, K, R, T, depth_maps, image_shape,
                               N: int = 5, band_m: float = 0.02,
                               fallback_near: float = 0.5, fallback_far: float = 6.0):
        """
        Build depth-centered samples (meters) along camera rays.
        Args:
            uv_int: [V,N,2] integer pixel coords
            K,R,T:  [V,3,3],[V,3,3],[V,3,1]
            depth_maps: [V,H,W] meters
            image_shape: (H,W)
            N: samples per vertex per view
            band_m: +/- band around measured depth
        Returns:
            pts_world: [V,N,Ns,3]
            t_vals:    [V,N,Ns]
            valid:     [V,N] (has measured depth)
        """
        H, W = image_shape
        dev = K.device
        V, Nverts = uv_int.shape[:2]

        # camera centers: Cw = -R^T T
        Rt = R.transpose(1, 2)                                                # [V,3,3]
        Cw = -torch.matmul(Rt, T)[..., 0]                                      # [V,3]

        # K^{-1}
        Kinv = torch.inverse(K)                                               # [V,3,3]

        # build pixel dirs (u,v,1) -> cam -> world
        u = uv_int[..., 0].float(); v = uv_int[..., 1].float()                 # [V,N]
        ones = torch.ones_like(u)
        pix = torch.stack([u, v, ones], dim=-1)                                # [V,N,3]
        cam_dirs = torch.matmul(Kinv[:, None], pix.unsqueeze(-1)).squeeze(-1)  # [V,N,3]
        ray_dirs_w = torch.matmul(Rt[:, None], cam_dirs.unsqueeze(-1)).squeeze(-1)
        ray_dirs_w = torch.nn.functional.normalize(ray_dirs_w, dim=-1)         # [V,N,3]

        # center depths
        d_center = depth_maps[torch.arange(V, device=dev)[:, None], v.long(), u.long()]  # [V,N]
        valid = d_center > 0

        t_min = torch.where(valid, (d_center - band_m).clamp_min(1e-3),
                            torch.full_like(d_center, fallback_near))
        t_max = torch.where(valid, d_center + band_m,
                            torch.full_like(d_center, fallback_far))

        t_lin = torch.linspace(0, 1, steps=N, device=dev)[None, None, :]       # [1,1,N]
        t_vals = (t_min[..., None] * (1 - t_lin) + t_max[..., None] * t_lin)   # [V,N,N]

        Cw_exp = Cw[:, None, None, :]                                          # [V,1,1,3]
        pts_world = Cw_exp + t_vals[..., None] * ray_dirs_w[:, :, None, :]     # [V,N,N,3]
        return pts_world, t_vals, valid

    # # ─────────────────────────────────────────────────────────────────────
    # # Main: use depth-based visibility & optional depth-centered sampling
    # # ─────────────────────────────────────────────────────────────────────
    # def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale,
    #                        prev_weight=None, prev_holder=None):
    #     """
    #     holder_feat_map: (V, C=featdim, Hf, Wf)
    #     batch['input_depths'][t]: (V, H, W) meters
    #     Returns as before:
    #         if cfg.use_viz_test:
    #             final_result: (V, 6890) bool visibility
    #             big_holder:  (V, 6890, embed_size)
    #         else:
    #             holder:      (6890, embed_size) averaged across views
    #     """
    #     dev = holder_feat_map.device
    #     smpl_vertice = batch['smpl_vertice'][t]             # [6890,3] or [1,6890,3]
    #     image_shape = batch['input_imgs'][t].shape[-2:]     # (H, W)

    #     input_R = batch['input_R'].reshape(-1, 3, 3).to(dev)    # [V,3,3]
    #     input_T = batch['input_T'].reshape(-1, 3, 1).to(dev)    # [V,3,1]
    #     input_K = batch['input_K'].reshape(-1, 3, 3).to(dev)    # [V,3,3]

    #     # Depth maps in meters for each input view
    #     depth_maps = batch['input_depths'][t].to(dev)           # [V,H,W]

    #     # ---- Depth-based visibility (z-test) ----
    #     use_depth_vis = getattr(cfg, 'use_depth_visibility', True)
    #     if use_depth_vis:
    #         result, z_cam, uv, uv_int = self.compute_visibility_from_depth(
    #             smpl_vertice.to(dev), input_K, input_R, input_T,
    #             depth_maps, image_shape,
    #             z_tol_m=getattr(cfg, 'depth_visibility_tol_m', 0.01)
    #         )
    #     else:
    #         # Fallback to rasterizer (legacy)
    #         result = self.compute_visibility_on_the_fly(
    #             smpl_vertice.to(dev), input_K, input_R, input_T, image_shape
    #         )
    #         # Compute uv for feature sampling anyway
    #         if smpl_vertice.dim() == 3:
    #             sv = smpl_vertice.squeeze(0).to(dev)
    #         else:
    #             sv = smpl_vertice.to(dev)
    #         Xc = torch.matmul(input_R[:, None], sv.unsqueeze(-1)).squeeze(-1) + input_T[:, None, :3, 0]
    #         Xc_h = torch.matmul(input_K[:, None], Xc.unsqueeze(-1)).squeeze(-1)
    #         uv = Xc_h[..., :2] / Xc_h[..., 2:].clamp_min(1e-6)

    #     # ---- Sample pixel-aligned features at uv ----
    #     latent = self.sample_from_feature_map(holder_feat_map,
    #                                           holder_feat_scale, image_shape, uv)
    #     latent = latent.permute(0, 2, 1)    # (V, 6890, embed_size)
    #     num_input = latent.shape[0]         # V

    #     # ---- Optional: depth-centered sampling band (for efficient NeRF queries) ----
    #     if getattr(cfg, 'use_depth_centered_sampling', False):
    #         # integer uv for depth lookups
    #         if 'uv_int' not in locals():
    #             # make if coming from rasterizer path
    #             u = uv[..., 0].round().long().clamp(0, image_shape[1]-1)
    #             v = uv[..., 1].round().long().clamp(0, image_shape[0]-1)
    #             uv_int = torch.stack([u, v], dim=-1)
    #         pts_world, t_vals, valid_depth = self.depth_centered_samples(
    #             uv_int, input_K, input_R, input_T, depth_maps, image_shape,
    #             N=getattr(cfg, 'depth_samples_N', 5),
    #             band_m=getattr(cfg, 'depth_band_m', 0.02),
    #             fallback_near=getattr(cfg, 'fallback_near_m', 0.5),
    #             fallback_far=getattr(cfg, 'fallback_far_m', 6.0)
    #         )
    #         # If you want to gate features further or compute view-dependent cues,
    #         # you can use pts_world middle sample here.

    #     # ---- Assemble outputs (same interface as before) ----
    #     if getattr(cfg, 'use_viz_test', False):
    #         final_result = result  # (V,6890) bool

    #         big_holder = torch.zeros((latent.shape[0], latent.shape[1],
    #                                   cfg.embed_size), device=dev)
    #         big_holder[final_result] = latent[final_result]

    #         if getattr(cfg, 'weight', 'cross_transformer') == 'cross_transformer':
    #             # Optionally, also return sampling outputs if requested
    #             if getattr(cfg, 'return_depth_samples', False) and getattr(cfg, 'use_depth_centered_sampling', False):
    #                 return final_result, big_holder, (pts_world, t_vals, valid_depth)
    #             return final_result, big_holder
    #     else:
    #         holder = latent.sum(0) / num_input  # (6890, embed_size)
    #         return holder



    def compute_visibility_on_the_fly(self, vertices, K, R, T, image_shape):
        """
        Compute per-vertex visibility using nvdiffrast and camera projection.
        Args:
            vertices: SMPL vertices [1, 6890, 3]
            K, R, T: camera intrinsics/extrinsics [3, 3, 3], [3, 3, 3], [3, 3, 1]
            image_shape: (H, W)
        Returns:
            visibility_mask: [3, 6890] (bool) visibility per view
        """
        H, W = image_shape
        device = vertices.device

        # Remove batch dim [1, 6890, 3] -> [6890, 3]
        vertices = vertices.squeeze(0)

        # Load SMPL .obj with only vertices and face indices
        obj_path = 'data/smplx/smpl/smpl_uv_neutral.obj'
        verts_list, faces_list, _ = load_obj(obj_path)
        

        # Convert to tensors
        faces = [ [v_idx for (v_idx, _) in face] for face in faces_list ]
        faces = torch.tensor(faces, dtype=torch.int32, device=device)  # [F, 3]

        # Create visibility mask [3, 6890]
        visibility_mask = torch.zeros((3, vertices.shape[0]), dtype=torch.bool, device=device)

        for view in range(3):
            R_i = R[view]
            T_i = T[view]
            K_i = K[view]

            # Construct 4x4 extrinsic matrix
            extr = torch.eye(4, device=device)
            extr[:3, :3] = R_i
            extr[:3, 3] = T_i[:, 0]

            # Compute near and far clipping planes
            near, far = compute_near_far(vertices, extr[:3])

            # Compute OpenGL projection matrix
            P_clip, translate = perspective_projection_opencv_to_opengl(
                K_i[0, 0], K_i[1, 1], K_i[0, 2], K_i[1, 2],
                near=near, far=far, width=W, height=H, device=device)


            # Project vertices to clip space
            pos_clip = world_to_clip_space(vertices, extr[:3], P_clip, translate)  # [6890, 4]
            pos_clip = pos_clip.unsqueeze(0).contiguous()  # [1, 6890, 4]

            # Rasterize
            rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=[W, H])

            face_ids = rast_out[..., 3][0]  # [H, W]
            visible_faces = face_ids.unique().long() - 1
            visible_faces = visible_faces[(visible_faces >= 0) & (visible_faces < faces.shape[0])]

            if visible_faces.numel() == 0:
                continue

            visible_verts = torch.unique(faces[visible_faces])
            visibility_mask[view, visible_verts] = True

        return visibility_mask  # [3, 6890]



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

            pixel_feat = self.get_pixel_aligned_feature(batch,
                                                        xyz[:, i:i + chunk],
                                                        pixel_feat_map,
                                                        pixel_feat_scale,
                                                        batchify=True)

            ret = self.net(pixel_feat, sp_input,
                           grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk],
                           light_pts[:, i:i + chunk],
                           holder=holder)

            all_ret.append(ret)

        all_ret = torch.cat(all_ret, 1)

        return all_ret


    def render(self, batch):

        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)


        xyz = pts.clone()
        light_pts = embedder.xyz_embedder(pts)


        pts = self.pts_to_can_pts(pts, batch)

        ray_d0 = batch['ray_d']


        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)
        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()


        light_pts = light_pts.view(sh[0], -1,
                                   embedder.xyz_dim)
        viewdir = viewdir.view(sh[0], -1,
                               embedder.view_dim)

        sp_input = self.prepare_sp_input(batch)
        grid_coords = self.get_grid_coords(pts, sp_input,
                                           batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)

        ### --- get feature maps from encoder
        image_list = batch['input_imgs']


        weight = None
        holder = None

        temporal_holders = []
        temporal_weights = []

        for t in range(cfg.time_steps):

            images = image_list[t].reshape(-1, *image_list[t].shape[2:])


            if t == 0:
                holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(
                    images)

            else:
                holder_feat_map, holder_feat_scale, _, _ = self.net.encoder(
                    images)


            ### --- paint the holder
            weight, holder = self.paint_neural_human(batch, t,
                                                     holder_feat_map,
                                                     holder_feat_scale,
                                                     weight, holder)

            if cfg.weight == 'cross_transformer':

                if cfg.cross_att_mode == 'cross_att':
                    temporal_holders.append(holder)
                    temporal_weights.append(weight)

        if cfg.time_steps == 1:
            holder = temporal_holders[0]

        if ray_o.size(1) <= 2048:

            pixel_feat = self.get_pixel_aligned_feature(batch, xyz,
                                                        pixel_feat_map,
                                                        pixel_feat_scale)

            raw = self.net(pixel_feat, sp_input, grid_coords, viewdir,
                           light_pts, holder=holder)

        else:
            raw = self.batchify_rays(sp_input, grid_coords, viewdir,
                                     light_pts,
                                     chunk=1024 * 32, net_c=None,
                                     batch=batch, xyz=xyz,
                                     pixel_feat_map=pixel_feat_map,
                                     pixel_feat_scale=pixel_feat_scale,
                                     holder=holder)
        #pdb.set_trace()

        raw = raw.reshape(-1, z_vals.size(2), 4)
        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)



        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)


        rgb_map = rgb_map.view(*sh[:-1], -1)
        acc_map = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])

        # # --- Debug: save cropped NeRF depth image (assuming B = 1) ---
        nerf_sigma = torch.relu(raw[..., 3])      # [N_rays, N_samples]
        nerf_z_vals = z_vals.clone()              # [N_rays, N_samples]
        # if not self.net.training:
        #     human_idx = batch.get("human_idx").item()
        #     import os
        #     import imageio
        #     import numpy as np

        #     # depth_map[0]: [N_rays] corresponding to mask_at_box == True
        #     depth_crop = self._make_cropped_depth(depth_map[0], batch)  # [h, w]

        #     # Normalize for visualization
        #     d = depth_crop
        #     d_min, d_max = float(np.min(d)), float(np.max(d))
        #     if d_max > d_min:
        #         d_vis = (d - d_min) / (d_max - d_min + 1e-8)
        #     else:
        #         d_vis = np.zeros_like(d, dtype=np.float32)

        #     os.makedirs("debug_NeRF", exist_ok=True)
        #     filename = "debug_NeRF/" + human_idx + "_nerf_depth_cropped.png"
        #     imageio.imwrite(
        #         filename,
        #         (d_vis * 255).astype(np.uint8)
        #     )
        # pdb.set_trace()

        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()
            B, Nrays = sh[0], sh[1]
            N_samples = nerf_sigma.shape[1]
            ret['nerf_sigma'] = nerf_sigma.view(B, Nrays, N_samples)
            ret['nerf_z_vals'] = nerf_z_vals.view(B, Nrays, N_samples)

        return ret


    def _make_cropped_depth(self, depth_pred, batch):
        """
        Reconstruct a full HxW depth image from masked 1D depth array and crop
        to the tight bounding box of the mask, mirroring _make_cropped_images.
        
        Args:
            depth_pred: [N_rays] depth values corresponding to mask_at_box == True
            batch: contains 'mask_at_box'
        Returns:
            depth_crop: [h, w] float32 cropped depth image
        """
        import numpy as np
        import cv2

        # mask_at_box: [H*W] -> [H, W] bool
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W).astype(bool)

        # depth_pred: [N_rays] (same order as mask_at_box[mask_at_box])
        depth_pred = depth_pred.detach().cpu().numpy().astype(np.float32)

        # Initialize full depth map (background = 0)
        depth_full = np.zeros((H, W), dtype=np.float32)
        depth_full[mask_at_box] = depth_pred

        # Crop tight bbox around the mask (same as RGB)
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        depth_crop = depth_full[y:y+h, x:x+w]

        return depth_crop
