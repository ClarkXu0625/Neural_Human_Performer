import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity
import os
import cv2
import matplotlib.pyplot as plt
from termcolor import colored
from lpips import LPIPS
import torch
import re
import subprocess
from collections import defaultdict
import pdb
import pandas as pd
import glob
import shutil

class Evaluator:
    def __init__(self):
        self.lpips_alex = []
        self.lpips_vgg = []
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.per_view_metrics = defaultdict(list) 

        # New: init LPIPS models
        self.lpips_alex_model = LPIPS(net='alex')
        self.lpips_vgg_model = LPIPS(net='vgg')
        # if torch.cuda.is_available():
        #     pdb.set_trace()
        #     self.lpips_alex_model = self.lpips_alex_model.cuda()
        #     self.lpips_vgg_model = self.lpips_vgg_model.cuda()
        if torch.cuda.is_available():
            #pdb.set_trace()  # good for checking device/activation during init
            self.lpips_alex_model = self.lpips_alex_model.cuda().eval()
            self.lpips_vgg_model = self.lpips_vgg_model.cuda().eval()
        else:
            self.lpips_alex_model = self.lpips_alex_model.eval()
            self.lpips_vgg_model = self.lpips_vgg_model.eval()


        result_dir = os.path.join(cfg.result_dir,
                                  'epoch_' + str(cfg.test.epoch),
                                  cfg.exp_folder_name)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def psnr_metric(self, img_pred, img_gt):

        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        if cfg.white_bkgd:
            img_gt = np.ones((H, W, 3))
        else:
            img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]

        result_dir = os.path.join(cfg.result_dir,
                                  'epoch_' + str(cfg.test.epoch),
                                  cfg.exp_folder_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        human_dir = os.path.join(result_dir, str(batch['human_idx'].item()))
        if not os.path.exists(human_dir):
            os.makedirs(human_dir)

        pred_dir = os.path.join(human_dir, 'pred')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        gt_dir = os.path.join(human_dir, 'gt')
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        input_dir = os.path.join(human_dir, 'input')
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{}_view{}.png'.format(pred_dir, frame_index,
                                           view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{}_view{}_gt.png'.format(gt_dir, frame_index,
                                              view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        for t in range(cfg.time_steps):
            for view in range(len(cfg.training_view)):
                tmp = batch['input_imgs'][t][0][view]

                tmp = tmp.data.detach().cpu().numpy().transpose(1, 2, 0)
                (tmp * 255).astype(np.uint8)
                plt.imsave(
                    '{}/frame{}_t_{}_view_{}.png'.format(input_dir, frame_index,
                                                         t, view), tmp)

        # compute the ssim
        #ssim = structural_similarity(img_pred, img_gt, multichannel=True)
        ssim = structural_similarity(
            img_pred,
            img_gt,
            win_size=min(img_pred.shape[0], img_pred.shape[1], 7),
            channel_axis=-1,
            data_range=1.0
        )

        return ssim
    

    def _to_lpips_tensor(self, img_np):
        """
        img_np: HxW (grayscale) or HxWxC in [0,1], C in {1,3,4}
        returns: 1x3xHxW tensor in [-1,1]
        """
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]  # drop alpha if present

        t = torch.from_numpy(img_np.astype(np.float32))  # H W C
        t = t.permute(2, 0, 1).unsqueeze(0)              # 1 C H W
        t = 2.0 * t - 1.0                                # [0,1]->[-1,1]
        return t.to(self.lpips_alex_model.net.parameters().__next__().device, non_blocking=True)
    
    def _make_cropped_images(self, rgb_pred, rgb_gt, batch):
        # Reconstruct full HxW images from masked rgb arrays
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W).astype(bool)

        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3), dtype=np.float32)
            img_gt   = np.ones((H, W, 3), dtype=np.float32)
        else:
            img_pred = np.zeros((H, W, 3), dtype=np.float32)
            img_gt   = np.zeros((H, W, 3), dtype=np.float32)

        # rgb_pred / rgb_gt are Nx3 in [0,1]
        img_pred[mask_at_box] = rgb_pred
        img_gt[mask_at_box]   = rgb_gt

        # Crop tight bbox around the mask (same as in ssim_metric)
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y+h, x:x+w]
        img_gt   = img_gt[y:y+h, x:x+w]
        return img_pred, img_gt

    def _make_cropped_depth(self, depth_pred, batch):
        """
        Reconstruct full HxW depth image from 1D depth_pred (N_rays),
        using mask_at_box to scatter into the image, and crop it
        exactly the same way as _make_cropped_images.
        """
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W).astype(bool)

        depth_pred = depth_pred.detach().cpu().numpy().astype(np.float32)

        depth_full = np.zeros((H, W), dtype=np.float32)
        depth_full[mask_at_box] = depth_pred

        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        depth_crop = depth_full[y:y+h, x:x+w]
        return depth_crop

    
    def _nerf_depth_from_sigma(self, sigma, z_vals):
        """
        Compute NeRF-style expected depth from per-ray densities and z samples.
        sigma:  [N_rays, N_samples] torch
        z_vals: [N_rays, N_samples] torch
        returns: depth: [N_rays] torch
        """
        # Ensure 2D
        if sigma.dim() == 3:
            # e.g., [B, N_rays, N_samples] -> flatten batch
            B, N_rays, N_samples = sigma.shape
            sigma = sigma.view(B * N_rays, N_samples)
            z_vals = z_vals.view(B * N_rays, N_samples)

        # Standard NeRF volume rendering math
        delta = z_vals[..., 1:] - z_vals[..., :-1]
        delta = torch.cat([delta, delta[..., -1:]], dim=-1)   # [N_rays, N_samples]

        alpha = 1.0 - torch.exp(-sigma * delta)               # [N_rays, N_samples]
        # transmittance T_i = Π_{j<i} (1 - α_j)
        T = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]                                           # [N_rays, N_samples]

        weights = T * alpha                                   # [N_rays, N_samples]

        depth = torch.sum(weights * z_vals, dim=-1)           # [N_rays]
        # Optional normalization:
        # depth = depth / (weights.sum(dim=-1) + 1e-10)
        return depth



    def lpips_metric(self, img_pred, img_gt, batch):
        """Inputs are numpy float32 in [0,1]. Convert to torch [-1,1]."""

        img_pred, img_gt = self._make_cropped_images(img_pred, img_gt, batch)
        

        ###### Save to flat folder for FID at end of epoch
        result_root = os.path.join(cfg.result_dir,
                                'epoch_' + str(cfg.test.epoch),
                                cfg.exp_folder_name)
        os.makedirs(result_root, exist_ok=True)

        flat_pred_dir = os.path.join(result_root, 'pred')
        flat_gt_dir   = os.path.join(result_root, 'gt')

        os.makedirs(flat_pred_dir, exist_ok=True)
        os.makedirs(flat_gt_dir, exist_ok=True)
        
        human = batch['human_idx'].item()
        frame = batch['frame_index'].item()
        view  = batch['cam_ind'].item()
        fname = f"{human}_frame{frame}_view{view}.png"

        cv2.imwrite(os.path.join(flat_pred_dir, fname), (img_pred[..., [2, 1, 0]] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(flat_gt_dir, fname),   (img_gt[..., [2, 1, 0]] * 255).astype(np.uint8))
        ##### end writing

        # Guard against empty crops (can happen if mask is empty)
        if img_pred.size == 0 or img_gt.size == 0:
            return float('nan'), float('nan')

        t_pred = self._to_lpips_tensor(img_pred)
        t_gt   = self._to_lpips_tensor(img_gt)

        with torch.no_grad():
            alex_val = float(self.lpips_alex_model(t_pred, t_gt).item())
            vgg_val  = float(self.lpips_vgg_model(t_pred,  t_gt).item())
        return alex_val, vgg_val
    
    def _compute_fid(self, pred_dir, gt_dir, device='cuda'):
        """
        Runs pytorch_fid and returns the numeric FID as float.
        Falls back to CPU if CUDA isn't available.
        """
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        # Run pytorch_fid and capture stdout
        proc = subprocess.run(
            ['python', '-m', 'pytorch_fid', '--device', device, pred_dir, gt_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        out = proc.stdout

        # Typical line contains: "FID: 12.3456"
        m = re.search(r'FID[:\s]+([0-9]*\.?[0-9]+)', out)
        if m:
            return float(m.group(1)), out
        else:
            # If format changes, still return full log for debugging
            return None, out
    # def _compute_fid(self, pred_dir, gt_dir, device='cuda'):
    #     """
    #     Verifies image dimensions and runs pytorch_fid.
    #     Returns (fid_value, log_output) or (None, error_log).
    #     """
    #     import PIL.Image as Image

    #     def get_image_shapes(folder):
    #         shapes = {}
    #         for fname in sorted(os.listdir(folder)):
    #             if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
    #                 fpath = os.path.join(folder, fname)
    #                 try:
    #                     img = Image.open(fpath)
    #                     shapes[fname] = img.size  # (W, H)
    #                 except Exception as e:
    #                     shapes[fname] = f"ERROR: {str(e)}"
    #         return shapes

    #     # Step 1: Check shapes in both folders
    #     pred_shapes = get_image_shapes(pred_dir)
    #     gt_shapes = get_image_shapes(gt_dir)

    #     # Step 2: Compare shapes between pred and gt
    #     mismatched = []
    #     for fname in pred_shapes:
    #         if fname in gt_shapes and pred_shapes[fname] != gt_shapes[fname]:
    #             mismatched.append((fname, pred_shapes[fname], gt_shapes[fname]))

    #     if mismatched:
    #         log = "[❌] Mismatched image sizes detected:\n"
    #         for fname, ps, gs in mismatched:
    #             log += f"  {fname}: pred {ps} vs gt {gs}\n"
    #         return None, log

    #     # Step 3: Fall back to CPU if needed
    #     if device == 'cuda' and not torch.cuda.is_available():
    #         device = 'cpu'

    #     # Step 4: Run FID subprocess
    #     proc = subprocess.run(
    #         ['python', '-m', 'pytorch_fid', '--device', device, pred_dir, gt_dir],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT,
    #         text=True,
    #         check=False
    #     )
    #     out = proc.stdout

    #     # Step 5: Parse FID score
    #     m = re.search(r'FID[:\s]+([0-9]*\.?[0-9]+)', out)
    #     if m:
    #         return float(m.group(1)), out
    #     else:
    #         return None, "[❌] FID subprocess failed.\n" + out
    

    def evaluate(self, output, batch):
        # allowed_views = {3, 8, 13}
        # view_idx = int(batch['cam_ind'].item())
        # if view_idx not in allowed_views:
        #     return
        #pdb.set_trace()

        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        # --- NEW: compute NeRF depth from density and save via viz_depth_map ---
        if 'nerf_sigma' in output and 'nerf_z_vals' in output:
            sigma = output['nerf_sigma']      # [B, N_rays, N_samples]
            z_vals = output['nerf_z_vals']    # [B, N_rays, N_samples]

            # Compute NeRF expected depth from densities
            nerf_depth_1d = self._nerf_depth_from_sigma(sigma, z_vals)  # [B*N_rays] or [B, N_rays]

            # Reshape: [B, N_rays] → take batch 0
            if nerf_depth_1d.dim() == 1:
                # rgb_map: [B, N_rays, 3] → get N_rays from rgb_map
                B, N_rays, _ = output['rgb_map'].shape
                nerf_depth_1d = nerf_depth_1d.view(B, N_rays)
            nerf_depth_1d = nerf_depth_1d[0]  # [N_rays]

            # Crop the depth using mask_at_box
            depth_crop = self._make_cropped_depth(nerf_depth_1d, batch)  # [h, w] numpy float32

            # Convert to torch for viz_depth_map
            depth_crop_torch = torch.from_numpy(depth_crop)[None, None]   # shape [1, 1, h, w]

            # Prepare debug directory
            result_root = os.path.join(
                cfg.result_dir,
                'epoch_' + str(cfg.test.epoch),
                cfg.exp_folder_name
            )
            debug_dir = os.path.join(result_root, "debug_nerf")
            os.makedirs(debug_dir, exist_ok=True)

            human_idx = int(batch['human_idx'].item())
            frame_idx = int(batch['frame_index'].item())
            view_idx  = int(batch['cam_ind'].item())

            base = f"human{human_idx}_frame{frame_idx}_view{view_idx}"

            # === Save visualization using your plotting function ===
            depth_png_path = os.path.join(debug_dir, f"{base}_depth.png")
            self.viz_depth_map(depth_crop_torch, fname=depth_png_path)

            # === Save raw .npy depth ===
            np.save(
                os.path.join(debug_dir, f"{base}_depth.npy"),
                depth_crop.astype(np.float32)
            )
        # --- END NEW BLOCK ---



        mse_str = 'mse: {}'.format(np.mean(self.mse))
        psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
        ssim_str = 'ssim: {}'.format(np.mean(self.ssim))

        print(mse_str)
        print(psnr_str)
        print(ssim_str)

        # new: LPIPS
        alex_val, vgg_val = self.lpips_metric(rgb_pred, rgb_gt, batch)
        self.lpips_alex.append(alex_val)
        self.lpips_vgg.append(vgg_val)

        print(f"LPIPS Alex: {alex_val:.4f}, VGG: {vgg_val:.4f}")
        print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

        human_idx = int(batch['human_idx'].item())
        cam_ind = int(batch['cam_ind'].item())

        # Save to per_view_metrics
        self.per_view_metrics[(human_idx, cam_ind)].append({
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'lpips_alex': alex_val,
            'lpips_vgg': vgg_val
        })

    def viz_depth_map(self, depth, fname="debug_depth.png", vmin=None, vmax=None):
        """
        depth: torch.Tensor
            - Either [B, 1, H, W] (image-like) 
            - Or [B, N] (per-ray depth; we will reshape if needed).
        Only uses batch index 0.
        """
        depth = depth.detach().float().cpu()

        if depth.ndim == 4:
            # [B, 1, H, W]
            d0 = depth[0, 0]  # [H, W]
        elif depth.ndim == 3 and depth.size(1) == 1:
            # [B, 1, N] -> not common, but handle
            # try to guess square-ish shape
            B, _, N = depth.shape
            side = int(N ** 0.5)
            d0 = depth[0, 0, :side * side].view(side, side)
        elif depth.ndim == 2:
            # [B, N] → try to reshape into (H, W) using target_hw if you have it
            B, N = depth.shape
            # naive guess: square
            side = int(N ** 0.5)
            d0 = depth[0, :side * side].view(side, side)
        else:
            raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

        d0 = d0.clone()
        valid = d0 > 0

        if valid.any():
            d_valid = d0[valid]
            lo = d_valid.min().item()
            hi = d_valid.max().item()
            # normalize valid region to [0,1]
            if hi > lo:
                d0[valid] = (d0[valid] - lo) / (hi - lo)
        else:
            # nothing valid; just zeros
            d0[:] = 0.0

        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)

        plt.figure(figsize=(5, 5))
        plt.imshow(d0.numpy(), cmap="magma", vmin=vmin, vmax=vmax)
        plt.colorbar(label="normalized depth")
        plt.title(os.path.basename(fname))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
    


    def summarize(self):

        result_root = os.path.join(cfg.result_dir,
                                'epoch_' + str(cfg.test.epoch),
                                cfg.exp_folder_name)
        os.makedirs(result_root, exist_ok=True)

        # Save per-sample arrays
        np.save(os.path.join(result_root, 'mse.npy'), self.mse)
        np.save(os.path.join(result_root, 'psnr.npy'), self.psnr)
        np.save(os.path.join(result_root, 'ssim.npy'), self.ssim)
        np.save(os.path.join(result_root, 'lpips_alex.npy'), self.lpips_alex)
        np.save(os.path.join(result_root, 'lpips_vgg.npy'), self.lpips_vgg)

        # Means
        exp_str   = f"experiment: {cfg.exp_name}"
        epoch_str = f"epoch: {cfg.test.epoch}"
        mse_str   = f"mse: {np.mean(self.mse):.6f}" if len(self.mse) else "mse: N/A"
        psnr_str  = f"psnr: {np.mean(self.psnr):.6f}" if len(self.psnr) else "psnr: N/A"
        ssim_str  = f"ssim: {np.mean(self.ssim):.6f}" if len(self.ssim) else "ssim: N/A"
        lpips_alex_str = f"lpips_alex: {np.mean(self.lpips_alex):.6f}" if len(self.lpips_alex) else "lpips_alex: N/A"
        lpips_vgg_str  = f"lpips_vgg: {np.mean(self.lpips_vgg):.6f}" if len(self.lpips_vgg) else "lpips_vgg: N/A"

        print(exp_str); print(epoch_str)
        print(mse_str); print(psnr_str); print(ssim_str)
        print(lpips_alex_str); print(lpips_vgg_str)

        # Compute FID once at the end on saved folders
        pred_dir = os.path.join(result_root, 'pred')
        gt_dir   = os.path.join(result_root, 'gt')
        # #new
        # os.makedirs(flat_pred_dir, exist_ok=True)
        # os.makedirs(flat_gt_dir, exist_ok=True)

        # # Copy all nested preds/GTs into flat folders
        # for human_id in os.listdir(result_root):
        #     human_dir = os.path.join(result_root, human_id)

        #     pred_dir = os.path.join(human_dir, 'pred')
        #     gt_dir   = os.path.join(human_dir, 'gt')

        #     for pred_file in glob.glob(os.path.join(pred_dir, '*.png')):
        #         fname = f"{human_id}_" + os.path.basename(pred_file)
        #         shutil.copyfile(pred_file, os.path.join(flat_pred_dir, fname))

        #     for gt_file in glob.glob(os.path.join(gt_dir, '*.png')):
        #         fname = f"{human_id}_" + os.path.basename(gt_file)
        #         shutil.copyfile(gt_file, os.path.join(flat_gt_dir, fname))
        #pdb.set_trace() #end

        # process the image in flat directory and compute fid
        # Collect sizes of all images
        def get_image_shapes(folder):
            return [cv2.imread(os.path.join(folder, f)).shape[:2]
                    for f in os.listdir(folder) if f.endswith('.png')]

        pred_shapes = get_image_shapes(pred_dir)
        gt_shapes = get_image_shapes(gt_dir)
        all_shapes = pred_shapes + gt_shapes
        max_H = max(h for h, w in all_shapes)
        max_W = max(w for h, w in all_shapes)
        target_size = (max_H, max_W)

        # Pad and overwrite
        for folder in [pred_dir, gt_dir]:
            for fname in os.listdir(folder):
                if not fname.endswith('.png'): continue
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)
                padded = pad_to_max_size(img, target_size)
                cv2.imwrite(fpath, padded)

        fid_str = "fid: N/A"
        fid_val = None
        fid_log = ""
        #pdb.set_trace()
        if os.path.exists(pred_dir) and os.path.exists(gt_dir):
            fid_val, fid_log = self._compute_fid(pred_dir, gt_dir, device='cuda')
            if fid_val is not None:
                fid_str = f"fid: {fid_val:.6f}"
                np.save(os.path.join(result_root, 'fid.npy'), np.array([fid_val], dtype=np.float32))
            else:
                fid_str = "fid: parse_failed"
        else:
            fid_log = "Warning: pred/gt folders not found, skipping FID."

        print(fid_str)

        # Write summary.txt (append FID too)
        with open(os.path.join(result_root, 'summary.txt'), 'w') as out:
            out.writelines([
                exp_str + "\n",
                epoch_str + "\n",
                mse_str + "\n",
                psnr_str + "\n",
                ssim_str + "\n",
                lpips_alex_str + "\n",
                lpips_vgg_str + "\n",
                fid_str + "\n",
            ])

        # Also save raw FID log (useful for debugging)
        with open(os.path.join(result_root, 'fid_log.txt'), 'w') as f:
            f.write(fid_log)

        # Reset metrics
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips_alex = []
        self.lpips_vgg = []

        # Flatten the metric dict
        records = []
        for (human, view), metrics_list in self.per_view_metrics.items():
            # Aggregate multiple entries (if multiple frames per view)
            avg = {
                'human_idx': human,
                'view': view,
                'mse': np.mean([m['mse'] for m in metrics_list]),
                'psnr': np.mean([m['psnr'] for m in metrics_list]),
                'ssim': np.mean([m['ssim'] for m in metrics_list]),
                'lpips_alex': np.mean([m['lpips_alex'] for m in metrics_list]),
                'lpips_vgg': np.mean([m['lpips_vgg'] for m in metrics_list]),
            }
            records.append(avg)

        df = pd.DataFrame(records).sort_values(by=['human_idx', 'view'])
        csv_path = os.path.join(result_root, 'per_view_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed per-view metrics to: {csv_path}")

    # def summarize(self):

    #     result_root = os.path.join(cfg.result_dir,
    #                                'epoch_' + str(cfg.test.epoch),
    #                                cfg.exp_folder_name)

    #     mse_path = os.path.join(result_root, 'mse.npy')
    #     psnr_path = os.path.join(result_root, 'psnr.npy')
    #     ssim_path = os.path.join(result_root, 'ssim.npy')
        

    #     os.system('mkdir -p {}'.format(result_root))
    #     metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
    #     np.save(mse_path, self.mse)
    #     np.save(psnr_path, self.psnr)
    #     np.save(ssim_path, self.ssim)

    #     exp_str = 'experiment: {}'.format(cfg.exp_name)
    #     epoch_str = 'epoch: {}'.format(cfg.test.epoch)
    #     mse_str = 'mse: {}'.format(np.mean(self.mse))
    #     psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
    #     ssim_str = 'ssim: {}'.format(np.mean(self.ssim))

    #     print(exp_str)
    #     print(epoch_str)
    #     print(mse_str)
    #     print(psnr_str)
    #     print(ssim_str)

    #     # new: save LPIPS results
    #     np.save(os.path.join(result_root, 'lpips_alex.npy'), self.lpips_alex)
    #     np.save(os.path.join(result_root, 'lpips_vgg.npy'), self.lpips_vgg)
    #     lpips_alex_str = f"lpips_alex: {np.mean(self.lpips_alex):.6f}"
    #     lpips_vgg_str  = f"lpips_vgg: {np.mean(self.lpips_vgg):.6f}"
    #     print(lpips_alex_str)
    #     print(lpips_vgg_str)

    #     # Update save summary.txt
    #     with open(os.path.join(result_root, 'summary.txt'), 'w') as out:
    #         out.writelines([
    #             exp_str + "\n",
    #             epoch_str + "\n",
    #             mse_str + "\n",
    #             psnr_str + "\n",
    #             ssim_str + "\n",
    #             lpips_alex_str + "\n",
    #             lpips_vgg_str + "\n"
    #         ])

    #     # Compute FID at the end
    #     pred_dir = os.path.join(result_root, "pred")
    #     gt_dir   = os.path.join(result_root, "gt")
    #     if os.path.exists(pred_dir) and os.path.exists(gt_dir):
    #         os.system(f'python -m pytorch_fid --device cuda {pred_dir} {gt_dir}')
    #     else:
    #         print("Warning: pred/gt folders not found, skipping FID.")

    #     # Reset metrics
    #     self.mse = []
    #     self.psnr = []
    #     self.ssim = []
    #     self.lpips_alex = []
    #     self.lpips_vgg = []

    #     # with open(os.path.join(result_root, 'summary.txt'), 'w') as out:
    #     #     out.writelines([exp_str, epoch_str, mse_str, psnr_str, ssim_str])
    #     # self.mse = []
    #     # self.psnr = []
    #     # self.ssim = []

def pad_to_max_size(img, target_size):
    """Pads a 3-channel image to (H, W) = target_size using black background."""
    h, w = img.shape[:2]
    target_h, target_w = target_size

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)  # black padding
    )