import torch

# ----------------------------------------------------------
# IBR weighting strategies
# ----------------------------------------------------------
@torch.no_grad()
def _weights_angle_softmax(self, cos_theta, valid_mask):
    """
    cos_theta:  [B, V, M]
    valid_mask: [B, V, M] bool
    returns:    [B, V, M] weights sum_v=1
    """
    beta = getattr(self, "color_weight_beta", 10.0)
    logits = beta * cos_theta

    # mask invalid views / pixels
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, -1e9)

    w = torch.softmax(logits, dim=1)  # along view dimension
    return w

@torch.no_grad()
def _weights_cos_power(self, cos_theta, valid_mask):
    """
    cos^gamma with renormalization.
    """
    gamma = getattr(self, "color_weight_gamma", 2.0)
    w = cos_theta.clamp(min=0.0).pow(gamma)
    if valid_mask is not None:
        w = w * valid_mask.float()

    w_sum = w.sum(dim=1, keepdim=True) + 1e-8
    w = w / w_sum
    return w

@torch.no_grad()
def _weights_uniform(self, cos_theta, valid_mask):
    """
    All valid views equally weighted, invalid = 0.
    """
    if valid_mask is not None:
        w = valid_mask.float()
    else:
        w = torch.ones_like(cos_theta)

    w_sum = w.sum(dim=1, keepdim=True) + 1e-8
    w = w / w_sum
    return w

@torch.no_grad()
def _compute_color_weights(self, cos_theta, valid_mask):
    """
    Dispatch to the selected weighting strategy.
    cos_theta:  [B, V, M]
    valid_mask: [B, V, M] bool
    """
    mode = getattr(self, "color_weight_mode", "angle_softmax")
    if mode == "angle_softmax":
        return self._weights_angle_softmax(cos_theta, valid_mask)
    elif mode == "cos_power":
        return self._weights_cos_power(cos_theta, valid_mask)
    elif mode == "uniform":
        return self._weights_uniform(cos_theta, valid_mask)
    else:
        raise ValueError(f"Unknown color_weight_mode: {mode}")

@torch.no_grad()
def compute_ibr_colors(
    self,
    xyz,            # [B, N, S, 3] sample points (S = #samples per ray, e.g. 5)
    ray_d,          # [B, N, 3] novel-view ray directions
    input_imgs0,    # [B, V, C, Hs, Ws] RGB input images (time=0)
    input_K,        # [B, V, 3, 3]
    input_R,        # [B, V, 3, 3] world -> cam
    input_T,        # [B, V, 3, 1]
):
    """
    Multi-view image-based color for each sample point.

    Returns:
        rgb_flat : [B, N*S, 3]  (same ordering as xyz.view(B, -1, 3))
    """
    import torch.nn.functional as F  # safe here

    B, N, S, _ = xyz.shape
    _, V, C, Hs, Ws = input_imgs0.shape
    M = N * S

    dev = xyz.device
    dtype = xyz.dtype

    # Flatten samples and ray directions to [B, M, 3]
    pts_flat = xyz.view(B, M, 3)             # [B, M, 3]
    ray_d_flat = ray_d[:, :, None, :].expand(B, N, S, 3).reshape(B, M, 3)
    d_n = F.normalize(ray_d_flat, dim=-1)    # [B, M, 3]

    # Containers over views
    colors_v = []        # list of [B, M, 3]
    cos_theta_v = []     # list of [B, M]
    valid_v = []         # list of [B, M] bool

    for v in range(V):
        Rv = input_R[:, v]          # [B, 3, 3]
        Tv = input_T[:, v]          # [B, 3, 1]
        Kv = input_K[:, v]          # [B, 3, 3]
        img_v = input_imgs0[:, v]   # [B, C, Hs, Ws]

        # ---- world -> camera ----
        Xc = torch.matmul(
            Rv.unsqueeze(1),            # [B,1,3,3]
            pts_flat.unsqueeze(-1)      # [B,M,3,1]
        ) + Tv.unsqueeze(1)             # [B,1,3,1]
        Xc = Xc.squeeze(-1)             # [B,M,3]

        z = Xc[..., 2]                  # [B,M]
        x = Xc[..., 0] / z.clamp_min(1e-6)
        y = Xc[..., 1] / z.clamp_min(1e-6)

        fx = Kv[:, 0, 0][:, None]
        fy = Kv[:, 1, 1][:, None]
        cx = Kv[:, 0, 2][:, None]
        cy = Kv[:, 1, 2][:, None]

        u = fx * x + cx                 # [B,M]
        v_img = fy * y + cy             # [B,M]

        # ---- pixel bounds mask ----
        xu = (2.0 * u / (Ws - 1.0)) - 1.0
        yv = (2.0 * v_img / (Hs - 1.0)) - 1.0

        inb = (xu >= -1.0) & (xu <= 1.0) & (yv >= -1.0) & (yv <= 1.0)
        in_front = z > 1e-6
        valid = inb & in_front          # [B,M]

        # ---- sample image color ----
        grid = torch.stack([xu, yv], dim=-1).view(B, M, 1, 2)  # [B,M,1,2]
        samples = F.grid_sample(
            img_v, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )                       # [B,C,M,1]
        color = samples[:, :, :, 0].permute(0, 2, 1)  # [B,M,C] -> [B,M,3]

        # ---- view direction from camera center to point ----
        # camera center in world: C = -R^T T
        Cw = -torch.matmul(Rv.transpose(1, 2), Tv).squeeze(-1)  # [B,3]
        d_v = F.normalize(pts_flat - Cw[:, None, :], dim=-1)    # [B,M,3]

        # cos angle between novel view dir and this camera's dir
        cos_theta = (d_n * d_v).sum(dim=-1).clamp(min=0.0)      # [B,M]

        colors_v.append(color)           # [B,M,3]
        cos_theta_v.append(cos_theta)    # [B,M]
        valid_v.append(valid)            # [B,M]

    # Stack over views
    colors = torch.stack(colors_v, dim=1)      # [B, V, M, 3]
    cos_theta = torch.stack(cos_theta_v, dim=1)  # [B, V, M]
    valid_mask = torch.stack(valid_v, dim=1)     # [B, V, M]

    # Compute weights per view
    w = self._compute_color_weights(cos_theta, valid_mask)      # [B, V, M]
    w = w.unsqueeze(-1)                                         # [B, V, M, 1]

    # Weighted sum over views
    rgb_flat = (w * colors).sum(dim=1)                          # [B, M, 3]

    return rgb_flat    # [B, N*S, 3]