import torch.nn as nn
import torch.nn.functional as F
import torch

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

from lib.config import cfg
from lib.networks.encoder import SpatialEncoder
import math
import time
import pdb

class SpatialKeyValue(nn.Module):

    def __init__(self):
        super(SpatialKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(256, 256, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))


def combine_interleaved(t, num_input=4, agg_type="average"):

    t = t.reshape(-1, num_input, *t.shape[1:])

    if agg_type == "average":
        t = torch.mean(t, dim=1)

    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = SpatialEncoder()

        if cfg.weight == 'cross_transformer':
            self.spatial_key_value_0 = None  #SpatialKeyValue()
            self.spatial_key_value_1 = None  #SpatialKeyValue()

        # self.xyzc_net = SparseConvNet()

        self.actvn = nn.ReLU()

        self.fc_ = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.fc_0 = nn.Conv1d(384, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)
        # nn.init.normal_(self.alpha_fc.weight, mean=0.0, std=1e-4)
        # nn.init.constant_(self.alpha_fc.bias, -3.0)        

        self.feature_fc = nn.Conv1d(256, 256, 1)

        # self.viewdir_dim = cfg.viewdir_dim  # set in config
        # self.view_fc = nn.Conv1d(256 + self.viewdir_dim, 128, 1)
        self.view_fc = nn.Conv1d(283, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.fc_3 = nn.Conv1d(256, 256, 1)
        self.fc_4 = nn.Conv1d(128, 128, 1)

        self.alpha_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)

        self.rgb_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.rgb_res_1 = nn.Conv1d(cfg.img_feat_size, 128, 1)
        # nn.init.constant_(self.alpha_fc.bias, 0.3)
        # self.pix_norm = nn.GroupNorm(1, cfg.img_feat_size)  # or LayerNorm over channels
        #self.rgb_hint_1x1 = nn.Conv1d(3, 256, 1)

    def forward(self, pixel_feat, viewdir, light_pts, holder=None):
        """
        Minimal-change forward (no rgb_hint):
        - same branches/heads as your original
        - density uses softplus(+0.3) to avoid opacity collapse
        - optional RGB sigmoid if self.apply_sigmoid_rgb is set True
        Shapes:
        pixel_feat : [B*n_input, C_img, N]
        viewdir    : [B, N, Dv]
        returns raw: [B, N, 4]  (RGB logits or [0,1] if flag enabled, + alpha>0)
        """
        B = light_pts.shape[0]
        n_input = int(pixel_feat.shape[0] // B)

        # ── shared trunk
        net = self.actvn(self.fc_(pixel_feat))          # [B*n,256,N]
        net = self.actvn(self.fc_1(net))                # [B*n,256,N]
        inter_net = self.actvn(self.fc_2(net))          # [B*n,256,N]

        # ── density (alpha) branch
        opa_net = combine_interleaved(inter_net, n_input, "average")  # [B,256,N]
        if hasattr(self, "alpha_res_0"):
            pix_alpha = self.actvn(self.alpha_res_0(pixel_feat))      # [B*n,256,N]
            pix_alpha = combine_interleaved(pix_alpha, n_input, "average")  # [B,256,N]
            opa_net = opa_net + pix_alpha
        opa_net = self.actvn(self.fc_3(opa_net))        # [B,256,N]
        alpha_raw = self.alpha_fc(opa_net)              # [B,1,N]
        alpha = F.softplus(alpha_raw + 0.3)             # strictly positive; small bias to avoid darkness
        #sigma = F.softplus(alpha) * 0.1

        # ── color (rgb) branch
        rgb_feat = self.feature_fc(inter_net)           # [B*n,256,N]
        if hasattr(self, "rgb_res_0"):
            rgb_feat = rgb_feat + self.rgb_res_0(pixel_feat)

        viewdir_rep = repeat_interleave(viewdir, n_input).transpose(1, 2)  # [B*n,Dv,N]

        rgb_feat = torch.cat((rgb_feat, viewdir_rep), dim=1)          # [B*n,256+Dv,N]
        rgb_feat = self.actvn(self.view_fc(rgb_feat))                  # [B*n,128,N]
        if hasattr(self, "rgb_res_1"):
            rgb_feat = rgb_feat + self.rgb_res_1(pixel_feat)

        rgb_feat = combine_interleaved(rgb_feat, n_input, "average")   # [B,128,N]
        rgb_feat = self.actvn(self.fc_4(rgb_feat))                     # [B,128,N]
        rgb = self.rgb_fc(rgb_feat)                                    # [B,3,N]


        raw = torch.cat((rgb, alpha), dim=1).transpose(1, 2)           # [B,N,4]
        #pdb.set_trace()
        return raw

  


    # black and white version
    # def forward(self, pixel_feat, viewdir, light_pts, holder=None):
    #     """
    #     Minimal-change forward:
    #     - removes xyzc_features, self.xyzc_net, and cross_attention
    #     - directly consumes pixel_feat
    #     - preserves original trunk -> inter_net -> {alpha/rgb} branches
    #     - preserves combine_interleaved('average') and head layers
    #     Shapes assumed:
    #     pixel_feat: [B*n_input, C_img, N]
    #     viewdir:    [B, N, Dv]
    #     light_pts:  [B, ...]  (unused here but kept for signature parity)
    #     """
    #     B = light_pts.shape[0]
    #     n_input = int(pixel_feat.shape[0] // B)

    #     # ── shared trunk from pixel features
    #     net = self.actvn(self.fc_(pixel_feat))     # [B*n, 256, N]
    #     net = self.actvn(self.fc_1(net))           # [B*n, 256, N]
    #     inter_net = self.actvn(self.fc_2(net))     # [B*n, 256, N]

    #     # ── density (alpha) branch
    #     opa_net = combine_interleaved(inter_net, n_input, "average")  # [B, 256, N]

    #     # if you had a residual from pixel features into alpha, keep it (no new modules added)
    #     if hasattr(self, "alpha_res_0"):
    #         pix_alpha = self.actvn(self.alpha_res_0(pixel_feat))      # [B*n, 256, N]
    #         pix_alpha = combine_interleaved(pix_alpha, n_input, "average")  # [B, 256, N]
    #         opa_net = opa_net + pix_alpha

    #     opa_net = self.actvn(self.fc_3(opa_net))   # [B, 256, N]
    #     alpha_raw = self.alpha_fc(opa_net)         # [B, 1,   N]

    #     # keep behavior minimal: return raw alpha as before.
    #     # (If you want strictly-positive densities, replace the next line with softplus.)
    #     alpha = alpha_raw
    #     # alpha = F.softplus(alpha_raw + 0.0)

    #     # ── color (rgb) branch
    #     rgb_feat = self.feature_fc(inter_net)      # [B*n, 256, N]
    #     if hasattr(self, "rgb_res_0"):
    #         rgb_feat = rgb_feat + self.rgb_res_0(pixel_feat)

    #     # concat viewdir (repeat per input to match [B*n, :, N])
    #     viewdir_rep = repeat_interleave(viewdir, n_input).transpose(1, 2)  # [B*n, Dv, N]
    #     rgb_feat = torch.cat((rgb_feat, viewdir_rep), dim=1)               # [B*n, 256+Dv, N]

    #     rgb_feat = self.actvn(self.view_fc(rgb_feat))  # [B*n, 128, N]
    #     if hasattr(self, "rgb_res_1"):
    #         rgb_feat = rgb_feat + self.rgb_res_1(pixel_feat)

    #     rgb_feat = combine_interleaved(rgb_feat, n_input, "average")  # [B, 128, N]
    #     rgb_feat = self.actvn(self.fc_4(rgb_feat))                    # [B, 128, N]
    #     rgb = self.rgb_fc(rgb_feat)                                   # [B,   3, N]

    #     # ── pack output [B, N, 4]
    #     raw = torch.cat((rgb, alpha), dim=1).transpose(1, 2)
    #     # pdb.set_trace()
    #     return raw


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(64, 64, 'subm0')
        self.down0 = stride_conv(64, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, grid_coords):

        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        '''

        '''

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())