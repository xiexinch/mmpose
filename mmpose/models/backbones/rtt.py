# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType


class lifter_res_block(nn.Module):

    def __init__(self, hidden=1024):
        super(lifter_res_block, self).__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x


@MODELS.register_module()
class LargeSimpleBaseline(nn.Module):

    def __init__(self, in_channels=17 * 2, channels=1024):
        super(LargeSimpleBaseline, self).__init__()

        self.upscale = nn.Linear(in_channels, channels)
        self.res_1 = lifter_res_block(hidden=channels)
        self.res_2 = lifter_res_block(hidden=channels)
        self.res_3 = lifter_res_block(hidden=channels)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_1(x))
        x = nn.LeakyReLU()(self.res_2(x))
        x = nn.LeakyReLU()(self.res_3(x))
        return tuple([x.unsqueeze(-1)])


@MODELS.register_module()
class L2(nn.Module):

    def __init__(self,
                 in_channels=17 * 2,
                 channels=1024,
                 gau_cfg: ConfigType = dict(
                     hidden_dims=512,
                     s=128,
                     expansion_factor=2,
                     dropout_rate=0.,
                     drop_path=0.,
                     act_fn='ReLU',
                     use_rel_bias=False,
                     pos_enc=False)):
        super(L2, self).__init__()

        self.upscale = nn.Linear(in_channels, channels)
        self.res_1 = lifter_res_block(hidden=channels)
        self.leaky_relu = nn.LeakyReLU()
        self.mlp = nn.Sequential(
            ScaleNorm(channels),
            nn.Linear(channels, gau_cfg['hidden_dims'], bias=False))

        self.mlp2 = nn.Sequential(
            ScaleNorm(gau_cfg['hidden_dims']),
            nn.Linear(
                gau_cfg['hidden_dims'], gau_cfg['hidden_dims'], bias=False))

        self.gau = RTMCCBlock(
            channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            attn_type='self-attn',
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.upscale(x)
        x = self.leaky_relu(self.res_1(x))
        x = self.mlp(x)
        x = self.mlp2(x)
        x = self.gau(x)
        return tuple([x.reshape(x.shape[0], -1, 1)])
