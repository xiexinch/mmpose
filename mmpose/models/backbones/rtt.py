# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.utils import digit_version
from torch import Tensor

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class KeypointPostionEncoder(nn.Module):

    def __init__(self, num_keypoints, num_channels):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_channels = num_channels
        self.embeding = nn.Linear(num_keypoints * 2, num_channels)

    def forward(self, x):
        return self.embeding(x)


@MODELS.register_module()
class RTT(BaseBackbone):

    def __init__(self,
                 num_keypoints: int,
                 embed_dims: int,
                 num_layers: int = 2):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.keypoint_encoder = KeypointPostionEncoder(num_keypoints,
                                                       embed_dims)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dims, nhead=8)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        x = self.keypoint_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return tuple([x.unsqueeze(-1)])


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


class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1, channels, bias=True)

        if digit_version(torch.__version__) < (1, 7, 0):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class LifterResBlock(nn.Module):

    def __init__(self, channels: int = 1024, num_layers: int = 2):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(num_layers)])

        self.ca = ChannelAttention(channels)

    def forward(self, x):
        inp = x
        for layer in self.layers:
            x = nn.LeakyReLU()(layer(x))
        x = self.ca(x) + inp
        return x


@MODELS.register_module()
class AttnLinearNet(nn.Module):

    def __init__(self,
                 in_channels: int = 17 * 2,
                 channels=1024,
                 num_linears: int = 2,
                 num_res_blocks: int = 2):
        super().__init__()
        self.sk_convert = nn.Linear(in_channels, in_channels)

        # upscale
        self.upscale = nn.Linear(in_channels, channels)
        self.res_layers = nn.ModuleList([
            LifterResBlock(channels=channels, num_layers=num_linears)
            for _ in range(num_res_blocks)
        ])

        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor):
        if x.dim() == 3:
            x = x.squeeze(-1)
        x = self.upscale(x)
        for layer in self.res_layers:
            x = self.act(layer(x))

        return tuple([x.unsqueeze(-1)])
