# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

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
