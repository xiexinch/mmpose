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


class TransformerLayer(nn.Module):

    def __init__(self, embed_dims: int, num_heads: int,
                 feedforward_channels: int):
        super().__init__()
        self.embed_dims = embed_dims

        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.GELU(),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x):
        x = self.self_attn(x, x, x)[0]
        x = self.cross_attn(x, x, x)[0]
        x = self.ffn(x)
        return x


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
            TransformerLayer(embed_dims, 4, embed_dims * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        x = self.keypoint_encoder(x)
        x_ = x.clone()
        for layer in self.layers:
            x = self.dropout(self.norm(layer(x))) + x_
        return tuple([x.unsqueeze(-1)])
