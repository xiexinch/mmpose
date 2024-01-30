# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn

from mmpose.models.utils.rtmcc_block import RTMCCBlock
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
class L3(nn.Module):

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
        super(L3, self).__init__()

        self.upscale = nn.Linear(in_channels, channels)
        self.res_1 = lifter_res_block(hidden=channels)
        self.res_2 = lifter_res_block(hidden=channels)
        self.res_3 = lifter_res_block(hidden=channels)

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
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, -1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_1(x))
        x = nn.LeakyReLU()(self.res_2(x))
        x = nn.LeakyReLU()(self.res_3(x))

        x = self.gau(x)
        return tuple([x.reshape(x.shape[0], -1, 1)])


class GraphAttention(nn.Module):

    def __init__(self,
                 in_channels: int = 1024,
                 out_channels: int = 1024,
                 num_heads: int = 8,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 adj_mat: List = [],
                 num_nodes: int = 133):
        super().__init__()
        self.is_concat = is_concat
        self.num_heads = num_heads

        if is_concat:
            assert out_channels % num_heads == 0
            self.channels = out_channels // num_heads
        else:
            self.channels = out_channels

        self.linear = nn.Linear(
            in_channels, self.channels * self.num_heads, bias=False)
        self.attn = nn.Linear(self.channels * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        if len(adj_mat) == 0:
            raise ValueError('adj_mat is empty')
        else:
            adj_mat = torch.tensor(
                adj_mat, dtype=torch.float32, requires_grad=False)
            assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == num_nodes
            assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == num_nodes
            assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.num_heads
            self.adj_mat = adj_mat

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        g = self.linear(x)
        g = g.view(B, N, self.num_heads, self.channels)
        g_repeat = g.repeat(1, N, 1, 1)
        g_repeat_interleave = g.repeat_interleave(N, dim=1)
        g_concat = torch.cat([g_repeat, g_repeat_interleave],
                             dim=-1).view(B, N, N, self.num_heads,
                                          self.channels * 2)
        e = self.activation(self.attn(g_concat)).squeeze(-1)

        e = e.masked_fill(self.adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)

        if self.is_concat:
            out = attn_res.reshape(B, N, self.num_heads * self.channels)
        else:
            out = attn_res.mean(dim=1)
        return out


class GraphTransformerLayer(nn.Module):

    def __init__(self,
                 token_dims: int = 1024,
                 num_heads: int = 8,
                 msa_drop_out: float = 0.1,
                 ga_drop_out: float = 0.6,
                 adj_mat: List = [],
                 num_nodes: int = 133,
                 mlp_ratio: int = 4):
        super().__init__()
        self.token_dims = token_dims

        # Graph Attention
        self.ga = GraphAttention(
            token_dims,
            token_dims,
            num_heads,
            is_concat=True,
            dropout=ga_drop_out,
            leaky_relu_negative_slope=0.2,
            adj_mat=adj_mat,
            num_nodes=num_nodes)

        # Multi Head Attention
        self.msa = nn.MultiheadAttention(token_dims, num_heads, msa_drop_out)

        # Residual Connection
        self.ln1 = nn.LayerNorm(token_dims * 2)
        self.ln2 = nn.LayerNorm(token_dims * 2)

        # MLP GELU
        self.mlp_gelu = nn.Sequential(
            nn.Linear(token_dims * 2, int(token_dims * mlp_ratio)), nn.GELU(),
            nn.Linear(int(token_dims * mlp_ratio), token_dims * 2))

        self.linear = nn.Linear(token_dims * 2, token_dims)

    def forward(self, inputs: torch.Tensor):
        feat_l = self.ga(inputs)
        feat_g = self.msa(inputs, inputs, inputs)[0]
        feat = torch.cat([feat_l, feat_g], dim=-1)
        x = self.ln1(feat) + feat
        x = self.ln2(self.mlp_gelu(x)) + x
        x = self.linear(x)
        return x


@MODELS.register_module()
class L4(nn.Module):

    def __init__(self,
                 num_keypoints: int = 133,
                 with_vis_scores: bool = False,
                 token_dims: int = 1024,
                 num_graph_layers: int = 3,
                 num_heads: int = 8,
                 msa_drop_out: float = 0.1,
                 ga_drop_out: float = 0.6,
                 adj_mat: List = []):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.token_dims = token_dims

        self.pos_dim = 3 if with_vis_scores else 2

        self.pos_w = nn.Parameter(torch.randn((self.pos_dim, token_dims)))
        self.pos_b = nn.Parameter(torch.randn((token_dims)))

        self.graph_layers = nn.ModuleList([
            GraphTransformerLayer(token_dims, num_heads, msa_drop_out,
                                  ga_drop_out, adj_mat, num_keypoints)
            for _ in range(num_graph_layers)
        ])

    def token_positional_encoding(self, inputs: torch.Tensor):
        assert inputs.ndim == 3

        x = torch.matmul(inputs, self.pos_w) + self.pos_b
        x[:, 0::2, :] = torch.sin(x[:, 0::2, :])
        x[:, 1::2, :] = torch.cos(x[:, 1::2, :])
        return x

    def forward(self, x: torch.Tensor):
        if self.ndim == 4:
            x = x.reshape(x.shape[0], self.num_keypoints, self.pos_dim)
        x = self.token_positional_encoding(x)

        for layer in self.graph_layers:
            x = layer(x)
            x = nn.GELU()(x)
        return tuple([x])
