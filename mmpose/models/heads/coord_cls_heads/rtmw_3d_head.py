# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMW3DHead(BaseHead):
    """Top-down head introduced in RTMPose-Wholebody (2023).

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        ps = 2
        self.ps = nn.PixelShuffle(ps)
        self.conv_dec = ConvModule(
            in_channels // ps**2,
            in_channels // 4,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))

        self.final_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        self.final_layer2 = ConvModule(
            in_channels // ps + in_channels // 4,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'] // 2, bias=False))

        self.mlp2 = nn.Sequential(
            ScaleNorm(flatten_dims * ps**2),
            nn.Linear(
                flatten_dims * ps**2, gau_cfg['hidden_dims'] // 2, bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        D = int(self.input_size[2] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        self.cls_z = nn.Linear(gau_cfg['hidden_dims'], D, bias=False)

    def forward(self, feats: Tuple[Tensor,
                                   Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward the network.

        The input is the feature map extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        # enc_b  n / 2, h, w
        # enc_t  n,     h, w
        enc_b, enc_t = feats

        feats_t = self.final_layer(enc_t)
        feats_t = torch.flatten(feats_t, 2)
        feats_t = self.mlp(feats_t)

        dec_t = self.ps(enc_t)
        dec_t = self.conv_dec(dec_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)

        feats_b = self.final_layer2(enc_b)
        feats_b = torch.flatten(feats_b, 2)
        feats_b = self.mlp2(feats_b)

        feats = torch.cat([feats_t, feats_b], dim=2)

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        pred_z = self.cls_z(feats)

        return pred_x, pred_y, pred_z

    def decode(self, x: Tensor, y: Tensor, z: Tensor,
               batch_data_samples: OptSampleList) -> InstanceList:
        root_z_list = []
        if batch_data_samples is not None:
            if 'root_z' in batch_data_samples[0].gt_instances:
                for data_sample in batch_data_samples:
                    root_z_list.append(data_sample.gt_instances.root_z)
                root_z = torch.from_numpy(np.stack(root_z_list))
            else:
                root_z = torch.ones(
                    x.shape[0], 1, device=x.device, dtype=x.dtype) * 5.6302552
        else:
            root_z = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return super().decode((x, y, z, root_z))

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """
        x, y, z = self.forward(feats)

        preds = self.decode(x, y, z, batch_data_samples)

        if test_cfg.get('output_heatmaps', False):
            raise NotImplementedError
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y, pred_z = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_z = torch.cat([
            d.gt_instance_labels.keypoint_z_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        with_z_labels = batch_data_samples[0].gt_instance_labels.with_z_labels[
            0]
        if with_z_labels:
            pred_simcc = (pred_x, pred_y, pred_z)
            gt_simcc = (gt_x, gt_y, gt_z)
        else:
            pred_simcc = (pred_x, pred_y, 0 * pred_z)
            gt_simcc = (gt_x, gt_y, torch.zeros_like(gt_z))

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        error = simcc_mpjpe(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        mpjpe = torch.tensor(error, device=gt_x.device)
        losses.update(mpjpe=mpjpe)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg


def simcc_mpjpe(output: Tuple[np.ndarray, np.ndarray, np.ndarray],
                target: Tuple[np.ndarray, np.ndarray, np.ndarray],
                simcc_split_ratio: float,
                mask: np.ndarray,
                thr: float = 0.05,
                normalize: Optional[np.ndarray] = None) -> float:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from 3D SimCC.

    Note:
        - PCK metric measures accuracy of the localization of the body joints.
        - The distances between predicted positions and the ground-truth ones
          are typically normalized by the bounding box size.

    Args:
        output (Tuple[np.ndarray, np.ndarray, np.ndarray]): Model predicted
            3D SimCC (x, y, z).
        target (Tuple[np.ndarray, np.ndarray, np.ndarray]): Groundtruth
            3D SimCC (x, y, z).
        simcc_split_ratio (float): SimCC split ratio for recovering actual
            coordinates.
        mask (np.ndarray[N, K]): Visibility mask for the target. False for
            invisible joints, and True for visible.
        thr (float): Threshold for PCK calculation. Default 0.05.
        normalize (Optional[np.ndarray[N, 3]]): Normalization factor for
            H, W, and Depth.

    Returns:
        Tuple[np.ndarray, float, int]:
        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    if len(output) == 3:
        pred_x, pred_y, pred_z = output
        gt_x, gt_y, gt_z = target
    else:
        pred_x, pred_y = output
        gt_x, gt_y = target
        pred_z, gt_z = np.zeros_like(pred_x), np.zeros_like(gt_x)

    N, _, Wx = pred_x.shape
    _, _, Wy = pred_y.shape
    _, _, Wz = pred_z.shape
    W, H, D = int(Wx / simcc_split_ratio), int(Wy / simcc_split_ratio), int(
        Wz / simcc_split_ratio)

    if normalize is None:
        normalize = np.tile(np.array([[H, W, D]]), (N, 1))

    pred_coords, _ = get_simcc_maximum(pred_x, pred_y, pred_z)
    pred_coords /= simcc_split_ratio
    gt_coords, _ = get_simcc_maximum(gt_x, gt_y, gt_z)
    gt_coords /= simcc_split_ratio

    return keypoint_mpjpe(pred_coords, gt_coords, mask)


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray,
                      simcc_z: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        simcc_z (np.ndarray): z-axis SimCC in shape (K, Wz) or (N, K, Wz)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 3) or (N, K, 3)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert isinstance(simcc_z, np.ndarray), ('simcc_z should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_z.ndim == 2 or simcc_z.ndim == 3, (
        f'Invalid shape {simcc_z.shape}')
    assert simcc_x.ndim == simcc_y.ndim == simcc_z.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape} != {simcc_z.ndim} ')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        simcc_z = simcc_z.reshape(N * K, -1)
    else:
        N = None

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    z_locs = np.argmax(simcc_z, axis=1)
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    max_val_z = np.amax(simcc_z, axis=1)

    vals = np.minimum(np.minimum(max_val_x, max_val_y), max_val_z)
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 3)
        vals = vals.reshape(N, K)

    return locs, vals
