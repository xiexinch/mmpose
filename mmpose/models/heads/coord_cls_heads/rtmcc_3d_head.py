# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_maximum as get_2d_simcc_maximum
from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCC3DHead(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

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
        input_size: Tuple[int, int, int],
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

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

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

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        pred_z = self.cls_z(feats)

        return pred_x, pred_y, pred_z

    def decode(self, batch_outputs: Union[Tensor,
                                          Tuple[Tensor]]) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        batch_output_np = to_numpy(batch_outputs, unzip=True)
        batch_keypoints = []
        batch_keypoints2d = []
        batch_scores = []
        for outputs in batch_output_np:
            keypoints_2d, keypoints, scores = _pack_and_call(
                outputs, self.decoder.decode)
            batch_keypoints2d.append(keypoints_2d)
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)

        preds = []
        for keypoints_2d, keypoints, scores in zip(batch_keypoints2d,
                                                   batch_keypoints,
                                                   batch_scores):
            pred = InstanceData(
                keypoints_2d=keypoints_2d,
                keypoints=keypoints,
                keypoint_scores=scores)
            preds.append(pred)

        return preds

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

        preds = self.decode((x, y, z))

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

        weight_z = torch.cat(
            [d.gt_instance_labels.weight_z for d in batch_data_samples],
            dim=0,
        )

        with_z_labels = [
            d.gt_instance_labels.with_z_label[0] for d in batch_data_samples
        ]

        N, K, _ = pred_x.shape
        keypoint_weights_ = keypoint_weights.clone()
        pred_simcc = (pred_x, pred_y, pred_z)
        gt_simcc = (gt_x, gt_y, gt_z)

        keypoint_weights = torch.cat([
            keypoint_weights[None, ...], keypoint_weights[None, ...],
            weight_z[None, ...]
        ])

        # calculate losses
        losses = dict()
        for i, loss_ in enumerate(self.loss_module):
            if loss_.loss_name == 'loss_bone':
                pred_coords = get_3d_coord(pred_x, pred_y, pred_z,
                                           with_z_labels)
                gt_coords = get_3d_coord(gt_x, gt_y, gt_z, with_z_labels)
                loss = loss_(pred_coords, gt_coords, keypoint_weights_)
            else:
                loss = loss_(pred_simcc, gt_simcc, keypoint_weights)
            losses[loss_.loss_name] = loss

        # calculate accuracy
        error = simcc_mpjpe(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights_) > 0,
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


def get_3d_coord(simcc_x, simcc_y, simcc_z, with_z_labels):
    N, K, W = simcc_x.shape
    # 过滤 z 轴
    for i, with_z in enumerate(with_z_labels):
        if not with_z:
            simcc_z[i] = torch.zeros_like(simcc_z[i])
    x_locs = simcc_x.reshape(N * K, -1).argmax(dim=1)
    y_locs = simcc_y.reshape(N * K, -1).argmax(dim=1)
    z_locs = simcc_z.reshape(N * K, -1).argmax(dim=1)

    locs = torch.stack((x_locs, y_locs, z_locs),
                       dim=-1).to(simcc_x).reshape(N, K, 3)
    return locs


def simcc_mpjpe(output: Tuple[np.ndarray, np.ndarray, np.ndarray],
                target: Tuple[np.ndarray, np.ndarray, np.ndarray],
                simcc_split_ratio: float,
                mask: np.ndarray,
                thr: float = 0.05) -> float:
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
        pred_coords, _ = get_simcc_maximum(pred_x, pred_y, pred_z)
        gt_coords, _ = get_simcc_maximum(gt_x, gt_y, gt_z)

    else:
        pred_x, pred_y = output
        gt_x, gt_y = target
        pred_coords, _ = get_2d_simcc_maximum(pred_x, pred_y)
        gt_coords, _ = get_2d_simcc_maximum(gt_x, gt_y)

    pred_coords /= simcc_split_ratio
    gt_coords /= simcc_split_ratio

    return keypoint_mpjpe(pred_coords, gt_coords, mask)


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray,
                      simcc_z: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
