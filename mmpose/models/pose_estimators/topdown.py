# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional

import torch
from mmengine.model import BaseModel
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, ForwardResults, InstanceList,
                                 OptConfigType, OptMultiConfig, OptSampleList,
                                 PixelDataList, SampleList)
from .base import BasePoseEstimator


@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples


@MODELS.register_module()
class TopdownPoseEstimator3D(TopdownPoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples


@MODELS.register_module()
class PoseDetectionLifter(BaseModel):

    def __init__(self, pose_estimator: ConfigType, pose_lifter: ConfigType,
                 **kwargs):
        super().__init__()
        self.pose_estimator = MODELS.build(pose_estimator)
        self.pose_lifter = MODELS.build(pose_lifter)

    def extract_feats(self, inputs: Tensor):
        return self.pose_estimator.backbone(inputs)

    def convert_2d_keypoings_to_image_space(self, keypoints, data_samples):
        input_center = Tensor(
            [d.metainfo['input_center'] for d in data_samples]).to(keypoints)
        input_scale = Tensor([d.metainfo['input_scale']
                              for d in data_samples]).to(keypoints)
        input_size = Tensor([d.metainfo['input_size']
                             for d in data_samples]).to(keypoints)
        keypoints[..., :2] = keypoints[..., :2] / input_size * input_scale \
            + input_center - 0.5 * input_scale
        return keypoints

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        raise NotImplementedError

    def encode_2d_keypoints(self, keypoints: Tensor):
        N, _, D = keypoints.shape
        root = torch.zeros((N, 1, D + 1),
                           dtype=torch.float32,
                           device=keypoints.device)
        if self.pose_lifter.head.decoder.keypoints_mean is not None:
            keypoints_mean = torch.from_numpy(
                self.pose_lifter.head.decoder.keypoints_mean).to(
                    keypoints.device)
            keypoints_std = torch.from_numpy(
                self.pose_lifter.head.decoder.keypoints_std).to(
                    keypoints.device)
            keypoints = (keypoints - keypoints_mean) / keypoints_std
        keypoints = keypoints.reshape(N, -1, 1)
        return keypoints, root

    def decode_3d_keypoints(self, keypoints: Tensor, target_root: Tensor):
        if self.pose_lifter.head.decoder.target_mean is not None:
            keypoints = keypoints + target_root
            keypoints_mean = torch.from_numpy(
                self.pose_lifter.head.decoder.target_mean).to(keypoints)
            keypoints_std = torch.from_numpy(
                self.pose_lifter.head.decoder.target_std).to(keypoints)
            keypoints = keypoints * keypoints_std + keypoints_mean
        if self.pose_lifter.head.decoder.remove_root:
            keypoints = torch.concat([target_root, keypoints], dim=1)
        scores = torch.ones_like(keypoints[..., 0], device=keypoints.device)
        return keypoints, scores

    def decode_simcc_pred(self, x: Tensor, y: Tensor):
        assert x.ndim == y.ndim
        if x.ndim == 3:
            N, K, Wx = x.shape
            simcc_x = x.reshape(N * K, -1)
            simcc_y = y.reshape(N * K, -1)
        else:
            N = None

        x_locs = torch.argmax(simcc_x, dim=1)
        y_locs = torch.argmax(simcc_y, dim=1)
        locs = torch.stack([x_locs, y_locs], dim=-1)
        max_val_x = torch.amax(simcc_x, dim=1)
        max_val_y = torch.amax(simcc_y, dim=1)

        vals = torch.stack([max_val_x, max_val_y], dim=-1).amax(dim=-1)
        mask2 = vals <= 0.
        mask2 = mask2.unsqueeze(-1).expand(-1, 2)
        locs[mask2] = -1.

        if N is not None:
            locs = locs.reshape(N, K, 2)
            vals = vals.reshape(N, K)

        if locs.ndim == 2:
            locs = locs.unsqueeze(0)
            vals = vals.unsqueeze(0)

        keypoints = locs / self.pose_estimator.head.decoder.simcc_split_ratio

        return keypoints, vals

    def predict(self, inputs: Tensor, data_samples: OptSampleList):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        img_feats = self.extract_feats(inputs)
        x, y = self.pose_estimator.head.forward(img_feats)
        keypoints, _ = self.decode_simcc_pred(x, y)
        keypoints = self.convert_2d_keypoings_to_image_space(
            keypoints, data_samples)
        keypoints, targets_root = self.encode_2d_keypoints(keypoints)
        kpts_feats = self.pose_lifter.backbone(keypoints)
        kpts_3d = self.pose_lifter.head(kpts_feats)
        keypoints, scores = self.decode_3d_keypoints(kpts_3d, targets_root)

        # Pack to data sample
        for data_sample, kpts, scores in zip(data_samples, keypoints, scores):
            data_sample.pred_instances = {
                'keypoints_3d': kpts,
                'keypoint_3d_scores': scores
            }

        return data_samples

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        img_feats = self.extract_feats(inputs)
        x, y = self.pose_estimator.head.forward(img_feats)
        keypoints, _ = self.decode_simcc_pred(x, y)
        if data_samples is not None:
            keypoints = self.convert_2d_keypoings_to_image_space(
                keypoints, data_samples)
        keypoints, targets_root = self.encode_2d_keypoints(keypoints)
        kpts_feats = self.pose_lifter.backbone(keypoints)
        kpts_3d = self.pose_lifter.head(kpts_feats)
        keypoints, _ = self.decode_3d_keypoints(kpts_3d, targets_root)
        return keypoints

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')
