# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.dataset import Compose
from mmengine.infer.infer import ModelType
from mmengine.logging import print_log
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmpose.evaluation.functional import nms
from mmpose.registry import DATASETS, INFERENCERS
from mmpose.structures import PoseDataSample, merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer
from .utils import default_det_models

try:
    from mmdet.apis.det_inferencer import DetInferencer
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='3dpose-estimation')
@INFERENCERS.register_module()
class Pose3DImageInferencer(BaseMMPoseInferencer):

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmpose',
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, Tuple]] = None,
                 **kwargs) -> None:

        init_default_scope('mmpose')
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self._init_detector(
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
            device=device)

    def _init_detector(
        self,
        det_model: Optional[Union[ModelType, str]] = None,
        det_weights: Optional[str] = None,
        det_cat_ids: Optional[Union[int, Tuple]] = None,
        device: Optional[str] = None,
    ):
        object_type = DATASETS.get(self.cfg.dataset_type).__module__.split(
            'datasets.')[-1].split('.')[0].lower()

        if det_model in ('whole_image', 'whole-image') or \
            (det_model is None and
                object_type not in default_det_models):
            self.detector = None

        else:
            det_scope = 'mmdet'
            if det_model is None:
                det_info = default_det_models[object_type]
                det_model, det_weights, det_cat_ids = det_info[
                    'model'], det_info['weights'], det_info['cat_ids']
            elif os.path.exists(det_model):
                det_cfg = Config.fromfile(det_model)
                det_scope = det_cfg.default_scope

            if has_mmdet:
                self.detector = DetInferencer(
                    det_model, det_weights, device=device, scope=det_scope)
            else:
                raise RuntimeError(
                    'MMDetection (v3.0.0 or above) is required to build '
                    'inferencers for top-down pose estimation models.')

            if isinstance(det_cat_ids, (tuple, list)):
                self.det_cat_ids = det_cat_ids
            else:
                self.det_cat_ids = (det_cat_ids, )

    def _init_pipeline(self, cfg: ConfigType) -> Callable:
        # 初始化用于数据预处理的管道
        # 这应该是一个能够处理输入图像并将其准备成适合模型输入的函数
        scope = cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)
        return Compose(cfg.test_dataloader.dataset.pipeline)

    def preprocess(self,
                   inputs: InputsType,
                   batch_size: int = 1,
                   bboxes: Optional[List] = None,
                   **kwargs):
        """Process the inputs into a model-feedable format.

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
            List[str or np.ndarray]: List of original inputs in the batch
        """
        for i, input in enumerate(inputs):
            bbox = bboxes[i] if bboxes else []
            data_infos = self.preprocess_single(
                input, index=i, bboxes=bbox, **kwargs)
            yield self.collate_fn(data_infos), [input]

    def preprocess_single(self,
                          input: InputType,
                          index: int,
                          bbox_thr: float = 0.3,
                          nms_thr: float = 0.3,
                          bboxes: Union[List[List], List[np.ndarray],
                                        np.ndarray] = []):
        """Process a single input into a model-feedable format.

        Args:
            input (InputType): Input given by user.
            index (int): index of the input
            bbox_thr (float): threshold for bounding box detection.
                Defaults to 0.3.
            nms_thr (float): IoU threshold for bounding box NMS.
                Defaults to 0.3.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """

        if isinstance(input, str):
            data_info = dict(img_path=input)
        else:
            data_info = dict(img=input, img_path=f'{index}.jpg'.rjust(10, '0'))
        data_info.update(self.model.dataset_meta)

        if self.cfg.data_mode == 'topdown':
            bboxes = []
            if self.detector is not None:
                try:
                    det_results = self.detector(
                        input, return_datasamples=True)['predictions']
                except ValueError:
                    print_log(
                        'Support for mmpose and mmdet versions up to 3.1.0 '
                        'will be discontinued in upcoming releases. To '
                        'ensure ongoing compatibility, please upgrade to '
                        'mmdet version 3.2.0 or later.',
                        logger='current',
                        level=logging.WARNING)
                    det_results = self.detector(
                        input, return_datasample=True)['predictions']
                pred_instance = det_results[0].pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]),
                    axis=1)

                label_mask = np.zeros(len(bboxes), dtype=np.uint8)
                for cat_id in self.det_cat_ids:
                    label_mask = np.logical_or(label_mask,
                                               pred_instance.labels == cat_id)

                bboxes = bboxes[np.logical_and(
                    label_mask, pred_instance.scores > bbox_thr)]
                bboxes = bboxes[nms(bboxes, nms_thr)]

            data_infos = []
            if len(bboxes) > 0:
                for bbox in bboxes:
                    inst = data_info.copy()
                    inst['bbox'] = bbox[None, :4]
                    inst['bbox_score'] = bbox[4:5]
                    data_infos.append(self.pipeline(inst))
            else:
                inst = data_info.copy()

                # get bbox from the image size
                if isinstance(input, str):
                    input = mmcv.imread(input)
                h, w = input.shape[:2]

                inst['bbox'] = np.array([[0, 0, w, h]], dtype=np.float32)
                inst['bbox_score'] = np.ones(1, dtype=np.float32)

                data_infos.append(self.pipeline(inst))

        else:  # bottom-up
            data_infos = [self.pipeline(data_info)]

        return data_infos

    @torch.no_grad()
    def forward(self,
                inputs: Union[dict, tuple],
                merge_results: bool = True,
                bbox_thr: float = -1):
        data_samples = self.model.test_step(inputs)
        if merge_results:
            data_samples = merge_data_samples(data_samples)
        if bbox_thr > 0:
            for ds in data_samples:
                if 'bbox_scores' in ds.pred_instances:
                    ds.pred_instances = ds.pred_instances[
                        ds.pred_instances.bbox_scores > bbox_thr]
        return [data_samples]

    def visualize(self,
                  inputs: list,
                  preds: List[PoseDataSample],
                  return_vis: bool = False,
                  show: bool = False,
                  draw_bbox: bool = False,
                  wait_time: float = 0,
                  radius: int = 3,
                  thickness: int = 1,
                  kpt_thr: float = 0.3,
                  num_instances: int = 1,
                  vis_out_dir: str = '',
                  window_name: str = '',
                  window_close_event_handler: Optional[Callable] = None,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            return_vis (bool): Whether to return images with predicted results.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (ms). Defaults to 0
            draw_bbox (bool): Whether to draw the bounding boxes.
                Defaults to False
            radius (int): Keypoint radius for visualization. Defaults to 3
            thickness (int): Link thickness for visualization. Defaults to 1
            kpt_thr (float): The threshold to visualize the keypoints.
                Defaults to 0.3
            vis_out_dir (str, optional): Directory to save visualization
                results w/o predictions. If left as empty, no file will
                be saved. Defaults to ''.
            window_name (str, optional): Title of display window.
            window_close_event_handler (callable, optional):

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if (not return_vis) and (not show) and (not vis_out_dir):
            return

        if getattr(self, 'visualizer', None) is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.radius = radius
        self.visualizer.line_width = thickness
        det_kpt_color = self.visualizer.kpt_color
        det_dataset_skeleton = self.visualizer.skeleton
        det_dataset_link_color = self.visualizer.link_color
        self.visualizer.det_kpt_color = det_kpt_color
        self.visualizer.det_dataset_skeleton = det_dataset_skeleton
        self.visualizer.det_dataset_link_color = det_dataset_link_color

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = mmcv.bgr2rgb(single_input)
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            # since visualization and inference utilize the same process,
            # the wait time is reduced when a video input is utilized,
            # thereby eliminating the issue of inference getting stuck.
            wait_time = 1e-5 if self._video_input else wait_time

            if num_instances < 0:
                num_instances = len(pred.pred_instances)

            visualization = self.visualizer.add_datasample(
                window_name,
                img,
                data_sample=pred,
                # det_data_sample=self._buffer['pose2d_results'],
                draw_gt=False,
                draw_bbox=draw_bbox,
                show=show,
                wait_time=wait_time,
                # dataset_2d=self.pose2d_model.model.
                # dataset_meta['dataset_name'],
                dataset_3d=self.model.dataset_meta['dataset_name'],
                kpt_thr=kpt_thr,
                num_instances=num_instances)
            results.append(visualization)

            if vis_out_dir:
                img_name = os.path.basename(pred.metainfo['img_path']) \
                    if 'img_path' in pred.metainfo else None
                self.save_visualization(
                    visualization,
                    vis_out_dir,
                    img_name=img_name,
                )

        if return_vis:
            return results
        else:
            return []
