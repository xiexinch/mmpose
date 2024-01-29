# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import get_local_path
from mmengine.logging import print_log

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS
from ..body3d import Human36mDataset


@DATASETS.register_module()
class H36MWholeBodyDataset(Human36mDataset):
    """H36MWholeBodyDataset dataset for pose estimation.

    H36M-WholeBody keypoints::

        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints

        In total, we have 133 keypoints for wholebody pose estimation.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        seq_step (int): The interval for extracting frames from the video.
            Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        multiple_target_step (int): The interval for merging sequence. Only
            valid when ``multiple_target`` is larger than 0. Default: 0.
        pad_video_seq (bool): Whether to pad the video so that poses will be
            predicted for every frame in the video. Default: ``False``.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        keypoint_2d_src (str): Specifies 2D keypoint information options, which
            should be one of the following options:

            - ``'gt'``: load from the annotation file
            - ``'detection'``: load from a detection
              result file of 2D keypoint
            - 'pipeline': the information will be generated by the pipeline

            Default: ``'gt'``.
        keypoint_2d_det_file (str, optional): The 2D keypoint detection file.
            If set, 2d keypoint loaded from this file will be used instead of
            ground-truth keypoints. This setting is only when
            ``keypoint_2d_src`` is ``'detection'``. Default: ``None``.
        factor_file (str, optional): The projection factors' file. If set,
            factor loaded from this file will be used instead of calculated
            factors. Default: ``None``.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def __init__(self,
                 ann_file: str,
                 data_root: str,
                 data_prefix: dict,
                 joint_2d_src: str,
                 normalize_with_dataset_stats: bool = False,
                 **kwargs):
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.joint_2d_src = joint_2d_src
        self.normalize_with_dataset_stats = normalize_with_dataset_stats

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

    def _load_ann_file(self, ann_file: str) -> dict:
        """Load annotation file to get image information.

        Args:
            ann_file (str): Annotation file path.

        Returns:
            dict: Annotation information.
        """

        with get_local_path(ann_file) as local_path:
            self.ann_data = json.load(open(local_path))
        with get_local_path(osp.join(self.data_root,
                                     self.joint_2d_src)) as local_path:
            self.joint_2d_ann = json.load(open(local_path))
        self._process_image_names(self.ann_data)

    def _process_image_names(self, ann_data: dict) -> List[str]:
        """Process image names."""
        image_folder = self.data_prefix['img']
        img_names = [ann_data[i]['image_path'] for i in ann_data]
        image_paths = []
        for image_name in img_names:
            scene, _, sub, frame = image_name.split('/')
            frame, suffix = frame.split('.')
            frame_id = f'{int(frame.split("_")[-1]) + 1:06d}'
            sub = '_'.join(sub.split(' '))
            path = f'{scene}/{scene}_{sub}/{scene}_{sub}_{frame_id}.{suffix}'
            if not osp.exists(osp.join(self.data_root, image_folder, path)):
                print_log(
                    f'Failed to read image {path}.',
                    logger='current',
                    level=logging.WARN)
                continue
            image_paths.append(path)
        self.image_names = image_paths

    def get_sequence_indices(self) -> List[List[int]]:
        self.ann_data['imgname'] = self.image_names
        return super().get_sequence_indices()

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        num_keypoints = self.metainfo['num_keypoints']

        img_names = np.array(self.image_names)
        num_imgs = len(img_names)

        scales = np.zeros(num_imgs, dtype=np.float32)
        centers = np.zeros((num_imgs, 2), dtype=np.float32)

        kpts_3d, kpts_2d = [], []
        for k in self.ann_data.keys():
            if not isinstance(self.ann_data[k], dict):
                continue
            ann, ann_2d = self.ann_data[k], self.joint_2d_ann[k]
            kpts_2d_i, kpts_3d_i = self._get_kpts(ann, ann_2d)
            kpts_3d.append(kpts_3d_i)
            kpts_2d.append(kpts_2d_i)

        kpts_3d = np.concatenate(kpts_3d, axis=0)
        kpts_2d = np.concatenate(kpts_2d, axis=0)
        kpts_visible = np.ones_like(kpts_2d[..., 0], dtype=np.float32)

        # Normalize 3D keypoints like H36M
        # Ref: https://github.com/open-mmlab/mmpose/blob/main/tools/dataset_converters/preprocess_h36m.py#L324 # noqa
        kpts_3d /= 1000.0

        if self.factor_file:
            with get_local_path(self.factor_file) as local_path:
                factors = np.load(local_path).astype(np.float32)
        else:
            factors = np.zeros((kpts_3d.shape[0], ), dtype=np.float32)

        instance_list = []
        for idx, frame_ids in enumerate(self.sequence_indices):
            expected_num_frames = self.seq_len
            if self.multiple_target:
                expected_num_frames = self.multiple_target

            assert len(frame_ids) == (expected_num_frames), (
                f'Expected `frame_ids` == {expected_num_frames}, but '
                f'got {len(frame_ids)} ')

            _img_names = img_names[frame_ids]
            _kpts_2d = kpts_2d[frame_ids]
            _kpts_3d = kpts_3d[frame_ids]
            _kpts_visible = kpts_visible[frame_ids]
            factor = factors[frame_ids].astype(np.float32)

            target_idx = [-1] if self.causal else [int(self.seq_len) // 2]
            if self.multiple_target > 0:
                target_idx = list(range(self.multiple_target))

            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': _kpts_2d,
                'keypoints_3d': _kpts_3d,
                'keypoints_visible': _kpts_visible,
                'keypoints_3d_visible': _kpts_visible,
                'scale': scales[idx],
                'center': centers[idx].astype(np.float32).reshape(1, -1),
                'factor': factor,
                'id': idx,
                'category_id': 1,
                'iscrowd': 0,
                'img_paths': list(_img_names),
                'img_ids': frame_ids,
                'lifting_target': _kpts_3d[target_idx],
                'lifting_target_visible': _kpts_visible[target_idx],
                'target_img_path': _img_names[target_idx],
            }

            if self.camera_param_file:
                _cam_param = self.get_camera_param(_img_names[0])
            else:
                # Use the max value of camera parameters in Human3.6M dataset
                _cam_param = {
                    'w': 1000,
                    'h': 1002,
                    'f': np.array([[1149.67569987], [1148.79896857]]),
                    'c': np.array([[519.81583718], [515.45148698]])
                }
            instance_info['camera_param'] = _cam_param
            instance_list.append(instance_info)

        image_list = []
        if self.data_mode == 'bottomup':
            for idx, img_name in enumerate(img_names):
                img_info = self.get_img_info(idx, img_name)
                image_list.append(img_info)

        return instance_list, image_list

    def _get_kpts(self, ann: dict,
                  ann_2d: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D keypoints and 3D keypoints from annotation."""
        kpts = ann['keypoints_3d']
        kpts_2d = ann_2d['keypoints_2d']
        kpts_3d = np.array([[j['x'], j['y'], j['z']] for _, j in kpts.items()],
                           dtype=np.float32)[np.newaxis, ...]
        kpts_2d = np.array([[j['x'], j['y']] for _, j in kpts_2d.items()],
                           dtype=np.float32)[np.newaxis, ...]
        return kpts_2d, kpts_3d


@DATASETS.register_module()
class H3WBDataset(BaseMocapDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def __init__(
        self,
        ann_file: str = '',
        seq_len: int = 1,
        multiple_target: int = 0,
        causal: bool = True,
        subset_frac: float = 1.0,
        camera_param_file: Optional[str] = None,
        data_mode: str = 'topdown',
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img=''),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        train_mode: bool = True,
    ):
        assert data_mode == 'topdown'

        self.camera_order_id = ['54138969', '55011271', '58860488', '60457274']
        if train_mode:
            self.subjects = ['S1', 'S5', 'S6']
        else:
            self.subjects = ['S7']

        super().__init__(
            ann_file=ann_file,
            seq_len=seq_len,
            multiple_target=multiple_target,
            causal=causal,
            subset_frac=subset_frac,
            camera_param_file=camera_param_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            data = np.load(local_path, allow_pickle=True)

        self.ann_data = data['train_data'].item()
        self._metadata = data['metadata'].item()

    def get_sequence_indices(self) -> List[List[int]]:
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:

        instance_list = []
        image_list = []

        instance_id = 0
        for subject in self.subjects:
            actions = self.ann_data[subject].keys()
            for act in actions:
                kpts_global = self.ann_data[subject][act]['global_3d']
                for cam in self.camera_order_id:
                    keypoints_2d = self.ann_data[subject][act][cam]['pose_2d']
                    keypoints_3d = self.ann_data[subject][act][cam][
                        'camera_3d']
                    num_keypoints = keypoints_2d.shape[1]

                    camera_param = self._metadata[subject][cam]
                    K = camera_param['K'][0]
                    f = (K[0, 0], K[1, 1])
                    c = (K[0, 2], K[1, 2])
                    camera_param = {
                        'K': camera_param['K'][0, :2, ...],
                        'R': camera_param['R'][0],
                        'T': camera_param['T'].reshape(3, 1),
                        'Distortion': camera_param['Distortion'][0],
                        'f': f,
                        'c': c
                    }

                    seq_step = 1
                    _len = (self.seq_len - 1) * seq_step + 1
                    _indices = list(
                        range(len(self.ann_data[subject][act]['frame_id'])))
                    seq_indices = [
                        _indices[i:(i + _len):seq_step]
                        for i in list(range(0,
                                            len(_indices) - _len + 1))
                    ]

                    for idx, frame_ids in enumerate(seq_indices):
                        expected_num_frames = self.seq_len
                        if self.multiple_target:
                            expected_num_frames = self.multiple_target

                        assert len(frame_ids) == (expected_num_frames), (
                            f'Expected `frame_ids` == {expected_num_frames}, but '  # noqa
                            f'got {len(frame_ids)} ')

                        _kpts_2d = keypoints_2d[frame_ids]
                        _kpts_3d = keypoints_3d[frame_ids]
                        _kpts_global = kpts_global[frame_ids]
                        target_idx = [-1] if self.causal else [
                            int(self.seq_len) // 2
                        ]
                        if self.multiple_target > 0:
                            target_idx = list(range(self.multiple_target))

                        instance_info = {
                            'num_keypoints':
                            num_keypoints,
                            'keypoints':
                            _kpts_2d,
                            'keypoints_3d':
                            _kpts_3d / 1000,
                            'keypoints_global':
                            _kpts_global / 1000,
                            'keypoints_visible':
                            np.ones_like(_kpts_2d[..., 0], dtype=np.float32),
                            'keypoints_3d_visible':
                            np.ones_like(_kpts_2d[..., 0], dtype=np.float32),
                            'scale':
                            np.zeros((1, 1), dtype=np.float32),
                            'center':
                            np.zeros((1, 2), dtype=np.float32),
                            'factor':
                            np.zeros((1, 1), dtype=np.float32),
                            'id':
                            instance_id,
                            'category_id':
                            1,
                            'iscrowd':
                            0,
                            'camera_param':
                            camera_param,
                            'img_paths': [
                                f'{subject}/{act}/{cam}/{i:06d}.jpg'
                                for i in frame_ids
                            ],
                            'img_ids':
                            frame_ids,
                            'lifting_target':
                            _kpts_3d[target_idx] / 1000,
                            'lifting_target_visible':
                            np.ones_like(_kpts_2d[..., 0],
                                         dtype=np.float32)[target_idx],
                        }
                        instance_list.append(instance_info)

                        if self.data_mode == 'bottomup':
                            for idx, img_name in enumerate(
                                    instance_info['img_paths']):
                                img_info = self.get_img_info(idx, img_name)
                                image_list.append(img_info)

                        instance_id += 1
        del self.ann_data
        return instance_list, image_list
