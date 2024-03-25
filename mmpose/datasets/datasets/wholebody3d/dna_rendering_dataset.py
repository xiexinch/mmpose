# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import get_local_path

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class DNARenderingDataset(BaseMocapDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 causal: bool = True,
                 subset_frac: float = 1.0,
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
                 max_refetch: int = 1000):
        self.hand_roots = [9, 10, 91, 112]
        self.data_mode = 'topdown'
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            subset_frac=subset_frac)

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            ann = np.load(local_path, allow_pickle=True)
        self.ann_data = ann['instances']

    def get_sequence_indices(self) -> List[List[int]]:
        assert self.seq_len == 1, 'Sequence length must be 1 for COCO dataset'
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        instance_list = []
        image_list = []

        instance_id = 0
        length = len(self.ann_data)
        sub_length = int(length * self.subset_frac)
        step = length // sub_length

        print(
            f'Loading {self.subset_frac * 100}% DNA Rendering annotations.....'
        )
        for idx, ann in enumerate(self.ann_data):
            if idx % step != 0:
                continue

            instance_info = {
                'img_path': ann['img_path'],
                'keypoints': ann['keypoints'],
                'keypoints_3d': ann['keypoints_3d'],
                'keypoints_visible': ann['keypoints_visible'],
                'bbox': ann['bbox'],
                'bbox_score': ann['bbox_score'],
                'cam_param': [ann['camera_param']]
            }

            instance_list.append(instance_info)
            instance_id += 1

        del self.ann_data
        return instance_list, image_list

    def _concnat_hand_roots(self, kpts, left_wrist_id, right_wrist_id,
                            left_hand_root_id, right_hand_root_id):
        left_wrist = kpts[..., left_wrist_id, :].copy()
        right_wrist = kpts[..., right_wrist_id, :].copy()
        kpts = np.concatenate([
            kpts[:, :left_hand_root_id, :], left_wrist[None, :],
            kpts[:, left_hand_root_id:right_hand_root_id, :],
            right_wrist[None, :], kpts[:, right_hand_root_id:, :]
        ],
                              axis=1)
        return kpts
