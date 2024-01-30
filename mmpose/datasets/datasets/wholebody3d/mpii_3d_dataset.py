# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from mmengine.fileio import get_local_path

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class MPII3DWBDataset(BaseMocapDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            data = np.load(local_path, allow_pickle=True)
        self.ann_data = list(data['data'])

    def get_sequence_indices(self) -> List[List[int]]:
        assert self.seq_len == 1, 'Sequence length must be 1 for COCO dataset'
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:

        instance_list = []

        for i, ann in enumerate(self.ann_data):
            num_keypoints = self.metainfo['num_keypoints']

            kpts_2d = np.array(
                ann['keypoints_2d'], dtype=np.float32).reshape(1, -1, 2)
            kpts_3d = np.array(
                ann['keypoints_3d'], dtype=np.float32).reshape(1, -1, 3)
            keypoints_visible = np.ones((1, num_keypoints), dtype=np.float32)
            camera_param = {
                'f': ann['camera_param']['focal'],
                'c': ann['camera_param']['princpt'],
            }
            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': kpts_2d,
                'keypoints_3d': kpts_3d,
                'keypoints_visible': keypoints_visible,
                # 'factors': np.zeros((kpts_3d.shape[0], ), dtype=np.float32),
                'id': i,
                'category_id': 1,
                'iscrowd': 0,
                'lifting_target': kpts_3d,
                'lifting_target_visible': keypoints_visible,
                'camera_param': camera_param
            }
            instance_list.append(instance_info)
        del self.ann_data
        return instance_list, []
