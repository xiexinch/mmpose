# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from mmengine.fileio import get_local_path

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS
from mmpose.structures import bbox_xywh2xyxy


@DATASETS.register_module()
class GTAHumanDataset(BaseMocapDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            data = np.load(local_path, allow_pickle=True)
        self.ann_data = list(data['data'])

    def get_sequence_indices(self) -> List[List[int]]:
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        instance_list = []
        num_keypoints = 133
        instance_id = 0
        for ann in self.ann_data:
            for i in range(ann['num_frames']):
                kpts_2d = ann['keypoints_2d'][i][None, ...].astype(np.float32)
                kpts_3d = ann['keypoints_3d'][i][None, ...].astype(np.float32)
                kpts_visible = ann['keypoint_weights'][i][None, ...].astype(
                    np.float32)
                img_path = ann['img_paths'][i]
                bbox = bbox_xywh2xyxy(ann['bboxes'][i][None,
                                                       ...].astype(np.float32))
                root = ann['roots'][i][None, ...].astype(np.float32)
                instance_info = {
                    'num_keypoints': num_keypoints,
                    'keypoints': kpts_2d,
                    'keypoints_3d': kpts_3d,
                    'keypoints_visible': kpts_visible,
                    'keypoints_3d_visible': kpts_visible,
                    'bbox': bbox,
                    'bbox_score': np.ones((1, )),
                    'root': root,
                    'img_path': img_path,
                    'lifting_target': kpts_3d,
                    'lifting_target_visible': kpts_visible,
                    'categoray_id': 1,
                    'is_crowd': 0,
                    'id': instance_id
                }
                instance_id += 1
                instance_list.append(instance_info)

        assert 0 < self.subset_frac <= 1
        if self.subset_frac < 1:
            length = len(instance_list)
            subset_length = int(length * self.subset_frac)
            step = length // subset_length
            subset_list = []
            for idx in range(0, length, step):
                subset_list.append(instance_list[idx])
            instance_list = subset_list
        del self.ann_data
        return instance_list, []
