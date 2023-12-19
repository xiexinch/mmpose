# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AgoraDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(
        from_file='configs/_base_/datasets/coco_wholebody.py')

    def parse_data_info(self, raw_data_info: dict) -> dict:
        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 2)

        keypoints_visible = np.ones((1, 133, 1), dtype=np.float32)

        keypoints_3d = np.array(
            ann['keypoints_3d'], dtype=np.float32).reshape(1, -1, 3)

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        camera_param = ann['camera_param']

        target_idx = [-1]

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            # 'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'keypoints_3d': keypoints_3d,
            'camera_param': camera_param,
            # 'area': area,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': np.array(ann['category_id']),
            'target_idx': target_idx,
            'lifting_target': keypoints_3d[target_idx],
            'lifting_target_visible': keypoints_visible[target_idx],
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        self.coco = None
        return data_list
