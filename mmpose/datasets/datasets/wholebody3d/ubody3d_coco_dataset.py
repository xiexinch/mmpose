# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

from mmpose.registry import DATASETS
from mmpose.utils import SimpleCamera
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class UBody3DCOCODataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ubody3d.py')

    def parse_data_info(self, raw_data_info: dict) -> dict:
        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['img_info']

        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 2)
        keypoints_visible = np.array(
            ann['keypoints_valid'], dtype=np.float32).reshape(-1, 1)

        keypoints_3d = np.array(
            ann['keypoints_3d'], dtype=np.float32).reshape(-1, 3)

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        if 'area' in ann:
            area = np.array(ann['area'], dtype=np.float32)
        else:
            area = np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None)
            area = np.array(area, dtype=np.float32)

        camera_param = ann['camera_param']
        camera_param = {
            'f': camera_param['focal'],
            'c': camera_param['princpt']
        }
        camera = SimpleCamera(camera_param)
        keypoints_3d_camera = camera.pixel_to_camera(keypoints_3d)

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'keypoints_3d': keypoints_3d,
            'keypoints_3d_cam': keypoints_3d_camera,
            'camera_param': camera_param,
            'area': area,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': np.array(ann['category_id']),
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info
