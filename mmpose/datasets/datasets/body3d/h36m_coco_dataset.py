# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine
import numpy as np

# from mmpose.codecs.utils import camera_to_image_coord
from mmpose.registry import DATASETS
from mmpose.utils import SimpleCamera
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class H36MCOCODataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h36m.py')

    def __init__(self, camera_param_file: str, **kwargs):
        self.camera_params = mmengine.load(
            osp.join(kwargs['data_root'], camera_param_file))
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        # calculate the stats of keypoints_3d
        z_max = np.max(
            [np.max(data['keypoints_3d'][..., 2:]) for data in data_list])
        z_min = np.min(
            [np.min(data['keypoints_3d'][..., 2:]) for data in data_list])
        # set the z_min to data_info
        for data_info in data_list:
            data_info['z_min'] = np.array([z_min])
            data_info['z_max'] = np.array([z_max])
        self.coco = None
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        keypoints_3d = raw_data_info['raw_ann_info'].get('keypoints_3d', None)
        if keypoints_3d is None:
            raise ValueError('keypoints_3d is required in data_info')
        _keypoints_3d = np.array(
            keypoints_3d, dtype=np.float32).reshape(1, -1, 4)
        subj, rest = osp.basename(data_info['img_path']).split('_', 1)
        _, rest = rest.split('.', 1)
        camera_name, rest = rest.split('_', 1)
        cam_key = (subj, camera_name)
        camera = SimpleCamera(self.camera_params[cam_key])
        kpts_3d = _keypoints_3d[..., :3]
        kpt_3d_cam = camera.world_to_camera(kpts_3d)
        keypoints_3d_visible = np.minimum(1, _keypoints_3d[..., 3])

        data_info['keypoints_3d'] = kpt_3d_cam
        data_info['keypoints_3d_visible'] = keypoints_3d_visible
        data_info['camera_param'] = self.camera_params[cam_key]
        return data_info
