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
        self.coco = None
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        keypoints_3d = raw_data_info['raw_ann_info'].get('keypoints_3d', None)
        if keypoints_3d is None:
            raise ValueError('keypoints_3d is required in data_info')
        _keypoints_3d = np.array(
            keypoints_3d, dtype=np.float32).reshape(1, -1, 4)
        keypoints_camera, keypoints_2d, cam_key = self._keypoint_world_to_gt(
            _keypoints_3d, self.camera_params, data_info['img_path'])
        keypoints_depth = keypoints_camera[..., 2]
        keypoints_pixel = np.concatenate(
            [keypoints_2d, keypoints_depth[..., None]], axis=-1)
        keypoints_3d_visible = np.minimum(1, _keypoints_3d[..., 3])

        data_info['keypoints_3d_gt'] = keypoints_camera
        data_info['keypoints_3d'] = keypoints_pixel
        data_info['keypoints_3d_visible'] = keypoints_3d_visible
        data_info['camera_param'] = self.camera_params[cam_key]
        return data_info

    def _keypoint_world_to_gt(self, keypoints, camera_params, image_name=None):
        """Project 3D keypoints from the world space to the camera space.

        Args:
            keypoints (np.ndarray): 3D keypoints in shape [..., 3]
            camera_params (dict): Parameters for all cameras.
            image_name (str): The image name to specify the camera.
        """
        subj, rest = osp.basename(image_name).split('_', 1)
        _, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        cam_key = (subj, camera)
        camera = SimpleCamera(camera_params[cam_key])
        keypoints_camera = camera.world_to_camera(keypoints[..., :3])
        keypoints_2d = camera.camera_to_pixel(keypoints_camera)

        return keypoints_camera, keypoints_2d, cam_key
