# Copyright (c) OpenMMLab. All rights reserved.

# from mmpose.codecs.utils import camera_to_image_coord
from mmpose.registry import DATASETS
from mmpose.utils import SimpleCamera
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class UBody3DCOCODataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ubody.py')

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        camera_param = data_info['camera_param']
        camera_param = {
            'f': camera_param['focal'],
            'c': camera_param['princpt']
        }
        camera = SimpleCamera(camera_param)
        keypoints_3d_camera = camera.pixel_to_camera(data_info['keypoints_3d'])

        data_info['keypoints_3d_cam'] = keypoints_3d_camera
        data_info['camera_param'] = camera_param

        return data_info
