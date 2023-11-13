# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class H36MCOCODataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h36m.py')

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        keypoints_3d = raw_data_info['raw_ann_info'].get('keypoints_3d', None)
        if keypoints_3d is None:
            raise ValueError('keypoints_3d is required in data_info')

        data_info['keypoints_3d'] = keypoints_3d

        return data_info
