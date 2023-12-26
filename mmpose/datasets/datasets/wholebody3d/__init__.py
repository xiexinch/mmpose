# Copyright (c) OpenMMLab. All rights reserved.
from .agora_dataset import AgoraDataset
from .coco_wholebody_3d import COCOWholebody3D
from .h3wb_dataset import H3WBDataset, H36MWholeBodyDataset
from .mpii_3d_dataset import MPII3DWBDataset
from .ubody3d_dataset import UBody3dDataset

__all__ = [
    'UBody3dDataset', 'H36MWholeBodyDataset', 'H3WBDataset', 'AgoraDataset',
    'COCOWholebody3D', 'MPII3DWBDataset'
]
