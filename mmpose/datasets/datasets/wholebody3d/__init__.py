# Copyright (c) OpenMMLab. All rights reserved.
from .h3wb_dataset import H36MWholeBodyDataset
from .mpii_3d_dataset import MPII3DWBDataset
from .mpii_3dhp_dataset import MPII3DHPWBDataset
from .ubody3d_dataset import UBody3dDataset

__all__ = [
    'UBody3dDataset', 'H36MWholeBodyDataset', 'MPII3DWBDataset',
    'MPII3DHPWBDataset'
]
