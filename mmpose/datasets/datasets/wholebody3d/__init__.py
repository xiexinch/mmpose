# Copyright (c) OpenMMLab. All rights reserved.
from .agora_dataset import AgoraDataset
from .h3wb_dataset import H3WBDataset, H36MWholeBodyDataset
from .ubody3d_dataset import UBody3dDataset

__all__ = [
    'UBody3dDataset', 'H36MWholeBodyDataset', 'H3WBDataset', 'AgoraDataset'
]
