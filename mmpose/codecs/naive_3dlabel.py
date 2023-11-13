# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class Naive3DLabel(BaseKeypointCodec):

    auxiliary_encode_keys = {'keypoints_3d'}

    label_mapping_table = dict(
        keypoint_x_labels='keypoint_x_labels',
        keypiont_y_labels='keypoint_y_labels',
        keypoint_z_labels='keypoint_z_labels',
        keypoint_weights='keypoint_weights',
    )

    def __init__(self,
                 input_size: tuple,
                 simcc_split_ratio: float = 2.0) -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio

    def encode(self,
               keypoints: np.ndarray,
               keypoints_3d: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None):
        """Encode keypoints to 3D labels."""

        encoded = dict(
            keypoint_x_labels=keypoints_3d[..., 0],
            keypoint_y_labels=keypoints_3d[..., 1],
            keypoint_z_labels=keypoints_3d[..., 2],
            keypoint_weights=keypoints_visible)
        return encoded

    def decode(self, keypoints, keypoints_3d, keypoints_visible):
        """Decode keypoints from 3D labels."""
        return keypoints_3d, keypoints_visible
