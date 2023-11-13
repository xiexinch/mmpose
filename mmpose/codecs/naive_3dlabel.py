# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class Naive3DLabel(BaseKeypointCodec):

    auxiliary_encode_keys = {'keypoints_3d'}

    label_mapping_table = dict(
        keypoint_x_labels='keypoint_x_labels',
        keypoint_y_labels='keypoint_y_labels',
        keypoint_z_labels='keypoint_z_labels',
        keypoint_weights='keypoint_weights',
    )

    def __init__(self,
                 input_size: tuple,
                 simcc_split_ratio: float = 2.0,
                 sigma: Union[float, int, Tuple[float]] = 6.0,
                 label_smooth_weight: float = 0.0,
                 normalize: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma, sigma], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_3d: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None):
        """Encode keypoints to 3D labels."""

        x_labels, y_labels, z_labels, keypoint_weights =  \
            self._generate_gaussian(keypoints_3d, keypoints_visible)
        encoded = dict(
            keypoint_x_labels=x_labels,
            keypoint_y_labels=y_labels,
            keypoint_z_labels=z_labels,
            keypoint_weights=keypoint_weights,
        )
        return encoded

    def decode(self, simcc_x: np.ndarray, simcc_y: np.ndarray,
               simcc_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints from 3D labels."""
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y, simcc_z)

        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        keypoints /= self.simcc_split_ratio
        return keypoints, scores

    def _map_coordinates(self,
                         keypoints: np.ndarray,
                         keypoints_visible: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray]:
        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = keypoints_visible.copy()

        return keypoints_split, keypoint_weights

    def _generate_gaussian(self,
                           keypoints: np.ndarray,
                           keypoints_visible: Optional[np.ndarray] = None
                           ) -> Tuple[np.ndarray]:
        N, K, _ = keypoints.shape
        w, h, d = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)
        D = np.around(d * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)
        target_z = np.zeros((N, K, D), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xyz grid
        x = np.arange(0, W, 1, np.float32)
        y = np.arange(0, H, 1, np.float32)
        z = np.arange(0, D, 1, np.float32)

        for n, k in product(range(N), range(K)):
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            left, top, near = mu - radius
            right, bottom, far = mu + radius + 1

            if left >= W or top >= H or near >= D or right < 0 or bottom < 0 or far < 0:  # noqa
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y, mu_z = mu

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * self.sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * self.sigma[1]**2))
            target_z[n, k] = np.exp(-((z - mu_z)**2) / (2 * self.sigma[2]**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]
            target_z /= norm_value[2]

        return target_x, target_y, target_z, keypoint_weights


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray,
                      simcc_z: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        simcc_z (np.ndarray): z-axis SimCC in shape (K, Wz) or (N, K, Wz)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 3) or (N, K, 3)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert isinstance(simcc_z, np.ndarray), ('simcc_z should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_z.ndim == 2 or simcc_z.ndim == 3, (
        f'Invalid shape {simcc_z.shape}')
    assert simcc_x.ndim == simcc_y.ndim == simcc_z.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape} != {simcc_z.ndim} ')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
        simcc_z = simcc_z.reshape(N * K, -1)
    else:
        N = None

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    z_locs = np.argmax(simcc_z, axis=1)
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    max_val_z = np.amax(simcc_z, axis=1)

    vals = np.minimum(np.minimum(max_val_x, max_val_y), max_val_z)
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 3)
        vals = vals.reshape(N, K)

    return locs, vals
