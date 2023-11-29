# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
# from mmpose.utils import SimpleCamera
from .base import BaseKeypointCodec
from .utils import pixel_to_camera


@KEYPOINT_CODECS.register_module()
class Naive3DLabel(BaseKeypointCodec):

    auxiliary_encode_keys = {
        'transformed_keypoints_3d', 'camera_param', 'keypoints_3d'
    }

    label_mapping_table = dict(
        keypoint_x_labels='keypoint_x_labels',
        keypoint_y_labels='keypoint_y_labels',
        keypoint_z_labels='keypoint_z_labels',
        keypoint_weights='keypoint_weights',
        keypoints_3d='keypoints_3d',
        keypoints_3d_gt='keypoints_3d_gt',
        keypoints_3d_visible='keypoints_3d_visible')

    instance_mapping_table = dict(
        bboxes='bboxes',
        keypoints_3d='keypoints_3d',
        keypoints_3d_gt='keypoints_3d_gt',
        keypoints_3d_visible='keypoints_3d_visible',
    )

    def __init__(self,
                 input_size: tuple,
                 simcc_split_ratio: float = 2.0,
                 sigma: Union[float, int, Tuple[float]] = 6.0,
                 label_smooth_weight: float = 0.0,
                 normalize: bool = True,
                 root_index: int = 0,
                 rootrel: bool = True,
                 test_mode: bool = False,
                 gt_field: str = 'keypoints_3d') -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize
        self.root_index = root_index
        self.rootrel = rootrel

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma, sigma], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

        self.test_mode = test_mode
        self.gt_field = gt_field

    def encode(self,
               keypoints: np.ndarray,
               keypoints_3d: np.ndarray,
               camera_param: dict,
               transformed_keypoints_3d: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints to 3D labels."""

        fx, fy = camera_param['f']
        cx, cy = camera_param['c']
        keypoints_3d_camera = pixel_to_camera(keypoints_3d, fx, fy, cx, cy)

        encoded = dict()
        if self.rootrel:
            root = keypoints_3d_camera[:, self.root_index, :]
            keypoints_3d_camera -= keypoints_3d_camera[
                ..., self.root_index:self.root_index + 1, :]
            encoded['root'] = root

        if not self.test_mode:
            x, y, z, weights = self._generate_gaussian(
                transformed_keypoints_3d, keypoints_visible)  # noqa
            encoded['keypoint_x_labels'] = x
            encoded['keypoint_y_labels'] = y
            encoded['keypoint_z_labels'] = z
            encoded['keypoint_weights'] = weights
        else:
            encoded['keypoints_3d_gt'] = keypoints_3d_camera
            encoded['keypoints_3d_visible'] = keypoints_visible
        return encoded

    def decode(self,
               simcc_x: np.ndarray,
               simcc_y: np.ndarray,
               simcc_z: np.ndarray,
               warp_mat: np.ndarray,
               z_max: np.ndarray,
               z_min: np.ndarray,
               target_root: Optional[np.ndarray] = None,
               camera_param: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints from 3D labels."""
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y, simcc_z)

        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        keypoints /= self.simcc_split_ratio

        # 1. z 轴坐标从 (0, d) 映射到 (z_min, z_max)
        z_max = z_max[0]
        z_min = z_min[0]
        # print(z_max, z_min, self.input_size[2])
        keypoints_z = keypoints[..., 2:] / self.input_size[2] * (z_max -
                                                                 z_min) + z_min
        # 2. 不处理 z
        # keypoints_z = keypoints[..., 2:]

        # 还原 xy 到原图空间
        keypoints_xy = keypoints[..., :2]
        warp_mat_homogeneous = np.vstack([warp_mat[0], [0, 0, 1]])
        warp_inv = warp_inv = np.linalg.inv(warp_mat_homogeneous)
        keypoints_xy = cv2.transform(keypoints_xy, warp_inv)[..., :2]
        # 拼接 xyz
        keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=-1)

        # 转换图像空间
        # camera = SimpleCamera(camera_param)
        fx, fy = camera_param['f']
        cx, cy = camera_param['c']
        keypoints = pixel_to_camera(keypoints, fx, fy, cx, cy)
        if target_root is not None and target_root.size > 0:
            keypoints = keypoints + target_root
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
