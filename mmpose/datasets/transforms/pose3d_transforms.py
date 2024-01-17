# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmpose.registry import TRANSFORMS
from mmpose.structures.keypoint import flip_keypoints_custom_center


@TRANSFORMS.register_module()
class RandomFlipAroundRoot(BaseTransform):
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        keypoints_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``keypoints``. Please
            refer to the docstring of the ``flip_keypoints_custom_center``
            function for more details.
        target_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``lifting_target``.
            Please refer to the docstring of the
            ``flip_keypoints_custom_center`` function for more details.
        flip_prob (float): Probability of flip. Default: 0.5.
        flip_camera (bool): Whether to flip horizontal distortion coefficients.
            Default: ``False``.
        flip_label (bool): Whether to flip labels instead of data.
            Default: ``False``.

    Required keys:
        - keypoints or keypoint_labels
        - lifting_target or lifting_target_label
        - keypoints_visible or keypoint_labels_visible (optional)
        - lifting_target_visible (optional)
        - flip_indices (optional)

    Modified keys:
        - keypoints or keypoint_labels (optional)
        - keypoints_visible or keypoint_labels_visible (optional)
        - lifting_target or lifting_target_label (optional)
        - lifting_target_visible (optional)
        - camera_param (optional)
    """

    def __init__(self,
                 keypoints_flip_cfg: dict,
                 target_flip_cfg: dict,
                 flip_prob: float = 0.5,
                 flip_camera: bool = False,
                 flip_label: bool = False):
        self.keypoints_flip_cfg = keypoints_flip_cfg
        self.target_flip_cfg = target_flip_cfg
        self.flip_prob = flip_prob
        self.flip_camera = flip_camera
        self.flip_label = flip_label

    def transform(self, results: Dict) -> dict:
        """The transform function of :class:`RandomFlipAroundRoot`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        if np.random.rand() <= self.flip_prob:
            if self.flip_label:
                assert 'keypoint_labels' in results
                assert 'lifting_target_label' in results
                keypoints_key = 'keypoint_labels'
                keypoints_visible_key = 'keypoint_labels_visible'
                target_key = 'lifting_target_label'
            else:
                assert 'keypoints' in results
                assert 'lifting_target' in results
                keypoints_key = 'keypoints'
                keypoints_visible_key = 'keypoints_visible'
                target_key = 'lifting_target'

            keypoints = results[keypoints_key]
            if keypoints_visible_key in results:
                keypoints_visible = results[keypoints_visible_key]
            else:
                keypoints_visible = np.ones(
                    keypoints.shape[:-1], dtype=np.float32)

            lifting_target = results[target_key]
            if 'lifting_target_visible' in results:
                lifting_target_visible = results['lifting_target_visible']
            else:
                lifting_target_visible = np.ones(
                    lifting_target.shape[:-1], dtype=np.float32)

            if 'flip_indices' not in results:
                flip_indices = list(range(self.num_keypoints))
            else:
                flip_indices = results['flip_indices']

            # flip joint coordinates
            _camera_param = deepcopy(results['camera_param'])

            keypoints, keypoints_visible = flip_keypoints_custom_center(
                keypoints,
                keypoints_visible,
                flip_indices,
                center_mode=self.keypoints_flip_cfg.get(
                    'center_mode', 'static'),
                center_x=self.keypoints_flip_cfg.get('center_x', 0.5),
                center_index=self.keypoints_flip_cfg.get('center_index', 0))
            lifting_target, lifting_target_visible = flip_keypoints_custom_center(  # noqa
                lifting_target,
                lifting_target_visible,
                flip_indices,
                center_mode=self.target_flip_cfg.get('center_mode', 'static'),
                center_x=self.target_flip_cfg.get('center_x', 0.5),
                center_index=self.target_flip_cfg.get('center_index', 0))

            results[keypoints_key] = keypoints
            results[keypoints_visible_key] = keypoints_visible
            results[target_key] = lifting_target
            results['lifting_target_visible'] = lifting_target_visible

            # flip horizontal distortion coefficients
            if self.flip_camera:
                assert 'camera_param' in results, \
                    'Camera parameters are missing.'

                assert 'c' in _camera_param
                _camera_param['c'][0] *= -1

                if 'p' in _camera_param:
                    _camera_param['p'][0] *= -1

                results['camera_param'].update(_camera_param)

        return results


@TRANSFORMS.register_module()
class RandomPerturb2DKeypoints(BaseTransform):

    def __init__(self,
                 pertur_prob: float = 0.5,
                 body_range: float = 10.0,
                 face_range: float = 2.0,
                 hand_range: float = 2.0,
                 body_indices: list = None,
                 hand_indices: list = None,
                 face_indices: list = None):
        """Add random perturbation to the input keypoints.

        Args:
           pertur_prob (float): Probability of perturbation
           max_perturbation (float): Maximum perturbation that can be added to
               each keypoint
        """
        self.pertur_prob = pertur_prob
        self.body_range = body_range
        self.face_range = face_range
        self.hand_range = hand_range
        self.body_indices = body_indices
        self.hand_indices = hand_indices
        self.face_indices = face_indices

    def _random_pertubation(self, keypoints: np.ndarray, indices: list,
                            pertur_range: float):
        """Add random perturbation to the input keypoints.

        Args:
            keypoints (np.ndarray): keypoints to be perturbed
            indices (list): indices of the keypoints to be perturbed
            pertur_range (float): maximum perturbation that can be added to
                each keypoint
        """
        keypoints = keypoints[:, indices]
        # Generate random perturbations for x and y coordinates
        perturbations = np.random.uniform(-pertur_range, pertur_range,
                                          keypoints.shape)
        # Apply the perturbations to the keypoints
        keypoints = keypoints + perturbations
        return keypoints

    def transform(self, results: dict) -> dict:

        keypoints = results['keypoints']

        prob = np.random.random()
        if prob > self.pertur_prob:
            return results

        perturbed_keypoints = keypoints.copy()

        if self.body_indices is not None:
            body_keypoints = self._random_pertubation(keypoints,
                                                      self.body_indices,
                                                      self.body_range)
            perturbed_keypoints[:, self.body_indices] = body_keypoints
        if self.hand_indices is not None:
            hand_keypoints = self._random_pertubation(keypoints,
                                                      self.hand_indices,
                                                      self.hand_range)
            perturbed_keypoints[:, self.hand_indices] = hand_keypoints
        if self.face_indices is not None:
            face_keypoints = self._random_pertubation(keypoints,
                                                      self.face_indices,
                                                      self.face_range)
            perturbed_keypoints[:, self.face_indices] = face_keypoints

        results['keypoints'] = perturbed_keypoints.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class RandomPerturbScoreBalance(RandomPerturb2DKeypoints):

    def _random_pertubation(self, keypoints: np.ndarray, indices: list,
                            pertur_range: float):
        """Add random perturbation to the input keypoints.

        Args:
            keypoints (np.ndarray): keypoints to be perturbed
            indices (list): indices of the keypoints to be perturbed
            pertur_range (float): maximum perturbation that can be added to
                each keypoint
        """
        keypoints = keypoints[:, indices]

        # Generate random perturbations for x, y, z coordinates
        perturbations = np.random.uniform(-pertur_range, pertur_range,
                                          keypoints.shape)

        # Apply the perturbations to the keypoints
        new_posision = keypoints + perturbations

        # Calculate distance between original position and new position
        distance = np.linalg.norm(new_posision - keypoints, axis=-1)
        rates = distance / pertur_range

        return keypoints, rates

    def transform(self, results: dict) -> dict:
        prob = np.random.random()
        if prob > self.pertur_prob:
            return results

        keypoints = results['keypoints']
        perturbed_keypoints = keypoints.copy()
        keypoints_visible = results['keypoints_visible'].copy()

        if self.body_indices is not None:
            body_keypoints, rates = self._random_pertubation(
                keypoints, self.body_indices, self.body_range)
            perturbed_keypoints[:, self.body_indices] = body_keypoints
            keypoints_score = 1 - rates
            keypoints_score[keypoints_score < 0] = 0
            keypoints_visible[:, self.body_indices] = keypoints_score

        if self.hand_indices is not None:
            hand_keypoints, rates = self._random_pertubation(
                keypoints, self.hand_indices, self.hand_range)
            perturbed_keypoints[:, self.hand_indices] = hand_keypoints
            keypoints_score = 1 - rates
            keypoints_score[keypoints_score < 0] = 0
            keypoints_visible[:, self.hand_indices] = keypoints_score
        if self.face_indices is not None:
            face_keypoints, rates = self._random_pertubation(
                keypoints, self.face_indices, self.face_range)
            perturbed_keypoints[:, self.face_indices] = face_keypoints
            keypoints_score = 1 - rates
            keypoints_score[keypoints_score < 0] = 0
            keypoints_visible[:, self.face_indices] = keypoints_score

        results['keypoints'] = perturbed_keypoints.astype(np.float32)
        results['keypoints_visible'] = keypoints_visible.astype(np.float32)
        results['lifting_target_visible'] = keypoints_visible.astype(
            np.float32)
        return results


@TRANSFORMS.register_module()
class RandomDropInput(BaseTransform):

    def __init__(self, prob: float = 0.5, drop_rate: float = 0.1):
        self.prob = prob
        self.drop_rate = drop_rate

    def transform(self, results: Dict) -> dict:
        if np.random.random() > self.prob:
            return results

        keypoints_visible = results['keypoints_visible']

        mask_indices = np.random.choice(
            keypoints_visible.shape[-1],
            int(keypoints_visible.shape[-1] * self.drop_rate),
            replace=False)
        keypoints_visible[:, mask_indices] = 0.0

        results['keypoints_visible'] = keypoints_visible
        results['lifting_target_visible'] = keypoints_visible
        return results


@TRANSFORMS.register_module()
class RandomHalfBody3D(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - keypoints
        - keypoints_visible
        - lifting_target_visible

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 9,
                 min_upper_keypoints: int = 2,
                 min_lower_keypoints: int = 3,
                 prob: float = 0.3,
                 upper_prioritized_prob: float = 0.7) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (num_upper < self.min_upper_keypoints
                        and num_lower < self.min_lower_keypoints):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_valid_ids if prefer_upper else lower_valid_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'],
            upper_body_ids=results['upper_body_ids'],
            lower_body_ids=results['lower_body_ids'])

        for indices in half_body_ids:
            if indices is None:
                continue
            results['keypoints_visible'][:, indices] = 0
            results['lifting_target_visible'][:, indices] = 0

        return results
