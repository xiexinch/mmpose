# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List, Tuple

import cv2
import h5py
import numpy as np

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


def get_bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Args:
        mask (np.ndarray): mask of the object, shape (H, W)

    Returns:
        bbox (np.ndarray): bounding box of the object, shape (1, 4),
            (x1, y1, x2, y2)
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(contours[0])
    bbox = np.array([x, y, x + w, y + h], dtype=np.float32)[None, :]
    return bbox


@DATASETS.register_module()
class DNARenderingDataset(BaseMocapDataset):

    def __init__(self, data_root: str, **kwargs):
        self.data_root = data_root
        super().__init__(data_root=data_root, **kwargs)

    def _load_ann_file(self, ann_file: str) -> dict:
        anno_files = os.listdir(osp.join(self.data_root, 'annotations'))
        ann_data = dict()
        img_data = dict()

        for file in anno_files:
            ann_data[file.replace('_annots.smc', '')] = h5py.File(file, 'r')
        self.ann_data = ann_data

        for key in ann_data.keys():
            file_path = osp.join(self.data_root, 'main', f'{key}.smc')
            img_data[key] = h5py.File(file_path, 'r')
        self.img_data = img_data

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:

        instance_list = []

        for key in self.ann_data.keys():
            anns = self.ann_data[key]
            imgs = self.img_data[key]

            cam_params = anns['Camera_Parameter']
            keypoints_2d_full = anns['Keypoints_2D']
            keypoints_3d = anns['Keypoints_3D']['keypoints3d']
            keypoints_3d_mask = anns['Keypoints_3D']['keypoints3d_mask'][(
            )].astype(bool)
            keypoints_3d = keypoints_3d[:, keypoints_3d_mask, :]
            object_masks = anns['Mask']

            cam_groups = imgs.keys()
            instance_id = 0
            for cam_group in cam_groups:
                cam_ids = imgs[cam_group].keys()
                for cam_id in cam_ids:
                    frame_ids = imgs[cam_group][cam_id]['color'].keys()
                    for frame_id in frame_ids:
                        img_bytes = imgs[cam_group][cam_id]['color'][frame_id][
                            ()]
                        kpts_2d = keypoints_2d_full[cam_id][int(frame_id)][
                            None, ..., :2].astype(np.float32)
                        kpts_3d_global = keypoints_3d[int(frame_id)][
                            None, ...].astype(np.float32)
                        kpts_3d = kpts_3d_global[..., :3]
                        kpts_visible = kpts_3d_global[..., 3]
                        cam_param = {
                            'RT':
                            cam_params[cam_id]['RT'][()].astype(np.float32),
                            'K':
                            cam_params[cam_id]['K'][()].astpye(np.float32),
                            'D': cam_params[cam_id]['D'][()].astype(np.float32)
                        }
                        cam_param['R'] = np.linalg.inv(cam_param['RT'])
                        kpts_3d_cam = kpts_3d @ cam_param['R']

                        mask_bytes = object_masks[cam_id]['mask'][frame_id][()]
                        mask = np.max(
                            cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR), 2)
                        bbox = get_bbox_from_mask(mask)

                        instance_info = {
                            'id': instance_id,
                            'img': img_bytes,
                            'keypoints': kpts_2d,
                            'keypoints_3d': kpts_3d_cam,
                            'keypoints_visible': kpts_visible,
                            'lifting_target': kpts_3d_cam,
                            'lifting_target_visible': kpts_visible,
                            'camera_param': cam_param,
                            'mask': mask,
                            'bbox': bbox,
                            'bbox_score': np.ones((1, ), dtype=np.float32)
                        }
                        instance_id += 1
                        instance_list.append(instance_info)

        return instance_list, []
