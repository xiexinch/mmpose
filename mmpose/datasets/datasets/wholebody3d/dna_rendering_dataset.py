# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import logging
import os
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import h5py
import numpy as np

from mmengine import ProgressBar, print_log
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
    return np.array([x, y, x + w, y + h], dtype=np.float32)[None, :]


@DATASETS.register_module()
class DNARenderingDataset(BaseMocapDataset):

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 causal: bool = True,
                 subset_frac: float = 1.0,
                 camera_param_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.data_root = data_root
        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        self._load_ann_file('')

        self.seq_len = 1
        self.causal = causal

        assert 0 < subset_frac <= 1, (
            f'Unsupported `subset_frac` {subset_frac}. Supported range '
            'is (0, 1].')
        self.subset_frac = subset_frac

        self.sequence_indices = self.get_sequence_indices()

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _load_ann_file(self, ann_file: str) -> dict:
        anno_files = os.listdir(osp.join(self.data_root, 'annotations'))
        ann_data = dict()
        img_data = dict()

        for file in anno_files:
            path = osp.join(self.data_root, 'annotations', file)
            ann_data[file.replace('_annots.smc', '')] = h5py.File(path, 'r')
        self.ann_data = ann_data

        for key in ann_data.keys():
            file_path = osp.join(self.data_root, 'main', f'{key}.smc')
            img_data[key] = h5py.File(file_path, 'r')
        self.img_data = img_data

    def get_sequence_indices(self) -> List[List[int]]:
        assert self.seq_len == 1, 'Sequence length must be 1 for COCO dataset'
        return []
    
    def load_data_list(self) -> List[dict]:
        asyncio.run(self._load_annotations())
        return self.data_list

    async def _load_annotations(self) -> Tuple[List[dict], List[dict]]:

        instance_list = []
        
        async def _async_create_instance(self, frame_id: str, cam_id: str, cam_group: str, imgs:dict, anns:dict):
            img_bytes = imgs[cam_group][cam_id]['color'][frame_id][()]
            if len(cam_id) == 1:
                cam_id = '0' + cam_id
            kpts_2d = anns['Keypoints_2D'][cam_id][int(frame_id)][None, ..., :2].astype(np.float32)
            kpts_3d_full = anns['Keypoints_3D']['keypoints3d']
            keypoints_3d_mask = anns['Keypoints_3D']['keypoints3d_mask'][()].astype(bool)
            kpts_3d_full = kpts_3d_full[:, keypoints_3d_mask, :]

            kpts_3d_global = kpts_3d_full[int(frame_id)][None, ...].astype(np.float32)
            kpts_3d = kpts_3d_global[..., :3]
            kpts_visible = kpts_3d_global[..., 3]
            
            cam_param = {
                'RT': anns['Camera_Parameter'][cam_id]['RT'][()].astype(np.float32),
                'K': anns['Camera_Parameter'][cam_id]['K'][()].astype(np.float32),
                'D': anns['Camera_Parameter'][cam_id]['D'][()].astype(np.float32)
            }
            cam_param['R'] = np.linalg.inv(cam_param['RT'])

            kpts_3d = np.concatenate([kpts_3d, np.ones((kpts_3d.shape[0], kpts_3d.shape[1], 1))], axis=-1)
            kpts_3d_cam = kpts_3d @ cam_param['R']
            kpts_3d_cam = kpts_3d_cam[..., :3]

            mask_bytes = anns['Mask'][str(int(cam_id))]['mask'][frame_id][()]
            mask = np.max(cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR), 2)
            # bbox = get_bbox_from_mask(mask)

            instance_info = {
                'img': img_bytes,
                'keypoints': kpts_2d,
                'keypoints_3d': kpts_3d_cam,
                'keypoints_visible': kpts_visible,
                'lifting_target': kpts_3d_cam,
                'lifting_target_visible': kpts_visible,
                'camera_param': cam_param,
                'mask': mask
            }
            instance_list.append(instance_info)

        progress_bar = ProgressBar(len(self.ann_data.keys()))
        for key in self.ann_data.keys():
            anns = self.ann_data[key]
            imgs = self.img_data[key]

            cam_groups = imgs.keys()
            for cam_group in cam_groups:
                cam_ids = imgs[cam_group].keys()
                for cam_id in cam_ids:
                    frame_ids = imgs[cam_group][cam_id]['color'].keys()
                    tasks = [_async_create_instance(self, frame_id, cam_id, cam_group, imgs, anns) for frame_id in frame_ids]
                    await asyncio.gather(*tasks)
            progress_bar.update()
        self.data_list = instance_list
        print_log(f'Loaded {len(instance_list)} samples', logger='current', level=logging.INFO)
