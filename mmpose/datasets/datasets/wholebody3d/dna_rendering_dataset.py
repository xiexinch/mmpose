# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine import ProgressBar
from mmengine.fileio import get_local_path

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class DNARenderingDataset(BaseMocapDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/h3wb.py')

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 causal: bool = True,
                 subset_frac: float = 1.0,
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
        self.hand_roots = [9, 10, 91, 112]
        self.data_mode = 'topdown'
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
            max_refetch=max_refetch,
            subset_frac=subset_frac)

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            ann = np.load(local_path, allow_pickle=True)
        self.ann_data = ann['data'][()]

    def get_sequence_indices(self) -> List[List[int]]:
        assert self.seq_len == 1, 'Sequence length must be 1 for COCO dataset'
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        instance_list = []
        image_list = []

        instance_id = 0
        length = len(self.ann_data.keys())
        sub_length = int(length * self.subset_frac)
        step = length // sub_length

        print(
            f'Loading {self.subset_frac * 100}% DNA Rendering annotations.....'
        )
        progress_bar = ProgressBar(sub_length)
        for idx, (_, ann) in enumerate(self.ann_data.items()):
            if idx % step != 0:
                continue
            kpts = ann['keypoints']
            kpts_3d = ann['keypoints_3d']
            kpts_visible = ann['keypoints_visible']
            l_wrist_id, r_wrist_id, l_hand_root_id, r_hand_root_id =\
                self.hand_roots
            kpts = self._concnat_hand_roots(kpts, l_wrist_id, r_wrist_id,
                                            l_hand_root_id, r_hand_root_id)
            kpts_3d = self._concnat_hand_roots(kpts_3d, l_wrist_id, r_wrist_id,
                                               l_hand_root_id, r_hand_root_id)
            kpts_visible = self._concnat_hand_roots(kpts_visible[..., None],
                                                    l_wrist_id, r_wrist_id,
                                                    l_hand_root_id,
                                                    r_hand_root_id).squeeze(-1)

            instance = {
                'img_path': ann['img'],
                'keypoints': kpts,
                'keypoints_3d': kpts_3d,
                'keypoints_visible': kpts_visible,
                'lifting_target': kpts_3d,
                'lifting_target_visible': kpts_visible,
                'id': instance_id,
                'bbox_score': np.ones((1, ), dtype=np.float32),
                'bbox': ann['bbox'],
            }
            instance_list.append(instance)
            instance_id += 1
            progress_bar.update()
        del self.ann_data
        return instance_list, image_list

    def _concnat_hand_roots(self, kpts, left_wrist_id, right_wrist_id,
                            left_hand_root_id, right_hand_root_id):
        left_wrist = kpts[..., left_wrist_id, :].copy()
        right_wrist = kpts[..., right_wrist_id, :].copy()
        kpts = np.concatenate([
            kpts[:, :left_hand_root_id, :], left_wrist[None, :],
            kpts[:, left_hand_root_id:right_hand_root_id, :],
            right_wrist[None, :], kpts[:, right_hand_root_id:, :]
        ],
                              axis=1)
        return kpts
