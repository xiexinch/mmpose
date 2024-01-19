# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class VectorSimilarityLoss(nn.Module):

    def __init__(self,
                 anchor_indices: List,
                 kpts_indices: List,
                 loss_weight=1.,
                 loss_name: str = 'vs_loss'):
        super().__init__()
        self.anchor_indices = anchor_indices
        self.kpts_indices = kpts_indices
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                target_weights: torch.Tensor = None,
                **kwargs):
        anchor_indices = self.anchor_indices
        kpts_indices = self.kpts_indices

        pred_anchors, pred_kpts = pred[:, anchor_indices], pred[:,
                                                                kpts_indices]
        label_anchors, label_kpts = label[:,
                                          anchor_indices], label[:,
                                                                 kpts_indices]

        num_anchors, num_kpts = len(anchor_indices), len(kpts_indices)
        pred_anchors = pred_anchors.repeat_interleave(num_kpts, dim=1)
        pred_kpts = pred_kpts.repeat_interleave(num_anchors, dim=1)
        label_anchors = label_anchors.repeat_interleave(num_kpts, dim=1)
        label_kpts = label_kpts.repeat_interleave(num_anchors, dim=1)

        pred_vecs = pred_kpts - pred_anchors
        label_vecs = label_kpts - label_anchors

        # cosine similarity
        cosine_similarity = F.cosine_similarity(pred_vecs, label_vecs, dim=-1)
        loss = 1 - cosine_similarity
        if target_weights is not None:
            target_weights = target_weights[:, kpts_indices]
            target_weights = target_weights.repeat_interleave(
                num_anchors, dim=1)
            loss = loss * target_weights
        loss = loss.mean() * self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
