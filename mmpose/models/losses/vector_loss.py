# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class VectorSimilarityLoss(nn.Module):

    def __init__(self,
                 kpts_indices: List,
                 loss_weight=1.,
                 loss_name: str = 'vs_loss'):
        super().__init__()
        self.kpts_indices = kpts_indices
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                target_weights: torch.Tensor = None,
                **kwargs):
        kpts_indices = self.kpts_indices

        pred_kpts = pred[:, kpts_indices]
        label_kpts = label[:, kpts_indices]

        pred_center = pred_kpts.mean(dim=1, keepdim=True)
        label_center = label_kpts.mean(dim=1, keepdim=True)

        pred_vecs = pred_kpts - pred_center
        label_vecs = label_kpts - label_center

        # cosine similarity
        cosine_similarity = F.cosine_similarity(pred_vecs, label_vecs, dim=-1)
        loss = 1 - cosine_similarity
        if target_weights is not None:
            target_weights = target_weights[:, kpts_indices]
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


@MODELS.register_module()
class FaceNurbsLoss(nn.Module):

    def __init__(self,
                 face_indices: List,
                 num_joints: int = 68,
                 curve_p: int = 2,
                 curve_q: int = 3,
                 face_resolution: int = 20,
                 iou_threshold: float = 0.5,
                 pca_n_components: int = 3,
                 loss_weight: float = 1.0,
                 loss_name: str = 'face_nurbs_loss'):
        super().__init__()
        assert len(face_indices) == num_joints

        self.face_indices = face_indices
        self.num_joints = num_joints
        self.knot_vector_u = self.generate_uniform_knot_vector(
            num_joints, curve_p)
        self.knot_vector_v = self.generate_uniform_knot_vector(
            num_joints, curve_q)
        self.curve_p = curve_p
        self.curve_q = curve_q
        self.face_resolution = face_resolution
        self.iou_threshold = iou_threshold
        self.pca_n_components = pca_n_components
        self.loss_weight = loss_weight
        self.Nu, self.Nv = self.pre_cal_uv_values(face_resolution)

        self._loss_name = loss_name

    # 定义B样条基函数
    def b_spline_basis(self, i, p, u, knot_vector):
        if p == 0:
            return 1.0 if knot_vector[i] <= u < knot_vector[i + 1] else 0.0
        else:
            denom1 = knot_vector[i + p] - knot_vector[i]
            denom2 = knot_vector[i + p + 1] - knot_vector[i + 1]
            term1 = 0.0 if denom1 == 0 else (
                (u - knot_vector[i]) /
                denom1) * self.b_spline_basis(i, p - 1, u, knot_vector)
            term2 = 0.0 if denom2 == 0 else (
                (knot_vector[i + p + 1] - u) /
                denom2) * self.b_spline_basis(i + 1, p - 1, u, knot_vector)
            return term1 + term2

    # 生成均匀节点向量
    def generate_uniform_knot_vector(self, n, p):
        knot_vector_length = n + p + 1
        start_knots = [0] * (p + 1)
        internal_knots = [
            i / (knot_vector_length - 2 * p - 1)
            for i in range(1, knot_vector_length - 2 * p - 1)
        ]
        end_knots = [1] * (p + 1)
        return start_knots + internal_knots + end_knots

    def pre_cal_uv_values(self, resolution):
        u_values = torch.linspace(0, 1, resolution)
        v_values = torch.linspace(0, 1, resolution)

        # 预计算基函数值
        Nu = torch.zeros(self.num_joints, resolution)
        Nv = torch.zeros(self.num_joints, resolution)
        for i in range(self.num_joints):
            for j, u in enumerate(u_values):
                Nu[i, j] = self.b_spline_basis(i, self.curve_p, u,
                                               self.knot_vector_u)
            for j, v in enumerate(v_values):
                Nv[i, j] = self.b_spline_basis(i, self.curve_q, v,
                                               self.knot_vector_v)
        Nu = Nu.unsqueeze(0).unsqueeze(-1)  # Shape: [1, n, resolution, 1]
        Nv = Nv.unsqueeze(0).unsqueeze(-2)  # Shape: [1, n, 1, resolution]
        return Nu, Nv

    # 批量处理NURBS曲面
    def nurbs_surface_batch(self, control_points, weights):
        device = control_points.device
        batch_size, n, _ = control_points.shape
        surface_points = torch.zeros(
            batch_size,
            self.face_resolution,
            self.face_resolution,
            3,
            device=device)

        Nu, Nv = self.Nu.to(device), self.Nv.to(device)
        W = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, n, 1, 1]

        # 批量计算曲面点
        for i in range(n):
            for j in range(n):
                temp = control_points[:, i, :].unsqueeze(-2).unsqueeze(
                    -2) * Nu[:, i, :, :] * Nv[:, j, :, :] * W[:, i]
                surface_points += temp

        # 重新整理surface_points的形状
        return surface_points.reshape(batch_size, -1, 3)

    def calculate_pca_similarity(self, surface1, surface2, n_components=3):
        """使用PyTorch的pca_lowrank计算两个曲面的相似度."""
        num_samples, num_points, num_coordinates = surface1.shape

        # 重塑数据以适应PCA
        surface1_reshaped = surface1.reshape(num_samples,
                                             num_points * num_coordinates)
        surface2_reshaped = surface2.reshape(num_samples,
                                             num_points * num_coordinates)

        # 组合重塑后的数据
        combined_data = torch.cat((surface1_reshaped, surface2_reshaped),
                                  dim=0)

        # 应用PyTorch的pca_lowrank
        U, S, V = torch.pca_lowrank(combined_data, q=n_components)

        # 将数据投影到主成分上
        pca_surface1 = torch.matmul(surface1_reshaped, V[:, :n_components])
        pca_surface2 = torch.matmul(surface2_reshaped, V[:, :n_components])

        # 计算主成分之间的欧氏距离
        distance = torch.norm(pca_surface1.mean(0) - pca_surface2.mean(0))
        return distance

    def calculate_iou(self, surface1, surface2, threshold=0.01):
        """计算两个曲面之间的IoU."""
        distances = torch.norm(surface1 - surface2, dim=2)
        intersection = torch.sum(distances < threshold)
        union = surface1.shape[1] + surface2.shape[1] - intersection
        iou = intersection.float() / union.float()
        return iou

    def forward(self, pred, label, target_weight):

        pred = pred[:, self.face_indices]
        label = label[:, self.face_indices]

        pred_surface = self.nurbs_surface_batch(pred, self.knot_vector_u,
                                                self.knot_vector_v,
                                                self.curve_p, self.curve_q,
                                                self.face_resolution)
        label_surface = self.nurbs_surface_batch(label, self.knot_vector_u,
                                                 self.knot_vector_v,
                                                 self.curve_p, self.curve_q,
                                                 self.face_resolution)

        pca_loss = 1 - self.calculate_pca_similarity(
            pred_surface, label_surface, self.pca_n_components)
        iou_loss = 1 - self.calculate_iou(pred_surface, label_surface,
                                          self.iou_threshold)

        loss = (pca_loss + iou_loss) * self.loss_weight

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
