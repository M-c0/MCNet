from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):  # 1.0
        super(MultiModalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets, camids, miss_m = False):
        """
        计算基于标签和摄像头 ID 的多模态对比损失。

        Args:
            features: Tensor, 样本特征向量，形状为 [batch_size*3, feature_dim]，3 表示 3 个模态。
            targets: Tensor, 样本标签，形状为 [batch_size]。
            camids: Tensor, 摄像头 ID，形状为 [batch_size]。
            temperature: float, 温度参数，用于调整相似性分布。

        Returns:
            loss: 对比损失的标量。
        """
        # batch_size, num_modalities, feature_dim = features.shape
        features = F.normalize(features, dim=-1)  # 对每个模态的特征向量进行归一化????????一起归一化还是每个向量的归一化

        # 展平模态维度以便于相似性计算
        # features = features.view(-1, feature_dim)  # [batch_size * num_modalities, feature_dim]
        # 生成标签对：每个样本的模态有相同的标签和 camid
        # targets_repeat = targets.repeat_interleave(num_modalities)  # [batch_size * num_modalities]
        if miss_m:
            targets_repeat = torch.cat([targets, targets], dim=0)
            camids_repeat = torch.cat([camids, camids], dim=0)
        else:
            targets_repeat = torch.cat([targets, targets, targets], dim=0)
            camids_repeat = torch.cat([camids, camids, camids], dim=0)
        # camids_repeat = camids.repeat_interleave(num_modalities)  # [batch_size * num_modalities]

        # 计算余弦相似度
        sim_matrix = torch.matmul(features, features.T)  # [batch_size * num_modalities, batch_size * num_modalities]
        sim_matrix = sim_matrix / self.temperature  # 调整相似性分布

        # 构建正负样本掩码
        target_mask = targets_repeat.unsqueeze(1) == targets_repeat.unsqueeze(0)
        camid_mask = camids_repeat.unsqueeze(1) == camids_repeat.unsqueeze(0)
        positive_mask = target_mask & camid_mask  # 正样本条件：标签相同且 camid 相同
        negative_mask = ~positive_mask  # 负样本为其补集

        # 排除自身对比
        identity_mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
        positive_mask = positive_mask & ~identity_mask

        # 计算对比损失
        exp_sim = torch.exp(sim_matrix)  # 计算相似性指数
        positive_sum = torch.sum(exp_sim * positive_mask, dim=1)  # 正样本相似性和
        denominator = torch.sum(exp_sim * negative_mask, dim=1) + positive_sum  # 全部相似性和（除自身）

        loss = -torch.log(positive_sum / denominator)  # 对比损失
        loss = loss.mean()  # 平均损失
        return loss
