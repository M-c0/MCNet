import math

import torch
import torch.nn as nn
from ..backbones.vit_pytorch import DropPath, Block
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_
from ..make_model import weights_init_classifier, weights_init_kaiming
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, get_attn=False):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if get_attn:
            return x, attn
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y, get_att=False):
        # q是x， 同时后面残差也是加 的q， 如果将common作为y， 模态作为q，符合CoEN
        if get_att:
            mid, attn = self.attn(self.norm1(x), self.norm11(y), get_att)
            x = x + self.drop_path(mid)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm11(y)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class Common(nn.Module):
    '''
        对所有patch的token concat后 24，3，128，768 进行处理，降维，得到一个 24，128， 768 的 向量作为公共特征
        怎么降维？  卷积？？

    '''

    def __init__(self, dim, num_classes):
        super().__init__()
        # 将 M*D 映射回 D
        self.proj = nn.Linear(3 * dim, 2 * dim)
        self.proj_1 = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.common_REDUCE = nn.Linear(2 * dim, dim)
        self.common_REDUCE.apply(weights_init_kaiming)

        self.block = Block(dim=768, num_heads=12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, M, N, D)
        Returns:
            public_feat: shape (B, N, D)
        """
        B, M, N, D = x.shape  # e.g. 24,3,129,768
        # 1) 把模态维度 M 和 token 维度 D 合并：先调整到 (B, N, M, D)
        x = x.permute(0, 2, 1, 3)  # (B, N, M, D)
        # 2) 展平成 (B, N, M*D)
        x = x.reshape(B, N, M * D)
        # 3) 线性降维回 D 维
        x = self.proj_1(self.proj(x))  # (B, N, D)
        x = self.act(x)
        x = self.norm(x)
        x = self.block(x)

        cls = x[:, 0, :]
        patch = x[:, 1:, :]

        row_sum = torch.sum(patch, dim=2)  # 64, 128
        num = (row_sum != 0).sum(dim=1).unsqueeze(-1)  # 64, 1
        patch = torch.sum(patch, dim=1) / num  # 64, 768
        common = self.common_REDUCE(torch.cat([cls, patch], dim=1))

        # x_cls_tk = x[:, 0]

        common_bn = self.bottleneck(common)
        socre = self.classifier(common_bn)
        return common_bn, socre, x


class Common_2(nn.Module):
    '''
        用一个block来提取公共特征，通过对比损失训练约束这个block
    '''

    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.gelu = nn.GELU()

        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.RGB_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NI_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.NI_REDUCE.apply(weights_init_kaiming)
        self.TI_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.TI_REDUCE.apply(weights_init_kaiming)
        self.common_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.common_REDUCE.apply(weights_init_kaiming)

        self.block = Block(dim=768, num_heads=12)

    def forward(self, x, y, z):
        # B, M, N, D = x.shape  # e.g. 24,3,129,768
        x = self.block(x)
        y = self.block(y)
        z = self.block(z)

        common_feat = x + y + z / 3

        x_cls = x[:, 0, :]
        x_patch = x[:, 1:, :]
        y_cls = y[:, 0, :]
        y_patch = y[:, 1:, :]
        z_cls = z[:, 0, :]
        z_patch = z[:, 1:, :]
        common_cls = common_feat[:, 0, :]
        common_patch = common_feat[:, 1:, :]

        row_sum = torch.sum(x_patch, dim=2)  # 64, 128
        num = (row_sum != 0).sum(dim=1).unsqueeze(-1)  # 64, 1
        x_patch = torch.sum(x_patch, dim=1) / num  # 64, 768
        y_patch = torch.sum(y_patch, dim=1) / num  # 64, 768
        z_patch = torch.sum(z_patch, dim=1) / num  # 64, 768
        common_patch = torch.sum(common_patch, dim=1) / num  # 64, 768
        xx = self.RGB_REDUCE(torch.cat([x_cls, x_patch], dim=-1))
        yy = self.RGB_REDUCE(torch.cat([y_cls, y_patch], dim=-1))
        zz = self.RGB_REDUCE(torch.cat([z_cls, z_patch], dim=-1))
        common = self.common_REDUCE(torch.cat([common_cls, common_patch], dim=-1))

        # x_cls_tk = x[:, 0]

        common_bn = self.bottleneck(common)
        socre = self.classifier(common_bn)
        return common, socre, torch.cat([xx, yy, zz], dim=0), common_feat


class Common_3(nn.Module):
    """
        使用PAT提取的交集mask，提取对应 patch 的 token 提取公共特征。
    """

    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.gelu = nn.GELU()

        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.RGB_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NI_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.NI_REDUCE.apply(weights_init_kaiming)
        self.TI_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.TI_REDUCE.apply(weights_init_kaiming)
        self.common_REDUCE = nn.Linear(2 * self.dim, self.dim)
        self.common_REDUCE.apply(weights_init_kaiming)

        self.block = Block(dim=768, num_heads=12)

    '''
        选取mask对应的 patch
        传入的是已经mask过的还是？？？？
    '''

    def forward(self, x, y, z, mask):
        x = self.block(x)
        y = self.block(y)
        z = self.block(z)

        common_feat = x + y + z / 3

        x_cls = x[:, 0, :]
        x_patch = x[:, 1:, :]
        y_cls = y[:, 0, :]
        y_patch = y[:, 1:, :]
        z_cls = z[:, 0, :]
        z_patch = z[:, 1:, :]
        common_cls = common_feat[:, 0, :]
        common_patch = common_feat[:, 1:, :]

        row_sum = torch.sum(x_patch, dim=2)  # 64, 128
        num = (row_sum != 0).sum(dim=1).unsqueeze(-1)  # 64, 1
        x_patch_avg = torch.sum(x_patch, dim=1) / num  # 64, 768
        y_patch_avg = torch.sum(y_patch, dim=1) / num  # 64, 768
        z_patch_avg = torch.sum(z_patch, dim=1) / num  # 64, 768
        common_patch = torch.sum(common_patch, dim=1) / num  # 64, 768
        xx = self.RGB_REDUCE(torch.cat([x_cls, x_patch_avg], dim=-1))
        yy = self.RGB_REDUCE(torch.cat([y_cls, y_patch_avg], dim=-1))
        zz = self.RGB_REDUCE(torch.cat([z_cls, z_patch_avg], dim=-1))
        common = self.common_REDUCE(torch.cat([common_cls, common_patch], dim=-1))

        # x_cls_tk = x[:, 0]

        common_bn = self.bottleneck(common)
        socre = self.classifier(common_bn)

        # 不同模态的特征取交集
        intersection = []
        for i in range(x.shape[0]):
            # 对第 i 个 batch，只取 mask[i]==True 的那些行
            intersection.append(torch.cat([x_patch[i, mask[i].squeeze(-1)].unsqueeze(0),
                                      y_patch[i, mask[i].squeeze(-1)].unsqueeze(0),
                                      z_patch[i, mask[i].squeeze(-1)].unsqueeze(0)], dim=0))  # 形状 (K, C)
        if self.training:
            return common, socre, torch.cat([xx, yy, zz], dim=0), intersection, common_feat
        return common, common_feat

class CrossAttModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_att_block = CrossAttentionBlock(dim, num_heads)
        # self.cross_att = CrossAttention(dim, num_heads)

    '''
        通过att map 来计算熵？  或者交叉熵也行？ 得到不同模态间的差异，差异大说明？ 差异小说明？？
        然后排序，计算cross attention
    '''

    def forward(self, rgb_feat, nir_feat, tir_feat, rgb_att, nir_att, tir_att, ):
        B, P = rgb_feat.size(0), rgb_feat.size(1)
        arange = torch.arange(B)

        # 计算每个模态的质量
        rgb_quality = self.attention_entropy(rgb_att)
        nir_quality = self.attention_entropy(nir_att)
        tir_quality = self.attention_entropy(tir_att)

        # 排序，得到质量最高的模态
        quality = torch.stack([rgb_quality, nir_quality, tir_quality], dim=0)
        _, idx = torch.sort(quality, dim=0, descending=True)

        feat = torch.stack([rgb_feat, nir_feat, tir_feat], dim=0)

        # 选取质量最高的模态计算交叉注意力
        feat_1 = feat[idx[0], arange]
        feat_2 = feat[idx[1], arange]
        feat_3 = feat[idx[2], arange]

        # feat_2 = feat_2 + self.cross_att(feat_2, feat_1)
        # feat_3 = feat_2 + self.cross_att(feat_3, feat_1)
        feat_2 = feat_2 + self.cross_att_block(feat_2, feat_1)
        feat_3 = feat_3 + self.cross_att_block(feat_3, feat_1)

        feat_ = torch.stack([feat_1, feat_2, feat_3], dim=0)
        inv_idx = idx.argsort(dim=0)
        rgb_feat = feat_[inv_idx[0], arange]
        nir_feat = feat_[inv_idx[1], arange]
        tir_feat = feat_[inv_idx[2], arange]

        return rgb_feat, nir_feat, tir_feat

    def attention_entropy(self, attn):
        # 只选取最后一个block或者最后合并的attention map
        # softmax
        length = len(attn)
        N = attn[0].shape[2] - 1
        B = attn[0].shape[0]
        H = attn[0].shape[1]
        k = 0
        last_map = attn[k]  # 64,12,129,129  每个阶段的att map
        for i in range(k + 1, length):
            last_map = torch.matmul(attn[i], last_map)
        last_map = last_map[:, :, 0, 1:]  # BS,12,128  每个头，class_token 和 patch 之间的相似度/att
        # 不需要归一化，本来att 就归一化了
        p = last_map  # .reshape(B, H*N)

        # attn: (B, H, L, L), assume already softmaxed over last dim
        # B, H, L = p.shape

        eps = 1e-8
        p = p.clamp(min=eps)
        entropy_per_head = - (p * p.log()).sum(dim=-1)  # (B, H)
        entropy_avg = entropy_per_head.mean(dim=1)  # (B,)
        # 归一化
        max_ent = math.log(p.shape[-1])  # p.shape[-1] 就是 N ，max_ent = log(N)   H_max = 1/n*log(1/n) = log(N)
        norm_ent = entropy_avg / max_ent  # (B,)
        quality = 1 - norm_ent  # (B,)  越大越好  熵越大，质量越差  加个1-H，就是越大越好

        return quality
