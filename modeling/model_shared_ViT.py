import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224, Block, CrossAttention
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from modeling.fusion_part.Common import Common, CrossAttentionBlock, CrossAttModule, Common_2, Common_3
from modeling.fusion_part.PartAtt import BlockMask, PAT
from modeling.vit_gan import vit_base_patch16_224_mm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        # No view
        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        x, att_list = self.base(x, cam_label=cam_label, view_label=view_label)  # list: tensor [24,129,768]
        global_feat = x[:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return x, cls_score, global_feat, att_list  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return x, feat, att_list
            else:
                return x, global_feat, att_list

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class mm_ReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(mm_ReID, self).__init__()
        self.backbone = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.num_head = 12
        self.mix_dim = 768

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.layer = cfg.MODEL.LAYER
        self.direct = cfg.MODEL.DIRECT

        self.classifier_ViT = nn.Linear(4 * self.mix_dim, self.num_classes, bias=False)
        self.classifier_ViT.apply(weights_init_classifier)
        self.bottleneck_ViT = nn.BatchNorm1d(4 * self.mix_dim)
        self.bottleneck_ViT.bias.requires_grad_(False)
        self.bottleneck_ViT.apply(weights_init_kaiming)

        self.miss = cfg.TEST.MISS

        self.fusion_layer = nn.Sequential(nn.Linear(768 * 4, 768), nn.GELU(), nn.Linear(768, 768 * 4))  # 融合层
        # self.fusion_layer.apply(weights_init_kaiming)  # 加了这个不收敛
        # self.common_pro = Common(768, self.num_classes)
        # self.common = Common_2(768, self.num_classes)
        self.common = Common_3(768, self.num_classes)

        self.rgb_block = Block(dim=768, num_heads=12)
        self.nir_block = Block(dim=768, num_heads=12)
        self.tir_block = Block(dim=768, num_heads=12)

        # self.cross_att_m = CrossAttModule(768, 12)

        self.cross_att_block_1 = CrossAttentionBlock(768, 12)
        self.cross_att_block_2 = CrossAttentionBlock(768, 12)
        self.cross_att_block_3 = CrossAttentionBlock(768, 12)

        # self.cross_att_1 = CrossAttention(768)
        # self.cross_att_2 = CrossAttention(768)
        # self.cross_att_3 = CrossAttention(768)

        # self.ratio = (1 / self.num_patches) * int(cfg.MODEL.HEAD_KEEP)
        self.PAT = PAT(ratio=0.1)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.RGB_REDUCE = nn.Linear(2 * self.mix_dim, self.mix_dim)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NI_REDUCE = nn.Linear(2 * self.mix_dim, self.mix_dim)
        self.NI_REDUCE.apply(weights_init_kaiming)
        self.TI_REDUCE = nn.Linear(2 * self.mix_dim, self.mix_dim)
        self.TI_REDUCE.apply(weights_init_kaiming)
        # self.common_REDUCE = nn.Linear(2 * self.mix_dim, self.mix_dim)
        # self.common_REDUCE.apply(weights_init_kaiming)
        # # PAT
        # self.classifier_com = nn.Linear(4 * self.mix_dim, self.num_classes, bias=False)
        # self.classifier_com.apply(weights_init_classifier)
        # self.bottleneck_com = nn.BatchNorm1d(4 * self.mix_dim)
        # self.bottleneck_com.bias.requires_grad_(False)
        # self.bottleneck_com.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(0.4)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dropout !!!!!!!!!!!!!!!!!!!!

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=None, label=None, view_label=None, v_cam=False):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_feat, NI_score, NI_global, NI_att_list = self.backbone(NI, cam_label=cam_label, view_label=view_label)
            TI_feat, TI_score, TI_global, TI_att_list = self.backbone(TI, cam_label=cam_label, view_label=view_label)
            RGB_feat, RGB_score, RGB_global, RGB_att_list = self.backbone(RGB, cam_label=cam_label,
                                                                          view_label=view_label)


            RGB_feats, NIR_feats, TIR_feats, mask, mask_inter, loss_bcc = self.PAT(RGB_feat=RGB_feat, RGB_attn=RGB_att_list,  # intersection=True 返回 交集 mask
                                                                  NIR_feat=NI_feat, NIR_attn=NI_att_list,
                                                                  TIR_feat=TI_feat, TIR_attn=TI_att_list, intersection=True)

            # RGB_feat, NI_feat, TI_feat, mask, loss_bcc = self.PAT(RGB_feat=RGB_feat, RGB_attn=RGB_att_list,
            #                                                       NIR_feat=NI_feat, NIR_attn=NI_att_list,
            #                                                       TIR_feat=TI_feat, TIR_attn=TI_att_list)

            # print(mask[0])
            # print(mask_inter[0])
            RGB_feat = self.rgb_block(RGB_feats)
            NI_feat = self.nir_block(NIR_feats)
            TI_feat = self.tir_block(TIR_feats)

            # common, common_socre, common_contras_glb, common_feat = self.common(RGB_feat, NI_feat, TI_feat)
            common, common_socre, common_contras_glb, common_loc, common_feat = self.common(RGB_feat, NI_feat, TI_feat, mask_inter)

            # 把交集中的每个 patch 来提取公共特征
            # 另起一个block，损失函数按 patch 的token来算，最后两个合并
            # 用原来的block，只选取 mask_inter对应的特征计算局部对比损失，外加计算一个总的损失


            # 增强每个模态的信息
            RGB_feat = self.cross_att_block_1(RGB_feat, common_feat)
            NI_feat = self.cross_att_block_2(NI_feat, common_feat)
            TI_feat = self.cross_att_block_3(TI_feat, common_feat)

            RGB_cls = RGB_feat[:, 0, :]
            NI_cls = NI_feat[:, 0, :]
            TI_cls = TI_feat[:, 0, :]

            RGB_patch = RGB_feat[:, 1:, :]  # 64, 128, 768
            NI_patch = NI_feat[:, 1:, :]
            TI_patch = TI_feat[:, 1:, :]

            # 如果使用置信度来搞的话，reduce网络是乱的，没办法每个模态都去融合
            row_sum = torch.sum(RGB_patch, dim=2)  # 64, 128
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)  # 64, 1
            RGB_patch = torch.sum(RGB_patch, dim=1) / num  # 64, 768
            NI_patch = torch.sum(NI_patch, dim=1) / num
            TI_patch = torch.sum(TI_patch, dim=1) / num
            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NI_REDUCE(torch.cat([NI_cls, NI_patch], dim=-1))
            tir = self.TI_REDUCE(torch.cat([TI_cls, TI_patch], dim=-1))

            ori = torch.cat([rgb, nir, tir, common], dim=-1)  # 24, 2304
            # ori = torch.cat([rgb, nir, tir], dim=-1)
            # ori = torch.cat([RGB_cls, NI_cls, TI_cls], dim=-1)  # 24, 2304
            # ori = self.fusion_layer(ori)

            ori_global = self.bottleneck_ViT(ori)
            ori_score = self.classifier_ViT(ori_global)

            if self.direct:
                return ori_score, ori
            else:
                return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, ori_score, ori, common, common_socre, common_contras_glb, common_loc, loss_bcc,  #   # common_cls, common_score  #

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_feat, NI_global, NI_att_list = self.backbone(NI, cam_label=cam_label, view_label=view_label)
            TI_feat, TI_global, TI_att_list = self.backbone(TI, cam_label=cam_label, view_label=view_label)
            RGB_feat, RGB_global, RGB_att_list = self.backbone(RGB, cam_label=cam_label, view_label=view_label)

            # RGB_feat, NI_feat, TI_feat = self.cross_att_m(RGB_feat, NI_feat, TI_feat, RGB_att_list, NI_att_list, TI_att_list)


            # common_input = torch.cat((NI_feat.unsqueeze(1), TI_feat.unsqueeze(1), RGB_feat.unsqueeze(1)), dim=1)
            # common_cls, common_score, common_feat = self.common_pro(common_input)

            RGB_feat, NI_feat, TI_feat, mask, mask_inter = self.PAT(RGB_feat=RGB_feat, RGB_attn=RGB_att_list,
                                                                  NIR_feat=NI_feat, NIR_attn=NI_att_list,
                                                                  TIR_feat=TI_feat, TIR_attn=TI_att_list, intersection=True)  # q = common_cash

            RGB_feat = self.rgb_block(RGB_feat)
            NI_feat = self.nir_block(NI_feat)
            TI_feat = self.tir_block(TI_feat)

            common, common_feat = self.common(RGB_feat, NI_feat, TI_feat, mask_inter)
            #
            # 增强每个模态的信息
            RGB_feat = self.cross_att_block_1(RGB_feat, common_feat)
            NI_feat = self.cross_att_block_2(NI_feat, common_feat)
            TI_feat = self.cross_att_block_3(TI_feat, common_feat)

            RGB_cls = RGB_feat[:, 0, :]
            NI_cls = NI_feat[:, 0, :]
            TI_cls = TI_feat[:, 0, :]

            RGB_patch = RGB_feat[:, 1:, :]  # 64, 128, 768
            NI_patch = NI_feat[:, 1:, :]
            TI_patch = TI_feat[:, 1:, :]

            row_sum = torch.sum(RGB_patch, dim=2)  # 64, 128
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)  # 64, 1
            RGB_patch = torch.sum(RGB_patch, dim=1) / num  # 64, 768
            NI_patch = torch.sum(NI_patch, dim=1) / num
            TI_patch = torch.sum(TI_patch, dim=1) / num

            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NI_REDUCE(torch.cat([NI_cls, NI_patch], dim=-1))
            tir = self.TI_REDUCE(torch.cat([TI_cls, TI_patch], dim=-1))
            # common_patch = common_feat[:, 1:, :]
            # common_patch = torch.sum(common_patch, dim=1) / num
            # common = self.common_REDUCE(torch.cat([common_cls, common_patch], dim=1))

            # ori = torch.cat([rgb, nir, tir], dim=-1)
            ori = torch.cat([rgb, nir, tir, common], dim=-1)
            # ori = torch.cat([RGB_global, NI_global, TI_global, common_feature], dim=-1)  # 24, 2304
            # ori = self.fusion_layer(ori)
            ori_global = self.bottleneck_ViT(ori)
            ori_score = self.classifier_ViT(ori_global)
            if self.neck_feat == 'after':
                pass
            else:
                ori_global = ori
            if v_cam:
                return ori_score
            return torch.cat([ori_global], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
    'vit_base_patch16_224_mm': vit_base_patch16_224_mm,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = mm_ReID(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building TOPReID===========')
    return model
