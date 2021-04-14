# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    maskrcnn-benchmark中所有模型共同模板类，支持 boxes， masks
    该类包括：
    - backbone  主干网络
    - rpn 区域推荐网络， 可选
    - heads 将feature和RPN中提取出来的proposals结合 以及 计算检测（box和分类）+分割（mask）

    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        '''
        build_backbone()  build_rpn()  build_roi_heads()重点需要了解的函数
        build_backbone主要是创建ResNet+FPN等特征提取网络
        
        '''

        # 创建骨干网络
        self.backbone = build_backbone(cfg)
        # 创建rpn
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # 创建roi_heads
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                训练阶段返回loss值
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
                测试阶段返回预测的结果（得分， 标签， mask）
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # 得到提取的图像特征
        features = self.backbone(images.tensors)
        # 通过rpn网络得到proposals和相应的loss值
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
