# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    对Faster-RCNN部分的loss进行计算
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        # gt 和 预测框之间的 IOU
        match_quality_matrix = boxlist_iou(target, proposal)

        # 预测边框和对应的gt的索引， 背景边框为-2 ， 模糊边框为-1 .eg:matched_idxs[4] = 6 :表示第5个预测边框所分配的GT的id为6
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets、
        # 获得 GT 的类别标签（这是一个BoxList对象）
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds

        # 将所有的背景边框和模糊边框的标签都对应成第一个gt的标签
        matched_targets = target[matched_idxs.clamp(min=0)]
        # 将对应的列表索引添加至gt列表中
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    # 计算出所有预测边框所对应的GT边框
    def prepare_targets(self, proposals, targets):
        # 类别标签列表
        labels = []
        # 回归box标签列表
        regression_targets = []
        # 分别对每一张图片进行操作
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # matched_targets为proposal所对应的label值
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            # matched_idxs里面包含有-1 和 -2的值
            matched_idxs = matched_targets.get_field("matched_idxs")

            # 获取每一个target所对应的label标签
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            # 将低于阈值的标签设置为背景类（因为在match_targets_to_proposals函数里面将低于阈值的和处于阈值之间的label设置成第一个GT的类别，现在要将它改成背景类）
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler   sampler 会将它忽视掉

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.
        这个函数用来生成正负样本，返回值为相应的proposals

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])

            BoxList类要看一下（已看完）
        返回值：
        proposal：# proposal中的数目是采样之后正负样本的总数，不是所有proposals的总数
        """

        # 得到GT的label和GT的regression（这个regression值不是单纯的坐标值，而是一些坐标转换的参数）
        # labels， regression都是列表类型
        labels, regression_targets = self.prepare_targets(proposals, targets)
        # 按照一定的方式选取背景框和目标边框，并返回其标签 得到列表类型
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        # 将已经被采样器选择用来训练的proposals提取出来
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        # proposal中的数目是采样之后正负样本的总数，不是所有proposals的总数
        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.
        要求提前调用好采样的函数（保证proposals和target的数目保持一致）
        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        # 保持列数不变，不断添加行数(行数为batch size图片数目)
        # class_logits 为预测的类别 shape:(batch size, num_subsample)
        class_logits = cat(class_logits, dim=0)
        # box_regression为预测的回归值  shape:(batch size, num_subsample * 4)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        # labels shape is (batch size, num_sub_sample)
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        # 计算分类的交叉熵损失
        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        # 获取正样本的索引和正样本的类别值
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            # 如果整个模型采用agnostic模型，即只分别含目标与不含目标两类
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:

            # 找到 box regression 预测值的目标框的索引
            # box regression预测值的维度(,4*class_num)
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        # 计算box的回归损失：smooth L1
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    # 判断前景和背景的matcher
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
