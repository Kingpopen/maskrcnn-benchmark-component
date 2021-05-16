# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


# inference过程用到的类
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results

    从一系列的类别分类得分，边框回归以及proposals中，计算post-processed boxes,
    以及应用NMS得到最后的结果。
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor, tensor]): x contains the class logits
                component logits and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # 这个地方先不用管它
        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

# 添加了零件分支的ROI 后处理类
class PostProcessor_component(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results

    从一系列的类别分类得分，边框回归以及proposals中，计算post-processed boxes,
    以及应用NMS得到最后的结果。
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor_component, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor, tensor]): x contains the class logits
                component logits and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        # ================== 2021 04 24 修改 =========================
        class_logits, component_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        # 照葫芦画瓢 计算零件的概率值
        component_prob = F.softmax(component_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # 这个地方先不用管它
        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]
        # ==================2021 04 24 修改 =========================
        # 照葫芦画瓢 获得零件的类别数
        num_components = component_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        # ==================2021 04 24 修改 =========================
        component_prob = component_prob.split(boxes_per_image, dim=0)

        results = []
        # ==================2021 04 24 修改 =========================
        # 照葫芦画瓢 给循环添加零件类别 per 表示一张图片  for训练是在遍历图片数
        for class_per_prob, component_per_prob, boxes_per_img, image_shape in zip(
            class_prob, component_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, class_per_prob, component_per_prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                # 对proposal进行筛选
                boxlist = self.filter_results(boxlist, num_classes, num_components)
            results.append(boxlist)
        return results

    # ==================2021 04 24 修改 =========================
    # 照葫芦画瓢 添加零件得分
    def prepare_boxlist(self, boxes, scores, component_scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field("component_scores", component_scores)
        return boxlist

    # ==================2021 04 24 修改 =========================
    # 这个地方要着重看一下
    def filter_results(self, boxlist, num_classes, num_components):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).

        筛选检测得到的目标框得分大于阈值 同时 进行NMS操作
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        # boxes中的 bbox 为回归后的坐标框
        #====================== 2021 04 27 修改=============================
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        # 获取proposals的零件得分
        component_scores = boxlist.get_field("component_scores").reshape(-1, num_components)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        # 得到得分大于阈值的bool
        inds_all = scores > self.score_thresh

        # 获取每一行中的component得分最大值及其下标 除掉了背景
        # component_scores shape is [num_per_image_proposals, ]
        component_scores, component_indexs = torch.max(component_scores[:, 1:], dim=-1)
        # 加1的原因是之前去掉了背景类别 现在要把index对应上
        component_indexs = component_indexs + 1

        for j in range(1, num_classes):
            # 第j类别中的大于阈值的item
            inds = inds_all[:, j].nonzero().squeeze(1)
            # 得到所有大于阈值的第j类别的得分
            scores_j = scores[inds, j]

            # 给第j类的object赋予零件类别得分 以及零件类别
            component_scores_j = component_scores[inds]
            component_index_j = component_indexs[inds]

            # 得到所有大于阈值得分的第j类别对应的box
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            # 添加零件得分 和 添加零件类别
            boxlist_for_class.add_field("component_scores", component_scores_j)
            boxlist_for_class.add_field("components", component_index_j)

            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        # 如果NMS之后检测的instance数目大于限制输出的instance数目
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result



def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    # NMS的阈值
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    # 每张图片的检测的最大instance数目
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    if cfg.MODEL.ROI_HEADS.COMPONENT_BRANCH:
        postprocessor = PostProcessor_component(
            score_thresh,
            nms_thresh,
            detections_per_img,
            box_coder,
            cls_agnostic_bbox_reg,
            bbox_aug_enabled
        )
    else:
        postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor



if __name__ == '__main__':
    x = [[[1, 4, 6, 8], [4, 1, 5, 9]],
         [[1, 2, 3, 4], [3, 1, 7, 9]]]

    x = torch.tensor(x, dtype=torch.float32)
    prob = F.softmax(x, -1)
    print("prob:", prob)
    print("the shape of prob:", prob.shape)

    class_prob = [[0.1, 0.5, 0.7, 0.1, 0.1],
                  [0.1, 0.6, 0.1, 0.1, 0.1],
                  [0.4, 0.1, 0.1, 0.3, 0.1]]
    class_prob = torch.tensor(class_prob, dtype=torch.float32)

    index = class_prob > 0.3
    print("index is:", index)

    inds = index[:, 0].nonzero().squeeze(1)
    print("inds:", inds)

    print("class_prob_index[:, 1:]:", class_prob[:, 1:])
    prob, index = torch.max(class_prob[:, 1:], dim=-1)
    print("prob is:", prob)
    print("index is:", index)

    x = [2, 1, 0, 1, 0, 1, 4]
    x = torch.tensor(x)
    x = x.nonzero().squeeze(1)
    print("x:", x)



