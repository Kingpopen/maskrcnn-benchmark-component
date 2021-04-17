# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# 正负样本的选择器（因为要权衡好正负样本的比例）
class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        batch_size_per_image（在配置文件中设置该参数） 是指每张图片挑选用于训练的proposal数目（如果实际数目小于这个值，那以实际数目为准）
        postive_fraction 是指batch_size_per_image中正样本个数的比例
        """

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        matched_idxs中包含每一个proposal的label值.(0为背景， -1为被忽视的类， positive值为相应的类别号)

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.0
        """
        pos_idx = []
        neg_idx = []
        # 批量处理  考虑到batch size维度的缘故
        for matched_idxs_per_image in matched_idxs:
            # 得到正样本的proposal index
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # 得到负样本的proposal index
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            # 正样本的数目
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # 从所有的正样本中随机挑选一定数目的正样本 得到的是 postive列表 的index
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            # 从所有的负样本中随机挑选一定数目的负样本
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # 得到用于训练正样本的 proposal index
            pos_idx_per_image = positive[perm1]
            # 得到用于训练负样本的 proposal index
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            # 将是用来训练的正样本proposal 设置为1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            # 将是用来训练的负样本proposal 设置为1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        # pos_idx列表中的每一个列表维度可能不太一样（内部每一个列表的维度取决于预测过程中的proposal个数，列表的数目是batch size数目）
        return pos_idx, neg_idx
