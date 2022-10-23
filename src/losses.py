import torch
import torch.nn as nn


def step_function(x, y, k=50):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    return torch.reciprocal(1 + torch.exp(k * (x - y)))


class OHEMBalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3, eps=1e-6, reduction='mean'):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt, mask):
        """
        :params: pred_prob, gt_prob, supervision_map
        """
        positive = (gt * mask)
        negative = ((1 - gt) * mask)

        no_positive = int(positive.sum())
        no_negative_expect = int(no_positive * self.negative_ratio)
        no_negative_current = int(negative.sum())
        no_negative = min(no_negative_expect, no_negative_current)

        loss = nn.functional.binary_cross_entropy(pred,
                                                  gt,
                                                  reduction=self.reduction)
        positive_loss = loss * positive
        negative_loss = loss * negative

        negative_loss, _ = torch.topk(negative_loss.view(-1), no_negative)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            no_positive + no_negative + self.eps)
        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        :params: appro binary map, gt_prob, supervision map
        """
        #         if pred.dim() == 4:
        #             pred = pred[:, 0, :, :]
        #             gt = gt[:, 0, :, :]
        #         assert pred.shape == gt.shape
        #         assert pred.shape == mask.shape

        #         if weights is not None:
        #             assert mask.shape == weights.shape
        #             mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class L1Loss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt, mask):
        if mask is not None:
            loss = (torch.abs(pred - gt) * mask).sum() / \
                (mask.sum() + self.eps)
        else:
            l1_loss_fn = torch.nn.L1Loss(reduction=self.reduction)
            loss = l1_loss_fn(pred, gt)
        return loss


class DBLoss(nn.Module):
    def __init__(self,
                 alpha=1.0,
                 beta=10.0,
                 reduction='mean',
                 negative_ratio=3,
                 eps=1e-6):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps

        self.ohem_loss = OHEMBalanceCrossEntropyLoss(self.negative_ratio,
                                                     self.eps, self.reduction)
        self.dice_loss = DiceLoss(self.eps)
        self.l1_loss = L1Loss(self.eps, self.reduction)

    def forward(self, preds, gts):
        """
        :params: preds (train mode): prob map, thresh map, binary map
        :params: gts (eval mode): prob map, thresh map
        """

        # predicts
        # prob_map, threshold_map, binary_map
        assert preds.dim() == 4
        assert gts.dim() == 4

        prob_pred = preds[:, 0, :, :]
        threshold_map = preds[:, 1, :, :]
        if preds.size(1) == 3:
            appro_binary_map = preds[:, 2, :, :]  # dim = 3

        # ground truths
        # prob_map, supervision_mask, threshold_map, text_area_map
        prob_gt_map = gts[0, :, :, :]  # 0/1
        supervision_mask = gts[1, :, :, :]  # 0/1
        threshold_gt_map = gts[2, :, :, :]  # 0.3 -> 0.7
        text_area_gt_map = gts[3, :, :, :]  # 0/1

        # losses
        prob_loss = self.ohem_loss(prob_pred, prob_gt_map, supervision_mask)
        threshold_loss = self.l1_loss(threshold_map, threshold_gt_map,
                                      text_area_gt_map)
        prob_threshold_loss = prob_loss + self.beta * threshold_loss
        if preds.size(1) == 3:
            binary_loss = self.dice_loss(appro_binary_map, prob_gt_map,
                                         supervision_mask)
            total_loss = self.alpha * binary_loss + prob_threshold_loss
            return prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss  # noqa
        else:
            return prob_threshold_loss


import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy import ndimage


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        assert pred.dim() == 4, pred.dim()
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class LeakyDiceLoss(nn.Module):
    '''
    Variation from DiceLoss.
    The coverage and union are computed separately.
    '''
    def __init__(self, eps=1e-6, coverage_scale=5.0):
        super(LeakyDiceLoss, self).__init__()
        self.eps = eps
        self.coverage_scale = coverage_scale

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape

        coverage = (pred * mask * gt).sum() / ((gt * mask).sum() + self.eps)
        assert coverage <= 1
        coverage = 1 - coverage
        excede = (pred * mask * gt).sum() / ((pred * mask).sum() + self.eps)
        assert excede <= 1
        excede = 1 - excede
        loss = coverage * self.coverage_scale + excede
        return loss, dict(coverage=coverage, excede=excede)


class InstanceDiceLoss(DiceLoss):
    '''
    DiceLoss normalized on each instance.
    Input:
        pred: (N, 1, H, W)
        gt: (N, 1, H, W)
        mask: (N, H, W)
    Note: This class assume that input tensors are on gpu,
        while cput computation is required to find union areas.
    '''
    REDUCTION = ['mean', 'sum', 'none']

    def __init__(self, threshold=0.3, iou_thresh=0.2, reduction=None,
                 max_regions=100, eps=1e-6):
        nn.Module.__init__(self)
        self.threshold = threshold
        self.iou_thresh = iou_thresh
        self.reduction = reduction
        if self.reduction is None:
            self.reduction = 'mean'
        assert self.reduction in self.REDUCTION
        self.max_regions = max_regions
        self.eps = eps

    def label(self, tensor_on_gpu, blur=None):
        '''
        Args:
            tensor_on_gpu: (N, 1, H, W)
            blur: Lambda. If exists, each instance will be blured using `blur`.
        '''
        tensor = tensor_on_gpu.cpu().detach().numpy()

        instance_maps = []
        instance_counts = []
        for batch_index in range(tensor_on_gpu.shape[0]):
            instance = tensor[batch_index]
            if blur is not None:
                instance = blur(instance)
            lable_map, instance_count = ndimage.label(instance[0])
            instance_count = min(self.max_regions, instance_count)
            instance_map = []
            for index in range(1, instance_count):
                instance = torch.from_numpy(
                        lable_map == index).to(tensor_on_gpu.device).type(torch.float32)
                instance_map.append(instance)
            instance_maps.append(instance_map)
        return instance_maps, instance_counts

    def iou(self, pred, gt):
        overlap = (pred * gt).sum()
        return max(overlap / pred.sum(), overlap / gt.sum())

    def replace_or_add(self, dest, value):
        if dest is None:
            return value
        if value is None:
            return dest
        return dest + value

    def forward(self, pred, gt, mask):
        # pred_label_maps: N, P, H, W, where P is the number of regions.
        torch.cuda.synchronize()
        pred_label_maps, _ = self.label(pred > self.threshold)
        gt_label_maps, _ = self.label(gt)

        losses = []
        for batch_index, gt_instance_maps in enumerate(gt_label_maps):
            pred_instance_maps = pred_label_maps[batch_index]
            if gt_instance_maps is None or pred_instance_maps is None:
                continue

            single_loss = None  # loss on a single image in a batch
            mask_not_matched = set(range(len(pred_instance_maps)))
            for gt_instance_map in gt_instance_maps:
                instance_loss = None  # loss on a specific gt region
                for instance_index, pred_instance_map in enumerate(pred_instance_maps):
                    if self.iou(pred_instance_map, gt_instance_map) > self.iou_thresh:
                        match_loss = self._compute(
                                pred[batch_index][0], gt[batch_index][0],
                                mask[batch_index] * (pred_instance_map + gt_instance_map > 0).type(torch.float32))
                        instance_loss = self.replace_or_add(instance_loss, match_loss)
                        if instance_index in mask_not_matched:
                            mask_not_matched.remove(instance_index)
                if instance_loss is None:
                    instance_loss = self._compute(
                            pred[batch_index][0], gt[batch_index][0],
                            mask[batch_index] * gt_instance_map)
                single_loss = self.replace_or_add(single_loss, instance_loss)

            '''Whether to compute single loss on instances which contrain no positive sample.
            if single_loss is None:
                single_loss = self._compute(
                        pred[batch_index][0], gt[batch_index][0],
                        mask[batch_index])
            '''

            for instance_index in mask_not_matched:
                single_loss = self.replace_or_add(
                        single_loss,
                        self._compute(
                            pred[batch_index][0], gt[batch_index][0],
                            mask[batch_index] * pred_instance_maps[instance_index]))

            if single_loss is not None:
                losses.append(single_loss)

        if self.reduction == 'none':
            loss = losses
        else:
            assert self.reduction in ['sum', 'mean']
            count = len(losses)
            loss = sum(losses)
            if self.reduction == 'mean':
                loss = loss / count
        return loss