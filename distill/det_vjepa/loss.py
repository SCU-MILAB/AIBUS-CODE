"""
FCOS losses for the ViT-G detector.

Assignment follows FCOS (Tian et al., 2019):
  * a location is positive for a GT box if it is inside the box, its distance to
    the box centre is within `center_radius * stride` (center sampling), and its
    max ltrb distance falls inside the level's regression range;
  * if several boxes claim the same location, the smallest one wins — implemented
    by assigning boxes in descending-area order so the smallest is written last.

Total loss = focal(cls) + GIoU(reg) / num_pos + BCE(centerness) / num_pos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e8


def regress_ranges_for(strides):
    """FCOS per-level ltrb ranges: (-1, 8*s0], (8*s0, 8*s1], ..., (8*sn, inf)."""
    ranges = []
    for i, s in enumerate(strides):
        low = -1.0 if i == 0 else 8.0 * strides[i - 1]
        high = INF if i == len(strides) - 1 else 8.0 * s
        ranges.append([low, high])
    return torch.tensor(ranges, dtype=torch.float32)


def ltrb_to_xyxy(points, dist):
    return torch.stack([
        points[:, 0] - dist[:, 0], points[:, 1] - dist[:, 1],
        points[:, 0] + dist[:, 2], points[:, 1] + dist[:, 3],
    ], dim=-1)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits:  [B, N, C] raw scores
    targets: [B, N]    with 0 = background and 1..C = class index + 1
    Returns the unreduced element-wise loss.
    """
    num_classes = logits.shape[-1]
    onehot = F.one_hot(targets.long(), num_classes + 1)[..., 1:].to(logits.dtype)

    ce = F.binary_cross_entropy_with_logits(logits, onehot, reduction='none')
    prob = logits.sigmoid()
    p_t = prob * onehot + (1 - prob) * (1 - onehot)
    alpha_t = alpha * onehot + (1 - alpha) * (1 - onehot)
    return alpha_t * ce * (1 - p_t).pow(gamma)


def paired_giou_loss(pred_xyxy, target_xyxy, weights=None, alpha=1.0):
    """GIoU or Alpha-GIoU, optionally averaged with positive-target weights."""
    lt = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_p = ((pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
              * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0))
    area_t = ((target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0)
              * (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0))
    union = (area_p + area_t - inter).clamp(min=1e-6)
    iou = inter / union

    lt_c = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb_c = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = (wh_c[:, 0] * wh_c[:, 1]).clamp(min=1e-6)

    penalty = (area_c - union) / area_c
    # Preserve the original arithmetic at alpha=1. Power overlap and penalty
    # separately (Alpha-IoU, Eq. 4); signed GIoU must not be exponentiated.
    loss = (1.0 - (iou - penalty) if alpha == 1.0 else
            1.0 - iou.clamp(0, 1).pow(alpha) + penalty.clamp(0, 1).pow(alpha))
    if weights is None:
        return loss.mean()
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


def instance_balanced_weights(instance_ids, base_weights=None):
    """Give each matched GT one unit of regression weight, across a whole batch."""
    _, inverse, counts = torch.unique(instance_ids, return_inverse=True, return_counts=True)
    if base_weights is None:
        return counts[inverse].to(torch.float32).reciprocal()
    totals = torch.zeros(len(counts), device=base_weights.device, dtype=base_weights.dtype)
    totals.scatter_add_(0, inverse, base_weights)
    return base_weights / totals[inverse].clamp(min=1e-6)


class FCOSLoss(nn.Module):
    def __init__(self, fpn_strides=(8, 16, 32), center_radius=1.5,
                 focal_alpha=0.25, focal_gamma=2.0,
                 reg_weight=1.0, ctr_weight=1.0, reg_centerness_weighted=False,
                 reg_l1_weight=0.0, reg_iou_alpha=1.0, ctr_target='centerness',
                 reg_normalization='positive'):
        super().__init__()
        if ctr_target not in ('centerness', 'iou'):
            raise ValueError('ctr_target must be centerness or iou')
        self.fpn_strides = tuple(fpn_strides)
        self.center_radius = center_radius
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reg_weight = reg_weight
        self.reg_l1_weight = reg_l1_weight
        self.reg_iou_alpha = reg_iou_alpha
        self.ctr_weight = ctr_weight
        self.ctr_target = ctr_target
        if reg_normalization not in ('positive', 'gt'):
            raise ValueError(f'unknown regression normalization: {reg_normalization}')
        self.reg_normalization = reg_normalization
        self.reg_centerness_weighted = reg_centerness_weighted
        self.register_buffer('regress_ranges', regress_ranges_for(self.fpn_strides),
                             persistent=False)

    def _match(self, points, ranges, radius, boxes, labels, return_gt_ids=False):
        """
        Assign targets for one image.

        points:  [N, 2] anchor centres in pixels
        ranges:  [N, 2] ltrb regression range of the level each location belongs to
        radius:  [N]    center-sampling radius in pixels
        Returns (labels [N] with 0=bg, target ltrb [N,4], target centerness [N]).
        """
        n = points.shape[0]
        tgt_labels = torch.zeros(n, dtype=torch.long, device=points.device)
        tgt_dist = torch.zeros(n, 4, device=points.device)
        tgt_ctr = torch.zeros(n, device=points.device)
        gt_ids = torch.full((n,), -1, dtype=torch.long, device=points.device) if return_gt_ids else None
        if boxes.numel() == 0:
            result = (tgt_labels, tgt_dist, tgt_ctr)
            return result + (gt_ids,) if return_gt_ids else result

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # Descending area → the smallest overlapping box is assigned last and wins.
        for i in torch.argsort(areas, descending=True).tolist():
            x1, y1, x2, y2 = boxes[i].tolist()
            left = points[:, 0] - x1
            top = points[:, 1] - y1
            right = x2 - points[:, 0]
            bottom = y2 - points[:, 1]
            dist = torch.stack([left, top, right, bottom], dim=1)

            max_dist = dist.max(dim=1).values
            positive = (
                (dist.min(dim=1).values > 0)                       # inside the box
                & (max_dist >= ranges[:, 0]) & (max_dist <= ranges[:, 1])
                & ((points[:, 0] - (x1 + x2) / 2).abs() < radius)   # center sampling
                & ((points[:, 1] - (y1 + y2) / 2).abs() < radius)
            )
            if not positive.any():
                continue

            tgt_labels[positive] = int(labels[i]) + 1
            if return_gt_ids:
                gt_ids[positive] = i
            tgt_dist[positive] = dist[positive]
            # Centerness is only meaningful inside the box, where all four
            # distances are positive, so the sqrt below never sees a negative.
            centerness = torch.sqrt(
                (torch.minimum(left, right) / torch.maximum(left, right).clamp(min=1e-6))
                * (torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp(min=1e-6))
            )
            tgt_ctr[positive] = centerness[positive]
        result = (tgt_labels, tgt_dist, tgt_ctr)
        return result + (gt_ids,) if return_gt_ids else result

    def forward(self, head_out, targets, anchor_points, anchor_strides, anchor_levels):
        """
        head_out:       raw (cls_logits, bbox_pred, ctr_logits) lists from the head
        targets:        list of per-image dicts with 'boxes' (xyxy px) and 'labels'
        anchor_points:  [N, 2]
        anchor_strides: [N]
        anchor_levels:  [N] long, index of the pyramid level of each location
        """
        cls_logits, bbox_pred, ctr_logits = head_out
        # fp32: under bf16 autocast the head outputs are bf16, and focal/BCE over
        # ~8k locations per image loses too much precision in bf16.
        cls = torch.cat([c.flatten(2).permute(0, 2, 1) for c in cls_logits], dim=1).float()
        dist = torch.cat([b.flatten(2).permute(0, 2, 1) for b in bbox_pred], dim=1).float()
        ctr = torch.cat([c.flatten(1) for c in ctr_logits], dim=1).float()

        ranges = self.regress_ranges.to(anchor_levels.device)[anchor_levels]
        radius = self.center_radius * anchor_strides.to(cls.device)

        matched = list(zip(*[
            self._match(anchor_points.to(cls.device), ranges, radius,
                        t['boxes'].to(cls.device), t['labels'].to(cls.device),
                        return_gt_ids=self.reg_normalization == 'gt')
            for t in targets
        ]))
        labels, dist_t, ctr_t = matched[:3]
        labels = torch.stack(labels)
        dist_t = torch.stack(dist_t)
        ctr_t = torch.stack(ctr_t)

        num_pos = (labels > 0).sum().clamp(min=1).to(cls.dtype)
        positive = labels > 0

        cls_loss = sigmoid_focal_loss(cls, labels, self.focal_alpha, self.focal_gamma)
        cls_loss = cls_loss.sum() / num_pos

        if positive.any():
            # `positive` is [B, N] while the anchor tensors are [N, ...], so
            # expand the anchors across the batch before masking.
            batch = cls.shape[0]
            points = anchor_points.to(cls.device).unsqueeze(0).expand(
                batch, -1, -1)[positive]
            strides = anchor_strides.to(cls.device).unsqueeze(0).expand(
                batch, -1)[positive].unsqueeze(-1)
            # The head predicts distances in units of the level stride and
            # decode() turns them into pixels by multiplying by the stride.  The
            # loss must apply the same scaling, otherwise it trains boxes that
            # are a factor of `stride` too large at inference.
            pred_boxes = ltrb_to_xyxy(points, dist[positive] * strides)
            target_boxes = ltrb_to_xyxy(points, dist_t[positive])
            weights = ctr_t[positive] if self.reg_centerness_weighted else None
            if self.reg_normalization == 'gt':
                # Local GT ids must not collide across different batch images.
                gt_ids = torch.stack(matched[3])
                width = max(len(t['boxes']) for t in targets)
                offsets = torch.arange(batch, device=cls.device).unsqueeze(1) * width
                weights = instance_balanced_weights((gt_ids + offsets)[positive], weights)
            reg_loss = paired_giou_loss(pred_boxes, target_boxes, weights=weights,
                                        alpha=self.reg_iou_alpha)
            if self.reg_l1_weight:
                # Relative edge errors keep small and large lesions comparable.
                # This training-only term adds no model state or decoding changes.
                size = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=1.0)
                normalizer = size.repeat(1, 2)
                edge_loss = F.smooth_l1_loss(
                    pred_boxes / normalizer, target_boxes / normalizer,
                    beta=0.1, reduction='none').mean(dim=1)
                edge_loss = (edge_loss.mean() if weights is None else
                             (edge_loss * weights).sum() / weights.sum().clamp(min=1e-6))
                reg_loss = reg_loss + self.reg_l1_weight * edge_loss
            quality_targets = ctr_t[positive]
            if self.ctr_target == 'iou':
                # Train the existing quality branch without changing decoding.
                # Detached labels prevent BCE from moving boxes to fit scores.
                with torch.no_grad():
                    intersection_wh = (torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
                                       - torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])).clamp(min=0)
                    intersection = intersection_wh.prod(dim=1)
                    pred_area = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=0).prod(dim=1)
                    target_area = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=0).prod(dim=1)
                    quality_targets = (intersection / (pred_area + target_area - intersection).clamp(min=1e-6)).clamp(0, 1)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr[positive], quality_targets, reduction='sum') / num_pos
        else:
            # Keep the graph connected so DDP does not flag the reg/ctr branches
            # as unused on images that contain no objects.
            reg_loss = dist.sum() * 0.0
            ctr_loss = ctr.sum() * 0.0

        total = cls_loss + self.reg_weight * reg_loss + self.ctr_weight * ctr_loss
        return {
            'loss': total,
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'ctr': ctr_loss.detach(),
            'num_pos': num_pos.detach(),
        }
