"""Official RF-DETR query detector with the project's unchanged V-JEPA ViT-G."""
from contextlib import nullcontext

import torch
from torch import nn

from ..model import build_vjepa_backbone, EMBED_DIM, PATCH_SIZE
from .vendor.rfdetr.models.backbone.projector import MultiScaleProjector
from .vendor.rfdetr.models.position_encoding import PositionEmbeddingSine
from .vendor.rfdetr.models.transformer import Transformer
from .vendor.rfdetr.models.lwdetr import LWDETR
from .vendor.rfdetr.models.matcher import HungarianMatcher
from .vendor.rfdetr.models.criterion import SetCriterion
from .vendor.rfdetr.utilities.tensors import NestedTensor
from .vendor.rfdetr.utilities.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class VJepaRFBackbone(nn.Module):
    """Adapt the final stride-16 token map to the official RF-DETR projector/joiner."""

    def __init__(self, resolution, checkpoint, hidden_dim, drop_path, grad_checkpoint,
                 freeze_backbone=False):
        super().__init__()
        self.resolution = resolution
        self.encoder = build_vjepa_backbone(checkpoint, resolution, num_frames=2,
                                            drop_path=drop_path, grad_checkpoint=grad_checkpoint)
        self.encoder.requires_grad_(not freeze_backbone)
        self.projector = MultiScaleProjector([EMBED_DIM], hidden_dim, [1.0],
                                            num_blocks=3, layer_norm=True)
        self.position = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    def extract_features(self, images):
        if images.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(f'expected square {self.resolution}-pixel images')
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if frozen:
            self.encoder.eval()
        with torch.no_grad() if frozen else nullcontext():
            video = images.unsqueeze(2).repeat(1, 1, 2, 1, 1)
            tokens = self.encoder(video)
        grid = self.resolution // PATCH_SIZE
        return tokens.reshape(images.shape[0], grid, grid, EMBED_DIM).permute(0, 3, 1, 2).contiguous()

    def forward(self, samples):
        features = self.projector([self.extract_features(samples.tensors)])
        nested = []
        positions = []
        for feature in features:
            mask = torch.zeros((feature.shape[0], *feature.shape[-2:]), dtype=torch.bool,
                               device=feature.device)
            item = NestedTensor(feature, mask)
            nested.append(item)
            positions.append(self.position(item, align_dim_orders=False).to(feature.dtype))
        return nested, positions, None


class VJepaRFDETR(nn.Module):
    """One foreground class; the upstream extra logit is not an output background box."""

    def __init__(self, resolution=768, checkpoint=None, hidden_dim=256,
                 decoder_layers=3, num_queries=300, group_detr=13,
                 self_heads=8, cross_heads=16, feedforward_dim=2048,
                 drop_path=.1, grad_checkpoint=True, freeze_backbone=False):
        super().__init__()
        if resolution <= 0 or resolution % 16:
            raise ValueError('resolution must be a positive multiple of 16')
        if not 0 < num_queries <= (resolution // 16) ** 2:
            raise ValueError('two-stage query count must not exceed stride-16 token count')
        if decoder_layers < 1 or group_detr < 1:
            raise ValueError('decoder_layers and group_detr must be positive')
        self.resolution = resolution
        backbone = VJepaRFBackbone(resolution, checkpoint, hidden_dim, drop_path,
                                  grad_checkpoint, freeze_backbone)
        transformer = Transformer(
            d_model=hidden_dim, sa_nhead=self_heads, ca_nhead=cross_heads,
            num_queries=num_queries, num_decoder_layers=decoder_layers,
            dim_feedforward=feedforward_dim, dropout=0., return_intermediate_dec=True,
            group_detr=group_detr, two_stage=True, num_feature_levels=1,
            dec_n_points=2, lite_refpoint_refine=True, bbox_reparam=True)
        self.detector = LWDETR(backbone, transformer, segmentation_head=None,
                               num_classes=2, num_queries=num_queries, aux_loss=True,
                               group_detr=group_detr, two_stage=True,
                               lite_refpoint_refine=True, bbox_reparam=True)

    @property
    def backbone(self):
        return self.detector.backbone.encoder

    def forward(self, images):
        return self.detector(images)

    def decode(self, outputs):
        # Only query/box format conversion. Existing evaluation handles all filtering,
        # NMS, original-image restoration, background completion and official metrics.
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'].float()) * self.resolution
        scores = outputs['pred_logits'][..., :1].float().sigmoid()
        return boxes, scores


def convert_targets(targets, resolution):
    converted = []
    for target in targets:
        labels = target['labels'].long()
        if (labels != 0).any():
            raise ValueError('RF-DETR comparison is lesion-only (training label 0)')
        boxes = target['boxes'].float()
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] < boxes[:, :2]).any():
            raise ValueError('invalid training boxes')
        normalized = box_xyxy_to_cxcywh(boxes) / resolution
        if (normalized < 0).any() or (normalized > 1).any():
            raise ValueError('training boxes must be in letterboxed model coordinates')
        converted.append({'boxes': normalized, 'labels': labels})
    return converted


class RFDETRCriterion(nn.Module):
    """Official Hungarian, IoU-aware BCE, L1/GIoU and auxiliary/encoder losses."""

    def __init__(self, resolution=768, decoder_layers=3, group_detr=13):
        super().__init__()
        self.resolution = resolution
        weights = {'loss_ce': 1., 'loss_bbox': 5., 'loss_giou': 2.}
        weights.update({f'{k}_{i}': v for i in range(decoder_layers - 1)
                        for k, v in list(weights.items()) if not k.rsplit('_', 1)[-1].isdigit()})
        weights.update({'loss_ce_enc': 1., 'loss_bbox_enc': 5., 'loss_giou_enc': 2.})
        self.criterion = SetCriterion(
            num_classes=2, matcher=HungarianMatcher(cost_class=2., cost_bbox=5., cost_giou=2.),
            weight_dict=weights, focal_alpha=.25, losses=['labels', 'boxes', 'cardinality'],
            group_detr=group_detr, ia_bce_loss=True)

    def forward(self, outputs, targets):
        def fp32(value):
            if torch.is_tensor(value):
                return value.float()
            if isinstance(value, dict):
                return {k: fp32(v) for k, v in value.items()}
            if isinstance(value, list):
                return [fp32(v) for v in value]
            return value
        raw = self.criterion(fp32(outputs), convert_targets(targets, self.resolution))
        total = sum(raw[k] * weight for k, weight in self.criterion.weight_dict.items() if k in raw)
        return {'loss': total, **{k: v.detach() for k, v in raw.items()}}
