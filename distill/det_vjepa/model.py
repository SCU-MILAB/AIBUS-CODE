"""
V-JEPA ViT-G as a detection backbone — full-parameter fine-tuning.

Backbone : `vit_giant_xformers` (patch 16, tubelet 2, RoPE) built from a V-JEPA
           checkpoint.  2D images are duplicated into a 2-frame pseudo-video
           because the patch embedding is a Conv3d with kernel (2, 16, 16), so
           T must be >= 2.  RoPE is generated at forward time from the actual
           T/H/W (src/models/utils/modules.py:RoPEAttention), so any resolution
           divisible by 16 works without pos-embed interpolation.

Neck     : ViTDet-style "simple feature pyramid" built from the final
           stride-16 token map by default. Optionally, each pyramid level
           can receive a different ViT block output at the same native
           stride-16 grid before spatial up/downsampling.

Head     : FCOS (anchor-free) — classification (focal), ltrb regression (GIoU),
           centerness.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

import src.models.vision_transformer as vit

PATCH_SIZE = 16
TUBELET_SIZE = 2
EMBED_DIM = 1408
VIT_G_DEPTH = 40


def load_detector_weights(model, state_dict, lesion_only=False):
    """Strict load; a two-class checkpoint can initialize a one-class lesion head."""
    state = dict(state_dict)
    if lesion_only:
        for key in ('head.cls_logits.weight', 'head.cls_logits.bias'):
            if state[key].shape[0] not in (1, 2):
                raise ValueError(f'{key}: expected 1 or 2 classes for lesion-only initialization')
            state[key] = state[key][:1]
    return model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_vjepa_backbone(checkpoint_path, resolution, num_frames=2,
                         drop_path=0.0, grad_checkpoint=True,
                         out_layers=None):
    """
    Build a ViT-G and load the V-JEPA encoder weights from `checkpoint_path`.

    The checkpoint stores the encoder under key 'encoder' with a 'model.'
    prefix, which is stripped here to match the bare VisionTransformer.

    Args:
        checkpoint_path: path to a V-JEPA checkpoint (e.g. best_vjepa_model*.pt)
        resolution:      square input resolution, must be divisible by 16
        num_frames:      temporal frames fed to the encoder (must be 2 — see
                         module docstring)
        drop_path:       stochastic depth rate (0.0 matches the checkpoint)
        grad_checkpoint: activation checkpointing inside the transformer blocks,
                         needed to fit full fine-tuning in memory
        out_layers:      optional zero-based block indices to return in order
    """
    if resolution % PATCH_SIZE != 0:
        raise ValueError(f'resolution {resolution} must be divisible by {PATCH_SIZE}')
    if num_frames != 2:
        raise ValueError(
            f'num_frames must be 2 (tubelet_size=2 → one temporal token); got {num_frames}'
        )

    model = vit.vit_giant_xformers(
        img_size=resolution,
        num_frames=num_frames,
        patch_size=PATCH_SIZE,
        tubelet_size=TUBELET_SIZE,
        uniform_power=True,
        use_rope=True,
        handle_nonsquare_inputs=True,
        drop_path_rate=drop_path,
        use_activation_checkpointing=grad_checkpoint,
        out_layers=out_layers,
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'encoder' not in ckpt:
            raise KeyError(
                f"checkpoint has no 'encoder' key; found {list(ckpt.keys())}"
            )
        state = ckpt['encoder']
        for prefix in ('module.', 'backbone.', 'model.'):
            state = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state.items()
            }

        # Without this, an encoder of the wrong size fails inside load_state_dict
        # with a hundred "size mismatch" lines and no hint at the cause.  Every
        # ViT-G checkpoint has patch_embed.proj.weight of shape (1408, 3, 2, 16, 16).
        embed = next((v.shape[0] for k, v in state.items()
                      if k.endswith('patch_embed.proj.weight')), None)
        if embed is None:
            raise KeyError(
                f'{checkpoint_path} has no patch_embed.proj.weight in its '
                f'"encoder" state dict — is this a V-JEPA ViT checkpoint?'
            )
        if embed != EMBED_DIM:
            arch = {1024: 'ViT-L', 1280: 'ViT-H'}.get(embed, f'embed_dim={embed}')
            raise ValueError(
                f'{checkpoint_path} looks like {arch} (embed_dim {embed}), but this '
                f'script only builds ViT-G (embed_dim {EMBED_DIM}, depth 40). '
                f'Use a ViT-G checkpoint, e.g. ckpts/vjepa_full/best_vjepa_model*.pt'
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if 'pos_embed' not in k]
        if missing or unexpected:
            raise RuntimeError(
                f'encoder weights did not match the model — '
                f'missing={missing[:5]} ({len(missing)}), '
                f'unexpected={unexpected[:5]} ({len(unexpected)})'
            )
    return model


# ---------------------------------------------------------------------------
# Neck — ViTDet-style simple feature pyramid
# ---------------------------------------------------------------------------

class SimpleFeaturePyramid(nn.Module):
    """
    Build a multi-scale pyramid from one or several stride-16 feature maps.

    By default, every level receives the same final-block map. Optionally,
    each level receives a map from a different ViT block. Richer levels are
    produced by transposed convolutions and coarser levels by stride-2
    convolutions, then a top-down FPN fuses them. 1x1 (not 3x3) is used for
    the base level because the ViT feature grid is not shifted w.r.t. the
    image (ViTDet, Sec. 3.2).
    """

    def __init__(self, in_dim=EMBED_DIM, hidden_dim=256, strides=(8, 16, 32),
                 base_stride=16):
        super().__init__()
        self.strides = tuple(strides)
        self.base_stride = base_stride

        laterals = []
        for s in self.strides:
            if s > base_stride:
                ratio = s // base_stride
                if ratio & (ratio - 1):
                    raise ValueError(f'stride {s} must be base_stride*2^k')
                layers, d = [], in_dim
                for _ in range(ratio.bit_length() - 1):
                    layers += [nn.Conv2d(d, hidden_dim, 3, stride=2, padding=1),
                               nn.GroupNorm(32, hidden_dim), nn.GELU()]
                    d = hidden_dim
            elif s == base_stride:
                layers = [nn.Conv2d(in_dim, hidden_dim, 1),
                          nn.GroupNorm(32, hidden_dim), nn.GELU()]
            else:
                ratio = base_stride // s
                if ratio & (ratio - 1):
                    raise ValueError(f'stride {s} must be base_stride/2^k')
                layers, d = [], in_dim
                for _ in range(ratio.bit_length() - 1):
                    layers += [nn.ConvTranspose2d(d, hidden_dim, 2, stride=2),
                               nn.GroupNorm(32, hidden_dim), nn.GELU()]
                    d = hidden_dim
            laterals.append(nn.Sequential(*layers))
        self.laterals = nn.ModuleList(laterals)

        self.out_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                          nn.GroupNorm(32, hidden_dim), nn.GELU())
            for _ in self.strides
        ])

    def forward(self, feat):
        """One stride-16 map, or one stride-16 map per output level."""
        if isinstance(feat, torch.Tensor):
            inputs = [feat] * len(self.laterals)
        else:
            inputs = list(feat)
            if len(inputs) != len(self.laterals):
                raise ValueError(
                    f'expected {len(self.laterals)} ViT feature maps, got {len(inputs)}'
                )
        feats = [lat(x) for lat, x in zip(self.laterals, inputs)]
        for i in range(len(feats) - 1, 0, -1):
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i], size=feats[i - 1].shape[-2:], mode='nearest'
            )
        return [conv(f) for conv, f in zip(self.out_convs, feats)]


# ---------------------------------------------------------------------------
# Head — FCOS
# ---------------------------------------------------------------------------

class Scale(nn.Module):
    """Per-level learnable scalar that rescales the predicted distances."""

    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """
    Shared-weight FCOS head applied to every pyramid level.

    The towers and output convolutions are shared across levels (standard FCOS);
    only the distance regression is rescaled per level via `Scale`.
    """

    def __init__(self, num_classes, in_channels=256, num_convs=4,
                 strides=(8, 16, 32), prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(strides)

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                           nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        for conv in (self.cls_logits, self.bbox_pred, self.centerness):
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 0)
        # Focal loss wants a low initial foreground probability.
        nn.init.constant_(self.cls_logits.bias,
                          -math.log((1 - prior_prob) / prior_prob))
        # Positive bias so the ReLU above is never dead at init and every level
        # starts out predicting a box of one stride.
        nn.init.constant_(self.bbox_pred.bias, 1.0)

    def forward(self, feats):
        cls_logits, bbox_pred, ctr_logits = [], [], []
        for level, x in enumerate(feats):
            cls_logits.append(self.cls_logits(self.cls_tower(x)))

            reg_feat = self.reg_tower(x)
            # Distances are in units of the level stride and decode() applies the
            # stride exactly once.  ReLU is deliberately used instead of exp():
            # with exp the gradient is proportional to the predicted distance, so
            # GIoU's pull towards larger boxes feeds back and the branch diverges
            # to the clamp within a few dozen steps.  ReLU is bounded below by 0
            # and its bias starts positive (1.0 = a box of one stride), so no
            # unit is dead and the head starts from a sane box size.
            dist = F.relu(self.bbox_pred(reg_feat))
            bbox_pred.append(self.scales[level](dist))
            ctr_logits.append(self.centerness(reg_feat))
        return cls_logits, bbox_pred, ctr_logits


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class VJepaDetector(nn.Module):
    """
    ViT-G backbone + simple feature pyramid + FCOS head, all trainable.

    By default all pyramid levels use the final ViT map. With out_layers,
    each level uses a selected block output before its existing transform.

    forward(images) returns raw head outputs for loss computation.
    decode(...) turns them into boxes/scores for evaluation.
    """

    def __init__(self, num_classes, resolution=640, num_frames=2,
                 checkpoint=None, fpn_strides=(8, 16, 32), fpn_dim=256,
                 head_convs=4, drop_path=0.0, grad_checkpoint=True,
                 out_layers=None):
        super().__init__()
        self.num_classes = num_classes
        self.resolution = resolution
        self.num_frames = num_frames
        self.fpn_strides = tuple(fpn_strides)
        self.out_layers = None if out_layers is None else tuple(out_layers)
        if self.out_layers is not None:
            if len(self.out_layers) != len(self.fpn_strides):
                raise ValueError(
                    'out_layers must contain one block index per fpn_strides level'
                )
            if (not self.out_layers
                    or any(not isinstance(i, int) or isinstance(i, bool)
                           for i in self.out_layers)
                    or any(a >= b for a, b in zip(self.out_layers, self.out_layers[1:]))
                    or self.out_layers[0] < 0
                    or self.out_layers[-1] != VIT_G_DEPTH - 1):
                raise ValueError(
                    f'out_layers must be increasing zero-based indices in '
                    f'[0, {VIT_G_DEPTH - 1}] and end at {VIT_G_DEPTH - 1}'
                )

        self.backbone = build_vjepa_backbone(
            checkpoint, resolution, num_frames=num_frames,
            drop_path=drop_path, grad_checkpoint=grad_checkpoint,
            out_layers=self.out_layers,
        )
        self.neck = SimpleFeaturePyramid(EMBED_DIM, fpn_dim, self.fpn_strides,
                                         base_stride=PATCH_SIZE)
        self.head = FCOSHead(num_classes, fpn_dim, head_convs, self.fpn_strides)

        # Anchor points / per-location stride — identical for every image at a
        # fixed resolution, so precompute once as registered buffers.
        locs, strides, levels = [], [], []
        for level, s in enumerate(self.fpn_strides):
            h = w = resolution // s
            ys = (torch.arange(h, dtype=torch.float32) + 0.5) * s
            xs = (torch.arange(w, dtype=torch.float32) + 0.5) * s
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            locs.append(torch.stack([xx, yy], dim=-1).reshape(-1, 2))
            strides.append(torch.full((h * w,), float(s)))
            levels.append(torch.full((h * w,), level, dtype=torch.long))
        self.register_buffer('anchor_points', torch.cat(locs, 0), persistent=False)
        self.register_buffer('anchor_strides', torch.cat(strides, 0), persistent=False)
        self.register_buffer('anchor_levels', torch.cat(levels, 0), persistent=False)

    def extract_features(self, images):
        """Return one stride-16 map by default, or selected block maps."""
        b = images.shape[0]
        grid = self.resolution // PATCH_SIZE
        # Duplicate each 2D image into a 2-frame pseudo-video (tubelet_size=2).
        video = images.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        tokens = self.backbone(video)          # [B, T*Hp*Wp, D], T = 1 here

        def to_map(x):
            return x.reshape(b, grid, grid, EMBED_DIM).permute(0, 3, 1, 2).contiguous()

        if self.out_layers is None:
            return to_map(tokens)
        if len(tokens) != len(self.out_layers):
            raise RuntimeError(
                f'backbone returned {len(tokens)} outputs for '
                f'{len(self.out_layers)} requested out_layers'
            )
        return [to_map(x) for x in tokens]

    def forward(self, images):
        return self.head(self.neck(self.extract_features(images)))

    def decode(self, head_out):
        """
        head_out → (boxes [B, N, 4] in input pixels, scores [B, N, C]).

        Scores are cls_sigmoid * centerness_sigmoid, as in FCOS.
        """
        cls_logits, bbox_pred, ctr_logits = head_out
        locs = self.anchor_points
        strides = self.anchor_strides.view(1, -1, 1)

        cls = torch.cat([c.flatten(2).permute(0, 2, 1) for c in cls_logits], 1)
        dist = torch.cat([b.flatten(2).permute(0, 2, 1) for b in bbox_pred], 1)
        ctr = torch.cat([c.flatten(1) for c in ctr_logits], 1)

        ltrb = dist * strides
        boxes = torch.stack([
            locs[:, 0] - ltrb[..., 0], locs[:, 1] - ltrb[..., 1],
            locs[:, 0] + ltrb[..., 2], locs[:, 1] + ltrb[..., 3],
        ], dim=-1)
        scores = cls.sigmoid() * ctr.sigmoid().unsqueeze(-1)
        return boxes, scores

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
