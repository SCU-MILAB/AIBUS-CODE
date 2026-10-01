"""CPU regression check for score preservation during bbox-only adaptation."""
from types import SimpleNamespace

import torch
from torch import nn

from distill.det_vjepa.loss import FCOSLoss
from distill.det_vjepa.model import FCOSHead, SimpleFeaturePyramid
from distill.det_vjepa.train import build_optimizer, configure_trainability


class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 64, 16, stride=16)
        self.neck = SimpleFeaturePyramid(in_dim=64)
        self.head = FCOSHead(1)

    def forward(self, images):
        return self.head(self.neck(self.backbone(images)))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = SmallDetector().train()
    configure_trainability(model, True, 'box_projection')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9223
    state = {k: v.clone() for k, v in model.state_dict().items()}
    frozen = {k for k, p in model.named_parameters() if not p.requires_grad}
    optimizer = build_optimizer(model, SimpleNamespace(lr_head=1e-3, lr_backbone=1e-4,
                                                      layer_decay=.85, weight_decay=.05), 1)
    images = torch.randn(2, 3, 128, 128)
    before = model(images)
    points, strides, levels = [], [], []
    for level, (stride, logits) in enumerate(zip(model.head.strides, before[0])):
        h, w = logits.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        points.append(torch.stack((x.flatten() + .5, y.flatten() + .5), 1) * stride)
        strides.append(torch.full((h * w,), stride))
        levels.append(torch.full((h * w,), level, dtype=torch.long))
    anchors = (torch.cat(points), torch.cat(strides), torch.cat(levels))
    empty = dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long))
    targets = [dict(boxes=torch.tensor([[20., 30., 90., 110.]]), labels=torch.tensor([0])), empty]
    loss = FCOSLoss()(before, targets, *anchors)
    assert loss['num_pos'] > 1 and torch.isfinite(loss['loss'])
    loss['loss'].backward()
    assert model.head.bbox_pred.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    optimizer.step()
    with torch.no_grad():
        after = model(images)
    assert all(torch.equal(a, b) for branch in (0, 2)
               for a, b in zip(before[branch], after[branch]))
    assert any(not torch.equal(a, b) for a, b in zip(before[1], after[1]))
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(state[k], model.state_dict()[k]) for k in frozen)
    model.load_state_dict(state, strict=True)
    optimizer.zero_grad(set_to_none=True)
    empty_loss = FCOSLoss()(model(images), [empty, empty], *anchors)['loss']
    assert torch.isfinite(empty_loss)
    empty_loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
               for p in model.parameters() if p.requires_grad)
    default_model = SmallDetector()
    configure_trainability(default_model, True)
    assert all(p.requires_grad for p in default_model.neck.parameters())
    assert all(p.requires_grad for p in default_model.head.parameters())
    independent = SmallDetector()
    configure_trainability(independent, False, 'box_projection')
    assert all(p.requires_grad for p in independent.backbone.parameters())
    print('PASS: 9,223 trainable parameters; boxes adapt, raw scores and frozen weights '
          'remain exact; empty targets remain connected; old state loads strictly; '
          'default and independent backbone masks preserved.')


if __name__ == '__main__':
    main()
