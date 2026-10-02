"""CPU checks for the prospective training-only center assignment experiment."""
import sys
import torch
from distill.det_vjepa.loss import FCOSLoss
from distill.det_vjepa.train import parse_args
from distill.det_vjepa.scripts import run_autoresearch_h4_h5 as loop


def main():
    torch.set_num_threads(1)
    y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    points = torch.stack((x.flatten() + .5, y.flatten() + .5), 1) * 8
    strides = torch.full((16,), 8.)
    levels = torch.zeros(16, dtype=torch.long)
    heads = ([torch.zeros(1, 1, 4, 4, requires_grad=True)],
             [torch.ones(1, 4, 4, 4, requires_grad=True)],
             [torch.zeros(1, 1, 4, 4, requires_grad=True)])
    target = [dict(boxes=torch.tensor([[0., 0., 28., 28.]]), labels=torch.tensor([0]))]
    legacy = FCOSLoss()(heads, target, points, strides, levels)
    explicit = FCOSLoss(center_radius=1.5)(heads, target, points, strides, levels)
    assert all(torch.equal(legacy[k], explicit[k]) for k in legacy)
    smaller = FCOSLoss(center_radius=1.0)(heads, target, points, strides, levels)
    assert legacy['num_pos'].item() == 9 and smaller['num_pos'].item() == 4
    smaller['loss'].backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for branch in heads for t in branch)
    empty = [dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long))]
    assert torch.isfinite(FCOSLoss(center_radius=1.0)(heads, empty, points, strides, levels)['loss'])
    command = loop.command_for('verify', loop.REFERENCE, loop.REFERENCE_SCORE,
                               dict(loop.BASE_RECIPE, center_radius=1.0))
    sys.argv = ['train'] + command[command.index('--data'):]
    args = parse_args()
    assert args.center_radius == 1.0 and args.train_minutes == 120
    assert args.val_max_images == 0 and args.conf_thres == .25 and args.iou_thres == .7
    assert loop.history()[1] >= 16
    print('PASS: default exact; positive count 9 -> 4; finite gradients and empty loss; real CLI and protocol defaults valid')


if __name__ == '__main__':
    main()
