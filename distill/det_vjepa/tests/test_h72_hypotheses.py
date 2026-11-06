import unittest
import subprocess
import types
from unittest.mock import patch

import torch
from torch import nn

from distill.det_vjepa import model as module
from distill.det_vjepa.loss import FCOSLoss, instance_balanced_weights, paired_giou_loss
from distill.det_vjepa.train import configure_trainability


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, module.EMBED_DIM, 16, stride=16)

    def forward(self, video):
        return self.proj(video.mean(2)).flatten(2).transpose(1, 2)


def tiny_detector(**kwargs):
    with patch.object(module, 'build_vjepa_backbone', side_effect=lambda *a, **k: TinyBackbone()):
        return module.VJepaDetector(1, resolution=32, fpn_dim=32,
                                  head_convs=1, **kwargs)


class HypothesisTests(unittest.TestCase):
    def test_default_loss_and_gradients_match_prechange_implementation(self):
        source = subprocess.check_output(
            ['git', 'show', 'HEAD:distill/det_vjepa/loss.py'], text=True)
        original = types.ModuleType('original_loss')
        exec(compile(source, 'original_loss.py', 'exec'), original.__dict__)
        torch.manual_seed(91)
        points = torch.tensor([[4., 4.], [12., 4.], [20., 4.]])
        targets = [{'boxes': torch.tensor([[0., 0., 16., 8.], [16., 0., 24., 8.]]),
                    'labels': torch.zeros(2, dtype=torch.long)}] * 2
        raw = tuple([torch.randn(2, channels, 1, 3, requires_grad=True)]
                    for channels in [1, 4, 1])
        duplicate = tuple([x.detach().clone().requires_grad_() for x in branch] for branch in raw)
        args = (targets, points, torch.full((3,), 8.), torch.zeros(3, dtype=torch.long))
        current = FCOSLoss(reg_weight=.5)(raw, *args)
        previous = original.FCOSLoss(reg_weight=.5)(duplicate, *args)
        for key in current:
            torch.testing.assert_close(current[key], previous[key], rtol=0, atol=0)
        current['loss'].backward()
        previous['loss'].backward()
        for first, second in zip(raw, duplicate):
            torch.testing.assert_close(first[0].grad, second[0].grad, rtol=0, atol=0)

    def test_full_gt_loss_matches_hand_computed_per_instance_average(self):
        points = torch.tensor([[4., 4.], [12., 4.], [20., 4.]])
        targets = [{'boxes': torch.tensor([[0., 0., 16., 8.], [16., 0., 24., 8.]]),
                    'labels': torch.zeros(2, dtype=torch.long)}]
        raw = ([torch.zeros(1, 1, 1, 3, requires_grad=True)],
               [torch.ones(1, 4, 1, 3, requires_grad=True)],
               [torch.zeros(1, 1, 1, 3, requires_grad=True)])
        result = FCOSLoss(reg_normalization='gt')(raw, targets, points,
                  torch.full((3,), 8.), torch.zeros(3, dtype=torch.long))
        pred = torch.cat([points - 8, points + 8], dim=1)
        first = paired_giou_loss(pred[:2], targets[0]['boxes'][:1].expand(2, 4))
        second = paired_giou_loss(pred[2:], targets[0]['boxes'][1:])
        torch.testing.assert_close(result['reg'], (first + second) / 2)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(raw[1][0].grad).all())

    def test_gradient_routes_preserve_outputs_and_cut_only_selected_paths(self):
        torch.manual_seed(7)
        images = torch.randn(2, 3, 32, 32)
        model = tiny_detector()
        configure_trainability(model, head_train_scope='box_projection')
        reference = model(images)
        for route, active in [('all', (True, True)), ('no_ctr', (True, False)),
                              ('reg_only', (False, False))]:
            model.head.backbone_gradient = route
            output = model(images)
            for old_branch, new_branch in zip(reference, output):
                for old, new in zip(old_branch, new_branch):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
            for branch_index, enabled in [(0, active[0]), (2, active[1])]:
                model.zero_grad(set_to_none=True)
                term = sum(x.sum() for x in output[branch_index])
                self.assertEqual(term.requires_grad, enabled)
                if enabled:
                    term.backward(retain_graph=True)
                    self.assertIsNotNone(model.backbone.proj.weight.grad)
            model.zero_grad(set_to_none=True)
            sum(x.sum() for x in output[1]).backward()
            self.assertIsNotNone(model.backbone.proj.weight.grad)
            self.assertIsNotNone(model.head.bbox_pred.weight.grad)

    def test_residual_old_checkpoint_identity_gradient_and_strict_loading(self):
        torch.manual_seed(7)
        baseline = tiny_detector()
        state = baseline.state_dict()
        images = torch.randn(2, 3, 32, 32)
        expected = baseline(images)
        for mode in ['token', 'image']:
            candidate = tiny_detector(box_residual=mode)
            module.load_detector_weights(candidate, state, allow_new_residual=True)
            configure_trainability(candidate, head_train_scope='box_projection')
            actual = candidate(images)
            for old_branch, new_branch in zip(expected, actual):
                for old, new in zip(old_branch, new_branch):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
            sum(x.sum() for x in actual[1]).backward()
            grad = candidate.head.box_residual.output.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0)
            self.assertTrue(all(p.requires_grad for p in candidate.head.box_residual.parameters()))
            reloaded = tiny_detector(box_residual=mode)
            module.load_detector_weights(reloaded, candidate.state_dict())
            broken = dict(state)
            broken.pop('head.bbox_pred.bias')
            with self.assertRaises(RuntimeError):
                module.load_detector_weights(candidate, broken, allow_new_residual=True)
            partial = dict(state)
            partial['head.box_residual.output.bias'] = candidate.head.box_residual.output.bias
            with self.assertRaises(RuntimeError):
                module.load_detector_weights(candidate, partial, allow_new_residual=True)

    def test_per_gt_weights_equalize_instances_not_positions(self):
        ids = torch.tensor([0, 0, 1, 2, 2, 2])
        weights = instance_balanced_weights(ids)
        torch.testing.assert_close(weights, torch.tensor([.5, .5, 1., 1/3, 1/3, 1/3]))
        losses = torch.tensor([0., 0., 1., 2., 2., 2.])
        self.assertAlmostEqual(((losses * weights).sum() / weights.sum()).item(), 1., places=6)
        weighted = instance_balanced_weights(ids, torch.tensor([.2, .8, .5, .1, .3, .6]))
        for group in [0, 1, 2]:
            self.assertAlmostEqual(weighted[ids == group].sum().item(), 1., places=6)

    def test_gt_ids_follow_overlap_winner_and_empty_backward_is_finite(self):
        criterion = FCOSLoss(reg_normalization='gt')
        points = torch.tensor([[8., 8.], [16., 16.], [80., 80.]])
        labels, distances, ctr, ids = criterion._match(
            points, torch.tensor([[-1., 1e8]]).repeat(3, 1),
            torch.full((3,), 100.),
            torch.tensor([[0., 0., 32., 32.], [4., 4., 12., 12.]]),
            torch.zeros(2, dtype=torch.long), return_gt_ids=True)
        self.assertEqual(ids.tolist(), [1, 0, -1])
        self.assertEqual(labels.tolist(), [1, 1, 0])
        for normalization in ['positive', 'gt']:
            criterion = FCOSLoss(reg_normalization=normalization)
            raw = tuple([torch.randn(2, channels, 1, 1, requires_grad=True)]
                        for channels in [1, 4, 1])
            empty = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}] * 2
            losses = criterion(raw, empty, torch.tensor([[4., 4.]]),
                               torch.tensor([8.]), torch.tensor([0]))
            losses['loss'].backward()
            self.assertTrue(torch.isfinite(losses['loss']))
            for branch in raw:
                self.assertIsNotNone(branch[0].grad)
                self.assertTrue(torch.isfinite(branch[0].grad).all())


if __name__ == '__main__':
    unittest.main()
