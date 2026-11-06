"""Integration invariants for the isolated RF-DETR comparison."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .model import VJepaRFDETR, RFDETRCriterion, convert_targets
from ..evaluation import evaluate_detector


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 1408, 16, stride=16)

    def forward(self, video):
        return self.proj(video.mean(2)).flatten(2).transpose(1, 2)


def tiny_model():
    with patch('distill.det_vjepa.rf_detr.model.build_vjepa_backbone',
               side_effect=lambda *args, **kwargs: TinyEncoder()):
        return VJepaRFDETR(resolution=64, checkpoint=None, hidden_dim=32,
                          decoder_layers=2, num_queries=4, group_detr=2,
                          self_heads=4, cross_heads=4, feedforward_dim=64)


class RFDETRTests(unittest.TestCase):
    def test_small_fixed_problem_loss_decreases(self):
        torch.manual_seed(11)
        model = tiny_model()
        model.train()
        criterion = RFDETRCriterion(resolution=64, decoder_layers=2, group_detr=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images = torch.randn(2, 3, 64, 64)
        targets = [{'boxes': torch.tensor([[12., 12., 44., 44.]]),
                    'labels': torch.tensor([0])}] * 2
        values = []
        for step in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)['loss']
            self.assertTrue(torch.isfinite(loss))
            values.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        self.assertLess(sum(values[-5:]) / 5, .5 * values[0])

    def test_target_conversion_known_boxes_and_empty(self):
        targets = [{'boxes': torch.tensor([[0., 0., 32., 16.]]),
                    'labels': torch.tensor([0])},
                   {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
        result = convert_targets(targets, 64)
        torch.testing.assert_close(result[0]['boxes'], torch.tensor([[.25, .125, .5, .25]]))
        self.assertEqual(result[1]['boxes'].shape, (0, 4))
        with self.assertRaises(ValueError):
            convert_targets([{'boxes': torch.tensor([[0., 0., 1., 1.]]),
                              'labels': torch.tensor([1])}], 64)

    def test_official_group_queries_loss_and_backbone_gradient(self):
        torch.manual_seed(3)
        model = tiny_model()
        model.train()
        raw = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(raw['pred_logits'].shape, (2, 8, 2))
        self.assertIn('enc_outputs', raw)
        self.assertEqual(len(raw['aux_outputs']), 1)
        criterion = RFDETRCriterion(resolution=64, decoder_layers=2, group_detr=2)
        targets = [{'boxes': torch.tensor([[8., 8., 40., 40.]]), 'labels': torch.tensor([0])},
                   {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
        losses = criterion(raw, targets)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
        model.eval()
        raw = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(raw['pred_logits'].shape, (2, 4, 2))
        boxes, scores = model.decode(raw)
        self.assertEqual(boxes.shape, (2, 4, 4))
        self.assertEqual(scores.shape, (2, 4, 1))

    def test_empty_batch_backward_and_decode_scale(self):
        criterion = RFDETRCriterion(resolution=64, decoder_layers=1, group_detr=1)
        raw = {'pred_logits': torch.zeros(2, 4, 2, requires_grad=True),
               'pred_boxes': torch.full((2, 4, 4), .25, requires_grad=True)}
        empty = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}] * 2
        loss = criterion(raw, empty)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(raw['pred_boxes'].grad)
        model = tiny_model()
        boxes, scores = model.decode({'pred_boxes': torch.tensor([[[.5, .5, .5, .25]]]),
                                    'pred_logits': torch.tensor([[[0., 100.]]])})
        torch.testing.assert_close(boxes, torch.tensor([[[16., 24., 48., 40.]]]))
        torch.testing.assert_close(scores, torch.tensor([[[.5]]]))

    def test_existing_evaluation_accepts_model_without_changes(self):
        model = tiny_model()
        def perfect(images):
            return {'pred_boxes': torch.tensor([[[.375, .375, .5, .5]]]),
                    'pred_logits': torch.tensor([[[10., -10.]]])}
        model.forward = perfect
        targets = [{'boxes': torch.tensor([[8., 8., 40., 40.]]), 'labels': torch.tensor([0]),
                    'letterbox': (1., 0., 0.), 'original_size': (64, 64),
                    'foreground_box': [0., 0., 64., 64.], 'path': 'test.png'}]
        metrics, predictions = evaluate_detector(model, [(torch.zeros(1, 3, 64, 64), targets)],
                                                 torch.device('cpu'), 2, lesion_only=True)
        self.assertAlmostEqual(metrics['lesion_AP50_95'], 1., places=6)
        self.assertEqual(metrics['num_images'], 1)
        self.assertEqual(len(predictions), 1)


if __name__ == '__main__':
    unittest.main()
