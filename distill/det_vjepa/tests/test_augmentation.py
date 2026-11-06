import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from distill.det_vjepa.dataset import augment_detection, YoloDetectionDataset, validate_augmentation


class AugmentationTests(unittest.TestCase):
    def test_translation_aligns_pixels_and_boxes(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[20:40, 20:40] = 255
        boxes = np.array([[20, 20, 40, 40]], np.float32)
        labels = np.array([0])
        with patch('numpy.random.uniform', side_effect=[0, 1, np.array([0.1, 0.1])]):
            result, moved, classes = augment_detection(image, boxes, labels, {'translate': 0.1})
        np.testing.assert_array_equal(moved, [[30, 30, 50, 50]])
        self.assertTrue(np.all(result[30:50, 30:50] == 255))
        np.testing.assert_array_equal(classes, labels)

    def test_rejects_truncated_lesion(self):
        image = np.zeros((100, 100, 3), np.uint8)
        boxes = np.array([[80, 80, 99, 99]], np.float32)
        with patch('numpy.random.uniform', side_effect=[0, 1, np.array([0.2, 0.2])]):
            result, moved, _ = augment_detection(image, boxes, np.array([0]), {'translate': 0.2})
        np.testing.assert_array_equal(result, image)
        np.testing.assert_array_equal(moved, boxes)

    def test_disabled_preserves_rng_and_negative_samples_work(self):
        image = np.zeros((64, 64, 3), np.uint8)
        boxes, labels = np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        np.random.seed(42)
        expected = np.random.rand()
        np.random.seed(42)
        result, _, _ = augment_detection(image, boxes, labels, {})
        np.testing.assert_array_equal(result, image)
        self.assertEqual(np.random.rand(), expected)
        result, moved, classes = augment_detection(image, boxes, labels,
            dict(degrees=3, translate=0.03, scale=0.1, brightness=0.08,
                 contrast=0.15, gamma=0.15, noise_std=0.01, blur_prob=1))
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(moved.shape, (0, 4))
        self.assertEqual(classes.shape, (0,))

    def test_validation_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images/val').mkdir(parents=True)
            (root / 'labels/val').mkdir(parents=True)
            path = root / 'images/val/sample.png'
            cv2.imwrite(str(path), np.full((60, 80, 3), 100, np.uint8))
            (root / 'labels/val/sample.txt').write_text('0 0.5 0.5 0.3 0.4\n')
            common = dict(root=directory, image_paths=[str(path)], imgsz=96,
                          augment=False, lesion_only=True)
            baseline = YoloDetectionDataset(**common)[0]
            augmented = YoloDetectionDataset(**common, augmentation=dict(
                degrees=3, translate=0.03, scale=0.1, brightness=0.08,
                contrast=0.15, gamma=0.15, noise_std=0.01, blur_prob=1))[0]
            for key in ('image', 'boxes', 'labels'):
                self.assertTrue(torch.equal(baseline[key], augmented[key]))
            np.testing.assert_array_equal(baseline['foreground_box'], augmented['foreground_box'])

    def test_invalid_magnitudes(self):
        for options in ({'scale': 1}, {'gamma': float('nan')}, {'translate': -0.1}):
            with self.assertRaises(ValueError):
                validate_augmentation(options)


if __name__ == '__main__':
    unittest.main()
