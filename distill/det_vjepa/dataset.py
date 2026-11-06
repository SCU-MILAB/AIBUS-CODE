"""
YOLO-format detection dataset.

Expected layout (same as ultralytics):

    <root>/
      images/<split>/*.jpg
      labels/<split>/*.txt        # "cls cx cy w h", all normalized to [0, 1]

Images are letterboxed to a square resolution with aspect ratio preserved, so
boxes keep their shape.  Labels are converted to absolute xyxy pixels of the
letterboxed image.
"""

import glob
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def load_data_yaml(path):
    """Return (root, split_specs, class_names) from a YOLO data yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get('path') or os.path.dirname(os.path.abspath(path))
    names = cfg['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return root, cfg, list(names)


def label_path_for(image_path):
    """<root>/images/<split>/x.jpg → <root>/labels/<split>/x.txt"""
    marker_in, marker_out = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    if marker_in not in image_path:
        raise ValueError(
            f'cannot derive a label path from {image_path!r} — '
            f'expected an "images" directory component'
        )
    return marker_out.join(image_path.rsplit(marker_in, 1)).rsplit('.', 1)[0] + '.txt'


def list_images(root, spec):
    """`spec` is a directory (relative to root) or a .txt file of image paths."""
    if str(spec).endswith('.txt'):
        list_path = spec if os.path.isabs(spec) else os.path.join(root, spec)
        with open(list_path) as f:
            entries = [line.strip() for line in f if line.strip()]
        return [e if os.path.isabs(e) else os.path.join(root, e) for e in entries]

    split_dir = spec if os.path.isabs(spec) else os.path.join(root, spec)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'split directory not found: {split_dir}')
    with os.scandir(split_dir) as it:
        files = sorted(e.path for e in it
                       if e.is_file() and e.name.lower().endswith(IMG_EXTS))
    return files


def letterbox(img, new_shape, color=114):
    """Resize preserving aspect ratio, then pad to `new_shape` (h, w)."""
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (new_shape[1] - new_w) / 2, (new_shape[0] - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, ratio, left, top


def read_labels(path, img_w, img_h):
    """Parse YOLO labels → (boxes [n,4] xyxy px, labels [n]) in the source size."""
    if not os.path.isfile(path):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = (float(v) for v in parts[:5])
            rows.append((int(cls), cx, cy, bw, bh))
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    arr = np.asarray(rows, np.float32)
    cx, cy, bw, bh = arr[:, 1] * img_w, arr[:, 2] * img_h, arr[:, 3] * img_w, arr[:, 4] * img_h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return boxes.astype(np.float32), arr[:, 0].astype(np.int64)


def foreground_box(img, black_threshold=3):
    """Envelope of substantial non-black regions in original-image coordinates.

    Keep disconnected tissue while filtering tiny noise and narrow overlays.
    Always retain the largest component, including for small source images.
    """
    mask = (img.max(axis=2) > black_threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = img.shape[:2]
    if count <= 1:
        return np.array([0, 0, width, height], np.float32)
    stats = stats[1:]
    keep = ((stats[:, cv2.CC_STAT_AREA] >= 64)
            & (stats[:, cv2.CC_STAT_WIDTH] >= max(4, width * 0.05)))
    keep[int(np.argmax(stats[:, cv2.CC_STAT_AREA]))] = True
    regions = stats[keep]
    return np.array([regions[:, 0].min(), regions[:, 1].min(),
                     (regions[:, 0] + regions[:, 2]).max(),
                     (regions[:, 1] + regions[:, 3]).max()], np.float32)


def validate_augmentation(options):
    """Validate optional training-only augmentation magnitudes."""
    limits = {'degrees': 15, 'translate': 0.25, 'scale': 0.5,
              'brightness': 0.5, 'contrast': 0.5, 'gamma': 0.5,
              'noise_std': 0.1, 'blur_prob': 1}
    unknown = set(options) - set(limits)
    if unknown:
        raise ValueError(f'unknown augmentation options: {sorted(unknown)}')
    for key, value in options.items():
        if not np.isfinite(value) or not 0 <= value <= limits[key]:
            raise ValueError(f'augmentation {key} must be in [0, {limits[key]}]')


def augment_detection(img, boxes, labels, options):
    """Mild affine and intensity augmentation; keep boxes aligned with pixels.

    Reject geometric transforms that substantially cut any positive box, rather
    than silently turning a partially visible lesion into a negative sample.
    All-zero options preserve the historical training pipeline and RNG stream.
    """
    h, w = img.shape[:2]
    degrees, translate, scale = (options.get(k, 0) for k in ('degrees', 'translate', 'scale'))
    if degrees or translate or scale:
        angle = np.random.uniform(-degrees, degrees)
        zoom = np.random.uniform(1 - scale, 1 + scale)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
        matrix[:, 2] += np.random.uniform(-translate, translate, 2) * [w, h]
        transformed = boxes.copy()
        accept = True
        if len(boxes):
            corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
            points = corners @ matrix[:, :2].T + matrix[:, 2]
            raw = np.concatenate((points.min(axis=1), points.max(axis=1)), axis=1)
            transformed = raw.copy()
            transformed[:, [0, 2]] = transformed[:, [0, 2]].clip(0, w)
            transformed[:, [1, 3]] = transformed[:, [1, 3]].clip(0, h)
            sides = transformed[:, 2:] - transformed[:, :2]
            raw_area = np.prod(raw[:, 2:] - raw[:, :2], axis=1)
            retained = np.prod(sides, axis=1) / np.maximum(raw_area, 1e-6)
            accept = bool(np.all((sides > 1).all(axis=1) & (retained >= 0.9)))
        if accept:
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
            boxes = transformed.astype(np.float32)

    brightness, contrast, gamma, noise = (options.get(k, 0) for k in
                                         ('brightness', 'contrast', 'gamma', 'noise_std'))
    if brightness or contrast or gamma or noise:
        pixels = img.astype(np.float32) / 255.0
        if contrast:
            pixels = (pixels - 0.5) * np.random.uniform(1 - contrast, 1 + contrast) + 0.5
        if brightness:
            pixels += np.random.uniform(-brightness, brightness)
        pixels = np.clip(pixels, 0, 1)
        if gamma:
            pixels = pixels ** np.random.uniform(1 - gamma, 1 + gamma)
        if noise:
            # Shared noise across channels preserves grayscale ultrasound appearance.
            pixels += np.random.normal(0, np.random.uniform(0, noise), (h, w, 1))
        img = (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)
    blur_prob = options.get('blur_prob', 0)
    if blur_prob and np.random.rand() < blur_prob:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=np.random.uniform(0.3, 0.8))
    return img, boxes, labels


class YoloDetectionDataset(Dataset):
    """Letterboxed YOLO-format detection dataset."""

    def __init__(self, root, image_paths, imgsz=640, augment=False,
                 flip_prob=0.5, skip_empty=False, lesion_only=False,
                 black_threshold=3, augmentation=None):
        self.root = root
        self.image_paths = image_paths
        self.imgsz = imgsz
        self.augment = augment
        self.flip_prob = flip_prob
        self.skip_empty = skip_empty
        self.lesion_only = lesion_only
        self.black_threshold = black_threshold
        self.augmentation = dict(augmentation or {})
        validate_augmentation(self.augmentation)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'failed to read image: {path}')
        h0, w0 = img.shape[:2]

        label_path = label_path_for(path)
        if self.lesion_only and not os.path.isfile(label_path):
            raise FileNotFoundError(f'missing annotation cannot be treated as background: {label_path}')
        boxes, labels = read_labels(label_path, w0, h0)
        if self.lesion_only and np.any((labels != 0) & (labels != 1)):
            raise ValueError(f'lesion-only mode expects original class ids 0/1: {label_path}')
        region = (foreground_box(img, self.black_threshold)
                  if self.lesion_only and not self.augment
                  else np.array([0, 0, w0, h0], np.float32))
        # Keep normal images as negative training examples, with no positive boxes.
        if self.lesion_only and self.augment:
            keep = labels == 0
            boxes, labels = boxes[keep], labels[keep]

        img, ratio, pad_x, pad_y = letterbox(img, (self.imgsz, self.imgsz))
        if boxes.shape[0]:
            boxes = boxes * ratio
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            # Drop boxes that fall outside the padded canvas or collapsed to a line.
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.imgsz)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.imgsz)
            min_side = 0.0 if self.lesion_only and not self.augment else 1.0
            keep = ((boxes[:, 2] - boxes[:, 0]) > min_side) & ((boxes[:, 3] - boxes[:, 1]) > min_side)
            boxes, labels = boxes[keep], labels[keep]

        if self.augment:
            img, boxes, labels = augment_detection(img, boxes, labels, self.augmentation)

        if self.augment and (boxes.shape[0] or self.lesion_only) and np.random.rand() < self.flip_prob:
            img = img[:, ::-1]
            boxes = np.stack([
                self.imgsz - boxes[:, 2], boxes[:, 1],
                self.imgsz - boxes[:, 0], boxes[:, 3],
            ], axis=1)

        # BGR uint8 HWC → RGB float32 CHW, ImageNet-normalized (matches the
        # normalization used by the V-JEPA fine-tuning pipeline in utils.py).
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

        return {
            'image': tensor,
            'boxes': torch.from_numpy(np.ascontiguousarray(boxes)).float(),
            'labels': torch.from_numpy(np.ascontiguousarray(labels)).long(),
            'path': path,
            'original_size': (w0, h0),
            'letterbox': (ratio, pad_x, pad_y),
            'foreground_box': region,
        }


def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    targets = [{'boxes': b['boxes'], 'labels': b['labels'], 'path': b['path'],
                'original_size': b['original_size'], 'letterbox': b['letterbox'],
                'foreground_box': b['foreground_box']}
               for b in batch]
    return images, targets


def build_datasets(data_yaml, imgsz=640, flip_prob=0.5, lesion_only=False,
                   black_threshold=3, augmentation=None):
    """Return (train_dataset, val_dataset, class_names)."""
    root, cfg, names = load_data_yaml(data_yaml)
    if lesion_only and len(names) != 2:
        raise ValueError('lesion-only mode requires a two-class yaml: class 0 lesion, class 1 background')
    train_paths = list_images(root, cfg['train'])
    val_paths = list_images(root, cfg['val'])
    train_set = YoloDetectionDataset(root, train_paths, imgsz, augment=True,
                                     flip_prob=flip_prob, lesion_only=lesion_only,
                                     black_threshold=black_threshold,
                                     augmentation=augmentation)
    val_set = YoloDetectionDataset(root, val_paths, imgsz, augment=False,
                                   lesion_only=lesion_only, black_threshold=black_threshold)
    return train_set, val_set, names
