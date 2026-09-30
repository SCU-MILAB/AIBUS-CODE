"""Regenerate background annotations from source pixels, preserving lesion rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import foreground_box, list_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/lx/dataset/6cls_detection'))
    parser.add_argument('--black_threshold', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.black_threshold <= 255:
        parser.error('black_threshold must be between 0 and 255')
    root = args.root.resolve()
    output = root / 'labels_2cls_remap'
    staging = root / 'labels_2cls_remap.building'
    entry = root / 'data_2cls_remap'
    config = root / 'data-2cls-remap.yaml'
    for path in (output, staging, entry, config):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite {path}')
    source_config = yaml.safe_load((root / 'data.yaml').read_text())
    splits = {'train': 'train2017', 'val': 'val2017'}
    jobs = []
    for split in splits.values():
        (staging / split).mkdir(parents=True)
        for image in list_images(str(root), f'images/{split}'):
            jobs.append((split, Path(image)))
    cv2.setNumThreads(1)

    def convert(job):
        split, image_path = job
        source = root / 'labels' / split / (image_path.stem + '.txt')
        raw = source.read_bytes()  # Missing labels are errors, not background.
        lines = raw.decode().splitlines(keepends=True)
        lesion, background = [], []
        for line in lines:
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5 or values[0] not in ('0', '1'):
                raise ValueError(f'invalid two-class annotation: {source}: {line!r}')
            (lesion if values[0] == '0' else background).append(line)
        # Preserve class 0 text exactly; no image decoding needed for lesion-only rows.
        text = ''.join(lesion)
        if background:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f'cannot read {image_path}')
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(float, foreground_box(image, args.black_threshold))
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'invalid foreground box for {image_path}')
            if text and not text.endswith('\n'):
                text += '\n'
            text += f'1 {(x1+x2)/(2*w):.10f} {(y1+y2)/(2*h):.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}\n'
        destination = staging / split / source.name
        destination.write_text(text)
        written = destination.read_text().splitlines(keepends=True)
        preserved = [line for line in written if line.split()[0] == '0']
        assert [line.rstrip('\r\n') for line in preserved] == [line.rstrip('\r\n') for line in lesion]
        return split, len(lesion), bool(background), not bool(lesion or background), hashlib.sha256(raw).hexdigest()

    summary = {split: dict(images=0, lesion_boxes=0, background_images=0, empty_images=0)
               for split in splits.values()}
    source_hash = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (split, lesion_count, background, empty, digest) in enumerate(executor.map(convert, jobs)):
            stats = summary[split]
            stats['images'] += 1
            stats['lesion_boxes'] += lesion_count
            stats['background_images'] += int(background)
            stats['empty_images'] += int(empty)
            source_hash.update(digest.encode())
            if (index + 1) % 20000 == 0:
                print(f'processed {index+1}/{len(jobs)}', flush=True)
    summary.update(black_threshold=args.black_threshold,
                   foreground_rule='union_area_ge64_width_ge_max4_5percent_keep_largest',
                   source_labels=str(root/'labels'), source_hash=source_hash.hexdigest(),
                   empty_policy='preserve empty; do not synthesize ground truth')
    (staging / 'generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    staging.rename(output)
    entry.mkdir()
    (entry / 'images').symlink_to('../images', target_is_directory=True)
    (entry / 'labels').symlink_to('../labels_2cls_remap', target_is_directory=True)
    cfg = dict(path=str(entry), train='images/train2017', val='images/val2017',
               names=source_config['names'])
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'data YAML: {config}', flush=True)


if __name__ == '__main__':
    main()
