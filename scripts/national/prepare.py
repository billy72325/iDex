"""Decode the parquet shards into a grouped, deduplicated manifest for all 1025 species.

Output: artifacts/national/data/images/<dex id>/<n>.png plus manifest.jsonl with
one row per image: {id, path, split, source_split, group, shiny, sha256}.

Splitting is done per species and per *group* (the publisher's file name without
its shiny suffix) so a normal/shiny pair of the same artwork never straddles
train and validation. Exact-duplicate bytes are kept once.
"""
import argparse
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from model import load_labels

SOURCES = Path('artifacts/national/sources/JJMack--pokemon-classification-gen1-9')
OUT = Path('artifacts/national/data')


def decode(raw, size):
    image = Image.open(io.BytesIO(raw))
    image.load()
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGBA')
        flat = Image.new('RGBA', image.size, (255, 255, 255, 255))
        flat.alpha_composite(image)
        image = flat.convert('RGB')
    else:
        image = image.convert('RGB')
    # Centre-square then resize so the on-device preprocessing (square crop, bilinear) matches.
    w, h = image.size
    side = min(w, h)
    image = image.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    return image.resize((size, size), Image.BILINEAR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=256, help='stored edge; training crops to 224')
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=20261109)
    a = p.parse_args()
    labels = load_labels()
    known = {x['id'] for x in labels}
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    errors = []
    for file in sorted(SOURCES.glob('*.parquet')):
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=64):
            for item in batch.to_pylist():
                try:
                    dex = int(item['label'])
                    if dex not in known:
                        continue
                    image = item.get('image', item.get('image_data'))
                    raw = image if isinstance(image, bytes) else image.get('bytes')
                    if not raw:
                        raise ValueError('no embedded bytes')
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    filename = item.get('file_name', '') or ''
                    group = re.sub(r'-s(?=\.[a-z]+$)', '', filename) or digest[:16]
                    target = OUT / 'images' / f'{dex:04d}' / f'{digest[:16]}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        decode(raw, a.size).save(target, optimize=True)
                    rows.append(dict(id=dex, path=str(target), source_split=file.stem, group=f'{dex}:{group}',
                                     shiny=bool(item.get('shiny')), sha256=digest))
                except Exception as e:
                    errors.append(dict(file=file.name, error=str(e)))
        print(f'{file.name}: {len(rows)} images so far', flush=True)
    if not rows:
        raise SystemExit('no images decoded; run acquire.py first')
    rng = random.Random(a.seed)
    by_species = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_species[r['id']][r['group']].append(i)
    for dex, groups in by_species.items():
        names = sorted(groups)
        rng.shuffle(names)
        n_val = max(1, round(len(names) * a.val_fraction)) if len(names) > 1 else 0
        for k, name in enumerate(names):
            split = 'val' if k < n_val else 'train'
            for i in groups[name]:
                rows[i]['split'] = split
    manifest = OUT / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'prepare_errors.json').write_text(json.dumps(errors, indent=2))
    counts = defaultdict(int)
    for r in rows:
        counts[r['id']] += 1
    missing = sorted(known - set(counts))
    thin = sorted(d for d, c in counts.items() if c < 5)
    summary = dict(images=len(rows), species_with_images=len(counts), missing_species=missing, species_under_5_images=thin,
                   train=sum(r['split'] == 'train' for r in rows), val=sum(r['split'] == 'val' for r in rows),
                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), errors=len(errors))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
