"""Turn crawled (and your own) photos into an extra manifest that train_v2.py can append.

Crawled images carry noisy labels (wrong species, fan art, screenshots of lists,
packaging). A trained model vets them: keep an image when the labelled species is
in the model's top-k or its probability is above --min-prob, and never trust the
model more than the label (we only *drop*, never relabel).

    python scripts/national/filter_crawl.py --checkpoint artifacts/national/runs/v2-efficientnet_v2_s-224/best.pt \
        --crawl artifacts/national/crawl --own artifacts/national/own --out artifacts/national/v2/extra_manifest.jsonl

Own photos (artifacts/national/own/<id>/*.jpg) are trusted: no model vetting, 20 %
held out as kind "own" validation so camera accuracy can be measured on them.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from labels import load_labels
from model import make_model


def phash(im, size=8):
    g = np.asarray(ImageOps.grayscale(im).resize((32, 32), Image.BILINEAR), dtype=np.float32)
    n = 32
    dct = np.cos(np.pi / n * (np.arange(n)[None, :] + .5) * np.arange(size)[:, None])
    low = dct @ g @ dct.T
    return int(''.join('1' if v > np.median(low[1:, 1:]) else '0' for v in low.flatten()), 2)


class Plain(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as src:
            im = ImageOps.exif_transpose(src).convert('RGB')
        im = TF.center_crop(TF.resize(im, self.size), self.size)
        return TF.to_tensor(im), i


def collect(folder, kind):
    rows = []
    for species in sorted(Path(folder).glob('[0-9]*')):
        try:
            dex = int(species.name)
        except ValueError:
            continue
        for f in sorted(species.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.heic', '.webp'):
                rows.append(dict(id=dex, kind=kind, path=str(f)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--crawl', type=Path, default=Path('artifacts/national/crawl'))
    p.add_argument('--own', type=Path, default=Path('artifacts/national/own'))
    p.add_argument('--out', type=Path, default=Path('artifacts/national/v2/extra_manifest.jsonl'))
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--min-prob', type=float, default=0.03)
    p.add_argument('--max-per-species', type=int, default=80)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261111)
    a = p.parse_args()
    labels = load_labels(); index = {x['id']: i for i, x in enumerate(labels)}
    rows = collect(a.crawl, 'crawl') + collect(a.own, 'own')
    if not rows:
        raise SystemExit('nothing under --crawl or --own')
    # exact + perceptual dedupe (crawl results repeat the same product shot endlessly)
    seen_sha, seen_ph, kept, dropped = set(), set(), [], defaultdict(int)
    for r in rows:
        try:
            raw = Path(r['path']).read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha in seen_sha:
                dropped['exact_dup'] += 1; continue
            with Image.open(r['path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                if min(im.size) < 120:
                    dropped['too_small'] += 1; continue
                ph = phash(im)
            if ph in seen_ph:
                dropped['near_dup'] += 1; continue
            seen_sha.add(sha); seen_ph.add(ph); r['sha256'] = sha; kept.append(r)
        except Exception:
            dropped['unreadable'] += 1
    # model vetting for crawled rows
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(state['config']['arch'], len(labels)); model.load_state_dict(state['model']); model = model.to(device).eval()
    crawl_rows = [r for r in kept if r['kind'] == 'crawl']
    verdict = {}
    if crawl_rows:
        loader = DataLoader(Plain([r['path'] for r in crawl_rows], state['config']['size']), batch_size=128, num_workers=a.workers)
        with torch.inference_mode():
            for x, idx in loader:
                probs = torch.softmax(model(x.to(device)).float(), dim=1).cpu()
                top = probs.topk(a.topk, dim=1).indices
                for k, i in enumerate(idx.tolist()):
                    truth = index[crawl_rows[i]['id']]
                    verdict[i] = bool((top[k] == truth).any()) or float(probs[k, truth]) >= a.min_prob
    final = []
    per_species = defaultdict(int)
    for i, r in enumerate(crawl_rows):
        if not verdict.get(i, False):
            dropped['model_disagrees'] += 1; continue
        if per_species[r['id']] >= a.max_per_species:
            dropped['over_cap'] += 1; continue
        per_species[r['id']] += 1; final.append(r)
    own_rows = [r for r in kept if r['kind'] == 'own']
    rng = random.Random(a.seed)
    for r in final:
        r['group'] = f'crawl:{r["id"]}:{r["sha256"][:12]}'; r['split'] = 'train'
    by_species = defaultdict(list)
    for r in own_rows:
        by_species[r['id']].append(r)
    for dex, group in by_species.items():
        rng.shuffle(group)
        n_val = round(len(group) * 0.2) if len(group) >= 5 else 0
        for k, r in enumerate(group):
            r['group'] = f'own:{dex}:{r["sha256"][:12]}'; r['split'] = 'val' if k < n_val else 'train'
    out_rows = final + own_rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in out_rows))
    summary = dict(crawled=len(crawl_rows), crawled_kept=len(final), own=len(own_rows), dropped=dict(dropped),
                   species_with_crawl=len(per_species), under_10_crawl=sorted(d for d, c in per_species.items() if c < 10),
                   out=str(a.out))
    a.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
