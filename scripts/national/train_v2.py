"""v2 training: camera-robust 1,025-class classifier with outlier exposure.

What changed from train.py
- Artwork is composited onto real photo backgrounds (COCO) or synthetic ones at
  random scale/position with perspective, blur, noise, JPEG and lighting jitter,
  so the model sees "a Pokémon somewhere in a photo" instead of clean white art.
- Real photos (Gen I web images) train alongside artwork.
- Negatives (COCO) are trained toward a uniform distribution (outlier exposure),
  which lowers the top softmax on non-Pokémon input so the app can abstain.
- No label smoothing; EMA weights; temperature fitted on validation and stored
  in the checkpoint config so the app threshold can be reasoned about.

Checkpoint schema is unchanged (model, config, labels, manifest_sha256), so
export.py and the Apple validation harness still apply.

    python scripts/national/train_v2.py train --arch efficientnet_v2_s --size 256 --epochs 40 --out artifacts/national/runs/v2-effv2s
    python scripts/national/train_v2.py predict --checkpoint artifacts/national/runs/v2-effv2s/best.pt --out artifacts/national/predictions/v2-effv2s-val.jsonl
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from model import load_labels, make_model

V2 = Path('artifacts/national/v2')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rows_from(manifest, split):
    wanted = set(split.split(','))
    return [r for r in (json.loads(s) for s in manifest.read_text().splitlines()) if r['split'] in wanted]


# ----------------------------------------------------------------------------- augmentation
class Backgrounds:
    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob('*.jpg'))

    def sample(self, size):
        r = random.random()
        if self.paths and r < 0.65:
            with Image.open(random.choice(self.paths)) as im:
                im = im.convert('RGB')
                w, h = im.size
                side = random.randint(int(min(w, h) * 0.4), min(w, h))
                x, y = random.randint(0, w - side), random.randint(0, h - side)
                bg = im.crop((x, y, x + side, y + side)).resize((size, size), Image.BILINEAR)
                if random.random() < 0.5:
                    bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
                return bg
        if r < 0.85:  # smooth gradient (table, wall, screen)
            a = np.random.randint(40, 230, 3); b = np.random.randint(40, 230, 3)
            t = np.linspace(0, 1, size)[:, None] if random.random() < 0.5 else np.linspace(0, 1, size)[None, :]
            arr = (a * (1 - t[..., None]) + b * t[..., None]).astype(np.uint8)
            arr = np.broadcast_to(arr, (size, size, 3)).copy()
            return Image.fromarray(arr)
        return Image.new('RGB', (size, size), tuple(int(v) for v in np.random.randint(20, 240, 3)))


def composite_artwork(art, backgrounds, size):
    """Place RGBA artwork on a background at random scale/position with camera-like distortions."""
    bg = backgrounds.sample(size)
    # Crop to the figure first: many sprites sit in a large transparent square.
    box = art.getchannel('A').getbbox()
    if box and (box[2] - box[0]) >= 8 and (box[3] - box[1]) >= 8:
        art = art.crop(box)
    w, h = art.size
    side = max(w, h)
    square = Image.new('RGBA', (side, side), (0, 0, 0, 0)); square.paste(art, ((side - w) // 2, (side - h) // 2)); art = square
    scale = random.uniform(0.45, 0.95)
    side = int(size * scale)
    fg = art.resize((side, side), Image.BILINEAR)
    fg = fg.rotate(random.uniform(-25, 25), resample=Image.BILINEAR, expand=False)
    if random.random() < 0.5:
        fg = fg.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:  # perspective
        startpoints = [(0, 0), (side, 0), (side, side), (0, side)]
        j = side * 0.18
        endpoints = [(random.uniform(0, j), random.uniform(0, j)), (side - random.uniform(0, j), random.uniform(0, j)),
                     (side - random.uniform(0, j), side - random.uniform(0, j)), (random.uniform(0, j), side - random.uniform(0, j))]
        fg = TF.perspective(fg, startpoints, endpoints, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    # lighting on the figure only (toys/cards are lit by the room)
    rgb = fg.convert('RGB'); alpha = fg.getchannel('A')
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.6, 1.3))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.7, 1.3))
    rgb = ImageEnhance.Color(rgb).enhance(random.uniform(0.6, 1.3))
    if random.random() < 0.3:  # soft shadow / glossy highlight band
        shade = Image.new('L', rgb.size, 0)
        band = np.linspace(0, 1, rgb.size[0]) ** random.uniform(1, 3) * random.uniform(40, 110)
        shade = Image.fromarray(np.broadcast_to(band.astype(np.uint8), (rgb.size[1], rgb.size[0])).copy())
        rgb = Image.composite(ImageOps.autocontrast(rgb), rgb, shade) if random.random() < 0.5 else Image.blend(rgb, Image.new('RGB', rgb.size, (0, 0, 0)), 0.15)
    # soften the cut-out edge so it reads as a photographed object
    alpha = alpha.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.6)))
    x = random.randint(-side // 12, size - side + side // 12)
    y = random.randint(-side // 12, size - side + side // 12)
    canvas = bg.copy()
    canvas.paste(rgb, (x, y), alpha)
    return canvas


def photo_like(im):
    """Degradations shared by composited artwork and real photos."""
    if random.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.2)))
    if random.random() < 0.25:  # motion blur
        k = random.choice([3, 5, 7]); kernel = [0.0] * (k * k)
        if random.random() < 0.5:
            for i in range(k): kernel[(k // 2) * k + i] = 1.0 / k
        else:
            for i in range(k): kernel[i * k + k // 2] = 1.0 / k
        im = im.filter(ImageFilter.Kernel((k, k), kernel, scale=1))
    if random.random() < 0.3:  # sensor noise
        arr = np.asarray(im).astype(np.float32)
        arr += np.random.normal(0, random.uniform(2, 10), arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if random.random() < 0.5:  # JPEG round trip
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=random.randint(35, 90)); buf.seek(0)
        im = Image.open(buf).convert('RGB')
    if random.random() < 0.2:  # screen moiré / scanline hint
        arr = np.asarray(im).astype(np.float32)
        period = random.choice([2, 3, 4]); amp = random.uniform(4, 14)
        arr[::period] -= amp
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if random.random() < 0.15:
        im = ImageOps.grayscale(im).convert('RGB')
    return im


class V2Dataset(Dataset):
    def __init__(self, rows, index, size, train, backgrounds=None):
        self.rows, self.index, self.size, self.train = rows, index, size, train
        self.backgrounds = backgrounds
        self.color = transforms.ColorJitter(0.35, 0.35, 0.3, 0.04)
        self.to_tensor = transforms.ToTensor()
        self.erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0.5)

    def __len__(self):
        return len(self.rows)

    def _train_image(self, r):
        if r['kind'] == 'art':
            with Image.open(r['path']) as im:
                art = im.convert('RGBA')
            if random.random() < 0.8:
                im = composite_artwork(art, self.backgrounds, self.size + 32)
            else:  # keep some clean artwork so the easy case stays easy
                box = art.getchannel('A').getbbox() or (0, 0) + art.size
                art = art.crop(box); w, h = art.size; side = int(max(w, h) * random.uniform(1.05, 1.4))
                bg = Image.new('RGB', (side, side), tuple(int(v) for v in np.random.randint(200, 256, 3)))
                bg.paste(art.convert('RGB'), ((side - w) // 2, (side - h) // 2), art.getchannel('A'))
                im = bg.resize((self.size + 32, self.size + 32), Image.BILINEAR)
            im = photo_like(im)
            im = TF.center_crop(im, self.size) if random.random() < 0.3 else transforms.RandomResizedCrop(self.size, scale=(0.55, 1.0), ratio=(0.85, 1.18))(im)
        else:  # photo or negative
            with Image.open(r['path']) as src:
                im = src.convert('RGB')
            im = transforms.RandomResizedCrop(self.size, scale=(0.45, 1.0), ratio=(0.8, 1.25))(im)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)
            im = photo_like(im)
        im = self.color(im)
        return im

    def _eval_image(self, r):
        with Image.open(r['path']) as src:
            if r['kind'] == 'art':
                art = src.convert('RGBA')
                im = Image.new('RGB', art.size, (255, 255, 255)); im.paste(art.convert('RGB'), (0, 0), art.getchannel('A'))
            else:
                im = src.convert('RGB')
        im = TF.resize(im, self.size); im = TF.center_crop(im, self.size)
        return im

    def __getitem__(self, i):
        r = self.rows[i]
        im = self._train_image(r) if self.train else self._eval_image(r)
        x = self.to_tensor(im)
        if self.train:
            x = self.erase(x)
        target = self.index[r['id']] if r['id'] is not None else -1
        return x, target


# ----------------------------------------------------------------------------- training
def evaluate(model, loader, device, temperature=1.0):
    model.eval()
    stats = defaultdict(lambda: [0, 0, 0.0])  # kind -> [correct, total, maxprob sum]
    logits_all, targets_all = [], []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for x, y, kinds in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            logits = model(x).float()
            probs = torch.softmax(logits / temperature, dim=1)
            top = probs.max(dim=1)
            for k in range(y.numel()):
                kind = kinds[k]
                stats[kind][1] += 1
                stats[kind][2] += top.values[k].item()
                if y[k] >= 0:
                    stats[kind][0] += int(top.indices[k].item() == y[k].item())
                    logits_all.append(logits[k].cpu()); targets_all.append(int(y[k]))
    out = {kind: dict(top1=(c / n if kind != 'neg' else None), n=n, mean_max_prob=s / n) for kind, (c, n, s) in stats.items()}
    return out, (torch.stack(logits_all) if logits_all else None, torch.tensor(targets_all))


def fit_temperature(logits, targets):
    """Single-scalar temperature scaling on validation logits."""
    if logits is None or len(targets) == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(logits / log_t.exp(), targets); loss.backward(); return loss
    opt.step(closure)
    return float(log_t.exp().item())


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


class EvalWrap(Dataset):
    def __init__(self, inner):
        self.inner = inner
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, i):
        x, y = self.inner[i]
        return x, y, self.inner.rows[i]['kind']


def collate_eval(batch):
    xs, ys, kinds = zip(*batch)
    return torch.stack(xs), torch.tensor(ys), list(kinds)


def train(args):
    device = torch.device('cuda')
    seed_all(args.seed)
    labels = load_labels()
    index = {x['id']: i for i, x in enumerate(labels)}
    manifest = V2 / 'manifest.jsonl'
    digest = hashlib.sha256(manifest.read_bytes())
    tr, va = rows_from(manifest, 'train'), rows_from(manifest, 'val')
    for extra in args.extra_manifest:
        digest.update(extra.read_bytes())
        tr += rows_from(extra, 'train'); va += rows_from(extra, 'val')
        print(f'extra manifest {extra}: +{len(rows_from(extra, "train"))} train, +{len(rows_from(extra, "val"))} val', flush=True)
    manifest_sha = digest.hexdigest()
    counts = defaultdict(int)
    for r in tr:
        counts[(r['id'], r['kind'])] += 1
    print(f'train {len(tr)} (art {sum(r["kind"]=="art" for r in tr)}, photo {sum(r["kind"]=="photo" for r in tr)}, '
          f'crawl {sum(r["kind"]=="crawl" for r in tr)}, own {sum(r["kind"]=="own" for r in tr)}, neg {sum(r["kind"]=="neg" for r in tr)}) '
          f'val {len(va)} classes {len(labels)}', flush=True)
    # Balanced sampling: per (species, kind) inverse-sqrt frequency; negatives weighted to ~4 % of batches.
    kind_boost = {'photo': 1.6, 'crawl': 1.3, 'own': 2.5, 'art': 1.0}
    pos_weight = {i: 1.0 / math.sqrt(counts[(r['id'], r['kind'])]) * kind_boost.get(r['kind'], 1.0)
                  for i, r in enumerate(tr) if r['kind'] != 'neg'}
    neg_total = sum(r['kind'] == 'neg' for r in tr)
    neg_share = 0.05  # negatives make up ~5 % of sampled batches
    neg_weight = (neg_share / (1 - neg_share)) * sum(pos_weight.values()) / neg_total if neg_total else 0.0
    weights = [pos_weight.get(i, neg_weight) for i in range(len(tr))]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(tr), replacement=True)
    backgrounds = Backgrounds(V2 / 'bg')
    train_loader = DataLoader(V2Dataset(tr, index, args.size, True, backgrounds), batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0, collate_fn=collate, prefetch_factor=4)
    val_loader = DataLoader(EvalWrap(V2Dataset(va, index, args.size, False)), batch_size=args.batch * 2, shuffle=False, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate_eval)
    model = make_model(args.arch, len(labels), pretrained=True)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if init['labels'] != labels or init['config']['arch'] != args.arch:
            raise ValueError('init checkpoint labels/arch mismatch')
        model.load_state_dict(init['model'], strict=True)
        print(f'initialised from {args.init_checkpoint}', flush=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    config = dict(arch=args.arch, size=args.size, classes=len(labels), epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed,
                  recipe='v2-composite-photo-oe', oe_weight=args.oe_weight, ema_decay=args.ema_decay,
                  extra_manifests=[str(x) for x in args.extra_manifest], init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'environment.json').write_text(json.dumps(dict(
        python=os.sys.version, torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), parameters=sum(p_.numel() for p_ in model.parameters()), config=config, manifest_sha256=manifest_sha), indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_loader); warmup = max(1, 2 * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ce = torch.nn.CrossEntropyLoss()
    start, best = 0, -1.0
    if args.resume and (args.out / 'last.pt').exists():
        state = torch.load(args.out / 'last.pt', map_location='cpu', weights_only=True)
        if state['manifest_sha256'] != manifest_sha:
            raise ValueError('resume manifest mismatch')
        model.load_state_dict(state['model_raw']); ema.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start, best = state['epoch'], state['best_val']
        print(f'resumed at epoch {start}, best score {best:.4f}', flush=True)
    began = time.time()
    for epoch in range(start, args.epochs):
        model.train(); tick = time.time(); loss_sum = seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            pos = y >= 0
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = ce(logits[pos], y[pos]) if pos.any() else logits.sum() * 0
            if (~pos).any():  # outlier exposure: push negatives toward uniform
                logp = torch.log_softmax(logits[~pos], dim=1)
                loss = loss + args.oe_weight * (-(logp.mean(dim=1))).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); scheduler.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema_decay).add_(pm.detach(), alpha=1 - args.ema_decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            loss_sum += loss.item() * y.numel(); seen += y.numel()
        scores, (vl, vt) = evaluate(ema, val_loader, device)
        art = scores.get('art', {}).get('top1') or 0.0
        photo = scores.get('photo', {}).get('top1')
        own = scores.get('own', {}).get('top1')
        # Selection score favours what the camera sees: own photos, then web photos, then artwork.
        parts = [(0.3, art)] + ([(0.4, photo)] if photo is not None else []) + ([(0.6, own)] if own is not None else [])
        score = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / max(seen, 1), val=scores, score=score, seconds=time.time() - tick,
                   elapsed_seconds=time.time() - began, lr=scheduler.get_last_lr()[0])
        with (args.out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        payload = dict(model={k: v.detach().cpu() for k, v in ema.state_dict().items()}, config=config, labels=labels,
                       manifest_sha256=manifest_sha, epoch=epoch + 1, best_val=max(best, score))
        temp = args.out / 'last.partial'
        torch.save(dict(payload, model_raw={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temp)
        temp.replace(args.out / 'last.pt')
        if score > best:
            best = score
            temperature = fit_temperature(vl, vt)
            payload['config'] = dict(config, temperature=temperature, val=scores)
            torch.save(payload, args.out / 'best.pt')
            print(f'  new best {score:.4f}; temperature {temperature:.3f}', flush=True)
        if args.max_minutes and (time.time() - began) / 60 > args.max_minutes:
            print('time budget reached; resume with --resume', flush=True); break
    (args.out / 'run_status.json').write_text(json.dumps(dict(best_score=best, epochs_completed=epoch + 1, elapsed_seconds=time.time() - began), indent=2))


def predict(args):
    device = torch.device('cuda')
    labels = load_labels(); index = {x['id']: i for i, x in enumerate(labels)}
    rows = rows_from(V2 / 'manifest.jsonl', args.split)
    for extra in args.extra_manifest:
        rows += rows_from(extra, args.split)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['labels'] != labels:
        raise ValueError('checkpoint labels mismatch')
    model = make_model(state['config']['arch'], len(labels)).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(state['model']); model.eval()
    T = state['config'].get('temperature', 1.0)
    loader = DataLoader(EvalWrap(V2Dataset(rows, index, state['config']['size'], False)), batch_size=128, num_workers=args.workers, collate_fn=collate_eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_class = defaultdict(lambda: [0, 0]); kind_stats = defaultdict(lambda: [0, 0]); neg_max = []
    with args.out.open('w') as f, torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        offset = 0
        for x, y, kinds in loader:
            logits = model(x.to(device).contiguous(memory_format=torch.channels_last)).float()
            probs = torch.softmax(logits, dim=1)          # app semantics: T = 1
            top = probs.topk(5, dim=1)
            for k in range(y.numel()):
                r = rows[offset + k]
                ids = [labels[i]['id'] for i in top.indices[k].tolist()]
                kind_stats[kinds[k]][1] += 1
                if r['id'] is None:
                    neg_max.append(top.values[k][0].item())
                else:
                    hit = int(ids[0] == r['id']); kind_stats[kinds[k]][0] += hit
                    per_class[r['id']][1] += 1; per_class[r['id']][0] += hit
                f.write(json.dumps(dict(path=r['path'], kind=kinds[k], truth=r['id'], top5=ids, probs=[round(v, 5) for v in top.values[k].tolist()])) + '\n')
            offset += y.numel()
    acc = {d: c / n for d, (c, n) in per_class.items()}
    summary = dict(rows=len(rows), temperature=T,
                   by_kind={k: dict(top1=(c / n if k != 'neg' else None), n=n) for k, (c, n) in kind_stats.items()},
                   negatives_max_prob=dict(mean=float(np.mean(neg_max)) if neg_max else None, p95=float(np.percentile(neg_max, 95)) if neg_max else None),
                   worst_species=sorted(acc.items(), key=lambda kv: kv[1])[:25],
                   suggested_app_threshold='pick a confidence just above negatives_max_prob.p95 (PokemonClassifier currently uses 0.25)')
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'predict'])
    p.add_argument('--arch', choices=['mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_v2_s'], default='efficientnet_v2_s')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', default='val')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=96)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--oe-weight', type=float, default=0.5)
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--max-minutes', type=float, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--extra-manifest', type=Path, action='append', default=[], help='additional manifest(s) from filter_crawl.py')
    p.add_argument('--init-checkpoint', type=Path, help='warm-start weights (same arch and labels)')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required')
    train(args) if args.action == 'train' else predict(args)


if __name__ == '__main__':
    main()
