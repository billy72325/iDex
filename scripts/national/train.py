"""Train / evaluate the 1025-class National Pokédex classifier on one CUDA GPU.

Checkpoints use the same schema as scripts/replacement (model, config, labels,
manifest_sha256) so export.py and the Apple validation harness keep working.

    python train.py train --arch efficientnet_b0 --epochs 20 --out artifacts/national/runs/effb0
    python train.py predict --arch efficientnet_b0 --checkpoint artifacts/national/runs/effb0/best.pt --out artifacts/national/predictions/effb0-val.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import load_labels, make_model

DATA = Path('artifacts/national/data')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rows_from(manifest, split):
    wanted = set(split.split(','))
    return [r for r in (json.loads(s) for s in manifest.read_text().splitlines()) if r['split'] in wanted]


class Shards(Dataset):
    def __init__(self, rows, index, size, train):
        self.rows = rows
        self.index = index
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), fill=255),
                transforms.ColorJitter(0.3, 0.3, 0.25, 0.03),
                transforms.RandomGrayscale(0.03),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, value=1.0),
            ])
        else:
            self.tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['path']) as im:
            x = self.tf(im.convert('RGB'))
        return x, self.index[r['id']]


def evaluate(model, loader, device):
    model.eval()
    total = correct = top5 = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x).float()
            loss_sum += criterion(logits, y).item()
            ranked = logits.topk(5, dim=1).indices
            correct += (ranked[:, 0] == y).sum().item()
            top5 += (ranked == y[:, None]).any(dim=1).sum().item()
            total += y.numel()
    return dict(loss=loss_sum / total, top1=correct / total, top5=top5 / total, n=total)


def train(args):
    device = torch.device('cuda')
    seed_all(args.seed)
    labels = load_labels()
    index = {x['id']: i for i, x in enumerate(labels)}
    manifest = DATA / 'manifest.jsonl'
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    tr, va = rows_from(manifest, 'train'), rows_from(manifest, 'val')
    print(f'train {len(tr)} val {len(va)} classes {len(labels)}', flush=True)
    counts = defaultdict(int)
    for r in tr:
        counts[r['id']] += 1
    # Class-balanced sampling softens the long tail (some species have very few images).
    weights = torch.tensor([1.0 / counts[r['id']] ** 0.5 for r in tr], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(tr), replacement=True)
    train_loader = DataLoader(Shards(tr, index, args.size, True), batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(Shards(va, index, args.size, False), batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    model = make_model(args.arch, len(labels), pretrained=True).to(device).to(memory_format=torch.channels_last)
    config = dict(arch=args.arch, size=args.size, classes=len(labels), epochs=args.epochs, lr=args.lr, batch=args.batch,
                  label_smoothing=args.label_smoothing, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'environment.json').write_text(json.dumps(dict(
        python=os.sys.version, torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), parameters=sum(p.numel() for p in model.parameters()), config=config,
        manifest_sha256=manifest_sha), indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    warmup = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    start, best = 0, 0.0
    if args.resume and (args.out / 'last.pt').exists():
        state = torch.load(args.out / 'last.pt', map_location='cpu', weights_only=True)
        if state['manifest_sha256'] != manifest_sha:
            raise ValueError('resume manifest mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start, best = state['epoch'], state['best_val']
        print(f'resumed at epoch {start}, best top1 {best:.4f}', flush=True)
    began = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        tick = time.time()
        loss_sum = seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); scheduler.step()
            loss_sum += loss.item() * y.numel(); seen += y.numel()
        scores = evaluate(model, val_loader, device)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / seen, val=scores, seconds=time.time() - tick,
                   elapsed_seconds=time.time() - began, lr=scheduler.get_last_lr()[0])
        with (args.out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=config, labels=labels,
                       manifest_sha256=manifest_sha, epoch=epoch + 1, best_val=max(best, scores['top1']))
        temp = args.out / 'last.partial'
        torch.save(dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temp)
        temp.replace(args.out / 'last.pt')
        if scores['top1'] > best:
            best = scores['top1']
            torch.save(payload, args.out / 'best.pt')
        if args.max_minutes and (time.time() - began) / 60 > args.max_minutes:
            print('time budget reached; last.pt can be resumed with --resume', flush=True)
            break
    (args.out / 'run_status.json').write_text(json.dumps(dict(best_top1=best, epochs_completed=epoch + 1,
                                                              elapsed_seconds=time.time() - began), indent=2))


def predict(args):
    device = torch.device('cuda')
    labels = load_labels()
    index = {x['id']: i for i, x in enumerate(labels)}
    rows = rows_from(DATA / 'manifest.jsonl', args.split)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['labels'] != labels:
        raise ValueError('checkpoint labels mismatch')
    model = make_model(state['config']['arch'], len(labels)).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(state['model']); model.eval()
    loader = DataLoader(Shards(rows, index, state['config']['size'], False), batch_size=128, num_workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_class = defaultdict(lambda: [0, 0])
    with args.out.open('w') as f, torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        offset = 0
        for x, y in loader:
            probs = torch.softmax(model(x.to(device).contiguous(memory_format=torch.channels_last)).float(), dim=1)
            top = probs.topk(5, dim=1)
            for k in range(y.numel()):
                r = rows[offset + k]
                ids = [labels[i]['id'] for i in top.indices[k].tolist()]
                per_class[r['id']][1] += 1
                per_class[r['id']][0] += int(ids[0] == r['id'])
                f.write(json.dumps(dict(path=r['path'], truth=r['id'], top5=ids, probs=[round(v, 5) for v in top.values[k].tolist()])) + '\n')
            offset += y.numel()
    acc = {d: c / n for d, (c, n) in per_class.items()}
    worst = sorted(acc.items(), key=lambda kv: kv[1])[:25]
    summary = dict(rows=len(rows), top1=sum(c for c, _ in per_class.values()) / len(rows), worst_species=worst)
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'predict'])
    p.add_argument('--arch', choices=['mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_v2_s'], default='efficientnet_b0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', default='val')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--max-minutes', type=float, default=0, help='0 = no budget')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required (the GX10 / DGX Spark container provides it)')
    train(args) if args.action == 'train' else predict(args)


if __name__ == '__main__':
    main()
