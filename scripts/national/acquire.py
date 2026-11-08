"""Download the pinned gen 1–9 dataset and TorchVision ImageNet weights. Nothing is executed.

Everything lands under artifacts/national/sources/ with a SHA-256 inventory.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from model import PRETRAINED

DATASET = 'JJMack/pokemon-classification-gen1-9'
REVISION = '959ff557b102fc3b4f912d0ca769273be5b9ca7f'
# Recorded by the original campaign (scripts/replacement/acquire.py).
EXPECTED = {
    'train.parquet': '74e32dc22e51789fb46b4873c86a1a1b4dcdbe77a1f30a379fd01597784b4314',
    'test.parquet': '4b812cf954e8e69479eaeb0f8c88d24b9d0796c898d88f76867c82fba1d74313',
    'validation.parquet': '6d0022b9d0bd12dd10a56ecc86f519d2968cbb384c6f396014a93ba36e9c8633',
}
ROOT = Path('artifacts/national/sources')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.partial')
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'iDex-national-training/1.0'})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                partial.replace(target)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + attempt * 3)
    digest = sha(target)
    if expected and digest != expected:
        raise ValueError(f'checksum mismatch for {target}: {digest}')
    row = dict(path=str(target), url=url, bytes=target.stat().st_size, sha256=digest,
               retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['all', 'data', 'pretrained'], nargs='?', default='all')
    a = p.parse_args()
    records = []
    if a.what in ('all', 'data'):
        folder = ROOT / DATASET.replace('/', '--')
        records.append(fetch(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/README.md', folder / 'README.md'))
        for name, digest in EXPECTED.items():
            records.append(fetch(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}', folder / name, digest))
    if a.what in ('all', 'pretrained'):
        for arch, (name, url) in PRETRAINED.items():
            records.append(fetch(url, ROOT / 'torchvision' / name))
    inventory = ROOT / 'inventory.jsonl'
    with inventory.open('a') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f'{len(records)} files recorded in {inventory}')


if __name__ == '__main__':
    main()
