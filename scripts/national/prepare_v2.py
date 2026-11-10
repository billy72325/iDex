"""Build the v2 manifest: artwork with alpha for compositing, real photos, and negatives.

Sources (all fetched by run_v2.sh):
  artifacts/national/sources/JJMack--pokemon-classification-gen1-9/*.parquet   1,025 species artwork + sprites
  artifacts/replacement/sources/keremberke--pokemon-classification/*.zip        web photos, Gen I (public domain per card)
  artifacts/replacement/sources/sgtsaughter--pokemon-classification-images-151/PokemonData.zip   web photos, Gen I (Apache-2.0 per card)
  artifacts/replacement/sources/coco128/coco128.zip                             128 natural photos: negatives + backgrounds

Output under artifacts/national/v2/:
  art/<id>/<hash>.png     RGBA artwork (alpha kept when the source has it, else a white-key mask)
  photo/<id>/<hash>.jpg   real photos, EXIF-upright, long edge 320
  neg/<hash>.jpg          negatives (no Pokémon), long edge 320
  bg/<hash>.jpg           backgrounds for compositing (same COCO images, long edge 512)
  manifest.jsonl          {id|null, kind: art|photo|neg, path, group, split}
  summary.json
"""
import argparse
import hashlib
import io
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from labels import load_labels

ART = Path('artifacts/national/sources/JJMack--pokemon-classification-gen1-9')
REPL = Path('artifacts/replacement/sources')
OUT = Path('artifacts/national/v2')


def normalize(s):
    s = str(s).lower().replace('♀', 'f').replace('♂', 'm').replace('female', 'f').replace('male', 'm')
    return re.sub('[^a-z0-9]', '', s)


def usable(path):
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def white_key_alpha(rgb, threshold=238):
    """Alpha from near-white pixels for artwork that lost its transparency upstream."""
    gray = rgb.convert('L')
    mask = gray.point(lambda v: 0 if v >= threshold else 255)
    # Keep the mask only when it looks like a figure on white (10–90 % coverage).
    coverage = sum(mask.histogram()[255:]) / (mask.width * mask.height)
    if not 0.08 <= coverage <= 0.92:
        return None
    return mask


def save_art(raw, target):
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        if src.mode in ('RGBA', 'LA') or (src.mode == 'P' and 'transparency' in src.info):
            im = src.convert('RGBA')
            alpha = im.getchannel('A')
            if alpha.getextrema()[0] < 255:          # genuine transparency
                rgb = Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), im).convert('RGB')
                out = rgb.copy(); out.putalpha(alpha)
            else:
                rgb = im.convert('RGB'); out = rgb.copy(); out.putalpha(white_key_alpha(rgb) or Image.new('L', rgb.size, 255))
        else:
            rgb = src.convert('RGB'); out = rgb.copy(); out.putalpha(white_key_alpha(rgb) or Image.new('L', rgb.size, 255))
    w, h = out.size
    side = max(w, h)
    canvas = Image.new('RGBA', (side, side), (255, 255, 255, 0))
    canvas.paste(out, ((side - w) // 2, (side - h) // 2))
    if side > 288:
        canvas = canvas.resize((288, 288), Image.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp.png'); canvas.save(tmp, optimize=True); tmp.replace(target)


def save_photo(raw, target, long_edge):
    with Image.open(io.BytesIO(raw)) as src:
        im = ImageOps.exif_transpose(src).convert('RGBA')
        im = Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), im).convert('RGB')
    w, h = im.size
    scale = long_edge / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp.jpg'); im.save(tmp, quality=92); tmp.replace(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--val-fraction', type=float, default=0.15)
    a = p.parse_args()
    labels = load_labels()
    by_name = {normalize(x['name']): x['id'] for x in labels}
    by_name.update({'nidoranf': 29, 'nidoranm': 32, 'mrmime': 122, 'farfetchd': 83, 'mimejr': 439, 'typenull': 772,
                    'flabebe': 669, 'sirfetchd': 865, 'mrrime': 866, 'porygonz': 474, 'hooh': 250, 'jangmoo': 782,
                    'hakamoo': 783, 'kommoo': 784, 'tapukoko': 785, 'tapulele': 786, 'tapubulu': 787, 'tapufini': 788})
    known = {x['id'] for x in labels}
    rows, seen, errors = [], set(), []
    stats = defaultdict(int)

    # --- artwork (RGBA) ---
    for file in sorted(ART.glob('*.parquet')):
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=64):
            for item in batch.to_pylist():
                try:
                    dex = int(item['label'])
                    if dex not in known:
                        continue
                    image = item.get('image_data', item.get('image'))
                    raw = image if isinstance(image, bytes) else image.get('bytes')
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    filename = item.get('file_name', '') or ''
                    group = re.sub(r'-s(?=\.[a-z]+$)', '', filename) or digest[:16]
                    target = OUT / 'art' / f'{dex:04d}' / f'{digest[:16]}.png'
                    if not usable(target):
                        save_art(raw, target)
                    rows.append(dict(id=dex, kind='art', path=str(target), group=f'art:{dex}:{group}', sha256=digest))
                    stats['art'] += 1
                except Exception as e:
                    errors.append(dict(source=file.name, error=str(e)))
        print(f'{file.name}: {stats["art"]} artwork', flush=True)

    # --- real photos (Gen I) and negatives ---
    archives = [(z, 'sgtsaughter') for z in sorted((REPL / 'sgtsaughter--pokemon-classification-images-151').glob('*.zip'))]
    archives += [(z, 'keremberke') for z in sorted((REPL / 'keremberke--pokemon-classification').glob('*.zip'))]
    archives += [(z, 'coco128') for z in sorted((REPL / 'coco128').glob('*.zip'))]
    unmapped = defaultdict(int)
    for archive, source in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                member = PurePosixPath(info.filename)
                if member.is_absolute() or '..' in member.parts or member.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    continue
                if info.file_size > 30_000_000 or info.file_size == 0:
                    continue
                try:
                    raw = z.read(info)
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    if source == 'coco128':
                        neg = OUT / 'neg' / f'{digest[:16]}.jpg'
                        bg = OUT / 'bg' / f'{digest[:16]}.jpg'
                        if not usable(neg):
                            save_photo(raw, neg, 320)
                        if not usable(bg):
                            save_photo(raw, bg, 512)
                        rows.append(dict(id=None, kind='neg', path=str(neg), group=f'neg:{digest[:16]}', sha256=digest))
                        stats['neg'] += 1
                        continue
                    dex = by_name.get(normalize(member.parent.name))
                    if dex is None:
                        unmapped[member.parent.name] += 1
                        continue
                    original = re.split(r'_(?:jpg|png|jpeg)\.rf\.', member.name)[0]
                    target = OUT / 'photo' / f'{dex:04d}' / f'{digest[:16]}.jpg'
                    if not usable(target):
                        save_photo(raw, target, 320)
                    rows.append(dict(id=dex, kind='photo', path=str(target), group=f'{source}:{dex}:{original}', sha256=digest, source=source))
                    stats['photo'] += 1
                except Exception as e:
                    errors.append(dict(source=f'{archive.name}:{info.filename}', error=str(e)))
        print(f'{archive.name}: photos {stats["photo"]} negatives {stats["neg"]}', flush=True)

    if stats['art'] == 0:
        raise SystemExit('no artwork decoded; run acquire first')

    # --- grouped split per (species, kind) so art-val and photo-val are both measurable ---
    rng = random.Random(a.seed)
    buckets = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        buckets[(r['id'], r['kind'])][r['group']].append(i)
    for key, groups in buckets.items():
        names = sorted(groups); rng.shuffle(names)
        n_val = max(1, round(len(names) * a.val_fraction)) if len(names) >= 4 else 0
        for k, name in enumerate(names):
            for i in groups[name]:
                rows[i]['split'] = 'val' if k < n_val else 'train'
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'prepare_errors.json').write_text(json.dumps(errors, indent=2))
    photo_species = {r['id'] for r in rows if r['kind'] == 'photo'}
    summary = dict(
        artwork=stats['art'], photos=stats['photo'], negatives=stats['neg'], backgrounds=len(list((OUT / 'bg').glob('*.jpg'))),
        species_with_photos=len(photo_species), unmapped_photo_folders=dict(sorted(unmapped.items())),
        train=sum(r['split'] == 'train' for r in rows), val=sum(r['split'] == 'val' for r in rows),
        val_art=sum(r['split'] == 'val' and r['kind'] == 'art' for r in rows),
        val_photo=sum(r['split'] == 'val' and r['kind'] == 'photo' for r in rows),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), errors=len(errors))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
