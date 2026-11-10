"""Crawl web photos of merchandise (figures, plush, cards) for every species.

Personal-research use only: the images stay in artifacts/ (ignored by git) and
are never redistributed. Search engines' terms apply; keep the rate modest.

    python scripts/national/crawl.py --per-species 60 --workers 6
    python scripts/national/crawl.py --ids 25,133,1025 --per-species 100   # a few species

Output: artifacts/national/crawl/<id>/<n>.jpg plus crawl_log.jsonl (one line per
species: queries run, images saved). Resumable: species that already hold at
least --per-species images are skipped.
"""
import argparse
import hashlib
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request
from PIL import Image

from labels import load_labels

OUT = Path('artifacts/national/crawl')
CATALOG = Path('iDex/Resources/pokemon-national.json')
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'

# Merchandise-oriented queries: what a phone camera actually sees.
QUERY_TEMPLATES_EN = ['{en} pokemon figure', '{en} pokemon plush', '{en} pokemon card', '{en} pokemon toy photo']
QUERY_TEMPLATES_JA = ['{ja} ぬいぐるみ', '{ja} フィギュア', '{ja} ポケモンカード']


def names():
    cat = {e['id']: e for e in json.loads(CATALOG.read_text(encoding='utf-8'))}
    return {x['id']: (cat[x['id']]['englishName'], cat[x['id']]['japaneseName']) for x in load_labels() if x['id'] in cat}


def search_urls(query, limit):
    """Image URLs for a query from DuckDuckGo (no key needed)."""
    urls = []
    try:
        from ddgs import DDGS  # pip install ddgs
        with DDGS() as d:
            for r in d.images(query, max_results=limit, safesearch='moderate'):
                if r.get('image'):
                    urls.append(r['image'])
    except Exception as e:
        print(f'  ddgs failed for {query!r}: {e}', file=sys.stderr)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def bing_fallback(query, limit, folder, hashes, per_species, saved):
    """When DuckDuckGo returns little, let icrawler download from Bing into a temp dir."""
    try:
        import shutil, tempfile
        from icrawler.builtin import BingImageCrawler  # pip install icrawler
        tmp = Path(tempfile.mkdtemp(prefix='bing-'))
        crawler = BingImageCrawler(storage={'root_dir': str(tmp)}, log_level=40, feeder_threads=1, parser_threads=1, downloader_threads=4)
        crawler.crawl(keyword=query, max_num=limit, filters=dict(type='photo'), min_size=(160, 160))
        for f in sorted(tmp.iterdir()):
            if saved >= per_species:
                break
            try:
                data = fetch_image_bytes(f.read_bytes())
            except Exception:
                continue
            if not data:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            (folder / f'{digest[:16]}.jpg').write_bytes(data)
            saved += 1
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        print(f'  bing failed for {query!r}: {e}', file=sys.stderr)
    return saved


def fetch_image_bytes(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        if min(im.size) < 160 or max(im.size) > 6000:
            return None
        im = im.convert('RGB')
        w, h = im.size
        scale = 640 / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=90)
        return buf.getvalue()


def fetch_image(url, timeout=12):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': 'https://www.bing.com/'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if int(r.headers.get('Content-Length', 0)) > 15_000_000:
            return None
        raw = r.read(15_000_000)
    return fetch_image_bytes(raw)


def crawl_species(dex, en, ja, per_species, log):
    folder = OUT / f'{dex:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    have = len(list(folder.glob('*.jpg')))
    if have >= per_species:
        return dict(id=dex, skipped=True, images=have)
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.jpg')}
    queries = [t.format(en=en) for t in QUERY_TEMPLATES_EN] + [t.format(ja=ja) for t in QUERY_TEMPLATES_JA]
    random.shuffle(queries)
    saved, tried = have, 0
    for q in queries:
        if saved >= per_species:
            break
        urls = search_urls(q, max(12, per_species // 3))
        for u in urls:
            if saved >= per_species:
                break
            tried += 1
            try:
                data = fetch_image(u)
            except Exception:
                continue
            if not data:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            (folder / f'{digest[:16]}.jpg').write_bytes(data)
            saved += 1
        time.sleep(random.uniform(0.6, 1.6))
    if saved < per_species // 2:
        for q in queries[:2]:
            saved = bing_fallback(q, per_species, folder, hashes, per_species, saved)
            if saved >= per_species:
                break
    row = dict(id=dex, name=en, queries=queries, tried=tried, images=saved)
    with log.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'#{dex:04d} {en}: {saved} images ({tried} tried)', flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--per-species', type=int, default=60)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--ids', help='comma-separated National Dex ids; default all')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'crawl_log.jsonl'
    table = names()
    ids = [int(x) for x in a.ids.split(',')] if a.ids else sorted(table)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda d: crawl_species(d, table[d][0], table[d][1], a.per_species, log), ids))
    total = sum(r['images'] for r in results)
    thin = [r['id'] for r in results if r['images'] < 10]
    print(json.dumps(dict(species=len(results), images=total, under_10=thin), indent=2))


if __name__ == '__main__':
    main()
