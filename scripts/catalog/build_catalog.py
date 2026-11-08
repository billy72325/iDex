#!/usr/bin/env python3
"""Rebuild the offline Kanto catalog from cached upstream CSVs and sprites."""
import concurrent.futures
import csv
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.catalog-cache'
RESOURCES = ROOT / 'iDex' / 'Resources'
CSV_BASE = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/'
MAX_ID = 1025
FILES = ['pokemon.csv', 'pokemon_species.csv', 'pokemon_species_names.csv', 'pokemon_species_flavor_text.csv', 'pokemon_types.csv', 'types.csv', 'pokemon_stats.csv', 'stats.csv']

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'iDex-development-catalog/1.0'})
            data = urllib.request.urlopen(request, timeout=40).read()
            path.write_bytes(data)
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def rows(file):
    return list(csv.DictReader(io.StringIO(download(CSV_BASE + file, CACHE / file).decode('utf-8'))))

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        csvs = dict(zip(FILES, pool.map(rows, FILES)))
    species_names = {(int(r['pokemon_species_id']), int(r['local_language_id'])): r for r in csvs['pokemon_species_names.csv']}
    type_names = {r['id']: r['identifier'] for r in csvs['types.csv']}
    generation_of = {int(r['id']): int(r['generation_id']) for r in csvs['pokemon_species.csv']}
    stat_names = {r['id']: r['identifier'] for r in csvs['stats.csv']}
    flavor = {}
    # Retain the latest available text in each language (CSV is ordered by version id).
    for r in csvs['pokemon_species_flavor_text.csv']:
        if int(r['species_id']) <= MAX_ID and r['language_id'] in ('1', '4', '9', '11', '12'):
            normalized = ' '.join(r['flavor_text'].replace('\f', ' ').split())
            flavor[(int(r['species_id']), int(r['language_id']))] = normalized.replace(' ', '') if r['language_id'] in ('4', '12') else normalized
    entries = []
    slug_by_id = {}
    for p in csvs['pokemon.csv']:
        number = int(p['id'])
        if not 1 <= number <= MAX_ID:
            continue
        en = species_names[(number, 9)]
        zh = species_names[(number, 12)]
        zht = species_names[(number, 4)]
        ja = species_names[(number, 11)]
        slug_by_id[number] = p['identifier']
        entry = {
            'id': number,
            'englishName': en['name'],
            'chineseName': zh['name'],
            'traditionalChineseName': zht['name'],
            'japaneseName': ja['name'],
            'types': [type_names[t['type_id']] for t in csvs['pokemon_types.csv'] if int(t['pokemon_id']) == number],
            'height': int(p['height']) / 10,
            'weight': int(p['weight']) / 10,
            'englishDescription': flavor[(number, 9)],
            # Upstream lacks Chinese text for 303 species and Japanese text for 127; fall back to English
            # (Chinese fallbacks are hand-translated in the committed catalog; keep those when rebuilding).
            'chineseDescription': flavor.get((number, 12), flavor[(number, 9)]),
            'traditionalChineseDescription': flavor.get((number, 4), flavor[(number, 9)]),
            'japaneseDescription': flavor.get((number, 11), flavor.get((number, 1), flavor[(number, 9)])),
            'englishGenus': en['genus'],
            'chineseGenus': zh['genus'],
            'traditionalChineseGenus': zht['genus'],
            'japaneseGenus': ja['genus'],
            'stats': [{'name': stat_names[s['stat_id']], 'value': int(s['base_stat'])} for s in csvs['pokemon_stats.csv'] if int(s['pokemon_id']) == number],
            'generation': generation_of[number]
        }
        entries.append(entry)
    assert len(entries) == MAX_ID
    assert all(p['chineseDescription'] and p['traditionalChineseName'] and p['traditionalChineseDescription'] and p['japaneseName'] and p['japaneseDescription']
               and p['japaneseGenus'] and len(p['stats']) == 6 for p in entries)
    RESOURCES.mkdir(parents=True, exist_ok=True)
    (RESOURCES / 'pokemon-national.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    assets = RESOURCES / 'PokemonSprites.xcassets'
    assets.mkdir(exist_ok=True)
    (assets / 'Contents.json').write_text(json.dumps({'info': {'author': 'xcode', 'version': 1}}, indent=2))
    def sprite(p):
        number = p['id']
        asset = assets / f'pokemon-{number:03d}.imageset'
        path = asset / 'sprite.png'
        slug = slug_by_id[number]
        pokeapi = f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{number}.png'
        # Pokémon Database black-white sprites stop at Gen 5 (#649); later species use PokéAPI default sprites.
        urls = [f'https://img.pokemondb.net/sprites/black-white/normal/{slug}.png', pokeapi] if number <= 649 else [pokeapi]
        for source in urls:
            try:
                data = download(source, path)
                if not data.startswith(b'\x89PNG'):
                    path.unlink()
                    raise ValueError('Not a PNG')
                break
            except Exception:
                if source == urls[-1]:
                    raise
        (asset / 'Contents.json').write_text(json.dumps({'images': [{'filename': 'sprite.png', 'idiom': 'universal'}], 'info': {'author': 'xcode', 'version': 1}}, indent=2))
        return number, source
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        sources = dict(pool.map(sprite, entries))
    (RESOURCES / 'sprite-sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    font_dir = RESOURCES / 'Fonts'
    font_dir.mkdir(exist_ok=True)
    for filename in ['PressStart2P-Regular.ttf', 'OFL.txt']:
        target = 'PressStart2P-OFL.txt' if filename == 'OFL.txt' else filename
        download('https://raw.githubusercontent.com/google/fonts/main/ofl/pressstart2p/' + filename, font_dir / target)
    fusion_url = 'https://github.com/TakWolf/fusion-pixel-font/releases/download/2026.10.16/fusion-pixel-font-12px-proportional-otf-v2026.10.16.zip'
    archive = download(fusion_url, CACHE / 'fusion-pixel-font.zip')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in zipped.namelist():
            if name.endswith(('zh_hans.otf', 'zh_tw.otf', '-ja.otf', 'OFL.txt', 'LICENSE.txt')):
                output_name = 'FusionPixel-OFL.txt'
                if name.endswith('zh_hans.otf'):
                    output_name = 'FusionPixel-12px-zh-Hans.otf'
                elif name.endswith('zh_tw.otf'):
                    output_name = 'FusionPixel-12px-zh-Hant.otf'
                elif name.endswith('-ja.otf'):
                    output_name = 'FusionPixel-12px-ja.otf'
                output = font_dir / output_name
                output.write_bytes(zipped.read(name))
    print(f'Built {len(entries)} English, Simplified Chinese, Traditional Chinese, and Japanese entries and {len(sources)} sprites.')
    print('Pikachu:', entries[24])
    print('Fonts:', [p.name for p in font_dir.iterdir()])

if __name__ == '__main__':
    main()
