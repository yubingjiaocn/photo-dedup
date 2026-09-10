"""Local-only real-event R&D ingestion, frozen feature baseline and blind panels."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import stat
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
sys.path.insert(0, str(REPO))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def merge_parts(day):
    directory = ROOT / 'acquisition' / day
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    for part_file in directory.glob('manifest-part-*.json'):
        part = json.loads(part_file.read_text())
        index, count = part['partition']
        if len(part['items']) != len(manifest['items']):
            raise ValueError('Partition inventory mismatch')
        for i, item in enumerate(part['items']):
            if i % count != index:
                continue
            if item['id'] != manifest['items'][i]['id']:
                raise ValueError('Partition ID mismatch')
            if item['status'] == 'done':
                manifest['items'][i] = item
    manifest['complete'] = all(i['status'] == 'done' for i in manifest['items'])
    dump(path, manifest)
    print('merged done', sum(i['status'] == 'done' for i in manifest['items']), '/', len(manifest['items']))


def ingest(day):
    source = ROOT / 'acquisition' / day
    manifest = json.loads((source / 'manifest.json').read_text())
    target = ROOT / 'originals' / day
    target.mkdir(parents=True, exist_ok=True)
    payload = source / 'whole-date.zip'
    if not zipfile.is_zipfile(payload):
        if not manifest['items'] or not all(i['status'] == 'done' for i in manifest['items']):
            raise ValueError('Individual download set is incomplete')
        payload = source / 'individual-downloads.zip'
        if not payload.exists():
            with zipfile.ZipFile(payload, 'x', compression=zipfile.ZIP_STORED) as archive:
                for item in manifest['items']:
                    downloaded = Path(item['file'])
                    if not downloaded.is_file() or downloaded.stat().st_size != item['bytes']:
                        raise ValueError('Downloaded media is missing or changed')
                    archive.write(downloaded, downloaded.name)
    records = []
    with zipfile.ZipFile(payload) as archive:
        seen = set()
        for item in archive.infolist():
            if item.is_dir():
                continue
            rel = Path(item.filename)
            mode = item.external_attr >> 16
            if (rel.is_absolute() or '..' in rel.parts or '\\' in item.filename
                    or stat.S_ISLNK(mode) or str(rel) in seen):
                raise ValueError('Unsafe archive member')
            seen.add(str(rel))
            if rel.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.heic', '.mp4', '.mov'}:
                raise ValueError(f'Unexpected media suffix: {rel.suffix}')
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if dest.read_bytes() != archive.read(item):
                    raise ValueError('Existing original differs')
            else:
                with archive.open(item) as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            row = {'file': str(dest), 'bytes': dest.stat().st_size}
            if rel.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                with Image.open(dest) as image:
                    exif = image.getexif()
                    inner = exif.get_ifd(34665) if 34665 in exif else {}
                    row.update(width=image.width, height=image.height,
                               captured=inner.get(36867) or exif.get(306),
                               camera=exif.get(272), kind='photo')
                    image.verify()
            else:
                row['kind'] = 'video' if rel.suffix.lower() in {'.mp4', '.mov'} else 'unsupported_photo'
            records.append(row)
    result = {'date': day, 'expected_ui_ids': len(manifest['items']),
              'downloaded_files': len(records),
              'complete_against_collected_ids': len(records) == len(manifest['items']),
              'counts': dict(Counter(r['kind'] for r in records)), 'files': records}
    dump(ROOT / 'ingestion' / f'{day}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
    if not result['complete_against_collected_ids']:
        raise ValueError('Incomplete whole date; do not claim complete event')


def slice_event(day, index=0):
    manifest = json.loads((ROOT / 'acquisition' / day / 'manifest.json').read_text())
    components, component, previous = [], [], None
    for item in manifest['items']:
        match = re.search(r'(\d{4}年\d+月\d+日 \d+:\d+:\d+)', item.get('label') or '')
        if not match:
            # Unknown entries are barriers, not guessed temporal neighbours.
            # Do not admit the component immediately touching that barrier.
            component = []
            previous = None
            continue
        timestamp = datetime.strptime(match[1], '%Y年%m月%d日 %H:%M:%S').timestamp()
        if previous is not None and abs(timestamp - previous) > 120:
            components.append(component)
            component = []
        component.append(item)
        previous = timestamp
    components.append(component)
    candidates = [c for c in components if 3 <= len(c) <= 20
                  and all(i['status'] == 'done' and i['label'].startswith('照片') for i in c)]
    if index >= len(candidates):
        raise ValueError('Requested complete temporal event not downloaded yet')
    selected = candidates[index]
    slice_name = 'narrow-slice' if index == 0 else f'narrow-slice-{index}'
    dest = ROOT / 'originals' / slice_name
    dest.mkdir(parents=True, exist_ok=True)
    for item in selected:
        target = dest / Path(item['file']).name
        if not target.exists():
            with Path(item['file']).open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    dump(ROOT / f'{slice_name}.json', {'source_date': day, 'split': 'development',
                                     'rule': 'first complete 3-20-frame 120s temporal component in full UI manifest',
                                     'items': selected})
    print('complete narrow event photos:', len(selected))


def prefix_event(day, limit=100, name='development-prefix'):
    manifest = json.loads((ROOT / 'acquisition' / day / 'manifest.json').read_text())
    prefix = []
    for item in manifest['items'][:limit]:
        if item['status'] != 'done':
            break
        with Image.open(item['file']) as image:
            outer = image.getexif()
            inner = outer.get_ifd(34665) if 34665 in outer else {}
            captured = inner.get(36867) or outer.get(306)
        if not captured:
            raise ValueError('Prefix requires native capture timestamps')
        prefix.append((datetime.strptime(captured, '%Y:%m:%d %H:%M:%S').timestamp(), item))
    prefix.sort(key=lambda row: row[0], reverse=True)
    cuts = [i for i in range(1, len(prefix)) if prefix[i - 1][0] - prefix[i][0] > 120]
    if not cuts:
        raise ValueError('No closed temporal boundary in downloaded prefix')
    selected = prefix[:cuts[-1]]  # exclude entire oldest/open-ended component
    dest = ROOT / 'originals' / name
    if dest.exists():
        raise ValueError('Frozen development prefix already exists')
    dest.mkdir(parents=True)
    for _, item in selected:
        with Path(item['file']).open('rb') as src, (dest / Path(item['file']).name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    dump(ROOT / f'{name}.json', {'date': day, 'split': 'development',
                                           'downloaded_contiguous_prefix': len(prefix),
                                           'photos': len(selected), 'open_tail_excluded': len(prefix) - len(selected),
                                           'items': [i for _, i in selected]})
    print('closed development prefix photos:', len(selected))


def pipeline(day):
    from src import config, stage0_inventory, stage1_features, stage2_cluster
    source = ROOT / 'originals' / day
    out = ROOT / 'cache' / day
    out.mkdir(parents=True, exist_ok=True)
    settings = config.load_config().as_dict()
    settings['paths'].update(root=str(source), db=str(out / 'inventory.sqlite'),
                             output_dir=str(out), models_dir=str(REPO / 'models'),
                             trash=str(out / 'unused-trash'))
    settings['features']['backend'] = 'torch'
    settings['features']['thumbnails']['max_px'] = 1024
    settings['cluster']['keeper_score_policy'] = 'per_member'
    cp = out / 'config.yaml'
    if cp.exists() and yaml.safe_load(cp.read_text()) != settings:
        raise ValueError('Frozen config drift')
    cp.write_text(yaml.safe_dump(settings, allow_unicode=True))
    outputs = {}
    for name, run in [('inventory', stage0_inventory.run), ('features', stage1_features.run),
                      ('cluster', stage2_cluster.run)]:
        start = time.perf_counter()
        value = run(config_path=str(cp))
        outputs[name] = {'seconds': time.perf_counter() - start, 'result': value}
        dump(out / 'baseline-run.json', outputs)
    print(json.dumps(outputs, default=str))


def panel(day, split):
    db = ROOT / 'cache' / day / 'inventory.sqlite'
    conn = sqlite3.connect(f'file:{db}?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    groups = conn.execute('SELECT id,group_type,member_count FROM groups WHERE member_count>1 ORDER BY id').fetchall()
    out = ROOT / 'panels' / day
    if (out / 'mapping.json').exists():
        raise ValueError('Panel exists: frozen input must not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    mapping, packet = [], []
    for group in groups:
        if group['group_type'] == 'sha_exact':
            continue  # exact copies are not semantic multi-image evidence
        rows = conn.execute('SELECT f.* FROM files f JOIN group_members gm ON gm.file_id=f.id WHERE gm.group_id=? ORDER BY f.exif_timestamp,f.id', (group['id'],)).fetchall()
        if len(rows) != group['member_count']:
            raise ValueError('Incomplete group membership')
        alias = f'G{group["id"]:03d}'
        frames, mapped = [], []
        for i, row in enumerate(rows):
            frame_alias = f'F{i + 1:02d}'
            with Image.open(row['path']) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                p = out / 'images' / f'{alias}-{frame_alias}.jpg'
                p.parent.mkdir(exist_ok=True)
                image.save(p, quality=93, exif=b'')
            frames.append({'alias': frame_alias, 'image': str(p),
                           'offset_seconds': (row['exif_timestamp'] or 0) - (rows[0]['exif_timestamp'] or 0)})
            mapped.append({'alias': frame_alias, 'id': row['id'], 'path': row['path']})
        # Contact sheets contain only downsampled derivatives, no native originals/EXIF.
        for start in range(0, len(frames), 12):
            part = frames[start:start + 12]
            sheet = Image.new('RGB', (4 * 300, ((len(part) + 3) // 4) * 320), '#222222')
            draw = ImageDraw.Draw(sheet)
            for j, frame in enumerate(part):
                with Image.open(frame['image']) as im:
                    im.thumbnail((296, 286))
                    x, y = (j % 4) * 300, (j // 4) * 320
                    sheet.paste(im, (x + (300 - im.width) // 2, y + 25))
                    draw.text((x + 5, y + 5), f'{alias} {frame["alias"]} +{frame["offset_seconds"]}s', fill='white')
            dest = out / 'sheets' / f'{alias}-{start // 12:02d}.jpg'
            dest.parent.mkdir(exist_ok=True)
            sheet.save(dest, quality=93)
        packet.append({'group_alias': alias, 'frames': frames})
        mapping.append({'group_alias': alias, 'date': day, 'split': split, 'db': str(db),
                        'group_id': group['id'], 'group_type': group['group_type'], 'frames': mapped})
    conn.close()
    dump(out / 'mapping.json', mapping)
    for i in range(0, len(packet), 10):
        dump(out / f'blind-{i // 10 + 1:02d}.json', {'status': 'unlabelled', 'groups': packet[i:i + 10]})
    dump(out / 'summary.json', {'date': day, 'split': split, 'multi_image_groups': len(packet),
                               'photos': sum(len(p['frames']) for p in packet),
                               'selection': 'all non-byte-exact multi-image baseline groups, complete date; no singleton padding'})
    print((out / 'summary.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['ingest', 'pipeline', 'panel', 'slice_event', 'prefix_event', 'merge_parts'])
    parser.add_argument('day')
    parser.add_argument('--split', choices=['development', 'holdout'], default='development')
    parser.add_argument('--slice-index', type=int, default=0)
    parser.add_argument('--max-prefix', type=int, default=100)
    parser.add_argument('--prefix-name', default='development-prefix')
    args = parser.parse_args()
    if args.action == 'prefix_event':
        prefix_event(args.day, args.max_prefix, args.prefix_name)
    elif args.action == 'slice_event':
        slice_event(args.day, args.slice_index)
    elif args.action == 'panel':
        panel(args.day, args.split)
    else:
        globals()[args.action](args.day)
