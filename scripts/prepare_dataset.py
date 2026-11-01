"""Prepare the exact shared DPD-Small release without changing official splits."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zlib

PROJECT = Path(__file__).resolve().parents[1]
SPLITS = {'train': 8000, 'test_seen': 2000, 'test_unseen': 1448}
COLUMNS = ['filename', 'crop_id', 'disease_id', 'composition_id',
           'crop_name', 'disease_name', 'composition_name']
SCHEMA_VERSION = 1


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe dataset path: {relative}')
    target = root.joinpath(*path.parts)
    if not target.absolute().is_relative_to(root.absolute()):
        raise ValueError(f'Dataset path escapes its directory: {relative}')
    return target


def find_7zip():
    for command in ('7z', '7zz', '7za'):
        if found := shutil.which(command):
            return found
    for key in ('ProgramFiles', 'ProgramFiles(x86)'):
        if os.environ.get(key):
            path = Path(os.environ[key]) / '7-Zip/7z.exe'
            if path.is_file():
                return str(path)
    return None


def extract(archive, stage, reference):
    stage.mkdir(parents=True, exist_ok=True)
    # Only the pinned archive is accepted, before any extraction takes place.
    needed = sum(item['size'] for item in reference['files'].values()) + 256 * 1024 * 1024
    if shutil.disk_usage(stage).free < needed:
        raise ValueError(f'Not enough free disk space; need at least {needed / 1024**3:.2f} GiB.')
    print('Extracting the archive. This can take several minutes...', flush=True)
    if tool := find_7zip():
        subprocess.run([tool, 'x', str(archive), '-o' + str(stage), '-y', '-bsp0'], check=True)
    else:
        import py7zr
        with py7zr.SevenZipFile(archive, mode='r') as source:
            for entry in source.list():
                safe_path(stage, entry.filename.replace('\\', '/'))
            source.extractall(path=stage)


def signatures(root, reference):
    expected = set(reference['files'])
    actual = set()
    result = {}
    for path in [root, *root.rglob('*')]:
        info = path.lstat()
        if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f'Dataset cannot contain symbolic links or junctions: {path}')
        if stat.S_ISREG(info.st_mode):
            relative = path.relative_to(root).as_posix()
            actual.add(relative)
            result[relative] = [info.st_size, info.st_mtime_ns]
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Dataset file list differs. Missing: {missing[:5]}; unexpected: {extra[:5]}')
    for relative, entry in reference['files'].items():
        if result[relative][0] != entry['size']:
            raise ValueError(f'Wrong file size: {relative}')
    return result


def verify_files(root, reference):
    from PIL import Image
    signatures(root, reference)
    contents = defaultdict(list)
    images = 0
    print('Checking every file checksum and decoding all images...', flush=True)
    for index, (relative, entry) in enumerate(reference['files'].items(), 1):
        path = safe_path(root, relative)
        crc = 0
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                crc = zlib.crc32(block, crc)
                digest.update(block)
        if f'{crc & 0xffffffff:08X}' != entry['crc32']:
            raise ValueError(f'Checksum mismatch: {relative}')
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            with Image.open(path) as image:
                image.load()
            images += 1
            contents[digest.hexdigest()].append(relative)
        if index % 2000 == 0:
            print(f'  Checked {index}/{len(reference["files"])} files', flush=True)
    return {'images_decoded': images,
            'identical_image_groups': [paths for paths in contents.values() if len(paths) > 1]}


def read_metadata(root):
    rows = {}
    mappings = {name: {} for name in ('crop', 'disease', 'composition')}
    compositions = {}
    filenames = set()
    for split, expected in SPLITS.items():
        path = root / f'DPD_Small_{split}.csv'
        with path.open(encoding='utf-8-sig', newline='') as source:
            raw = list(csv.reader(source))
        if len(raw) != expected:
            raise ValueError(f'{split}: expected {expected} rows, found {len(raw)}')
        records = []
        for index, values in enumerate(raw, 1):
            if len(values) != 7:
                raise ValueError(f'{split} row {index}: expected seven columns (original CSV has no header)')
            record = dict(zip(COLUMNS, values))
            filename = record['filename']
            if '/' in filename or '\\' in filename or filename in filenames:
                raise ValueError(f'Duplicate or invalid image filename across splits: {filename}')
            filenames.add(filename)
            image_path = safe_path(root, 'DPD_Small_dataset/' + filename)
            if not image_path.is_file():
                raise ValueError(f'Image referenced by CSV is missing: {filename}')
            for category in mappings:
                label = int(record[category + '_id'])
                name = record[category + '_name']
                if label < 0 or mappings[category].get(label, name) != name:
                    raise ValueError(f'Conflicting {category} label: {label}')
                mappings[category][label] = name
                record[category + '_id'] = label
            pair = (record['crop_id'], record['disease_id'])
            label = record['composition_id']
            if compositions.get(label, pair) != pair:
                raise ValueError(f'Conflicting crop/disease composition: {label}')
            compositions[label] = pair
            record['image_path'] = 'DPD_Small_dataset/' + filename
            records.append(record)
        rows[split] = records
    for category, count in [('crop', 55), ('disease', 175), ('composition', 333)]:
        if set(mappings[category]) != set(range(count)):
            raise ValueError(f'Unexpected {category} IDs; must preserve the official mapping.')
        if len(set(mappings[category].values())) != count:
            raise ValueError(f'Multiple {category} IDs have the same name.')
    image_files = {p.name for p in (root / 'DPD_Small_dataset').iterdir() if p.is_file()}
    if image_files != filenames:
        raise ValueError('CSV image list does not exactly cover the extracted image directory.')
    train = {row['composition_id'] for row in rows['train']}
    seen = {row['composition_id'] for row in rows['test_seen']}
    unseen = {row['composition_id'] for row in rows['test_unseen']}
    if len(train) != 323 or train != seen or len(unseen) != 10 or train & unseen:
        raise ValueError('Official seen/unseen composition split is inconsistent.')
    return rows, mappings, compositions


def make_outputs(rows, mappings, compositions, reference, validation):
    output = {}
    for split, records in rows.items():
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['image_path', *COLUMNS], lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
        output[split + '.csv'] = stream.getvalue().encode('utf-8')
    labels = {category: {str(k): v for k, v in sorted(values.items())}
              for category, values in mappings.items()}
    labels['composition_components'] = {str(k): {'crop_id': v[0], 'disease_id': v[1]}
                                        for k, v in sorted(compositions.items())}
    labels['seen_composition_ids'] = sorted({r['composition_id'] for r in rows['train']})
    labels['unseen_composition_ids'] = sorted({r['composition_id'] for r in rows['test_unseen']})
    output['label_map.json'] = json_bytes(labels)
    split_by_path = {r['image_path']: split for split, rs in rows.items() for r in rs}
    duplicates = [{'paths': paths, 'splits': sorted({split_by_path[p] for p in paths})}
                  for paths in validation['identical_image_groups']]
    report = {'counts': {s: len(rs) for s, rs in rows.items()},
              'images_decoded': validation['images_decoded'],
              'identical_image_groups': duplicates,
              'policy': 'Original splits retained. No validation split or image modifications.'}
    output['validation_report.json'] = json_bytes(report)
    fingerprint = hashlib.sha256(json_bytes({
        'schema_version': SCHEMA_VERSION, 'archive_sha256': reference['archive_sha256'],
        'outputs': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(output.items())}
    })).hexdigest()
    output['dataset_info.json'] = json_bytes({
        'schema_version': SCHEMA_VERSION, 'dataset_fingerprint': fingerprint,
        'archive_sha256': reference['archive_sha256'], 'image_root': '../extracted',
        'counts': report['counts'], 'validation_split': None,
        'crop_classes': 55, 'disease_classes': 175, 'composition_classes': 333,
        'seen_composition_classes': 323, 'unseen_composition_classes': 10,
        'official_csv_has_header': False,
    })
    return output, fingerprint


def prepare(project, archive, force=False):
    dataset = project / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    reference_path = dataset / 'metadata/dataset_reference.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    for relative in reference['files']:
        safe_path(dataset, relative)
    reference_hash = sha256(reference_path)
    state_path = dataset / '.setup_state.json'
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        state = {}
    if not archive.is_file():
        raise ValueError(f'Put the completed DPD_Small.7z in {dataset / "raw"}, then run setup again.')
    stat = archive.stat()
    archive_signature = [str(archive.resolve()), stat.st_size, stat.st_mtime_ns]
    if stat.st_size != reference['archive_size']:
        raise ValueError('Archive size differs from the shared release. Download may be incomplete.')
    cached = (state.get('schema_version') == SCHEMA_VERSION
              and state.get('reference_sha256') == reference_hash)
    if force or not cached or state.get('archive_signature') != archive_signature:
        print('Checking archive SHA-256...', flush=True)
        if sha256(archive) != reference['archive_sha256']:
            raise ValueError('Archive SHA-256 differs from the shared release. Do not use this archive.')
    extracted = dataset / 'extracted'
    validation = None
    if not extracted.exists():
        stage = dataset / '.extracting'
        if stage.is_dir():
            try:
                validation = verify_files(stage, reference)
                print('Recovered a fully extracted dataset.', flush=True)
            except (ValueError, OSError):
                print('Incomplete extraction found. Re-extracting into the staging folder...', flush=True)
        if validation is None:
            extract(archive, stage, reference)
            validation = verify_files(stage, reference)
        stage.rename(extracted)
    current = signatures(extracted, reference)
    if validation is None:
        if not force and cached and state.get('file_signatures') == current:
            validation = state['validation']
            print('Existing data unchanged; reusing the successful full check.', flush=True)
        else:
            validation = verify_files(extracted, reference)
    rows, mappings, compositions = read_metadata(extracted)
    outputs, fingerprint = make_outputs(rows, mappings, compositions, reference, validation)
    prepared = dataset / 'prepared'
    for name, contents in outputs.items():
        atomic_write(prepared / name, contents)
    atomic_write(state_path, json_bytes({
        'schema_version': SCHEMA_VERSION, 'reference_sha256': reference_hash,
        'archive_signature': archive_signature, 'file_signatures': current,
        'validation': validation, 'dataset_fingerprint': fingerprint,
    }))
    print('\nDATASET READY', flush=True)
    print('Train: 8000 | Test seen: 2000 | Test unseen: 1448')
    print(f'Dataset fingerprint: {fingerprint}')
    print(f'Metadata: {prepared}')
    duplicates = len(validation['identical_image_groups'])
    if duplicates:
        print(f'NOTE: {duplicates} identical-image groups retained; see validation_report.json.')
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional external archive; original is not moved or copied.')
    parser.add_argument('--verify', action='store_true', help='Rehash archive and fully check images again.')
    args = parser.parse_args()
    archive = args.archive.resolve() if args.archive else PROJECT / 'dataset/raw/DPD_Small.7z'
    try:
        prepare(PROJECT, archive, args.verify)
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as error:
        print(f'\nDATASET NOT READY: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
