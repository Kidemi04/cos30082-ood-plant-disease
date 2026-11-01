"""Shared, framework-independent access to the prepared official splits."""
import csv
import json
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parent


def load_split(split, project_dir=PROJECT):
    """Return records with local absolute image paths and original integer labels."""
    if split not in {'train', 'test_seen', 'test_unseen'}:
        raise ValueError('Choose train, test_seen, or test_unseen.')
    dataset = Path(project_dir) / 'dataset'
    manifest = dataset / 'prepared' / f'{split}.csv'
    if not manifest.is_file():
        raise FileNotFoundError('Run setup_dataset.bat before loading the dataset.')
    with manifest.open(encoding='utf-8', newline='') as source:
        records = list(csv.DictReader(source))
    root = (dataset / 'extracted').resolve()
    for record in records:
        relative = PurePosixPath(record['image_path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
            raise ValueError(f'Invalid prepared image path: {record["image_path"]}')
        record['image_path'] = root.joinpath(*relative.parts)
        for category in ('crop', 'disease', 'composition'):
            record[category + '_id'] = int(record[category + '_id'])
    return records


def load_labels(project_dir=PROJECT):
    return json.loads((Path(project_dir) / 'dataset/prepared/label_map.json').read_text(encoding='utf-8'))
