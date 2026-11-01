import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_dataset.py'
spec = importlib.util.spec_from_file_location('prepare_dataset', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def reference_for(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            files[path.relative_to(root).as_posix()] = {
                'size': len(data), 'crc32': f'{zlib.crc32(data):08X}'
            }
    return {'files': files, 'archive_sha256': 'test-release'}


class DatasetPreparationTests(unittest.TestCase):
    def test_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in ('../outside', '/absolute', 'C:/outside', 'a\\..\\outside'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    prepare.safe_path(root, path)

    def test_changed_content_with_same_size_fails_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'data.csv'
            path.write_bytes(b'original')
            reference = reference_for(root)
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                prepare.verify_files(root, reference)

    def test_bad_image_fails_even_when_its_checksum_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bad.jpg').write_bytes(b'not an image')
            with self.assertRaises(OSError):
                prepare.verify_files(root, reference_for(root))

    def test_extra_or_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'expected.csv'
            path.write_bytes(b'expected')
            reference = reference_for(root)
            extra = root / 'extra.csv'
            extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                prepare.signatures(root, reference)
            extra.unlink()
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                prepare.signatures(root, reference)

    def test_archive_mismatch_cannot_start_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root / 'dataset/metadata'
            meta.mkdir(parents=True)
            archive = root / 'wrong.7z'
            archive.write_bytes(b'wrong archive')
            (meta / 'dataset_reference.json').write_text(json.dumps({
                'archive_size': archive.stat().st_size, 'archive_sha256': 'incorrect', 'files': {}
            }))
            with patch.object(prepare, 'extract') as extract:
                with self.assertRaisesRegex(ValueError, 'Archive SHA-256 differs'):
                    prepare.prepare(root, archive)
                extract.assert_not_called()
            self.assertFalse((root / 'dataset/.setup_state.json').exists())

    def test_py7zr_fallback_extracts_and_validates(self):
        import py7zr
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            Image.new('RGB', (8, 8), 'green').save(source / 'leaf.png')
            (source / 'labels.csv').write_text('leaf.png,0\n')
            reference = reference_for(source)
            archive = root / 'tiny.7z'
            with py7zr.SevenZipFile(archive, 'w') as target:
                for path in source.iterdir():
                    target.write(path, path.name)
            with patch.object(prepare, 'find_7zip', return_value=None):
                prepare.extract(archive, root / 'stage', reference)
            result = prepare.verify_files(root / 'stage', reference)
            self.assertEqual(result['images_decoded'], 1)


if __name__ == '__main__':
    unittest.main()
