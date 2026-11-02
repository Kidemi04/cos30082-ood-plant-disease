from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

try:
    import torch
    from torch.utils.data import DataLoader
    import preprocessing
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch and torchvision are not installed')
class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def image(self, name, size, mode='RGB', **save):
        path = self.root / name
        Image.new(mode, size, 'green' if mode == 'RGB' else (0, 128, 0, 128)).save(path, **save)
        return path

    def test_transforms_give_fixed_size_normalised_tensors(self):
        image = preprocessing.load_image(self.image('wide.jpg', (900, 300)))
        for transform in (preprocessing.train_transform(), preprocessing.eval_transform()):
            tensor = transform(image)
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(preprocessing.eval_transform()(image), preprocessing.eval_transform()(image)))

    def test_load_image_converts_rgba_and_keeps_large_jpeg_above_min_size(self):
        self.assertEqual(preprocessing.load_image(self.image('alpha.png', (50, 40), 'RGBA')).mode, 'RGB')
        large = preprocessing.load_image(self.image('large.jpg', (4000, 3000)), min_size=256)
        self.assertEqual(large.mode, 'RGB')
        self.assertLess(large.size[0], 4000)
        self.assertGreaterEqual(min(large.size), 256)

    def test_load_image_applies_exif_orientation(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees
        self.assertEqual(preprocessing.load_image(self.image('rotated.jpg', (60, 30), exif=exif)).size, (30, 60))

    def test_dataset_items_collate_into_batches(self):
        records = [{'image_path': self.image(f'{i}.jpg', (300, 200)), 'filename': f'{i}.jpg', 'crop_id': i,
                    'disease_id': i + 1, 'composition_id': i + 2} for i in range(3)]
        dataset = preprocessing.DPDDataset(records, preprocessing.eval_transform())
        batch = next(iter(DataLoader(dataset, batch_size=3)))
        self.assertEqual(tuple(batch['image'].shape), (3, 3, 224, 224))
        self.assertEqual(batch['composition'].tolist(), [2, 3, 4])
        self.assertEqual(batch['filename'], ['0.jpg', '1.jpg', '2.jpg'])

    def test_balanced_sampler_oversamples_rare_classes(self):
        records = [{'composition_id': 0}] * 90 + [{'composition_id': 1}] * 10
        draws = Counter(records[i]['composition_id'] for i in preprocessing.balanced_sampler(
            records, power=1.0, num_samples=20000, generator=torch.Generator().manual_seed(0)))
        self.assertAlmostEqual(draws[1] / 20000, 0.5, delta=0.02)

    def test_denormalise_inverts_normalise(self):
        tensor = torch.rand(3, 8, 8)
        normalised = (tensor - torch.tensor(preprocessing.IMAGENET_MEAN).view(-1, 1, 1)) \
            / torch.tensor(preprocessing.IMAGENET_STD).view(-1, 1, 1)
        self.assertTrue(torch.allclose(preprocessing.denormalise(normalised), tensor, atol=1e-6))


@unittest.skipIf(torch is None, 'PyTorch and torchvision are not installed')
@unittest.skipUnless((PROJECT / 'dataset/metadata/validation_split.csv').is_file()
                     and (PROJECT / 'dataset/prepared/train.csv').is_file(), 'dataset is not prepared')
class RealDatasetTests(unittest.TestCase):
    def test_build_datasets_loads_real_images(self):
        datasets = preprocessing.build_datasets()
        self.assertEqual({name: len(d) for name, d in datasets.items()},
                         {'train': 6779, 'val': 1221, 'test_seen': 2000, 'test_unseen': 1448})
        for name in ('train', 'test_unseen'):
            self.assertEqual(tuple(datasets[name][0]['image'].shape), (3, 224, 224))
        self.assertEqual(len(preprocessing.build_datasets(use_validation=False)['train']), 8000)


if __name__ == '__main__':
    unittest.main()
