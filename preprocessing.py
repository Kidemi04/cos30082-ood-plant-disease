"""Shared input pipeline: image loading, preprocessing, augmentation and PyTorch datasets for DPD-Small.

All models use these transforms, so differences in results come from the models, not the input pipeline.
"""
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from dataset_access import load_split, load_validation_split

IMAGE_SIZE = 224
RESIZE_RATIO = 0.875  # evaluation: resize the shorter side to 256, then centre-crop 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)  # OpenAI CLIP and BioCLIP pretraining statistics
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ORIENTATION = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
               5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
               8: Image.Transpose.ROTATE_90}  # EXIF orientation tag -> transpose that displays the image upright


def load_image(path, min_size=None):
    """Open an image as RGB with its EXIF orientation applied.

    min_size lets the JPEG decoder skip full resolution (some images are 6000 px wide) while keeping
    both sides at least that large, which makes loading much faster without changing the result visibly.
    """
    with Image.open(path) as image:
        if min_size and image.format in ('JPEG', 'MPO'):
            image.draft('RGB', (min_size, min_size))
        # Not ImageOps.exif_transpose: it rewrites the EXIF block and fails on some malformed DPD-Small files.
        method = ORIENTATION.get(image.getexif().get(0x0112))
        image = image.convert('RGB')
        return image.transpose(method) if method else image


AUGMENT_STRENGTHS = ('standard', 'strong')


def train_transform(image_size=IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, strength='standard'):
    """Training augmentation: random crop and scale, flips and mild colour jitter.

    Hue jitter stays small because colour (yellowing, rust, mildew) is itself a disease symptom.

    strength='strong' is the field-robustness variant (Member E): tighter random crops (viewpoint, framing),
    stronger lighting and contrast changes, perspective warps, occasional blur (focus, camera shake) and
    random erasing (occlusion, busy backgrounds). Hue jitter stays at 0.02 for the same reason as above.
    The default 'standard' pipeline is unchanged.
    """
    if strength not in AUGMENT_STRENGTHS:
        raise ValueError(f'strength must be one of {AUGMENT_STRENGTHS}, got {strength!r}')
    if strength == 'standard':
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.25, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomPerspective(distortion_scale=0.3, p=0.3),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random'),
    ])


def eval_transform(image_size=IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Deterministic preprocessing for validation, test and the user interface."""
    return T.Compose([
        T.Resize(round(image_size / RESIZE_RATIO), antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def denormalise(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo Normalize for displaying an image tensor (C, H, W) or batch (N, C, H, W)."""
    shape = (-1, 1, 1)
    return (tensor * torch.tensor(std).view(shape) + torch.tensor(mean).view(shape)).clamp(0, 1)


class DPDDataset(Dataset):
    """Images with crop, disease and composition targets.

    Each item is a dict: image, crop, disease, composition (official IDs) and filename, which
    evaluate.save_predictions needs. The default DataLoader collate batches it.
    """

    def __init__(self, records, transform=None, image_size=IMAGE_SIZE):
        self.records = records
        self.transform = transform
        self.min_size = round(image_size / RESIZE_RATIO)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = load_image(record['image_path'], self.min_size)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'crop': record['crop_id'], 'disease': record['disease_id'],
                'composition': record['composition_id'], 'filename': record['filename']}


def build_datasets(image_size=IMAGE_SIZE, use_validation=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Datasets for the official splits; train and val come from the shared validation manifest.

    With use_validation=False, train is the full official train split (for a final retrain after model
    selection) and there is no val entry. mean/std change only the final Normalize step, for backbones
    pretrained with other statistics (for example CLIP_MEAN/CLIP_STD); every other step is unchanged.
    """
    if use_validation:
        train, val = load_validation_split()
    else:
        train, val = load_split('train'), None
    evaluation = eval_transform(image_size, mean, std)
    datasets = {'train': DPDDataset(train, train_transform(image_size, mean, std), image_size),
                'test_seen': DPDDataset(load_split('test_seen'), evaluation, image_size),
                'test_unseen': DPDDataset(load_split('test_unseen'), evaluation, image_size)}
    if val is not None:
        datasets['val'] = DPDDataset(val, evaluation, image_size)
    return datasets


def balanced_sampler(records, key='composition_id', power=0.5, num_samples=None, generator=None):
    """Oversample rare classes: each image is drawn with weight 1 / count(class) ** power.

    power=0 is the natural distribution, 1 is fully class-balanced, 0.5 (square-root sampling) is a
    middle ground that avoids repeating single-image compositions too often. Use with shuffle=False.
    """
    counts = Counter(record[key] for record in records)
    weights = torch.tensor([counts[record[key]] ** -power for record in records], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples or len(records), replacement=True, generator=generator)
