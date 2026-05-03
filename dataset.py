# dataset.py
# Dataset and DataLoader helpers for vehicle image classification.

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def get_transforms(augment=False):
    """Create the transform pipeline used by the classifier."""
    base = [
        transforms.Resize((config.resize_y, config.resize_x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)


class VehicleDataset(Dataset):
    """Loads images from class folders or class-name file prefixes."""

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir=config.DATA_DIR, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_transforms(augment)
        self.class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}
        self.samples = self._scan_dir()

        if not self.samples:
            raise RuntimeError(f"No labelled images found in {root_dir}")

    def _scan_dir(self):
        samples = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.VALID_EXT:
                    continue
                fpath = os.path.join(dirpath, fname)
                label = self._label_from_path(fpath)
                if label is not None:
                    samples.append((fpath, self.class_to_idx[label]))
        return samples

    def _label_from_path(self, fpath):
        parent = os.path.basename(os.path.dirname(fpath)).lower()
        if parent in self.class_to_idx:
            return parent
        return self._label_from_filename(os.path.basename(fpath))

    def _label_from_filename(self, fname):
        stem = os.path.splitext(fname)[0]
        prefix = stem.split("_")[0].lower()
        return prefix if prefix in self.class_to_idx else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def vehicle_dataloader(root_dir=config.DATA_DIR,
                       batch_size=config.batchsize,
                       shuffle=True,
                       augment=False,
                       num_workers=0):
    dataset = VehicleDataset(root_dir=root_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
