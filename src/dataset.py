"""OpenImagesV7 dataset for dVAE training.

Uses `fiftyone` to download the requested split into a local directory and
then iterates over the on-disk JPEGs with a standard `torch.utils.data` API.
fiftyone is only required for the first download; after that, the script
falls back to a pure-filesystem scan so training does not depend on it.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import PIL.Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from dall_e import map_pixels

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _scan_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS)


def download_openimages_v7(
    root: str,
    split: str,
    max_samples: Optional[int] = None,
) -> Path:
    """Ensure an OpenImagesV7 split exists under ``root/<split>``.

    Calls fiftyone the first time, then re-uses the on-disk cache. We only
    fetch images themselves (``label_types=[]``); the dVAE does not use labels.
    """
    target = Path(root) / split
    if target.exists() and any(target.rglob("*.jpg")):
        return target

    import fiftyone.zoo as foz

    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        max_samples=max_samples,
        label_types=[],
        dataset_dir=str(Path(root) / "_fo"),
    )
    src = Path(ds.first().filepath).parent
    target.mkdir(parents=True, exist_ok=True)
    # symlink files into the split dir rather than copy
    for p in src.iterdir():
        if p.suffix.lower() in VALID_EXTS:
            link = target / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
    return target


@dataclass
class CropConfig:
    """Train-time crop policy.

    ``size`` is either an int (square) or a (H, W) tuple.
    """
    size: int | tuple[int, int] = 256
    random_crop: bool = True
    horizontal_flip: bool = True

    @property
    def hw(self) -> tuple[int, int]:
        if isinstance(self.size, int):
            return (self.size, self.size)
        return self.size


class OpenImagesV7(Dataset):
    """Image-only dataset over a directory of files.

    Resizes the short side to match the target then random- or center-crops.
    Pixels are mapped through DALL-E's ``map_pixels`` so that the encoder
    sees the same distribution it was pretrained on.
    """

    def __init__(self, root: str | Path, crop: CropConfig, training: bool = True):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.files = _scan_images(self.root)
        if not self.files:
            raise RuntimeError(f"no images found under {self.root}")
        self.crop = crop
        self.training = training

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int) -> PIL.Image.Image:
        path = self.files[idx]
        img = PIL.Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = self._load(idx)
        except (OSError, PIL.UnidentifiedImageError):
            # A handful of OpenImages files are corrupted; resample.
            return self.__getitem__((idx + 1) % len(self))

        H, W = self.crop.hw
        target_short = min(H, W)
        s = min(img.size)
        if s < target_short:
            r = target_short / s
            new_size = (round(r * img.size[1]), round(r * img.size[0]))
            img = TF.resize(img, new_size, interpolation=PIL.Image.LANCZOS)

        if self.training and self.crop.random_crop:
            # Random crop of exactly (H, W).
            iw, ih = img.size
            if ih < H or iw < W:
                img = TF.resize(img, (max(ih, H), max(iw, W)),
                                interpolation=PIL.Image.LANCZOS)
                iw, ih = img.size
            top  = random.randint(0, ih - H)
            left = random.randint(0, iw - W)
            img = TF.crop(img, top, left, H, W)
            if self.crop.horizontal_flip and random.random() < 0.5:
                img = TF.hflip(img)
        else:
            iw, ih = img.size
            top  = max(0, (ih - H) // 2)
            left = max(0, (iw - W) // 2)
            img = TF.crop(img, top, left, H, W)

        x = T.ToTensor()(img)  # (3, H, W) in [0, 1]
        return map_pixels(x.unsqueeze(0)).squeeze(0)
