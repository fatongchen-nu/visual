from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class TactileSegDataset(Dataset):
    """Dataset that expects:
    root/images/*.jpg|png
    root/masks/*.png with pixel classes 0/1/2.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Sequence[Path],
        image_size: int = 512,
        is_train: bool = True,
    ) -> None:
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.is_train = is_train
        self.image_size = image_size

        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomCrop(width=image_size, height=image_size, p=1.0),
        ]
        eval_transforms = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        ]

        self.transforms = A.Compose(
            train_transforms if is_train else eval_transforms,
            additional_targets={"mask": "mask"},
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {self.image_paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {self.mask_paths[index]}")

        transformed = self.transforms(image=image, mask=mask)
        image_t = transformed["image"].astype(np.float32) / 255.0
        mask_t = transformed["mask"].astype(np.int64)

        image_t = torch.from_numpy(np.transpose(image_t, (2, 0, 1)))
        mask_t = torch.from_numpy(mask_t)
        return image_t, mask_t


def _collect_pairs(root: Path, image_dir: str, mask_dir: str) -> Tuple[List[Path], List[Path]]:
    image_root = root / image_dir
    mask_root = root / mask_dir
    image_paths = sorted([p for p in image_root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    paired_images: List[Path] = []
    paired_masks: List[Path] = []
    for img in image_paths:
        mask = mask_root / f"{img.stem}.png"
        if mask.exists():
            paired_images.append(img)
            paired_masks.append(mask)

    if not paired_images:
        raise RuntimeError(f"No image/mask pairs found under {root}")
    return paired_images, paired_masks


def create_splits(
    root: str,
    image_dir: str = "images",
    mask_dir: str = "masks",
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    images, masks = _collect_pairs(Path(root), image_dir, mask_dir)
    train_images, val_images, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
    )
    return train_images, train_masks, val_images, val_masks
