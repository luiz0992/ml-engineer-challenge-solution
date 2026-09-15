"""COCO detection dataset for fine-tuning RT-DETR.

The challenge ships a 1,000-image COCO val2017 subset. Fine-tuning on a
held-out slice of that subset is what we have; splitting it 800/200 keeps
evaluation off the images used for the gradient.

RT-DETR wants boxes as normalised ``cxcywh`` and class indices in ``0..79``.
COCO category IDs are non-contiguous ``1..90``; submitting those as class
labels trains against the wrong space and yields a plausible near-zero mAP.
The contiguous mapping is built from the annotation file so it matches the
pretrained head.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DETECTION_IMAGE_SIZE = 640
NUM_COCO_CLASSES = 80


class CocoLayoutError(RuntimeError):
    """Raised when the COCO subset is missing or unreadable."""


def resolve_coco_root(data_dir: Path | str = "data") -> Path:
    """Locate the extracted COCO val2017 subset."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "coco_val2017",
        data_dir / "coco",
        data_dir,
    ]
    for candidate in candidates:
        images = candidate / "val2017"
        annotations = candidate / "annotations" / "instances_val2017.json"
        if images.is_dir() and annotations.is_file():
            return candidate
    raise CocoLayoutError(
        f"Could not find a COCO subset under {data_dir}. Expected "
        f"`val2017/` and `annotations/instances_val2017.json`. Run:\n"
        f"  python scripts/setup/download_datasets.py --dataset coco_sample"
    )


def _contiguous_category_map(coco: Any) -> dict[int, int]:
    """Map COCO category IDs onto the dense 0..79 index the model emits."""
    cat_ids = sorted(int(c) for c in coco.getCatIds())
    return {cat_id: index for index, cat_id in enumerate(cat_ids)}


def _load_image_as_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize and scale to ``[0, 1]`` without ImageNet normalisation.

    RT-DETR is trained with ``do_normalize=False``. Applying ImageNet statistics
    here would silently shift the input distribution.
    """
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CocoDetectionSubset(Dataset):
    """COCO images and boxes in the layout RT-DETR's loss expects."""

    def __init__(
        self,
        root: Path,
        image_ids: Sequence[int],
        *,
        image_size: int = DETECTION_IMAGE_SIZE,
        split: str = "val2017",
        annotation_file: str = "instances_val2017.json",
    ) -> None:
        from pycocotools.coco import COCO

        self.root = root
        self.image_dir = root / split
        self.image_ids = list(image_ids)
        self.image_size = image_size
        self.coco = COCO(str(root / "annotations" / annotation_file))
        self.cat_to_label = _contiguous_category_map(self.coco)
        if len(self.cat_to_label) != NUM_COCO_CLASSES:
            raise CocoLayoutError(
                f"Expected {NUM_COCO_CLASSES} COCO categories, found {len(self.cat_to_label)}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        original_width, original_height = image.size
        pixel_values = _load_image_as_tensor(image, self.image_size)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            if annotation.get("iscrowd", 0):
                continue
            x, y, width, height = annotation["bbox"]
            if width <= 1 or height <= 1:
                continue
            boxes.append(
                [
                    (x + width / 2) / original_width,
                    (y + height / 2) / original_height,
                    width / original_width,
                    height / original_height,
                ]
            )
            labels.append(self.cat_to_label[int(annotation["category_id"])])

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros(0, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros(0, 4, dtype=torch.float32),
        }
        return pixel_values, target


def coco_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> dict[str, Any]:
    """Stack images; keep labels as a list of dicts, which RT-DETR requires."""
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def build_coco_dataloaders(
    data_dir: Path | str = "data",
    *,
    image_size: int = DETECTION_IMAGE_SIZE,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """Split the local COCO subset into train and validation loaders."""
    from pycocotools.coco import COCO

    root = resolve_coco_root(data_dir)
    coco = COCO(str(root / "annotations" / "instances_val2017.json"))
    image_dir = root / "val2017"
    # The annotation file covers all of val2017; the challenge ships a subset.
    # Only keep IDs whose files are actually on disk.
    image_ids = sorted(
        int(info["id"])
        for info in coco.loadImgs(coco.getImgIds())
        if (image_dir / info["file_name"]).is_file()
    )
    if len(image_ids) < 4:
        raise CocoLayoutError(f"COCO subset at {root} has only {len(image_ids)} images")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(image_ids), generator=generator).tolist()
    split_at = max(1, min(len(image_ids) - 1, int(len(image_ids) * train_fraction)))
    train_ids = [image_ids[i] for i in permutation[:split_at]]
    val_ids = [image_ids[i] for i in permutation[split_at:]]

    train_set = CocoDetectionSubset(root, train_ids, image_size=image_size)
    val_set = CocoDetectionSubset(root, val_ids, image_size=image_size)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": coco_collate,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    logger.info(
        "COCO subset from %s: %d train / %d val images",
        root,
        len(train_set),
        len(val_set),
    )
    return train_loader, val_loader, len(train_set)
