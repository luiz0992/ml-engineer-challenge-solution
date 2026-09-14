"""Tiny-ImageNet dataset construction.

Wraps :class:`torchvision.datasets.ImageFolder` with three additions that
matter for correctness and for serving:

1. **A validation-layout guard.** The dataset ships its validation split as a
   flat ``val/images/`` directory plus ``val_annotations.txt``. Pointing
   ``ImageFolder`` at it yields one class and labels every image ``0``, with no
   error raised — training appears to work and validation accuracy is
   meaningless. :func:`assert_val_is_restructured` turns that silent corruption
   into an explicit failure with instructions.

2. **Train/val class-index agreement.** ``ImageFolder`` assigns indices by
   sorting directory names *within each split independently*. If the two splits
   ever disagree, validation labels are permuted relative to training ones.
   :func:`build_datasets` checks the mappings match.

3. **Human-readable class names.** ``words.txt`` maps WordNet IDs to English
   labels, which the API returns instead of raw ``n01443537`` identifiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.data.augmentation import (
    AugmentationConfig,
    MixUpCutMix,
    build_eval_transform,
    build_train_transform,
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 200
NATIVE_IMAGE_SIZE = 64


class DatasetLayoutError(RuntimeError):
    """Raised when the dataset on disk is not in the expected layout."""


def assert_val_is_restructured(root: Path) -> None:
    """Verify the validation split is in ``ImageFolder`` layout.

    Raises :class:`DatasetLayoutError` if the split is still in its distributed
    form, rather than allowing a silent collapse to a single class.
    """
    val_dir = root / "val"

    if not val_dir.is_dir():
        raise DatasetLayoutError(
            f"No validation directory at {val_dir}. Run:\n"
            f"  python scripts/setup/download_datasets.py --dataset tiny_imagenet"
        )

    if (val_dir / "images").is_dir():
        raise DatasetLayoutError(
            f"{val_dir} is still in its distributed layout (a flat `images/` directory "
            f"plus `val_annotations.txt`).\n\n"
            f"torchvision's ImageFolder would silently discover a single class here and "
            f"label all 10,000 validation images 0, making validation accuracy "
            f"meaningless without raising any error.\n\n"
            f"Restructure it with:\n"
            f"  python scripts/setup/download_datasets.py --dataset tiny_imagenet --force"
        )

    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    if len(class_dirs) != NUM_CLASSES:
        raise DatasetLayoutError(
            f"Expected {NUM_CLASSES} class directories in {val_dir}, found "
            f"{len(class_dirs)}. The dataset may be partially extracted; re-run the "
            f"download script with --force."
        )


def resolve_root(data_dir: Path | str = "data") -> Path:
    """Locate the Tiny-ImageNet root beneath ``data_dir``.

    Tolerates the redundant ``tiny-imagenet-200/tiny-imagenet-200/`` nesting
    that results from extracting the archive without flattening, so datasets
    prepared by other tooling still load.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "tiny-imagenet-200",
        data_dir / "tiny-imagenet-200" / "tiny-imagenet-200",
        data_dir,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate

    raise DatasetLayoutError(
        f"Could not find Tiny-ImageNet under {data_dir}. Expected a directory "
        f"containing `train/` and `val/`. Run:\n"
        f"  python scripts/setup/download_datasets.py --dataset tiny_imagenet"
    )


def load_class_names(root: Path) -> dict[str, str]:
    """Map WordNet IDs to human-readable names from ``words.txt``.

    Returns an empty mapping if the file is absent; readable names are a
    presentation nicety and their absence must not block training.
    """
    words = root / "words.txt"
    if not words.is_file():
        logger.warning("words.txt not found at %s; class names unavailable", words)
        return {}

    mapping: dict[str, str] = {}
    with words.open(encoding="utf-8") as handle:
        for line in handle:
            wnid, _, names = line.partition("\t")
            if names:
                # Entries list synonyms: "n01443537\tgoldfish, Carassius auratus"
                mapping[wnid.strip()] = names.split(",")[0].strip()
    return mapping


@dataclass(slots=True)
class TinyImageNet:
    """Prepared Tiny-ImageNet splits and their label metadata."""

    train: ImageFolder
    val: ImageFolder
    root: Path
    #: Ordered so that ``class_names[i]`` is the name for model output index ``i``.
    class_names: list[str]
    wnids: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.wnids)


def build_datasets(
    data_dir: Path | str = "data",
    cfg: AugmentationConfig | None = None,
) -> TinyImageNet:
    """Construct the training and validation datasets with their transforms."""
    cfg = cfg or AugmentationConfig()
    root = resolve_root(data_dir)
    assert_val_is_restructured(root)

    train = ImageFolder(root / "train", transform=build_train_transform(cfg))
    val = ImageFolder(root / "val", transform=build_eval_transform(cfg))

    # ImageFolder sorts directory names per split. Disagreement would permute
    # validation labels relative to training ones, which shows up as a model
    # that trains well but validates at chance.
    if train.class_to_idx != val.class_to_idx:
        mismatched = {
            wnid
            for wnid in set(train.class_to_idx) | set(val.class_to_idx)
            if train.class_to_idx.get(wnid) != val.class_to_idx.get(wnid)
        }
        raise DatasetLayoutError(
            f"Train and validation class indices disagree for {len(mismatched)} classes "
            f"(e.g. {sorted(mismatched)[:5]}). Validation labels would be permuted "
            f"relative to training labels."
        )

    wnids = list(train.classes)
    names = load_class_names(root)
    class_names = [names.get(wnid, wnid) for wnid in wnids]

    logger.info(
        "Loaded Tiny-ImageNet from %s: %d train / %d val images, %d classes",
        root,
        len(train),
        len(val),
        len(wnids),
    )
    return TinyImageNet(train=train, val=val, root=root, class_names=class_names, wnids=wnids)


def build_dataloaders(
    data_dir: Path | str = "data",
    cfg: AugmentationConfig | None = None,
    *,
    batch_size: int = 256,
    num_workers: int = 8,
    use_mixing: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, TinyImageNet]:
    """Build train and validation loaders.

    MixUp/CutMix is applied in the collate function so it runs in the worker
    processes, off the training loop's critical path. It is never applied to
    the validation loader, which must measure clean accuracy.

    ``drop_last`` is set on the training loader because MixUp pairs a batch
    with a reversed copy of itself; a trailing batch of size 1 would mix an
    example with itself and contribute a degenerate gradient.
    """
    cfg = cfg or AugmentationConfig()
    data = build_datasets(data_dir, cfg)

    collate_fn = None
    if use_mixing and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0):
        mixer = MixUpCutMix(
            num_classes=data.num_classes,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mix_prob,
            label_smoothing=cfg.label_smoothing,
        )
        collate_fn = mixer.as_collate_fn()

    train_loader = DataLoader(
        data.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,  # type: ignore[arg-type]
    )
    val_loader = DataLoader(
        data.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, data
