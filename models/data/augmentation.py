"""Custom data augmentation pipeline for Tiny-ImageNet fine-tuning.

The challenge requires a *custom* augmentation pipeline, so the components that
carry the interesting design decisions are implemented here directly rather
than assembled from stock presets:

* :class:`PCALighting` — AlexNet-style colour jitter along the RGB covariance
  eigenbasis. Perturbs illumination without distorting hue relationships the
  way naive per-channel jitter does. Not available in torchvision.
* :class:`MixUpCutMix` — a batch-level collator implementing both MixUp and
  CutMix with correct area-corrected label mixing and label smoothing.

Stock torchvision ops are used where they are already correct and well tested
(cropping, flipping, erasing); reimplementing those would add risk, not value.

Design notes specific to Tiny-ImageNet
--------------------------------------
Source images are 64x64, which constrains the pipeline in two ways:

1. **Crop scale is much gentler than the ImageNet default.** The usual
   ``RandomResizedCrop`` lower bound of ``0.08`` retains an 18x18 region of a
   64x64 image, which frequently contains no part of the labelled object. The
   default here is ``0.65``, following the common practice for low-resolution
   datasets.

2. **Normalisation statistics depend on the backbone.** When fine-tuning
   ImageNet-pretrained weights the input distribution must match what the
   backbone saw during pretraining, so ImageNet statistics are the default.
   Tiny-ImageNet's own statistics are provided for training from scratch;
   using the wrong one silently costs accuracy rather than raising.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor
from torchvision.transforms import v2

# --- Normalisation statistics ---------------------------------------------
#: Channel statistics of ImageNet-1k. Correct choice when fine-tuning a
#: backbone pretrained on ImageNet, which is the default path here.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Channel statistics computed over the Tiny-ImageNet training split. Correct
#: choice only when training from scratch.
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)

#: Eigen-decomposition of the ImageNet RGB covariance matrix, used by
#: :class:`PCALighting`. From Krizhevsky et al. (2012), section 4.1.
_PCA_EIGVALS = torch.tensor([0.2175, 0.0188, 0.0045])
_PCA_EIGVECS = torch.tensor(
    [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]
)


class PCALighting(v2.Transform):
    """Jitter illumination along the RGB covariance eigenbasis.

    Adds ``sum_i (alpha_i * lambda_i) * v_i`` to every pixel, where ``v_i`` and
    ``lambda_i`` are the eigenvectors and eigenvalues of the ImageNet RGB
    covariance matrix and ``alpha_i ~ N(0, sigma)`` is drawn once per image.

    Because the perturbation follows the directions along which natural image
    colour actually varies, it models plausible lighting changes. Independent
    per-channel jitter, by contrast, produces colour casts that do not occur in
    natural images and can push samples off the data manifold.

    Expects a float tensor in ``[0, 1]`` with shape ``(..., 3, H, W)`` and must
    be applied *before* normalisation.
    """

    def __init__(self, alpha_std: float = 0.1) -> None:
        super().__init__()
        if alpha_std < 0:
            raise ValueError(f"alpha_std must be non-negative, got {alpha_std}")
        self.alpha_std = alpha_std

    def transform(self, inpt: Tensor, params: dict) -> Tensor:  # noqa: ARG002 - v2.Transform API
        if self.alpha_std == 0.0:
            return inpt
        if not isinstance(inpt, Tensor) or inpt.shape[-3] != 3:
            return inpt

        alpha = torch.randn(3, device=inpt.device, dtype=inpt.dtype) * self.alpha_std
        eigvals = _PCA_EIGVALS.to(device=inpt.device, dtype=inpt.dtype)
        eigvecs = _PCA_EIGVECS.to(device=inpt.device, dtype=inpt.dtype)

        # (3,3) @ (3,) -> (3,), one offset per channel.
        offset = eigvecs @ (alpha * eigvals)
        return (inpt + offset.view(3, 1, 1)).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha_std={self.alpha_std})"


@dataclass(slots=True)
class AugmentationConfig:
    """Declarative description of the augmentation pipeline.

    Exposed as a dataclass so the exact configuration can be serialised into
    the model card and training run metadata, making runs reproducible.
    """

    #: Network input resolution. Tiny-ImageNet is natively 64x64; upsampling to
    #: 224 lets us reuse ImageNet-pretrained weights without surgery on the
    #: patch embedding or stem.
    image_size: int = 224

    #: Lower/upper area fraction for RandomResizedCrop. See module docstring
    #: for why the lower bound is far above the ImageNet default.
    crop_scale: tuple[float, float] = (0.65, 1.0)
    crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)

    horizontal_flip_prob: float = 0.5

    #: RandAugment strength. ``num_ops=2, magnitude=9`` is the standard setting
    #: and is well matched to a short fine-tune.
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    use_randaugment: bool = True

    pca_lighting_std: float = 0.1

    #: Random erasing probability. Applied after normalisation, per the
    #: original paper.
    random_erasing_prob: float = 0.25

    #: Batch-level regularisation. Both are enabled; one is chosen at random
    #: per batch. See :class:`MixUpCutMix`.
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    label_smoothing: float = 0.1

    normalisation: Literal["imagenet", "tiny_imagenet"] = "imagenet"

    #: Populated in ``__post_init__``; not set by callers.
    mean: tuple[float, float, float] = field(init=False)
    std: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ValueError(f"image_size must be at least 32, got {self.image_size}")
        if not 0.0 < self.crop_scale[0] <= self.crop_scale[1] <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < lo <= hi <= 1, got {self.crop_scale}")

        if self.normalisation == "imagenet":
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        else:
            self.mean, self.std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD


def build_train_transform(cfg: AugmentationConfig) -> v2.Compose:
    """Compose the training-time augmentation pipeline.

    Order matters and is deliberate:

    1. Geometric ops (crop, flip) — cheapest, and reduce the pixel count that
       later ops must process.
    2. RandAugment — operates on uint8, which is what its magnitude scale
       assumes.
    3. Convert PIL -> tensor, then to float in ``[0, 1]``.
    4. PCA lighting — defined on unnormalised values.
    5. Normalise.
    6. Random erasing — applied last so erased regions are zero *in normalised
       space*, i.e. the channel mean, which is the intended behaviour.
    """
    steps: list[v2.Transform] = [
        v2.RandomResizedCrop(
            size=cfg.image_size,
            scale=cfg.crop_scale,
            ratio=cfg.crop_ratio,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
    ]

    if cfg.use_randaugment:
        steps.append(
            v2.RandAugment(
                num_ops=cfg.randaugment_num_ops,
                magnitude=cfg.randaugment_magnitude,
            )
        )

    # ImageFolder yields PIL images; ToImage converts to a uint8 tensor without
    # rescaling, and ToDtype(scale=True) then maps [0, 255] -> [0, 1].
    steps.append(v2.ToImage())
    steps.append(v2.ToDtype(torch.float32, scale=True))

    if cfg.pca_lighting_std > 0:
        steps.append(PCALighting(alpha_std=cfg.pca_lighting_std))

    steps.append(v2.Normalize(mean=cfg.mean, std=cfg.std))

    if cfg.random_erasing_prob > 0:
        steps.append(v2.RandomErasing(p=cfg.random_erasing_prob, value=0.0))

    return v2.Compose(steps)


def build_eval_transform(cfg: AugmentationConfig) -> v2.Compose:
    """Compose the deterministic evaluation pipeline.

    Uses the standard resize-then-centre-crop protocol with a 0.875 ratio, so
    reported accuracy is comparable with published ImageNet-style numbers.
    """
    resize_to = math.floor(cfg.image_size / 0.875)
    return v2.Compose(
        [
            v2.Resize(resize_to, antialias=True),
            v2.CenterCrop(cfg.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


class MixUpCutMix:
    """Batch collator applying MixUp or CutMix with soft targets.

    Both methods interpolate pairs of examples and their labels, which
    regularises the network and calibrates its confidence. They are
    complementary — MixUp blends globally, CutMix splices a rectangular patch —
    so one is selected at random per batch, following the DeiT/timm recipe.

    Two details are easy to get wrong and are handled explicitly:

    * **Area correction.** CutMix's mixing coefficient must be recomputed from
      the *realised* patch area after the box is clipped to the image bounds.
      Using the sampled ``lam`` directly biases the targets, because a box
      sampled near an edge covers less area than requested.
    * **Label smoothing composes with mixing.** Targets are built as smoothed
      one-hot vectors *before* interpolation, so the two regularisers combine
      rather than one overwriting the other.

    The batch is paired against a reversed copy of itself rather than a fresh
    permutation: this is the standard trick, costs no extra indexing, and gives
    each example a different partner.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing

    def _smooth(self, targets: Tensor) -> Tensor:
        """Convert integer labels to smoothed one-hot vectors."""
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        soft = torch.full(
            (targets.size(0), self.num_classes),
            off,
            device=targets.device,
            dtype=torch.float32,
        )
        return soft.scatter_(1, targets.unsqueeze(1), on)

    def _cutmix_box(self, height: int, width: int, lam: float) -> tuple[int, int, int, int]:
        """Sample a patch covering ``1 - lam`` of the image, clipped to bounds."""
        ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)

        cy = int(torch.randint(height, (1,)).item())
        cx = int(torch.randint(width, (1,)).item())

        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        return y1, y2, x1, x2

    def __call__(self, images: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Mix ``images`` and return them with soft targets.

        Always returns soft targets, even when no mixing is applied, so the
        training loop has a single unconditional loss path.
        """
        soft = self._smooth(targets)

        if torch.rand(1).item() >= self.prob:
            return images, soft

        flipped_images = images.flip(0)
        flipped_soft = soft.flip(0)

        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

        if use_cutmix:
            height, width = images.shape[-2:]
            y1, y2, x1, x2 = self._cutmix_box(height, width, lam)
            images = images.clone()
            images[..., y1:y2, x1:x2] = flipped_images[..., y1:y2, x1:x2]
            # Recompute lam from the realised area; the sampled value is only
            # an estimate once the box has been clipped.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
        else:
            images = images.mul(lam).add_(flipped_images, alpha=1.0 - lam)

        mixed_targets = soft.mul(lam).add_(flipped_soft, alpha=1.0 - lam)
        return images, mixed_targets

    def as_collate_fn(self, default_collate: object = None) -> object:
        """Return a ``collate_fn`` suitable for ``DataLoader``.

        Mixing runs in the worker process as part of collation, which keeps it
        off the training loop's critical path.
        """
        from torch.utils.data import default_collate as _default_collate

        collate = default_collate or _default_collate

        def collate_fn(batch: Iterable[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
            images, targets = collate(list(batch))  # type: ignore[operator]
            return self(images, targets)

        return collate_fn
