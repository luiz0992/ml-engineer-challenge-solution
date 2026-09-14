"""Tiny-ImageNet classification fine-tuning pipeline.

Concrete :class:`~models.training.pipeline.TrainingPipeline` implementing the
techniques the challenge requires — mixed precision, gradient clipping, and
learning-rate scheduling — plus layer-wise LR decay and the custom augmentation
pipeline from :mod:`models.data.augmentation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from models.data.augmentation import AugmentationConfig
from models.data.tiny_imagenet import build_dataloaders
from models.training.config import LearningRateStrategy, TrainConfig
from models.training.pipeline import TrainingPipeline
from models.training.utility import (
    AdaptiveLogger,
    count_parameters,
    create_lr_scheduler,
    create_scheduler_params,
    get_layerwise_lr_decay_params,
    get_optimizer,
)

logger = AdaptiveLogger(__name__)


def soft_target_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy against soft (probability-vector) targets.

    ``F.cross_entropy`` accepts class indices; MixUp and CutMix produce
    interpolated probability vectors, so the loss is computed directly as
    ``-sum(target * log_softmax(logits))``.

    ``log_softmax`` is used rather than ``log(softmax(...))`` because it is
    numerically stable — the naive form underflows to ``-inf`` for confident
    predictions, which is exactly the situation late in training.
    """
    return torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1).mean()


def accuracy_topk(
    logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    """Top-k accuracy. Accepts either class indices or soft targets."""
    if targets.ndim > 1:
        targets = targets.argmax(dim=-1)

    max_k = min(max(topk), logits.size(-1))
    _, predicted = logits.topk(max_k, dim=-1, largest=True, sorted=True)
    correct = predicted.eq(targets.view(-1, 1).expand_as(predicted))

    results: dict[str, float] = {}
    for k in topk:
        k_eff = min(k, max_k)
        # Metric names avoid "@": MLflow permits only alphanumerics, underscore,
        # dash, period, space, colon, and slash, and rejects the batch mid-run.
        results[f"acc_top{k}"] = correct[:, :k_eff].any(dim=-1).float().mean().item()
    return results


def build_model(
    name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> torch.nn.Module:
    """Create a timm backbone with a freshly initialised classifier head.

    Passing ``num_classes`` to ``timm.create_model`` replaces the pretrained
    1000-way head with a correctly sized, randomly initialised one; the
    backbone weights are retained.
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    total, trainable = count_parameters(model)
    logger.info(
        "Built %s (pretrained=%s): %.1fM parameters, %.1fM trainable",
        name,
        pretrained,
        total / 1e6,
        trainable / 1e6,
    )
    return model


class ClassificationPipeline(TrainingPipeline):
    """Fine-tune an ImageNet-pretrained backbone on Tiny-ImageNet."""

    def __init__(
        self,
        config: TrainConfig,
        accelerator: Accelerator,
        results_path: Path,
        **kwargs: Any,
    ) -> None:
        aug_cfg = AugmentationConfig(
            image_size=config.dataloader.image_size,
            crop_scale=tuple(config.dataloader.crop_scale),  # type: ignore[arg-type]
            use_randaugment=config.dataloader.use_randaugment,
            randaugment_num_ops=config.dataloader.randaugment_num_ops,
            randaugment_magnitude=config.dataloader.randaugment_magnitude,
            pca_lighting_std=config.dataloader.pca_lighting_std,
            random_erasing_prob=config.dataloader.random_erasing_prob,
            mixup_alpha=config.dataloader.mixup_alpha,
            cutmix_alpha=config.dataloader.cutmix_alpha,
            mix_prob=config.dataloader.mix_prob,
            label_smoothing=config.dataloader.label_smoothing,
            normalisation=config.dataloader.normalisation,  # type: ignore[arg-type]
        )

        train_loader, val_loader, dataset = build_dataloaders(
            data_dir=config.paths.data_dir,
            cfg=aug_cfg,
            batch_size=config.dataloader.batch_size_train,
            num_workers=config.dataloader.dataloader_num_workers,
            pin_memory=config.dataloader.pin_memory,
        )
        self.class_names = dataset.class_names
        self.wnids = dataset.wnids
        self.augmentation_config = aug_cfg

        model = build_model(
            config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            drop_path_rate=config.model.drop_path_rate,
        )

        if config.model.num_classes != dataset.num_classes:
            raise ValueError(
                f"Config declares num_classes={config.model.num_classes} but the dataset has "
                f"{dataset.num_classes} classes. A mismatch would silently train against the "
                "wrong label space."
            )

        if config.model.channels_last:
            # torch's stubs omit the memory_format overload on Module.to, though
            # it is valid at runtime.
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

        param_groups = self._build_param_groups(model, config)
        optimizer = get_optimizer(
            config.training.optimizer_name,
            params=param_groups,
            lr=config.training.head_learning_rate,
            weight_decay=config.training.optimizer_weight_decay,
            fused=config.training.fused_optimizer,
            betas=tuple(config.training.optimizer_betas),
            eps=config.training.optimizer_eps,
        )

        scheduler_params = create_scheduler_params(
            warmup_ratio=config.training.warmup_ratio,
            batch_size_train=config.dataloader.batch_size_train,
            dataset_length=len(dataset.train),
            num_epochs=config.training.epochs,
            train_subset_num_batches=config.training.train_subset_num_batches,
        )
        logger.info(
            "Scheduler: %d warmup steps of %d total (%d steps/epoch)",
            scheduler_params.num_warmup_steps,
            scheduler_params.num_training_steps,
            scheduler_params.steps_per_epoch,
        )
        lr_scheduler = create_lr_scheduler(
            scheduler_name=config.training.learning_rate_scheduler,
            optimizer=optimizer,
            scheduler_params=scheduler_params,
            start_epoch_at=config.training.start_epoch_at,
        )

        if config.model.compile_model:
            logger.info("Compiling model with torch.compile")
            model = torch.compile(model)  # type: ignore[assignment]

        super().__init__(
            config=config,
            accelerator=accelerator,
            exp_name=config.experiment_name,
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            num_epochs=config.training.epochs,
            eval_interval=config.training.eval_interval,
            save_interval=config.training.save_interval,
            results_path=results_path,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            run_name=config.run_name,
            **kwargs,
        )

    @staticmethod
    def _build_param_groups(model: torch.nn.Module, config: TrainConfig) -> list[dict[str, Any]]:
        """Construct optimizer parameter groups per the configured LR strategy."""
        strategy = config.training.learning_rate_strategy
        weight_decay = config.training.optimizer_weight_decay

        if strategy is LearningRateStrategy.llrd:
            groups = get_layerwise_lr_decay_params(
                model,
                encoder_lr=config.training.encoder_learning_rate,
                head_lr=config.training.head_learning_rate,
                layer_decay=config.training.layer_decay,
                weight_decay=weight_decay,
            )
            lrs = [g["lr"] for g in groups]
            logger.info(
                "LLRD: %d parameter groups, lr range %.2e..%.2e",
                len(groups),
                min(lrs),
                max(lrs),
            )
            return groups

        if strategy is LearningRateStrategy.frozen_backbone:
            head_names = {n for n, _ in model.named_parameters() if _is_head_param(n)}
            for name, param in model.named_parameters():
                param.requires_grad = name in head_names
            trainable = [p for p in model.parameters() if p.requires_grad]
            logger.info("Frozen backbone: training %d head tensors", len(trainable))
            return [
                {
                    "params": trainable,
                    "lr": config.training.head_learning_rate,
                    "weight_decay": weight_decay,
                }
            ]

        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": config.training.encoder_learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    # --- TrainingPipeline contract ----------------------------------------
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass over an ``(images, targets)`` batch.

        Training batches carry soft targets from MixUp/CutMix; validation
        batches carry integer labels. Both are handled by promoting integer
        labels to one-hot, so there is a single unconditional loss path.
        """
        images, targets = batch
        if self.config.model.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        logits = model(images)

        if targets.ndim == 1:
            soft = F.one_hot(targets, num_classes=logits.size(-1)).to(logits.dtype)
        else:
            soft = targets.to(logits.dtype)

        return soft_target_cross_entropy(logits, soft), logits, targets

    def compute_metrics(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        return accuracy_topk(logits, targets, topk=(1, 5))

    def calculate_primary_metric(
        self,
        eval_loss: float,  # noqa: ARG002 - required by the ABC; selection uses accuracy
        metrics: dict[str, float],
    ) -> float:
        """Top-1 accuracy is the metric used for model selection."""
        return metrics.get("acc_top1", 0.0)

    def is_higher_better_primary_metric(self) -> bool:
        return True

    def get_primary_metric_name(self) -> str:
        return "acc_top1"


def _is_head_param(name: str) -> bool:
    """Whether a parameter belongs to the classifier head."""
    return any(name.startswith(prefix) for prefix in ("head", "fc", "classifier"))
