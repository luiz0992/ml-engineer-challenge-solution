"""Supervised contrastive fine-tune of a dedicated similarity backbone.

This is a third model, not the classifier with its head removed. Classification
features collapse intra-class variation *by construction*; a contrastive loss
does the opposite — it pulls same-class images together while still leaving
room for pose and appearance to differ.

The backbone is still Tiny-ImageNet domain-specific (same ViT-Small as the
classifier) so the extra cost is one training run and one extra export, not a
second preprocessing pipeline.
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
from models.training.config import TrainConfig
from models.training.pipeline import TrainingPipeline
from models.training.utility import (
    AdaptiveLogger,
    count_parameters,
    create_lr_scheduler,
    create_scheduler_params,
    get_optimizer,
)

logger = AdaptiveLogger(__name__)


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    ``embeddings`` must already be L2-normalised. Integer labels identify
    positives; MixUp/CutMix soft targets are not used on this path.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (batch, dim) embeddings, got {tuple(embeddings.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected integer labels of shape (batch,), got {tuple(labels.shape)}")

    batch = embeddings.size(0)
    similarity = embeddings @ embeddings.T / temperature
    # Mask self-similarity so an example is never its own positive.
    self_mask = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    positives = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = positives & ~self_mask
    has_positive = positives.any(dim=1)
    if not bool(has_positive.any()):
        # A batch with no same-class pairs has nothing to pull together.
        return embeddings.new_zeros(())

    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    # Mean over positives per anchor, then over anchors that had a positive.
    # `where`, not multiply: the diagonal is -inf from the self-mask, and
    # `-inf * 0` is NaN.
    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(
        dim=1
    ) / positives.sum(dim=1).clamp(min=1)
    return -positive_log_prob[has_positive].mean()


def retrieval_recall_at_k(
    embeddings: Tensor,
    labels: Tensor,
    *,
    k: int = 5,
) -> dict[str, float]:
    """Leave-one-out recall@k on a validation batch of unit-norm embeddings."""
    if embeddings.size(0) < 2:
        return {f"recall_at{k}": 0.0}

    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-1.0)
    k_eff = min(k, embeddings.size(0) - 1)
    neighbours = similarity.topk(k_eff, dim=-1).indices
    neighbour_labels = labels[neighbours]
    hits = (neighbour_labels == labels.unsqueeze(1)).any(dim=1).float()
    return {f"recall_at{k}": hits.mean().item()}


class EmbeddingPipeline(TrainingPipeline):
    """Fine-tune a backbone as a metric-learning embedder."""

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
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            mix_prob=0.0,
            label_smoothing=0.0,
            normalisation=config.dataloader.normalisation,  # type: ignore[arg-type]
        )
        train_loader, val_loader, dataset = build_dataloaders(
            data_dir=config.paths.data_dir,
            cfg=aug_cfg,
            batch_size=config.dataloader.batch_size_train,
            num_workers=config.dataloader.dataloader_num_workers,
            use_mixing=False,
            pin_memory=config.dataloader.pin_memory,
        )
        self.class_names = dataset.class_names
        self.wnids = dataset.wnids
        self.temperature = 0.07

        model = timm.create_model(
            config.model.name,
            pretrained=config.model.pretrained,
            num_classes=0,
            drop_path_rate=config.model.drop_path_rate,
        )
        total, trainable = count_parameters(model)
        logger.info(
            "Built embedder %s: %.1fM parameters, %.1fM trainable, %d-d features",
            config.model.name,
            total / 1e6,
            trainable / 1e6,
            model.num_features,
        )

        optimizer = get_optimizer(
            config.training.optimizer_name,
            params=[p for p in model.parameters() if p.requires_grad],
            lr=config.training.encoder_learning_rate,
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
        lr_scheduler = create_lr_scheduler(
            scheduler_name=config.training.learning_rate_scheduler,
            optimizer=optimizer,
            scheduler_params=scheduler_params,
            start_epoch_at=config.training.start_epoch_at,
        )

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

    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        images, targets = batch
        if targets.ndim > 1:
            targets = targets.argmax(dim=-1)
        embeddings = F.normalize(model(images), p=2.0, dim=-1)
        loss = supervised_contrastive_loss(embeddings, targets, temperature=self.temperature)
        return loss, embeddings, targets

    def compute_metrics(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        return retrieval_recall_at_k(logits, targets, k=5)

    def calculate_primary_metric(
        self,
        eval_loss: float,  # noqa: ARG002 - required by the ABC
        metrics: dict[str, float],
    ) -> float:
        return metrics.get("recall_at5", 0.0)

    def is_higher_better_primary_metric(self) -> bool:
        return True

    def get_primary_metric_name(self) -> str:
        return "recall_at5"
