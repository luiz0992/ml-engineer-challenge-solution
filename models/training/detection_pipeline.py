"""Fine-tune RT-DETR on the local COCO subset.

The published checkpoint is already a strong detector. A short fine-tune on
the challenge's 1,000-image subset adapts it to the images we actually serve
without pretending we have the full COCO train2017 dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import Tensor

from models.data.coco import DETECTION_IMAGE_SIZE, build_coco_dataloaders
from models.optimisation.export_detection import DEFAULT_DETECTION_MODEL
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


class DetectionPipeline(TrainingPipeline):
    """Fine-tune a pretrained RT-DETR on the local COCO subset."""

    def __init__(
        self,
        config: TrainConfig,
        accelerator: Accelerator,
        results_path: Path,
        **kwargs: Any,
    ) -> None:
        from transformers import RTDetrForObjectDetection

        model_name = config.model.name or DEFAULT_DETECTION_MODEL
        model = RTDetrForObjectDetection.from_pretrained(model_name)
        total, trainable = count_parameters(model)
        logger.info(
            "Loaded detector %s: %.1fM parameters, %.1fM trainable",
            model_name,
            total / 1e6,
            trainable / 1e6,
        )
        raw_labels = model.config.id2label or {}
        self.id2label = {int(k): v for k, v in raw_labels.items()}
        self.class_names = [self.id2label[i] for i in range(len(self.id2label))]
        self.wnids: list[str] = []

        train_loader, val_loader, dataset_length = build_coco_dataloaders(
            data_dir=config.paths.data_dir,
            image_size=config.dataloader.image_size or DETECTION_IMAGE_SIZE,
            batch_size=config.dataloader.batch_size_train,
            num_workers=config.dataloader.dataloader_num_workers,
            pin_memory=config.dataloader.pin_memory,
            seed=config.training.seed,
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
            dataset_length=dataset_length,
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
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        scores = torch.sigmoid(outputs.logits)
        # Dummy targets keep the ABC contract; detection metrics use scores.
        dummy_targets = scores.new_zeros(scores.size(0), dtype=torch.int64)
        return outputs.loss, scores, dummy_targets

    def compute_metrics(self, logits: Tensor, targets: Tensor) -> dict[str, float]:  # noqa: ARG002
        """Peak confidence is a cheap proxy while full mAP runs offline.

        Computing COCO mAP inside the training loop needs pycocotools over the
        whole split and would dominate a short fine-tune. The dedicated
        `scripts/evaluate_detector.py` remains the source of published mAP.
        """
        peak = logits.amax(dim=-1).amax(dim=-1)
        return {
            "mean_peak_confidence": peak.mean().item(),
            "frac_confident": (peak > 0.5).float().mean().item(),
        }

    def calculate_primary_metric(
        self,
        eval_loss: float,
        metrics: dict[str, float],  # noqa: ARG002
    ) -> float:
        return -eval_loss

    def is_higher_better_primary_metric(self) -> bool:
        return True

    def get_primary_metric_name(self) -> str:
        return "neg_eval_loss"
