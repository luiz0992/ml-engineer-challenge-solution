"""Training entrypoint.

Usage::

    uv run python -m models.training.train                       # defaults
    uv run python -m models.training.train training.epochs=10    # override
    uv run python -m models.training.train --config-name smoke   # alternate config

Multi-GPU runs go through Accelerate::

    uv run accelerate launch -m models.training.train
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig, OmegaConf

from models.training.config import TaskName, TrainConfig, register_configs
from models.training.utility import (
    AdaptiveLogger,
    resolve_precision,
    resolve_tracking_uri,
    set_pytorch_optimized_configs,
    set_seed,
)

logger = AdaptiveLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _generate_run_name(config: TrainConfig) -> str:
    """Derive a descriptive run name when none is configured."""
    if config.run_name:
        return config.run_name
    backbone = config.model.name.split(".")[0]
    return (
        f"{backbone}-bs{config.dataloader.batch_size_train}"
        f"-lr{config.training.head_learning_rate:g}"
        f"-ep{config.training.epochs}"
        f"-{config.training.precision}"
    )


def build_pipeline(config: TrainConfig, accelerator: Accelerator, results_path: Path) -> Any:
    """Instantiate the pipeline for the configured task."""
    if config.task is TaskName.classification:
        from models.training.classification_pipeline import ClassificationPipeline

        return ClassificationPipeline(
            config=config, accelerator=accelerator, results_path=results_path
        )
    if config.task is TaskName.embedding:
        from models.training.embedding_pipeline import EmbeddingPipeline

        return EmbeddingPipeline(config=config, accelerator=accelerator, results_path=results_path)
    if config.task is TaskName.detection:
        from models.training.detection_pipeline import DetectionPipeline

        return DetectionPipeline(config=config, accelerator=accelerator, results_path=results_path)

    raise NotImplementedError(f"Task {config.task!r} has no training pipeline.")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train_classifier")
def main(cfg: DictConfig) -> float:
    """Run training and return the primary metric, for sweep integration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Materialise the structured config so attribute access is type-checked and
    # unknown keys have already been rejected by Hydra.
    config: TrainConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    config.run_name = _generate_run_name(config)

    set_seed(config.training.seed, deterministic_cuda=config.training.deterministic_cuda)
    set_pytorch_optimized_configs(deterministic_cuda=config.training.deterministic_cuda)

    mixed_precision = resolve_precision(config.training.precision)
    results_path = Path(config.paths.results_path) / config.run_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Resolve tracking explicitly and export it, so MLflow and Accelerate agree
    # and no ambient value from the operator's shell can capture the run.
    tracking_uri = resolve_tracking_uri(
        enabled=config.tracking.enabled,
        configured_uri=config.tracking.tracking_uri,
        inherit_env_uri=config.tracking.inherit_env_uri,
        results_path=results_path,
    )
    if tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        logger.warning("Experiment tracking disabled; metrics will be logged to the console only")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="mlflow" if tracking_uri else None,
        project_config=ProjectConfiguration(
            project_dir=str(results_path),
            automatic_checkpoint_naming=False,
        ),
    )

    logger.info("Run name: %s", config.run_name)
    logger.info("Device: %s | mixed_precision=%s", accelerator.device, mixed_precision)
    logger.info("Results path: %s", results_path)

    pipeline = build_pipeline(config, accelerator, results_path)
    summary = pipeline.run()

    if accelerator.is_main_process:
        _write_run_summary(results_path, config, summary, pipeline)

    return float(summary.get(f"best_{pipeline.get_primary_metric_name()}", 0.0))


def _write_run_summary(
    results_path: Path, config: TrainConfig, summary: dict[str, float], pipeline: Any
) -> None:
    """Persist metrics, config, and label metadata beside the checkpoints.

    The label list is written here because the serving stack needs the exact
    index-to-name mapping the model was trained with. Deriving it again at
    serving time by listing directories would silently break if the dataset on
    the serving host differed.
    """
    from models.training.config import to_container

    (results_path / "metrics.json").write_text(json.dumps(summary, indent=2))
    (results_path / "config.json").write_text(
        json.dumps(to_container(config), indent=2, default=str)
    )

    labels = getattr(pipeline, "class_names", None)
    if labels:
        (results_path / "labels.json").write_text(
            json.dumps(
                {"class_names": labels, "wnids": getattr(pipeline, "wnids", [])},
                indent=2,
            )
        )

    logger.info("Wrote run summary to %s", results_path)
    for key, value in sorted(summary.items()):
        logger.info("  %s = %.6f", key, value)


if __name__ == "__main__":
    register_configs()
    main()
