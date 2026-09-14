"""Training configuration schema.

Structured Hydra/OmegaConf configuration, mirroring the Delphos training
package: dataclasses declare the schema and are registered with Hydra's
:class:`~hydra.core.config_store.ConfigStore`, so a typo in a YAML key is a
startup error rather than a silently ignored setting.

Fields marked ``MISSING`` have no safe default and must be supplied by YAML or
a command-line override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class OptimizerName(StrEnum):
    """Supported optimizers."""

    adamw = "adamw"
    sgd = "sgd"


class LearningRateStrategy(StrEnum):
    """How the learning rate is distributed across a pretrained backbone."""

    #: Layer-wise decay: deeper layers train faster than earlier ones.
    llrd = "llrd"
    #: A single rate for every trainable parameter.
    uniform = "uniform"
    #: Freeze the backbone entirely; train only the classifier head.
    frozen_backbone = "frozen_backbone"


class Precision(StrEnum):
    """Training precision. ``amp_bf16`` is preferred on Ampere and later."""

    amp_bf16 = "amp_bf16"
    amp_fp16 = "amp_fp16"
    fp32 = "fp32"


class TaskName(StrEnum):
    """Which model this run trains."""

    classification = "classification"
    detection = "detection"
    embedding = "embedding"


@dataclass
class ModelConfig:
    """Backbone selection and head configuration."""

    #: timm identifier. Defaults to a ViT-Small pretrained on ImageNet-21k and
    #: fine-tuned on ImageNet-1k, which transfers well to Tiny-ImageNet since
    #: the 200 classes are an ImageNet-1k subset.
    name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    pretrained: bool = True
    num_classes: int = 200
    #: Stochastic depth. A mild regulariser that matters on a small dataset.
    drop_path_rate: float = 0.1
    #: Channels-last memory format, which lets cuDNN and TensorCores use their
    #: preferred layout for convolutional backbones.
    channels_last: bool = False
    #: torch.compile the model. Large speedup after a one-time warmup cost.
    compile_model: bool = False


@dataclass
class TrainingConfig:
    """Optimisation, scheduling, and loop-control settings."""

    epochs: int = MISSING
    #: Learning rate for the classifier head.
    head_learning_rate: float = MISSING
    #: Base learning rate for the backbone; scaled per layer under LLRD.
    encoder_learning_rate: float = MISSING
    learning_rate_strategy: LearningRateStrategy = LearningRateStrategy.llrd
    #: Per-layer LR multiplier under LLRD. 0.75 is the standard ViT value.
    layer_decay: float = 0.75

    optimizer_name: OptimizerName = OptimizerName.adamw
    optimizer_weight_decay: float = 0.05
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    fused_optimizer: bool = True

    learning_rate_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    step_schedulers_every_batch: bool = True

    precision: Precision = Precision.amp_bf16
    #: Gradient-norm clipping threshold. Required by the challenge brief and a
    #: genuine safeguard against loss spikes early in fine-tuning.
    max_grad_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False

    #: 0-indexed, matching PyTorch and HuggingFace. 0 is a fresh run.
    start_epoch_at: int = 0
    eval_interval: str = "1ep"
    save_interval: int = 1
    #: Cap batches per epoch for smoke tests. 0 means use the whole split.
    train_subset_num_batches: int = 0
    eval_subset_num_batches: int = 0

    seed: int = 42
    deterministic_cuda: bool = False


@dataclass
class DataloaderConfig:
    """Data loading and augmentation settings."""

    batch_size_train: int = 128
    batch_size_validation: int = 256
    dataloader_num_workers: int = 8
    dataloader_num_workers_eval: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    image_size: int = 224
    crop_scale: tuple[float, float] = (0.65, 1.0)
    use_randaugment: bool = True
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    pca_lighting_std: float = 0.1
    random_erasing_prob: float = 0.25
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    label_smoothing: float = 0.1
    normalisation: str = "imagenet"


@dataclass
class TrackingConfig:
    """Experiment-tracking settings.

    The tracking URI is taken from configuration rather than inherited from the
    ambient ``MLFLOW_TRACKING_URI``. Inheriting it means whatever endpoint
    happens to be exported in the operator's shell silently captures the run —
    which, on a developer machine configured for another project, points
    experiments at unrelated infrastructure and fails on an unsupported URI
    scheme. Defaulting to a local file store keeps a fresh clone reproducible
    with no setup.
    """

    enabled: bool = True
    #: ``null`` resolves to a ``file://`` store beneath ``paths.results_path``.
    #: Set to a real server (``http://...``) to track centrally.
    tracking_uri: str | None = None
    #: Honour an externally exported MLFLOW_TRACKING_URI instead of the above.
    #: Off by default; see the class docstring.
    inherit_env_uri: bool = False


@dataclass
class PathsConfig:
    """Filesystem locations."""

    data_dir: str = "data"
    results_path: str = "models/artifacts/runs"
    artifacts_dir: str = "models/artifacts"


@dataclass
class TrainConfig:
    """Root configuration for a training run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    task: TaskName = TaskName.classification
    experiment_name: str = "tiny-imagenet-classification"
    run_name: str | None = None
    continue_training: bool = False
    continue_folder: str | None = None


def register_configs() -> None:
    """Register the schema with Hydra's ConfigStore.

    Registering the dataclass as the base schema is what makes unknown YAML
    keys and wrong value types fail at startup instead of being ignored.
    """
    cs = ConfigStore.instance()
    cs.store(name="train_config_schema", node=TrainConfig)


def to_container(config: Any) -> dict[str, Any]:
    """Resolve a config to a plain dictionary for tracking and serialisation."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
        if isinstance(resolved, dict):
            return resolved  # type: ignore[return-value]
        raise TypeError(f"Expected config to resolve to a dict, got {type(resolved).__name__}")

    from dataclasses import asdict, is_dataclass

    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)

    raise TypeError(f"Cannot convert {type(config).__name__} to a container")
