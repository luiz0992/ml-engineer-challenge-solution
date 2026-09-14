"""Shared training utilities.

Follows the conventions established in the Delphos training package: an
:class:`AdaptiveLogger` that routes through Accelerate when a distributed state
exists, explicit scheduler-step arithmetic that survives resumption, and
optimizer construction driven by a config enum rather than free-form strings.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate.state import PartialState
from torch.optim import SGD, AdamW, Optimizer
from transformers import get_scheduler

from models.training.config import OptimizerName


class AdaptiveLogger:
    """Route to the Accelerate logger once a distributed state exists.

    Accelerate's logger suppresses output on non-main ranks, which is what we
    want during training. Before :class:`~accelerate.Accelerator` is
    constructed there is no distributed state, and reaching for the Accelerate
    logger then would swallow messages emitted during setup.
    """

    def __init__(self, name: str) -> None:
        self._standard = logging.getLogger(name)
        self._accelerate = get_accelerate_logger(name)

    def _active(self) -> Any:
        if PartialState._shared_state:
            return self._accelerate
        return self._standard

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active(), name)


logger = AdaptiveLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerParams:
    """Optimizer-step counts required to build a learning-rate schedule."""

    num_warmup_steps: int
    num_training_steps: int
    steps_per_epoch: int


def set_seed(seed: int, deterministic_cuda: bool = False) -> None:
    """Seed every RNG that affects training.

    ``deterministic_cuda`` trades roughly 10-20% throughput for bitwise
    reproducibility. It is off by default: for this workload run-to-run
    variance is far smaller than the accuracy differences we care about, and
    the benchmark numbers in ``benchmarks/`` are more useful when measured on
    the fast path that production would actually use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_pytorch_optimized_configs(
    enable_tf32: bool = True, deterministic_cuda: bool = False
) -> None:
    """Enable TF32 matmuls and the cuDNN autotuner.

    TF32 gives a substantial speedup on Ampere and later with no meaningful
    accuracy cost for this workload. The cuDNN benchmark autotuner is only
    enabled when determinism is not requested, since it selects algorithms
    non-deterministically; it is a clear win here because our input shapes are
    fixed, so the one-time search cost is amortised immediately.
    """
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if not deterministic_cuda:
        torch.backends.cudnn.benchmark = True


def parse_precision(precision: str) -> str:
    """Map config precision strings to Accelerate's ``mixed_precision`` values."""
    mapping = {"amp_bf16": "bf16", "amp_fp16": "fp16", "fp32": "no"}
    return mapping.get(precision, "no")


def resolve_precision(precision: str) -> str:
    """Pick a mixed-precision mode, downgrading when hardware cannot support it.

    bf16 is preferred over fp16 on Ampere and later: it has the same exponent
    range as fp32, so it needs no loss scaling and cannot produce the
    scaler-underflow stalls that fp16 training is prone to. On hardware without
    bf16 support we fall back to fp16 rather than silently training in a dtype
    the device will emulate slowly.
    """
    if precision == "amp_bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 unsupported on this GPU; falling back to fp16")
        return "fp16"
    if precision in {"amp_bf16", "amp_fp16"} and not torch.cuda.is_available():
        logger.warning("No CUDA device available; disabling mixed precision")
        return "no"
    return parse_precision(precision)


def calculate_steps_per_epoch(
    *, dataset_length: int, batch_size_train: int, train_subset_num_batches: int
) -> int:
    """Optimizer steps in one epoch, given the logical training batch size."""
    if batch_size_train <= 0:
        raise ValueError(f"batch_size_train must be positive, got {batch_size_train}")
    if train_subset_num_batches > 0:
        return train_subset_num_batches
    return max(1, dataset_length // batch_size_train)


def create_scheduler_params(
    *,
    warmup_ratio: float,
    batch_size_train: int,
    dataset_length: int,
    num_epochs: int,
    train_subset_num_batches: int = 0,
) -> SchedulerParams:
    """Derive schedule lengths from the dataset and batch shape.

    Warmup is expressed as a fraction of total training steps rather than the
    token budget used for language-model pretraining: token counts are not a
    meaningful unit for a fixed-size image dataset, where every sample costs
    the same.
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    steps_per_epoch = calculate_steps_per_epoch(
        dataset_length=dataset_length,
        batch_size_train=batch_size_train,
        train_subset_num_batches=train_subset_num_batches,
    )
    num_training_steps = steps_per_epoch * num_epochs
    return SchedulerParams(
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
        steps_per_epoch=steps_per_epoch,
    )


def create_lr_scheduler(
    *,
    scheduler_name: str,
    optimizer: Optimizer,
    scheduler_params: SchedulerParams,
    start_epoch_at: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a scheduler that resumes at the correct point.

    ``start_epoch_at`` is 0-indexed: 0 is a fresh run, N resumes after N
    completed epochs. PyTorch schedulers count optimizer steps rather than
    epochs, so ``last_epoch`` is derived from completed steps. Passing the
    epoch number directly would restart the warmup ramp on every resume.
    """
    if start_epoch_at < 0:
        raise ValueError(f"start_epoch_at must be non-negative, got {start_epoch_at}")

    prior_completed_steps = start_epoch_at * scheduler_params.steps_per_epoch
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=scheduler_params.num_warmup_steps,
        num_training_steps=scheduler_params.num_training_steps,
        scheduler_specific_kwargs={"last_epoch": prior_completed_steps - 1},
    )


def get_optimizer(
    optimizer_name: OptimizerName,
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    lr: float,
    weight_decay: float,
    fused: bool = False,
    **kwargs: Any,
) -> Optimizer:
    """Construct the optimizer named by the config."""
    use_fused = fused and torch.cuda.is_available()
    if fused and not use_fused:
        logger.warning("Fused optimizer requested but CUDA is unavailable; using the standard path")

    match optimizer_name:
        case OptimizerName.adamw:
            return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused, **kwargs)
        case OptimizerName.sgd:
            kwargs.setdefault("momentum", 0.9)
            kwargs.setdefault("nesterov", True)
            kwargs.pop("betas", None)
            kwargs.pop("eps", None)
            return SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
        case _:
            raise ValueError(f"Optimizer name {optimizer_name} not recognized.")


def get_layerwise_lr_decay_params(
    model: torch.nn.Module,
    *,
    encoder_lr: float,
    head_lr: float,
    layer_decay: float,
    weight_decay: float,
    no_decay_patterns: tuple[str, ...] = ("bias", "norm", "LayerNorm", "cls_token", "pos_embed"),
    min_lr: float = 1e-8,
) -> list[dict[str, Any]]:
    """Build parameter groups with layer-wise learning-rate decay (LLRD).

    Earlier layers of a pretrained backbone encode general features that
    transfer directly, while later layers are more task-specific. Scaling the
    learning rate by ``layer_decay ** (depth - i)`` lets the head adapt quickly
    without destroying the low-level features underneath — the standard recipe
    for fine-tuning vision transformers, and worth noticeably more accuracy than
    a uniform rate on a small dataset such as Tiny-ImageNet.

    Parameters matching ``no_decay_patterns`` are excluded from weight decay.
    Regularising biases and normalisation scales toward zero fights the
    normalisation itself and reliably costs accuracy.
    """
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError(f"layer_decay must be in (0, 1], got {layer_decay}")

    layer_ids = _assign_layer_ids(model)
    max_layer_id = max(layer_ids.values(), default=0)

    groups: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = layer_ids.get(name, max_layer_id)
        is_head = layer_id == max_layer_id
        base_lr = head_lr if is_head else encoder_lr
        scaled_lr = max(base_lr * (layer_decay ** (max_layer_id - layer_id)), min_lr)

        decays = not any(pattern in name for pattern in no_decay_patterns)
        key = f"layer{layer_id}_{'decay' if decays else 'nodecay'}"

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": scaled_lr,
                "weight_decay": weight_decay if decays else 0.0,
            }
        groups[key]["params"].append(param)

    return list(groups.values())


def _assign_layer_ids(model: torch.nn.Module) -> dict[str, int]:
    """Map parameter names to a depth index for LLRD.

    Handles the two backbone families used here: timm/HF vision transformers
    (``blocks.N`` / ``encoder.layer.N``) and torchvision ResNets
    (``layerN``). Anything unrecognised is treated as head-level so it receives
    the full head learning rate — the safe default, since under-training the
    head is more damaging than over-training a stray parameter.
    """
    import re

    block_patterns = (
        re.compile(r"blocks\.(\d+)\."),
        re.compile(r"encoder\.layer\.(\d+)\."),
        re.compile(r"layers\.(\d+)\."),
        re.compile(r"^layer(\d)\."),
    )
    stem_prefixes = ("patch_embed", "cls_token", "pos_embed", "conv1", "bn1", "embeddings")

    layer_ids: dict[str, int] = {}
    max_block = 0

    for name, _ in model.named_parameters():
        block_id: int | None = None
        for pattern in block_patterns:
            match = pattern.search(name)
            if match:
                block_id = int(match.group(1)) + 1  # reserve 0 for the stem
                break

        if block_id is not None:
            layer_ids[name] = block_id
            max_block = max(max_block, block_id)
        elif any(name.startswith(prefix) for prefix in stem_prefixes):
            layer_ids[name] = 0

    # Everything not yet assigned (classifier head, final norm) sits above the
    # deepest block.
    head_id = max_block + 1
    for name, _ in model.named_parameters():
        layer_ids.setdefault(name, head_id)

    return layer_ids


def resolve_tracking_uri(
    *,
    enabled: bool,
    configured_uri: str | None,
    inherit_env_uri: bool,
    results_path: Path,
) -> str | None:
    """Determine the MLflow tracking URI, or ``None`` to disable tracking.

    Resolution order:

    1. Tracking disabled by config -> ``None``.
    2. An explicit ``configured_uri`` -> used as given.
    3. ``inherit_env_uri`` and a supported ``MLFLOW_TRACKING_URI`` -> inherited.
    4. Otherwise -> a local SQLite store beneath ``results_path``.

    The environment variable is validated before use. MLflow only accepts a
    fixed set of schemes, and an unsupported value (for example a SageMaker
    ARN, which requires a plugin) otherwise raises deep inside tracker startup
    after the model and data are already loaded. Checking here turns that into
    a warning and a working local fallback.

    SQLite rather than a ``file://`` store: MLflow 3 placed the filesystem
    backend in maintenance mode and raises unless an opt-out flag is set.
    SQLite is equally self-contained and remains fully supported.
    """
    if not enabled:
        return None

    if configured_uri:
        return configured_uri

    if inherit_env_uri:
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            scheme = env_uri.split(":", 1)[0].lower() if ":" in env_uri else "file"
            if scheme in SUPPORTED_MLFLOW_SCHEMES:
                return env_uri
            logger.warning(
                "Ignoring MLFLOW_TRACKING_URI with unsupported scheme %r "
                "(supported: %s); falling back to a local SQLite store",
                scheme,
                ", ".join(sorted(SUPPORTED_MLFLOW_SCHEMES)),
            )

    results_path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(results_path / 'mlflow.db').resolve()}"


#: Schemes MLflow's tracking store registry accepts natively. Anything else
#: needs a plugin and must not be assumed available.
SUPPORTED_MLFLOW_SCHEMES = frozenset(
    {"file", "http", "https", "postgresql", "mysql", "sqlite", "mssql", "databricks"}
)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    """Flatten nested config dictionaries for experiment-tracker hyperparameters."""
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def create_missing_dirs(path: Path) -> None:
    """Create ``path`` and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
