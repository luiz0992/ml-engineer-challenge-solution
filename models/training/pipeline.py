"""Abstract training pipeline built on HuggingFace Accelerate.

Structure and operational behaviour follow the Delphos training package:

* Accelerate owns mixed precision, gradient accumulation, and device placement.
* MLflow tracking is optional — absent ``MLFLOW_TRACKING_URI`` the run logs to
  the console and continues, rather than failing.
* SIGTERM checkpoints and exits cleanly, so a pre-empted spot instance or a
  ``docker stop`` does not discard an epoch of work.
* Non-finite losses are tolerated individually but abort the run after
  ``MAX_CONSECUTIVE_NAN`` in a row, distinguishing a transient bad batch from
  genuine divergence.
* Learning-rate schedulers are deliberately kept out of ``accelerator.prepare``:
  the prepared wrapper auto-steps on every ``optimizer.step()``, which conflicts
  with the explicit per-batch/per-epoch stepping controlled by
  ``step_schedulers_every_batch``.

Adapted from the Delphos original in two places, both driven by the workload:

1. Batches are ``(images, targets)`` tuples rather than HuggingFace-style kwarg
   dicts, so the forward pass is delegated to :meth:`TrainingPipeline.compute_loss`.
2. Evaluation reports task metrics (top-1/top-5 accuracy) alongside loss.
   Accuracy, not a loss-derived quantity, is the primary metric for
   classification, so :meth:`calculate_primary_metric` receives both.
"""

from __future__ import annotations

import math
import os
import re
import signal
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.training.config import TrainConfig
from models.training.utility import AdaptiveLogger, flatten_dict

logger = AdaptiveLogger(__name__)

#: Consecutive non-finite losses tolerated before aborting. A single bad batch
#: is recoverable; a sustained run of them means the model has diverged and
#: further compute is wasted.
MAX_CONSECUTIVE_NAN = 10


def parse_eval_interval(eval_interval: int | str) -> tuple[str, int]:
    """Parse an eval interval into ``(unit, value)``.

    Accepts epoch-based (``"1ep"``) and batch-based (``"100ba"``) forms. A bare
    integer is treated as batch-based.
    """
    if isinstance(eval_interval, int):
        return ("ba", eval_interval)

    match = re.match(r"^(\d+)(ep|ba)$", str(eval_interval))
    if not match:
        raise ValueError(
            f"Unsupported eval_interval format: {eval_interval!r}. "
            "Expected '<N>ep' (epoch-based) or '<N>ba' (batch-based)."
        )
    return (match.group(2), int(match.group(1)))


#: Characters MLflow permits in a metric name. Anything else raises and aborts
#: the run at the first log call, potentially hours in.
_INVALID_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_\-. :/]")


def sanitise_metric_name(name: str) -> str:
    """Replace characters that experiment trackers reject in metric names.

    A defensive measure so a subclass returning, say, ``"acc@1"`` degrades to a
    renamed metric rather than killing a long training run at its first
    evaluation.
    """
    return _INVALID_METRIC_CHARS.sub("_", name)


class TrainingPipeline(ABC):
    """Base training loop. Subclasses define the model, loss, and metrics."""

    def __init__(
        self,
        config: TrainConfig,
        accelerator: Accelerator,
        exp_name: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
        eval_interval: int | str,
        save_interval: int,
        results_path: Path,
        gradient_accumulation_steps: int = 1,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        | Sequence[torch.optim.lr_scheduler.LRScheduler]
        | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = config
        self.accelerator = accelerator
        self.exp_name = exp_name
        self.run_name = run_name
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.results_path = results_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    # --- Subclass contract -------------------------------------------------
    @abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Run the forward pass.

        Returns ``(loss, logits, targets)``. Logits and targets are returned so
        the base class can compute metrics without a second forward pass.
        """

    @abstractmethod
    def compute_metrics(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute task metrics for one evaluation batch."""

    @abstractmethod
    def calculate_primary_metric(self, eval_loss: float, metrics: dict[str, float]) -> float:
        """Reduce evaluation results to the single metric used for model selection."""

    @abstractmethod
    def is_higher_better_primary_metric(self) -> bool: ...

    @abstractmethod
    def get_primary_metric_name(self) -> str: ...

    # --- Data --------------------------------------------------------------
    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self) -> DataLoader:
        return self.validation_loader

    # --- Checkpointing -----------------------------------------------------
    def _get_checkpoint_load_path(self) -> str | None:
        """Resolve the checkpoint to resume from, if any."""
        if not self.config.continue_training:
            return None

        continue_folder = self.config.continue_folder or "latest-epoch"
        checkpoint_path = self.results_path / "accelerate_states" / "checkpoints" / continue_folder

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"continue_training=True but no checkpoint at {checkpoint_path}. "
                "Cannot resume without a valid checkpoint."
            )

        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        return str(checkpoint_path)

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist training state and a plain weights export.

        ``accelerator.save_state`` is a collective and must run on every rank;
        gating it on ``is_main_process`` deadlocks under FSDP or DeepSpeed.
        Filesystem manipulation stays on the main process only.
        """
        accelerator = self.accelerator
        base_dir = self.results_path / "accelerate_states"
        checkpoint_dir = base_dir / "checkpoints" / f"epoch_{epoch}"

        if accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Barrier: every rank must see the directory before save_state writes.
        accelerator.wait_for_everyone()

        accelerator.save_state(str(checkpoint_dir))

        if not accelerator.is_main_process:
            return

        logger.info("Saved accelerator state to %s", checkpoint_dir)
        self._save_weights(base_dir / "weights" / f"epoch_{epoch}")

        # Atomic symlink swap: create a temporary link then os.replace, which is
        # atomic on POSIX, so `latest-epoch` is never missing even if the
        # process dies mid-update.
        latest = base_dir / "checkpoints" / "latest-epoch"
        tmp_link = latest.parent / f".latest-epoch-tmp-{epoch}"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        tmp_link.symlink_to(f"epoch_{epoch}")
        # Path.replace is atomic on POSIX, so `latest-epoch` is never absent.
        tmp_link.replace(latest)

        self._upload_artifacts_to_mlflow(checkpoint_dir, epoch)

    def _save_weights(self, save_path: Path) -> None:
        """Export unwrapped model weights for serving and ONNX conversion.

        Written as safetensors rather than a pickled ``state_dict``.
        ``torch.load`` on a pickle executes arbitrary code during
        deserialisation, so the artefact the serving stack consumes must not be
        one — the serving path should never need ``weights_only=False``.

        ``save_model`` (rather than ``save_file``) handles tied/shared tensors,
        which safetensors cannot represent directly and which occur in several
        timm backbones.
        """
        from safetensors.torch import save_model

        save_path.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        save_model(unwrapped, str(save_path / "model.safetensors"))
        logger.info("Saved model weights to %s", save_path)

    def _upload_artifacts_to_mlflow(self, checkpoint_dir: Path, epoch: int) -> None:
        """Best-effort artefact upload; never fails the run."""
        if not self._mlflow_enabled():
            return

        try:
            import mlflow

            tracker = self.accelerator.get_tracker("mlflow", unwrap=True)
            run_id = tracker.info.run_id
            if checkpoint_dir.exists():
                mlflow.log_artifacts(
                    str(checkpoint_dir),
                    artifact_path=f"checkpoints/epoch_{epoch}",
                    run_id=run_id,
                )
                logger.info("Uploaded checkpoint to MLflow: checkpoints/epoch_%d", epoch)
        except Exception:
            logger.exception("Failed to upload artifacts to MLflow")

    # --- Tracking ----------------------------------------------------------
    def _mlflow_enabled(self) -> bool:
        """MLflow is active when a tracking URI or ARN is configured."""
        return bool(os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_ARN"))

    def _build_hyperparameters(self) -> dict[str, Any]:
        """Flatten the config for tracker hyperparameter logging."""
        from models.training.config import to_container

        try:
            return flatten_dict(to_container(self.config))
        except (TypeError, ValueError):
            # Loud: without hyperparameters the tracked run cannot be reproduced.
            logger.error(
                "Failed to resolve config for hyperparameter logging -- the tracked run "
                "will have no hyperparameters and is not fully reproducible",
                exc_info=True,
            )
            return {}

    # --- Evaluation --------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, model: torch.nn.Module) -> tuple[float, dict[str, float]]:
        """Run the evaluation loop, returning mean loss and mean metrics."""
        model.eval()
        eval_dataloader = self.get_eval_dataloader()
        eval_subset = self.config.training.eval_subset_num_batches
        total = eval_subset if eval_subset > 0 else len(eval_dataloader)

        total_loss = 0.0
        metric_sums: dict[str, float] = {}
        num_batches = 0

        progress = tqdm(
            total=total, desc="Eval", disable=not self.accelerator.is_local_main_process
        )

        for step, batch in enumerate(eval_dataloader):
            if 0 < eval_subset <= step:
                break

            loss, logits, targets = self.compute_loss(model, batch)

            # Gather across ranks so the reported figure reflects the whole
            # eval set, not just this process's shard.
            gathered_loss = self.accelerator.gather_for_metrics(loss.detach().float())
            batch_loss = (
                gathered_loss.mean().item()
                if isinstance(gathered_loss, Tensor)
                else float(loss.detach())
            )

            gathered_logits = self.accelerator.gather_for_metrics(logits.detach())
            gathered_targets = self.accelerator.gather_for_metrics(targets.detach())
            for name, value in self.compute_metrics(gathered_logits, gathered_targets).items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value

            total_loss += batch_loss
            num_batches += 1
            progress.update(1)
            progress.set_postfix(loss=batch_loss)

        progress.close()
        model.train()

        if num_batches == 0:
            logger.warning("Evaluation had 0 batches")
            return float("inf"), {}

        return total_loss / num_batches, {k: v / num_batches for k, v in metric_sums.items()}

    def _log_eval_metrics(
        self,
        eval_loss: float,
        metrics: dict[str, float],
        step: int,
        best_metrics: list[float],
    ) -> float:
        """Log evaluation results and return the primary metric."""
        metric_name = self.get_primary_metric_name()
        metric_value = self.calculate_primary_metric(eval_loss, metrics)

        if not self.accelerator.is_main_process:
            return metric_value

        formatted = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        logger.info("Eval loss=%.6f %s", eval_loss, formatted)
        logger.info("%s: %.6f", metric_name, metric_value)

        best_metrics.append(metric_value)
        self.accelerator.log(
            {
                "loss/eval": eval_loss,
                sanitise_metric_name(metric_name): metric_value,
                **{sanitise_metric_name(k): v for k, v in metrics.items()},
            },
            step=step,
        )

        valid = [m for m in best_metrics if math.isfinite(m)]
        if valid:
            best = max(valid) if self.is_higher_better_primary_metric() else min(valid)
            logger.info("Best %s so far: %.6f", metric_name, best)

        return metric_value

    # --- Training ----------------------------------------------------------
    def run(self) -> dict[str, float]:
        """Execute training. Returns a summary of the final and best metrics."""
        sigterm_received = False

        def _sigterm_handler(signum: int, frame: object) -> None:  # noqa: ARG001 - signal API
            nonlocal sigterm_received
            sigterm_received = True
            logger.warning("SIGTERM received -- will checkpoint and exit after the current step")

        prev_handler = signal.signal(signal.SIGTERM, _sigterm_handler)

        if self._mlflow_enabled():
            init_kwargs: dict[str, Any] = {}
            if self.run_name:
                init_kwargs["run_name"] = self.run_name
            self.accelerator.init_trackers(
                project_name=self.exp_name,
                config=self._build_hyperparameters(),
                init_kwargs={"mlflow": init_kwargs},
            )
            logger.info(
                "MLflow tracking enabled for experiment=%s run=%s", self.exp_name, self.run_name
            )
        else:
            logger.warning("Experiment tracking is disabled -- metrics will not be recorded")

        model = self.model
        optimizer = self.optimizer
        if optimizer is not None:
            model, optimizer, self.train_loader, self.validation_loader = self.accelerator.prepare(
                model, optimizer, self.train_loader, self.validation_loader
            )
        else:
            model = self.accelerator.prepare(model)

        # Schedulers are stepped manually; see the module docstring.
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        if self.lr_scheduler is not None:
            schedulers = (
                list(self.lr_scheduler)
                if isinstance(self.lr_scheduler, Sequence)
                else [self.lr_scheduler]
            )

        eval_unit, eval_value = parse_eval_interval(self.eval_interval)

        checkpoint_path = self._get_checkpoint_load_path()
        if checkpoint_path:
            self.accelerator.load_state(checkpoint_path)

        train_subset = self.config.training.train_subset_num_batches
        step_every_batch = self.config.training.step_schedulers_every_batch
        max_grad_norm = self.config.training.max_grad_norm

        best_metrics: list[float] = []
        global_batch_idx = 0
        consecutive_nan_count = 0
        last_saved_epoch = -1
        start_epoch = self.config.training.start_epoch_at
        final_metrics: dict[str, float] = {}
        final_eval_loss = float("inf")

        logger.info("Starting training for %d epochs", self.num_epochs)

        try:
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                model.train()
                epoch_loss = 0.0
                num_steps = 0
                epoch_batch_idx = 0

                # Accumulate on-device and materialise once per optimizer step:
                # calling .item() per microbatch forces a host-device sync and
                # would report only the last microbatch rather than the true
                # accumulation-window average.
                running_loss = torch.zeros((), device=self.accelerator.device)
                accum_count = 0

                # Respect train_subset_num_batches so the progress bar reflects
                # the work actually scheduled, not the full split.
                available_steps = math.ceil(
                    len(self.get_train_dataloader()) / self.gradient_accumulation_steps
                )
                train_total = (
                    min(train_subset, available_steps) if train_subset > 0 else available_steps
                )
                progress = tqdm(
                    total=train_total,
                    desc=f"Epoch {epoch}",
                    disable=not self.accelerator.is_local_main_process,
                )

                for batch in self.get_train_dataloader():
                    if 0 < train_subset <= epoch_batch_idx:
                        break

                    with self.accelerator.accumulate(model):
                        loss, _, _ = self.compute_loss(model, batch)

                        if not torch.isfinite(loss):
                            consecutive_nan_count += 1
                            if consecutive_nan_count >= MAX_CONSECUTIVE_NAN:
                                raise RuntimeError(
                                    f"{MAX_CONSECUTIVE_NAN} consecutive non-finite losses -- "
                                    "aborting training"
                                )
                            logger.warning(
                                "Non-finite loss at step %d (%d consecutive) -- skipping batch",
                                global_batch_idx,
                                consecutive_nan_count,
                            )
                            if optimizer is not None:
                                optimizer.zero_grad(set_to_none=True)
                            continue
                        consecutive_nan_count = 0

                        self.accelerator.backward(loss)

                        if optimizer is not None:
                            # clip_grad_norm_ is a no-op unless gradients are
                            # synced, so this correctly clips the full
                            # accumulated gradient rather than each microbatch.
                            if max_grad_norm is not None and max_grad_norm > 0:
                                self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        running_loss += loss.detach().float()
                        accum_count += 1

                    if self.accelerator.sync_gradients:
                        gathered = self.accelerator.gather(
                            running_loss / max(accum_count, 1)
                        ).mean()
                        step_loss = gathered.item()
                        running_loss = torch.zeros_like(running_loss)
                        accum_count = 0

                        epoch_loss += step_loss
                        num_steps += 1
                        global_batch_idx += 1
                        epoch_batch_idx += 1

                        progress.update(1)
                        progress.set_postfix(loss=step_loss)
                        self.accelerator.log({"loss/train_step": step_loss}, step=global_batch_idx)

                        if optimizer is not None:
                            self.accelerator.log(
                                {
                                    f"lr/pg{i}": pg["lr"]
                                    for i, pg in enumerate(optimizer.param_groups)
                                },
                                step=global_batch_idx,
                            )

                        if step_every_batch:
                            for scheduler in schedulers:
                                scheduler.step()

                        if eval_unit == "ba" and global_batch_idx % eval_value == 0:
                            final_eval_loss, final_metrics = self._evaluate(model)
                            self._log_eval_metrics(
                                final_eval_loss, final_metrics, global_batch_idx, best_metrics
                            )

                        if sigterm_received:
                            break

                progress.close()

                if not step_every_batch:
                    for scheduler in schedulers:
                        scheduler.step()

                if num_steps == 0:
                    # Logging 0.0 here would falsify the tracked history for an
                    # epoch that never produced an optimizer step.
                    logger.warning(
                        "Epoch %d produced 0 optimizer steps -- skipping train-loss log", epoch
                    )
                else:
                    avg_train_loss = epoch_loss / num_steps
                    logger.info("Epoch %d: train_loss=%.6f", epoch, avg_train_loss)
                    self.accelerator.log({"loss/train": avg_train_loss}, step=global_batch_idx)

                if eval_unit == "ep" and (epoch + 1) % eval_value == 0:
                    final_eval_loss, final_metrics = self._evaluate(model)
                    self._log_eval_metrics(
                        final_eval_loss, final_metrics, global_batch_idx, best_metrics
                    )

                if (epoch + 1) % self.save_interval == 0:
                    self.accelerator.wait_for_everyone()
                    self._save_checkpoint(epoch)
                    last_saved_epoch = epoch

                if sigterm_received:
                    if last_saved_epoch != epoch:
                        logger.warning("SIGTERM: checkpointing epoch %d before exit", epoch)
                        self.accelerator.wait_for_everyone()
                        self._save_checkpoint(epoch)
                    else:
                        logger.warning("SIGTERM: epoch %d already checkpointed, exiting", epoch)
                    break

            self.accelerator.wait_for_everyone()
        finally:
            try:
                self.accelerator.end_training()
            except Exception:
                logger.exception("Failed to finalise accelerator (tracking run may be orphaned)")
            signal.signal(signal.SIGTERM, prev_handler)

        logger.info("Training complete")

        valid = [m for m in best_metrics if math.isfinite(m)]
        summary = {
            "final_eval_loss": final_eval_loss,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
        if valid:
            summary[f"best_{self.get_primary_metric_name()}"] = (
                max(valid) if self.is_higher_better_primary_metric() else min(valid)
            )
        return summary
