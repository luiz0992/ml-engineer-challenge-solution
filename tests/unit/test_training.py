"""Tests for training-loop primitives that do not need a GPU or a dataset.

Skipped unless torch is installed — the modules under test import it at
module level. CI's preprocessing-equivalence job installs torch and fails
if these skip; the default job is allowed to skip them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="require torchvision")

import torch

from models.training.classification_pipeline import accuracy_topk, soft_target_cross_entropy
from models.training.embedding_pipeline import retrieval_recall_at_k, supervised_contrastive_loss
from models.training.pipeline import parse_eval_interval, sanitise_metric_name
from models.training.utility import calculate_steps_per_epoch, create_scheduler_params

pytestmark = pytest.mark.unit


class TestSoftTargetCrossEntropy:
    def test_one_hot_matches_hard_labels(self) -> None:
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        hard = torch.tensor([0, 1])
        soft = torch.nn.functional.one_hot(hard, num_classes=2).float()
        assert torch.isclose(
            soft_target_cross_entropy(logits, soft),
            torch.nn.functional.cross_entropy(logits, hard),
            atol=1e-5,
        )

    def test_confident_correct_is_near_zero(self) -> None:
        logits = torch.tensor([[20.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert soft_target_cross_entropy(logits, targets).item() < 1e-6


class TestAccuracyTopk:
    def test_perfect_top1(self) -> None:
        logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 1.0]])
        targets = torch.tensor([1, 0])
        metrics = accuracy_topk(logits, targets, topk=(1, 2))
        assert metrics["acc_top1"] == 1.0
        assert metrics["acc_top2"] == 1.0

    def test_soft_targets_use_argmax(self) -> None:
        logits = torch.tensor([[0.0, 5.0]])
        targets = torch.tensor([[0.1, 0.9]])
        assert accuracy_topk(logits, targets, topk=(1,))["acc_top1"] == 1.0


class TestContrastiveLoss:
    def test_identical_positives_beat_random(self) -> None:
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            dim=-1,
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(embeddings, labels)
        generator = torch.Generator().manual_seed(0)
        random = supervised_contrastive_loss(
            torch.nn.functional.normalize(torch.randn(4, 2, generator=generator), dim=-1),
            labels,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(random)
        assert loss.item() < random.item()

    def test_no_positives_is_zero_not_nan(self) -> None:
        embeddings = torch.nn.functional.normalize(torch.eye(3), dim=-1)
        labels = torch.tensor([0, 1, 2])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert loss.item() == 0.0
        assert torch.isfinite(loss)

    def test_recall_is_perfect_for_identical_pairs(self) -> None:
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            dim=-1,
        )
        labels = torch.tensor([0, 0, 1, 1])
        assert retrieval_recall_at_k(embeddings, labels, k=1)["recall_at1"] == 1.0


class TestSchedulerArithmetic:
    def test_subset_overrides_dataset_length(self) -> None:
        assert (
            calculate_steps_per_epoch(
                dataset_length=10_000, batch_size_train=128, train_subset_num_batches=8
            )
            == 8
        )

    def test_warmup_is_a_fraction_of_total_steps(self) -> None:
        params = create_scheduler_params(
            warmup_ratio=0.1,
            batch_size_train=32,
            dataset_length=320,
            num_epochs=2,
        )
        assert params.steps_per_epoch == 10
        assert params.num_training_steps == 20
        assert params.num_warmup_steps == 2

    def test_rejects_a_full_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_ratio"):
            create_scheduler_params(
                warmup_ratio=1.0,
                batch_size_train=32,
                dataset_length=320,
                num_epochs=1,
            )


class TestEvalInterval:
    def test_parses_epoch_and_batch_forms(self) -> None:
        assert parse_eval_interval("1ep") == ("ep", 1)
        assert parse_eval_interval("100ba") == ("ba", 100)
        assert parse_eval_interval(50) == ("ba", 50)

    def test_rejects_unknown_units(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            parse_eval_interval("1step")

    def test_sanitises_mlflow_rejected_characters(self) -> None:
        assert sanitise_metric_name("acc@1") == "acc_1"
