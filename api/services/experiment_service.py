"""Experiment configuration and variant routing for the serving path.

Bridges :mod:`models.validation.ab_testing` into request handling: an
experiment declares how traffic splits across model versions, and this resolves
each caller to a version before inference.

Configuration is loaded from a JSON file rather than code, so an experiment can
be started, reweighted, or stopped by changing a mounted file rather than
rebuilding and redeploying the image.

Two safety properties:

**A misconfigured experiment never breaks serving.** If the file is malformed,
names a version that is not loaded, or has weights that do not sum to one, the
experiment is dropped with a loud log and traffic goes to the active version.
An experiment is an optimisation; failing requests over one would be absurd.

**Assignment is validated against loaded models.** A variant pointing at a
version that was never loaded would 404 for that share of traffic, so variants
are checked at load time and the experiment is rejected if any is unservable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from api.logging_config import get_logger
from models.validation.ab_testing import Experiment, Variant

logger = get_logger(__name__)


@dataclass(slots=True)
class ExperimentRegistry:
    """Active experiments, keyed by the model they apply to."""

    experiments: dict[str, Experiment]

    @classmethod
    def empty(cls) -> ExperimentRegistry:
        return cls(experiments={})

    @classmethod
    def load(cls, path: Path, *, available_versions: dict[str, set[str]]) -> ExperimentRegistry:
        """Load experiments, dropping any that cannot be served.

        ``available_versions`` maps a model name to the versions currently
        loaded. A variant referencing anything else is unservable, and the
        whole experiment is rejected rather than routing a share of traffic to
        a 404.
        """
        if not path.exists():
            logger.info("no_experiments_configured", path=str(path))
            return cls.empty()

        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.error(
                "experiment_config_unreadable",
                path=str(path),
                error=str(exc),
                impact="all traffic goes to the active model version",
            )
            return cls.empty()

        experiments: dict[str, Experiment] = {}

        for entry in raw.get("experiments", []):
            try:
                model_name = entry["model_name"]
                experiment = Experiment(
                    name=entry["name"],
                    variants=[
                        Variant(
                            name=v["name"],
                            model_version=v["model_version"],
                            weight=float(v["weight"]),
                        )
                        for v in entry["variants"]
                    ],
                    enabled=bool(entry.get("enabled", True)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.error(
                    "experiment_invalid",
                    experiment=entry.get("name", "<unnamed>"),
                    error=str(exc),
                    impact="experiment ignored; traffic goes to the active version",
                )
                continue

            loaded = available_versions.get(model_name, set())
            unservable = [
                v.model_version for v in experiment.variants if v.model_version not in loaded
            ]
            if unservable:
                logger.error(
                    "experiment_references_unloaded_versions",
                    experiment=experiment.name,
                    model=model_name,
                    missing=unservable,
                    available=sorted(loaded),
                    impact="experiment ignored; traffic goes to the active version",
                )
                continue

            experiments[model_name] = experiment
            logger.info(
                "experiment_active",
                experiment=experiment.name,
                model=model_name,
                variants={v.name: v.weight for v in experiment.variants},
            )

        return cls(experiments=experiments)

    def resolve_version(self, model_name: str, user_id: str) -> tuple[str | None, str | None]:
        """Return ``(model_version, variant_name)`` for a caller.

        ``(None, None)`` means no experiment applies and the active version
        should serve, which is the overwhelmingly common path.
        """
        experiment = self.experiments.get(model_name)
        if experiment is None:
            return None, None

        variant = experiment.assign(user_id)
        return variant.model_version, variant.name

    @property
    def active_count(self) -> int:
        return len(self.experiments)
