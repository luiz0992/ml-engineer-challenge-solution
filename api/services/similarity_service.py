"""Image similarity search.

Embeds a query image with the fine-tuned backbone and retrieves its nearest
neighbours from a FAISS index by cosine similarity.

**Cosine, not Euclidean.** Embeddings are unit-norm (normalisation is baked into
the exported graph), and inner product on unit vectors *is* cosine similarity.
Raw L2 distance on un-normalised transformer features is dominated by vector
magnitude, which tracks image contrast rather than content.

**What this retrieves.** Features optimised for classification deliberately
collapse intra-class variation — that is what makes a classifier work. Results
are therefore *semantically* similar (same category, comparable pose and colour)
rather than visually near-duplicate. For "show me more like this" that is
usually the intent; for near-duplicate detection it is not, and a perceptual
hash would be the right tool.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from api.config import Settings
from api.exceptions import ModelUnavailableError
from api.logging_config import get_logger
from api.models.responses import (
    ModelProvenance,
    SimilarImage,
    SimilarityResponse,
)
from api.services.model_service import ModelService
from api.utils.image_processing import PreprocessConfig, preprocess_image
from api.utils.validators import validate_image_upload

logger = get_logger(__name__)


class SimilarityIndex:
    """A FAISS index with the metadata needed to interpret its results."""

    def __init__(
        self, index: Any, labels: list[int], paths: list[str], metadata: dict[str, Any]
    ) -> None:
        self.index = index
        self.labels = labels
        self.paths = paths
        self.metadata = metadata

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @classmethod
    def load(cls, artifacts_dir: Path) -> SimilarityIndex:
        """Load an index and its manifest from disk.

        The manifest is mandatory: FAISS returns row numbers, and without the
        label and path mapping a result is an integer with no meaning.
        """
        import faiss

        index_path = artifacts_dir / "similarity.index"
        manifest_path = artifacts_dir / "similarity_manifest.json"
        metadata_path = artifacts_dir / "similarity_metadata.json"

        missing = [p for p in (index_path, manifest_path, metadata_path) if not p.exists()]
        if missing:
            raise ModelUnavailableError(
                "Similarity index not found. Build it with "
                "`python scripts/build_similarity_index.py`.",
                details={"missing": [str(p) for p in missing]},
            )

        index = faiss.read_index(str(index_path))
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads(metadata_path.read_text())

        if index.ntotal != len(manifest["labels"]):
            # A mismatch means the index and manifest came from different runs,
            # which would attach the wrong label to every result.
            raise ModelUnavailableError(
                f"Similarity index has {index.ntotal} vectors but the manifest "
                f"describes {len(manifest['labels'])}. They are out of sync."
            )

        logger.info("similarity_index_loaded", vectors=int(index.ntotal))
        return cls(index, manifest["labels"], manifest["paths"], metadata)


class SimilarityService:
    """Serves nearest-neighbour queries against the image index."""

    def __init__(
        self,
        model_service: ModelService,
        index: SimilarityIndex,
        settings: Settings,
        class_names: list[str],
    ) -> None:
        self.models = model_service
        self.index = index
        self.settings = settings
        self.class_names = class_names

    async def find_similar(
        self,
        image_bytes: bytes,
        *,
        correlation_id: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> SimilarityResponse:
        """Return the ``top_k`` most similar indexed images."""
        started = time.perf_counter()

        validate_image_upload(
            image_bytes,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )

        model = self.models.get_embedder()

        def _work() -> tuple[np.ndarray, np.ndarray]:
            config = PreprocessConfig(image_size=model.image_size)
            array = preprocess_image(image_bytes, config)
            embedding = self.models.run(model, array[None, ...])

            # FAISS requires contiguous float32; a non-contiguous array is
            # either silently copied or rejected depending on the build.
            query = np.ascontiguousarray(embedding.astype(np.float32))
            return self.index.index.search(query, top_k)

        scores, indices = await asyncio.to_thread(_work)

        results: list[SimilarImage] = []
        for score, row in zip(scores[0], indices[0], strict=True):
            # FAISS pads with -1 when fewer than k vectors exist.
            if row < 0 or float(score) < min_similarity:
                continue

            label_index = self.index.labels[int(row)]
            results.append(
                SimilarImage(
                    rank=len(results) + 1,
                    similarity=round(float(score), 6),
                    label=(
                        self.class_names[label_index]
                        if label_index < len(self.class_names)
                        else str(label_index)
                    ),
                    class_id=int(label_index),
                    reference=self.index.paths[int(row)],
                )
            )

        return SimilarityResponse(
            results=results,
            index_size=self.index.size,
            inference_time_ms=round((time.perf_counter() - started) * 1000, 2),
            correlation_id=correlation_id,
            provenance=ModelProvenance(
                model_name=model.name,
                model_version=model.version,
                backend=model.backend.value,
            ),
        )
