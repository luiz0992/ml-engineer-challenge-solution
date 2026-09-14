"""Tests for image similarity search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from api.exceptions import ModelUnavailableError
from api.services.similarity_service import SimilarityIndex
from tests.conftest import make_image_bytes

pytestmark = pytest.mark.unit


class TestIndexLoading:
    def test_loads_index_and_manifest(self, similarity_artifacts: Path) -> None:
        index = SimilarityIndex.load(similarity_artifacts)

        assert index.size == 50
        assert len(index.labels) == 50
        assert len(index.paths) == 50

    def test_missing_index_explains_how_to_build_one(self, artifacts_dir: Path) -> None:
        with pytest.raises(ModelUnavailableError) as exc_info:
            SimilarityIndex.load(artifacts_dir)

        assert "build_similarity_index" in exc_info.value.message

    def test_index_and_manifest_must_agree(self, similarity_artifacts: Path) -> None:
        """A size mismatch means they came from different runs.

        Loading anyway would attach the wrong label and path to every single
        result -- a silent corruption that looks like a working search.
        """
        manifest = json.loads((similarity_artifacts / "similarity_manifest.json").read_text())
        manifest["labels"] = manifest["labels"][:10]
        (similarity_artifacts / "similarity_manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ModelUnavailableError, match="out of sync"):
            SimilarityIndex.load(similarity_artifacts)


class TestSimilaritySearch:
    async def test_returns_ranked_neighbours(self, similarity_service: Any) -> None:
        response = await similarity_service.find_similar(
            make_image_bytes(), correlation_id="c", top_k=5
        )

        assert len(response.results) == 5
        assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]

        similarities = [r.similarity for r in response.results]
        assert similarities == sorted(similarities, reverse=True)

    async def test_similarity_is_a_cosine(self, similarity_service: Any) -> None:
        """Both query and index vectors are unit-norm, so scores lie in [-1, 1].

        A value outside that range would mean the normalisation was lost, and
        rankings would be ordered by vector magnitude rather than similarity.
        """
        response = await similarity_service.find_similar(
            make_image_bytes(), correlation_id="c", top_k=10
        )

        assert all(-1.0 <= r.similarity <= 1.0 for r in response.results)

    async def test_results_carry_labels_and_references(self, similarity_service: Any) -> None:
        """FAISS returns row numbers; a result must be interpretable."""
        response = await similarity_service.find_similar(
            make_image_bytes(), correlation_id="c", top_k=3
        )

        for result in response.results:
            assert result.label
            assert isinstance(result.class_id, int)
            assert result.reference.endswith(".JPEG")

    async def test_top_k_is_honoured(self, similarity_service: Any) -> None:
        for k in (1, 5, 20):
            response = await similarity_service.find_similar(
                make_image_bytes(), correlation_id="c", top_k=k
            )
            assert len(response.results) == k

    async def test_requesting_more_than_the_index_holds(self, similarity_service: Any) -> None:
        """FAISS pads with -1; those rows must be dropped, not returned.

        A -1 row number would index the manifest from the end and silently
        return the wrong image.
        """
        response = await similarity_service.find_similar(
            make_image_bytes(), correlation_id="c", top_k=100
        )

        assert len(response.results) <= 50
        assert all(r.class_id >= 0 for r in response.results)

    async def test_minimum_similarity_filters_results(self, similarity_service: Any) -> None:
        unfiltered = await similarity_service.find_similar(
            make_image_bytes(seed=3), correlation_id="c", top_k=20
        )
        filtered = await similarity_service.find_similar(
            make_image_bytes(seed=3), correlation_id="c", top_k=20, min_similarity=0.5
        )

        assert len(filtered.results) <= len(unfiltered.results)
        assert all(r.similarity >= 0.5 for r in filtered.results)

    async def test_reports_index_size_and_provenance(self, similarity_service: Any) -> None:
        response = await similarity_service.find_similar(
            make_image_bytes(), correlation_id="trace-1", top_k=3
        )

        assert response.index_size == 50
        assert response.correlation_id == "trace-1"
        assert response.provenance.model_name == "tiny-imagenet-embedder"
        assert response.inference_time_ms > 0

    async def test_rejects_an_invalid_image(self, similarity_service: Any) -> None:
        from api.exceptions import InvalidImageError

        with pytest.raises(InvalidImageError):
            await similarity_service.find_similar(b"not an image", correlation_id="c")

    async def test_identical_queries_return_identical_results(
        self, similarity_service: Any
    ) -> None:
        """Retrieval must be deterministic, or results are untrustworthy."""
        image = make_image_bytes(seed=7)

        first = await similarity_service.find_similar(image, correlation_id="a", top_k=5)
        second = await similarity_service.find_similar(image, correlation_id="b", top_k=5)

        assert [r.reference for r in first.results] == [r.reference for r in second.results]


class TestIndexConstruction:
    def test_rejects_non_normalised_embeddings(self, tmp_path: Path) -> None:
        """Inner product equals cosine similarity only for unit vectors.

        Building an index from un-normalised embeddings produces rankings
        ordered by magnitude, which correlates with image contrast rather than
        content -- and nothing would raise at query time.
        """
        from models.optimisation.export_embedding import build_index

        vectors = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)

        with pytest.raises(ValueError, match="unit-norm"):
            build_index(
                vectors,
                labels=list(range(20)),
                paths=[f"{i}.jpg" for i in range(20)],
                output_dir=tmp_path,
                source_model="test",
                image_size=224,
            )

    def test_builds_a_searchable_index(self, tmp_path: Path) -> None:
        from models.optimisation.export_embedding import build_index

        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((30, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        metadata = build_index(
            vectors,
            labels=list(range(30)),
            paths=[f"{i}.jpg" for i in range(30)],
            output_dir=tmp_path,
            source_model="test",
            image_size=224,
        )

        assert metadata.num_vectors == 30
        assert metadata.metric == "cosine"
        assert (tmp_path / "similarity.index").exists()
        assert (tmp_path / "similarity_manifest.json").exists()

    def test_a_vector_retrieves_itself_first(self, tmp_path: Path) -> None:
        """The basic correctness property of an exact index."""
        import faiss

        from models.optimisation.export_embedding import build_index

        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((30, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        build_index(
            vectors,
            labels=list(range(30)),
            paths=[f"{i}.jpg" for i in range(30)],
            output_dir=tmp_path,
            source_model="test",
            image_size=224,
        )

        index = faiss.read_index(str(tmp_path / "similarity.index"))
        scores, indices = index.search(np.ascontiguousarray(vectors[5:6]), 1)

        assert indices[0][0] == 5
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


class TestIndexMaintenance:
    """Incremental updates, so the index is not a static artefact."""

    def test_append_adds_without_rebuilding(self, similarity_artifacts: Path) -> None:
        from models.optimisation.export_embedding import append_to_index

        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((5, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        metadata = append_to_index(
            similarity_artifacts,
            vectors,
            labels=[1, 2, 3, 4, 5],
            paths=[f"new_{i}.jpg" for i in range(5)],
        )

        assert metadata.num_vectors == 55

    def test_append_rejects_non_normalised_vectors(self, similarity_artifacts: Path) -> None:
        """Mixed normalisation makes scores incomparable across the index.

        Only the new rows would be wrong, and nothing raises at query time.
        """
        from models.optimisation.export_embedding import append_to_index

        with pytest.raises(ValueError, match="unit-norm"):
            append_to_index(
                similarity_artifacts,
                np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32),
                labels=[1, 2, 3],
                paths=["a.jpg", "b.jpg", "c.jpg"],
            )

    def test_append_rejects_mismatched_dimensions(self, similarity_artifacts: Path) -> None:
        """Different dimensions mean the embeddings came from another model."""
        from models.optimisation.export_embedding import append_to_index

        vectors = np.zeros((2, 16), dtype=np.float32)
        vectors[:, 0] = 1.0

        with pytest.raises(ValueError, match="dimension"):
            append_to_index(similarity_artifacts, vectors, labels=[1, 2], paths=["a.jpg", "b.jpg"])

    def test_append_keeps_index_and_manifest_aligned(self, similarity_artifacts: Path) -> None:
        import faiss

        from models.optimisation.export_embedding import append_to_index

        rng = np.random.default_rng(1)
        vectors = rng.standard_normal((7, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        append_to_index(
            similarity_artifacts,
            vectors,
            labels=list(range(7)),
            paths=[f"x_{i}.jpg" for i in range(7)],
        )

        index = faiss.read_index(str(similarity_artifacts / "similarity.index"))
        manifest = json.loads((similarity_artifacts / "similarity_manifest.json").read_text())

        assert index.ntotal == len(manifest["labels"]) == len(manifest["paths"])

    def test_labels_can_be_changed_without_re_embedding(self, similarity_artifacts: Path) -> None:
        """Relabelling is a manifest edit; the vectors do not change.

        This was previously documented as requiring a full rebuild, which was
        simply untrue.
        """
        from models.optimisation.export_embedding import update_index_labels

        changed = update_index_labels(similarity_artifacts, {0: 99, 1: 98})

        assert changed == 2
        manifest = json.loads((similarity_artifacts / "similarity_manifest.json").read_text())
        assert manifest["labels"][0] == 99
        assert manifest["labels"][1] == 98

    def test_label_update_rejects_rows_outside_the_index(self, similarity_artifacts: Path) -> None:
        """A row beyond the end would desynchronise manifest from index."""
        from models.optimisation.export_embedding import update_index_labels

        with pytest.raises(IndexError, match="outside the index"):
            update_index_labels(similarity_artifacts, {9999: 1})
