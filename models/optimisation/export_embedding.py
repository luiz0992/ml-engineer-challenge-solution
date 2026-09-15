"""Image similarity search: embedding export and index construction.

The third model is a dedicated ViT-Small fine-tuned with supervised
contrastive loss (:mod:`models.training.embedding_pipeline`). Classification
features collapse intra-class variation by construction; a contrastive
objective does the opposite.

A run trained as ``task: embedding`` already has ``num_classes=0`` and loads
strictly. A classification run is still accepted: the head is dropped with
``strict=False`` so an older artefact layout keeps working.

**Why cosine similarity.** Embeddings are L2-normalised and the index uses inner
product, which for unit vectors is exactly cosine similarity. Raw L2 distance on
un-normalised features is dominated by vector magnitude, which for transformer
features correlates with image contrast rather than content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: ViT-Small's penultimate feature width.
EMBEDDING_DIM = 384

#: Index size. Enough to make retrieval meaningful across all 200 classes while
#: keeping the artefact small; the full 100k training set would be a 150 MB
#: index for a demonstration.
DEFAULT_INDEX_SIZE = 20_000


@dataclass(slots=True)
class IndexMetadata:
    """Describes a built similarity index."""

    num_vectors: int
    embedding_dim: int
    metric: str
    source_model: str
    image_size: int
    index_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_vectors": self.num_vectors,
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "source_model": self.source_model,
            "image_size": self.image_size,
            "index_type": self.index_type,
        }


def build_normalised_embedder(backbone: Any) -> Any:
    """Wrap a backbone so it emits unit-norm embeddings.

    Module level rather than nested inside the exporter, so verification can
    construct exactly the transformation that was exported. When this lived
    inside `export_embedding_onnx` it was unreachable, and a verifier written
    against the bare backbone compared un-normalised features to normalised
    ones -- reporting a max difference of 7.59 between vectors that cannot
    differ by more than 2.0, which is a broken harness rather than a broken
    artefact.

    L2 normalisation is baked into the graph so the serving code cannot forget
    it: an un-normalised query against a normalised index ranks by magnitude
    rather than similarity.
    """
    import torch
    from torch import nn

    class _NormalisedEmbedder(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.backbone = inner

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.backbone(images)
            return torch.nn.functional.normalize(features, p=2.0, dim=-1)

    return _NormalisedEmbedder(backbone)


def build_embedding_model(run_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load a dedicated embedding run, or drop the head of a classifier run.

    A ``task: embedding`` checkpoint already has ``num_classes=0``. A
    classification run is still accepted: ``strict=False`` ignores head
    tensors so an older artefact layout keeps working. Either way the return
    value is checked so a genuinely incomplete load still fails.
    """
    import timm
    from safetensors.torch import load_file

    config = json.loads((run_dir / "config.json").read_text())
    model_name = config["model"]["name"]

    weights_root = run_dir / "accelerate_states" / "weights"
    candidates = sorted(weights_root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    if not candidates:
        raise FileNotFoundError(f"No weights under {weights_root}")

    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    state_dict = load_file(candidates[-1] / "model.safetensors")

    # A dedicated embedding run has no classifier head, so the load is strict.
    # A classification run still works: the head tensors are unexpected and
    # ignored, which is the original reuse path.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected_non_head = [
        k for k in unexpected if not k.startswith(("head.", "fc.", "classifier."))
    ]
    if missing or unexpected_non_head:
        raise RuntimeError(
            f"Unexpected weight mismatch loading the embedding backbone. "
            f"Missing: {missing}. Unexpected (non-head): {unexpected_non_head}."
        )

    model.eval()
    logger.info("Built embedding model from %s (%d-d features)", model_name, model.num_features)
    return model, {
        "model_name": model_name,
        "image_size": config["dataloader"]["image_size"],
        "embedding_dim": model.num_features,
    }


def export_embedding_onnx(
    run_dir: Path,
    output_path: Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Export the backbone as an ONNX embedding extractor.

    L2 normalisation is baked into the graph so the serving code cannot forget
    it. An un-normalised query against a normalised index silently returns
    rankings ordered by vector magnitude rather than similarity.
    """
    import torch

    from models.optimisation.export import ExportError

    model, metadata = build_embedding_model(run_dir)
    wrapper = build_normalised_embedder(model).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = metadata["image_size"]
    dummy = torch.randn(2, 3, size, size)

    logger.info("Exporting embedding model to ONNX")
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output_path),
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch_size"}, "embeddings": {0: "batch_size"}},
            opset_version=18,
            do_constant_folding=True,
        )
    except Exception as exc:
        raise ExportError(f"Embedding ONNX export failed: {exc}") from exc

    _consolidate(output_path)

    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))

    max_diff = None
    if verify:
        max_diff = _verify_embeddings(wrapper, output_path, size)
        logger.info("PyTorch vs ONNX Runtime max |diff| = %.3e", max_diff)
        if max_diff > 1e-4:
            raise ExportError(
                f"Embedding export diverges from PyTorch (max |diff| = {max_diff:.3e})"
            )

    metadata["max_abs_diff"] = max_diff
    return metadata


def _consolidate(onnx_path: Path) -> None:
    """Inline external tensor data so the artefact is a single file."""
    import onnx

    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not sidecar.exists():
        return
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()


def _verify_embeddings(wrapper: Any, onnx_path: Path, image_size: int) -> float:
    """Compare PyTorch and ONNX embeddings, and confirm they are unit-norm."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    generator = torch.Generator().manual_seed(0)
    max_diff = 0.0

    for batch_size in (1, 3):
        sample = torch.randn(batch_size, 3, image_size, image_size, generator=generator)
        with torch.no_grad():
            expected = wrapper(sample).numpy()
        actual = session.run(None, {input_name: sample.numpy()})[0]

        max_diff = max(max_diff, float(np.abs(expected - actual).max()))

        # Normalisation must survive the export: an un-normalised query against
        # a normalised index ranks by magnitude rather than similarity.
        norms = np.linalg.norm(actual, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(f"Exported embeddings are not unit-norm: {norms}")

    return max_diff


def build_index(
    embeddings: np.ndarray,
    labels: list[int],
    paths: list[str],
    output_dir: Path,
    *,
    source_model: str,
    image_size: int,
) -> IndexMetadata:
    """Build and persist a FAISS index over the supplied embeddings.

    ``IndexFlatIP`` performs exact search. An approximate index (IVF, HNSW)
    would be faster above roughly a million vectors, but at 20,000 exact search
    takes under a millisecond and avoids both a training step and the recall
    loss that approximation brings. Choosing an approximate index before the
    exact one is too slow is premature.
    """
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # FAISS requires contiguous memory; a sliced or transposed array will
    # otherwise be silently copied or rejected depending on version.
    embeddings = np.ascontiguousarray(embeddings)

    norms = np.linalg.norm(embeddings, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Embeddings are not unit-norm, so inner product will not equal cosine similarity."
        )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "similarity.index"))

    # Labels and paths are stored alongside: the index returns row numbers, and
    # without this mapping a result is an integer with no meaning.
    (output_dir / "similarity_manifest.json").write_text(
        json.dumps({"labels": labels, "paths": paths}, indent=2)
    )

    metadata = IndexMetadata(
        num_vectors=int(index.ntotal),
        embedding_dim=int(embeddings.shape[1]),
        metric="cosine",
        source_model=source_model,
        image_size=image_size,
        index_type="IndexFlatIP",
    )
    (output_dir / "similarity_metadata.json").write_text(json.dumps(metadata.as_dict(), indent=2))

    logger.info(
        "Built FAISS index: %d vectors of dimension %d (%s)",
        metadata.num_vectors,
        metadata.embedding_dim,
        metadata.index_type,
    )
    return metadata


def append_to_index(
    artifacts_dir: Path,
    embeddings: np.ndarray,
    labels: list[int],
    paths: list[str],
) -> IndexMetadata:
    """Add vectors to an existing index without rebuilding it.

    ``IndexFlatIP`` supports incremental ``add``, so new images cost only their
    own embedding time rather than a full re-embed of the corpus. This is the
    difference between adding a hundred images in seconds and re-running a
    twenty-minute job.

    The manifest is extended in the same order, because FAISS assigns row
    numbers sequentially and any divergence between index position and manifest
    position silently mislabels every result after the first mismatch. The two
    are written together and their lengths are checked afterwards.

    An approximate index (IVF, HNSW) would need retraining as the distribution
    shifts, which is one more reason exact search is the right default until
    the corpus is large enough to require otherwise.
    """
    import faiss

    index_path = artifacts_dir / "similarity.index"
    manifest_path = artifacts_dir / "similarity_manifest.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No index at {index_path}. Build one first with scripts/build_similarity_index.py."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    norms = np.linalg.norm(embeddings, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "New embeddings are not unit-norm; inner product would no longer "
            "equal cosine similarity for these rows only, making their scores "
            "incomparable with the rest of the index."
        )

    index = faiss.read_index(str(index_path))
    if index.d != embeddings.shape[1]:
        raise ValueError(
            f"Index has dimension {index.d} but the new embeddings have "
            f"{embeddings.shape[1]}. They come from a different model."
        )

    manifest = json.loads(manifest_path.read_text())

    before = int(index.ntotal)
    index.add(embeddings)
    manifest["labels"].extend(labels)
    manifest["paths"].extend(paths)

    if index.ntotal != len(manifest["labels"]):
        raise RuntimeError(
            f"Index and manifest diverged: {index.ntotal} vectors against "
            f"{len(manifest['labels'])} entries. Neither has been written."
        )

    # Written only after the consistency check, so a failure leaves the
    # existing index and manifest intact rather than half-updated.
    faiss.write_index(index, str(index_path))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    metadata_path = artifacts_dir / "similarity_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["num_vectors"] = int(index.ntotal)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Appended %d vectors to the index (%d -> %d)",
        len(labels),
        before,
        index.ntotal,
    )
    return IndexMetadata(
        num_vectors=int(index.ntotal),
        embedding_dim=int(index.d),
        metric=metadata["metric"],
        source_model=metadata["source_model"],
        image_size=metadata["image_size"],
        index_type=metadata["index_type"],
    )


def update_index_labels(
    artifacts_dir: Path,
    updates: dict[int, int],
    *,
    path_updates: dict[int, str] | None = None,
) -> int:
    """Change the labels of indexed vectors without rebuilding the index.

    The FAISS index stores only vectors; labels and paths live in the manifest
    beside it. Relabelling is therefore a manifest edit and needs no re-embedding
    at all — a distinction worth making explicit, because "the index must be
    rebuilt to change a label" was previously stated as a limitation and is
    simply not true.

    ``updates`` maps row number to new class index. Rows are validated against
    the index size first: a row number beyond the end would extend the manifest
    and silently desynchronise it from the index, mislabelling every result
    after that point.

    Returns the number of rows changed.
    """
    import faiss

    index_path = artifacts_dir / "similarity.index"
    manifest_path = artifacts_dir / "similarity_manifest.json"

    if not index_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"No index or manifest in {artifacts_dir}")

    index = faiss.read_index(str(index_path))
    manifest = json.loads(manifest_path.read_text())

    size = int(index.ntotal)
    out_of_range = [row for row in {**updates, **(path_updates or {})} if not 0 <= row < size]
    if out_of_range:
        raise IndexError(
            f"Row numbers outside the index (size {size}): {sorted(out_of_range)[:10]}. "
            f"Writing them would desynchronise the manifest from the index and "
            f"mislabel every subsequent result."
        )

    changed = 0
    for row, label in updates.items():
        if manifest["labels"][row] != label:
            manifest["labels"][row] = label
            changed += 1

    for row, path in (path_updates or {}).items():
        if manifest["paths"][row] != path:
            manifest["paths"][row] = path
            changed += 1

    if len(manifest["labels"]) != size or len(manifest["paths"]) != size:
        raise RuntimeError(
            f"Manifest length ({len(manifest['labels'])} labels, "
            f"{len(manifest['paths'])} paths) does not match the index ({size}). "
            f"Nothing has been written."
        )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Updated %d manifest entries (no re-embedding required)", changed)
    return changed
