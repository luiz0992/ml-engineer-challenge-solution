#!/usr/bin/env python3
"""Write tiny synthetic artefacts so Compose can start without trained weights.

Used by CI's end-to-end job. The graphs match the serving contract — dynamic
batch, named inputs, the right ranks — in a few kilobytes rather than 260 MB.

All three models are written, not just the classifier. An end-to-end job that
publishes only a classifier exercises only `/classify`: `/detect` and
`/similar` return 503, which a test asserting "not deployed" passes happily
while the code path it was meant to cover never runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _classifier(path: Path, *, classes: int, size: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.standard_normal((3, classes)).astype(np.float32), name="weight"
    )
    bias = numpy_helper.from_array(np.zeros(classes, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["projected"]),
            helper.make_node("Add", ["projected", "bias"], ["logits"]),
        ],
        "synthetic_classifier",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", classes])],
        [weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _detector(path: Path, *, classes: int, queries: int, size: int) -> None:
    """A detector matching RT-DETR's exported contract.

    Two outputs, ``(scores, boxes)``, both rank 3 with a dynamic batch axis.
    Weights are zero so the bias decides the answer: query 0 is a confident
    centre box, the rest fall below any sensible threshold. That makes the
    decode path — threshold, rank, cap, and the box transform back into the
    uploaded image's pixel space — observable end to end.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    score_bias = np.zeros(queries * classes, dtype=np.float32)
    score_bias[0] = 0.9  # query 0, class 0
    score_bias[classes] = 0.2  # query 1, class 0: below threshold

    box_bias = np.tile(np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32), queries)

    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["pixel_values"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "score_weight"], ["score_proj"]),
            helper.make_node("Add", ["score_proj", "score_bias"], ["score_flat"]),
            helper.make_node("Reshape", ["score_flat", "score_shape"], ["scores"]),
            helper.make_node("MatMul", ["flat", "box_weight"], ["box_proj"]),
            helper.make_node("Add", ["box_proj", "box_bias"], ["box_flat"]),
            helper.make_node("Reshape", ["box_flat", "box_shape"], ["boxes"]),
        ],
        "synthetic_detector",
        [
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, ["batch", 3, size, size]
            )
        ],
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", queries, classes]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["batch", queries, 4]),
        ],
        [
            numpy_helper.from_array(
                np.zeros((3, queries * classes), dtype=np.float32), name="score_weight"
            ),
            numpy_helper.from_array(score_bias, name="score_bias"),
            numpy_helper.from_array(
                np.zeros((3, queries * 4), dtype=np.float32), name="box_weight"
            ),
            numpy_helper.from_array(box_bias, name="box_bias"),
            numpy_helper.from_array(
                np.array([-1, queries, classes], dtype=np.int64), name="score_shape"
            ),
            numpy_helper.from_array(np.array([-1, queries, 4], dtype=np.int64), name="box_shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _embedder(path: Path, *, dim: int, size: int) -> None:
    """An embedder emitting unit-norm vectors, as the real export does.

    L2 normalisation is part of the graph rather than the serving code, so the
    synthetic model has to normalise too — an un-normalised query against a
    normalised index ranks by magnitude rather than similarity.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(1)
    weight = numpy_helper.from_array(rng.standard_normal((3, dim)).astype(np.float32), name="w")

    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "w"], ["raw"]),
            helper.make_node("ReduceL2", ["raw"], ["norm"], keepdims=1, axes=[1]),
            helper.make_node("Div", ["raw", "norm"], ["embeddings"]),
        ],
        "synthetic_embedder",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", dim])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _similarity_index(root: Path, *, dim: int, vectors: int, classes: int, size: int) -> None:
    """A FAISS index with the manifest that gives its row numbers meaning."""
    import faiss

    rng = np.random.default_rng(2)
    embeddings = rng.standard_normal((vectors, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings))
    faiss.write_index(index, str(root / "similarity.index"))

    (root / "similarity_manifest.json").write_text(
        json.dumps(
            {
                "labels": [i % classes for i in range(vectors)],
                "paths": [f"train/class_{i % classes}/img_{i}.JPEG" for i in range(vectors)],
            }
        )
    )
    (root / "similarity_metadata.json").write_text(
        json.dumps(
            {
                "num_vectors": vectors,
                "embedding_dim": dim,
                "metric": "cosine",
                "source_model": "synthetic",
                "image_size": size,
                "index_type": "IndexFlatIP",
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--detection-classes", type=int, default=4)
    parser.add_argument("--detection-queries", type=int, default=4)
    # RT-DETR's fixed input size. The serving code reads it from
    # detection_labels.json, so the graph and the metadata must agree.
    parser.add_argument("--detection-image-size", type=int, default=640)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--index-vectors", type=int, default=50)
    args = parser.parse_args(argv)

    root = args.artifacts_dir
    onnx_dir = root / "onnx"
    _classifier(onnx_dir / "classifier_fp32.onnx", classes=args.classes, size=args.image_size)
    _classifier(onnx_dir / "classifier_int8.onnx", classes=args.classes, size=args.image_size)

    _detector(
        onnx_dir / "detector_fp32.onnx",
        classes=args.detection_classes,
        queries=args.detection_queries,
        size=args.detection_image_size,
    )
    (root / "detection_labels.json").write_text(
        json.dumps(
            {
                "id2label": {str(i): f"object_{i}" for i in range(args.detection_classes)},
                "num_classes": args.detection_classes,
                "image_size": args.detection_image_size,
                "max_detections": args.detection_queries,
            },
            indent=2,
        )
    )

    _embedder(onnx_dir / "embedder_fp32.onnx", dim=args.embedding_dim, size=args.image_size)
    _similarity_index(
        root,
        dim=args.embedding_dim,
        vectors=args.index_vectors,
        classes=args.classes,
        size=args.image_size,
    )

    (root / "labels.json").write_text(
        json.dumps(
            {
                "class_names": [f"class_{i}" for i in range(args.classes)],
                "wnids": [f"n{i:08d}" for i in range(args.classes)],
                "image_size": args.image_size,
            },
            indent=2,
        )
    )
    (root / "metrics.json").write_text(json.dumps({"final_acc_top1": 0.0}, indent=2))
    print(f"Synthetic artefacts written to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
