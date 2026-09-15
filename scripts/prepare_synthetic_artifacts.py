#!/usr/bin/env python3
"""Write tiny synthetic artefacts so Compose can start without trained weights.

Used by CI's end-to-end job. The graphs match the serving contract — dynamic
batch, named inputs, the right ranks — in a few kilobytes rather than 87 MB.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    root = args.artifacts_dir
    onnx_dir = root / "onnx"
    _classifier(onnx_dir / "classifier_fp32.onnx", classes=args.classes, size=args.image_size)
    _classifier(onnx_dir / "classifier_int8.onnx", classes=args.classes, size=args.image_size)

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
