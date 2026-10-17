#!/usr/bin/env python3
"""
Public-artifact launcher for canonical PatScout Table 5 Qwen inference.

The canonical inference implementation historically expects an artifact
tree rooted at experiments/. The released artifacts are stored under
reproducibility/artifacts/. This launcher validates the released layout,
creates a temporary compatibility tree using symbolic links, and invokes
the canonical inference script without modifying adapters or inputs.

Use --check-only to validate paths without importing torch or running
model inference.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONDITIONS = ["hard_only", "mixed_u"]
SEEDS = [13, 42, 77]
EXPECTED_ROWS = 1642


def parse_args():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifacts",
        type=Path,
        default=(
            project_root
            / "reproducibility/artifacts"
        ),
        help=(
            "Released reproducibility artifact root. "
            "Default: reproducibility/artifacts"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "reproducibility/generated/"
              "table5_qwen_recomputed"
        ),
        help="Persistent output directory",
    )

    parser.add_argument(
        "--legacy-script",
        type=Path,
        default=(
            project_root
            / "experiments/"
              "run_qwen_dense_hard_recomputed.py"
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate the public artifact mapping without "
            "running Qwen inference"
        ),
    )

    return parser.parse_args()


def validate_public_layout(artifacts):
    artifacts = artifacts.resolve()

    adapter_root = (
        artifacts
        / "table4/qwen_adapters"
    )

    dense_manifest = (
        artifacts
        / "table5/common/"
          "test_dense_hard_u.parquet"
    )

    qwen_artifacts = (
        artifacts
        / "table5/qwen"
    )

    required = [
        adapter_root,
        dense_manifest,
        qwen_artifacts,
    ]

    for path in required:
        if not path.exists():
            raise RuntimeError(
                f"Required public artifact missing: {path}"
            )

    adapters = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            adapter = (
                adapter_root
                / f"{condition}_seed{seed}"
            )

            for name in [
                "adapter_config.json",
                "adapter_model.safetensors",
            ]:
                path = adapter / name

                if not path.is_file():
                    raise RuntimeError(
                        f"Adapter artifact missing: {path}"
                    )

            config = json.loads(
                (adapter / "adapter_config.json").read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(config, dict):
                raise RuntimeError(
                    f"Invalid adapter config: {adapter}"
                )

            adapters.append(adapter)

    predictions = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            path = (
                qwen_artifacts
                / (
                    f"qwen3_reranker_{condition}_seed{seed}"
                    "_dense_hard_predictions.parquet"
                )
            )

            if not path.is_file():
                raise RuntimeError(
                    f"Public prediction/template missing: {path}"
                )

            predictions.append(path)

    return {
        "artifacts": artifacts,
        "adapter_root": adapter_root,
        "dense_manifest": dense_manifest,
        "qwen_artifacts": qwen_artifacts,
        "adapters": adapters,
        "predictions": predictions,
    }


def create_link(source, destination):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"Bridge destination already exists: {destination}"
        )

    destination.symlink_to(
        source.resolve(),
        target_is_directory=source.is_dir(),
    )


def build_bridge(layout, output):
    bridge = Path(
        tempfile.mkdtemp(
            prefix="patscout-table5-qwen-bridge-"
        )
    )

    control_root = (
        bridge
        / "experiments/model_generalization_v1/"
          "qwen3_reranker_0.6b_control"
    )

    dense_root = (
        bridge
        / "experiments/dense_hard_v1"
    )

    control_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    dense_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Support both historical adapter lookup styles:
    # T4/adapters/<condition>_seed<seed>
    # T4/<condition>_seed<seed>
    create_link(
        layout["adapter_root"],
        control_root / "adapters",
    )

    for adapter in layout["adapters"]:
        create_link(
            adapter,
            control_root / adapter.name,
        )

    create_link(
        layout["dense_manifest"],
        dense_root / "test_dense_hard_u.parquet",
    )

    create_link(
        layout["qwen_artifacts"],
        dense_root / "qwen_frozen_evaluation",
    )

    output = output.resolve()
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_link(
        output,
        dense_root / "qwen_frozen_evaluation_recomputed",
    )

    return bridge


def main():
    args = parse_args()

    layout = validate_public_layout(
        args.artifacts
    )

    if not args.legacy_script.is_file():
        raise RuntimeError(
            f"Canonical inference script missing: "
            f"{args.legacy_script}"
        )

    print("Public artifact mapping validated.")
    print("Adapters:", layout["adapter_root"])
    print("Dense-Hard:", layout["dense_manifest"])
    print("Table 5 Qwen artifacts:", layout["qwen_artifacts"])
    print("Adapter runs:", len(layout["adapters"]))
    print("Prediction/template files:", len(layout["predictions"]))

    if args.check_only:
        bridge = build_bridge(
            layout,
            args.output,
        )

        expected_bridge_paths = [
            (
                bridge
                / "experiments/model_generalization_v1/"
                  "qwen3_reranker_0.6b_control/adapters"
            ),
            (
                bridge
                / "experiments/dense_hard_v1/"
                  "test_dense_hard_u.parquet"
            ),
            (
                bridge
                / "experiments/dense_hard_v1/"
                  "qwen_frozen_evaluation"
            ),
            (
                bridge
                / "experiments/dense_hard_v1/"
                  "qwen_frozen_evaluation_recomputed"
            ),
        ]

        try:
            for path in expected_bridge_paths:
                if not path.exists():
                    raise RuntimeError(
                        f"Compatibility path missing: {path}"
                    )

            print("Temporary legacy tree: PASSED")
            print("Path bridge check: PASSED")
            print("Inference performed: NO")

        finally:
            shutil.rmtree(
                bridge,
                ignore_errors=True,
            )

        return

    bridge = build_bridge(
        layout,
        args.output,
    )

    environment = os.environ.copy()
    environment["XARTH_ARTIFACT_ROOT"] = str(
        bridge
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"

    command = [
        sys.executable,
        "-u",
        str(args.legacy_script.resolve()),
    ]

    print("Compatibility root:", bridge)
    print("Output:", args.output.resolve())
    print("Executing:", " ".join(command))

    try:
        result = subprocess.run(
            command,
            cwd=args.legacy_script.resolve().parents[1],
            env=environment,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Canonical Qwen inference failed with "
                f"return code {result.returncode}"
            )

    finally:
        shutil.rmtree(
            bridge,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
