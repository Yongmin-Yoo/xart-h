#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SEEDS = [13, 42, 77]
TRAINING_CONDITIONS = ["hard_only", "mixed_u"]
EVALUATION_CONDITIONS = ["hard_u", "random_u"]
METRICS = ["roc_auc", "pairwise_accuracy"]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    order = np.argsort(p_values)
    ordered = p_values[order]

    adjusted_ordered = np.empty(count, dtype=float)
    running = 0.0

    for rank, value in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        adjusted_ordered[rank] = running

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def finite_bootstrap_p(distribution):
    distribution = np.asarray(distribution, dtype=float)
    repetitions = len(distribution)

    lower = (
        np.count_nonzero(distribution <= 0) + 1
    ) / (repetitions + 1)

    upper = (
        np.count_nonzero(distribution >= 0) + 1
    ) / (repetitions + 1)

    return min(1.0, 2.0 * min(lower, upper))


def discover_files(artifacts, model):
    if model == "minilm":
        root = artifacts / "table4/minilm_control"

        def filename(condition, seed):
            return (
                root
                / f"{condition}_seed{seed}_ordered_predictions.parquet"
            )

    elif model == "deberta":
        root = (
            artifacts
            / "table4/deberta_control/ordered_evaluation/predictions"
        )

        def filename(condition, seed):
            return (
                root
                / f"{condition}_seed{seed}_ordered_predictions.parquet"
            )

    elif model == "qwen":
        root = artifacts / "table4/qwen_control"

        def filename(condition, seed):
            return (
                root
                / (
                    f"qwen3_reranker_{condition}_seed{seed}_"
                    "ordered_predictions.parquet"
                )
            )

    else:
        raise ValueError(model)

    files = {
        (condition, seed): filename(condition, seed)
        for condition in TRAINING_CONDITIONS
        for seed in SEEDS
    }

    missing = [
        str(path)
        for path in files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing prediction files:\n"
            + "\n".join(missing)
        )

    return files


def load_ensemble(artifacts, model):
    files = discover_files(artifacts, model)
    frames = []

    required_columns = {
        "source_row_id",
        "query_id",
        "candidate_type",
        "evaluation_condition",
        "training_condition",
        "seed",
        "cited_probability",
        "prediction_row_id",
    }

    for (condition, seed), path in files.items():
        frame = pd.read_parquet(path)

        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise RuntimeError(
                f"{path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if len(frame) != 9896:
            raise RuntimeError(
                f"Unexpected row count in {path}: {len(frame)}"
            )

        if set(frame["training_condition"].unique()) != {condition}:
            raise RuntimeError(
                f"Training condition mismatch in {path}"
            )

        if set(frame["seed"].unique()) != {seed}:
            raise RuntimeError(
                f"Seed mismatch in {path}"
            )

        if frame["cited_probability"].isna().any():
            raise RuntimeError(
                f"NaN probability in {path}"
            )

        if not frame["cited_probability"].between(0, 1).all():
            raise RuntimeError(
                f"Probability outside [0,1] in {path}"
            )

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    key_columns = [
        "source_row_id",
        "query_id",
        "candidate_type",
        "evaluation_condition",
        "training_condition",
    ]

    seed_counts = (
        combined.groupby(
            key_columns,
            dropna=False,
        )["seed"]
        .nunique()
    )

    if not (seed_counts == 3).all():
        bad = seed_counts[seed_counts != 3]
        raise RuntimeError(
            f"{model}: incomplete seed alignment: "
            f"{len(bad)} rows"
        )

    ensemble = (
        combined.groupby(
            key_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            cited_probability=(
                "cited_probability",
                "mean",
            ),
            seed_probability_std=(
                "cited_probability",
                "std",
            ),
            seed_count=("seed", "nunique"),
        )
    )

    expected_rows = 9896 * 2
    if len(ensemble) != expected_rows:
        raise RuntimeError(
            f"{model}: expected {expected_rows} ensemble rows, "
            f"found {len(ensemble)}"
        )

    return ensemble


def cell_frame(ensemble, training, evaluation):
    frame = ensemble[
        (ensemble["training_condition"] == training)
        & (ensemble["evaluation_condition"] == evaluation)
    ].copy()

    negative_type = evaluation

    expected_types = {"cited", negative_type}
    observed_types = set(frame["candidate_type"].unique())

    if observed_types != expected_types:
        raise RuntimeError(
            f"Unexpected candidate types for "
            f"{training}/{evaluation}: {observed_types}"
        )

    frame["target"] = (
        frame["candidate_type"] == "cited"
    ).astype(np.int8)

    if len(frame) != 4948:
        raise RuntimeError(
            f"Expected 4,948 rows for {training}/{evaluation}; "
            f"found {len(frame)}"
        )

    if frame["query_id"].nunique() != 1642:
        raise RuntimeError(
            f"Expected 1,642 queries for "
            f"{training}/{evaluation}"
        )

    return frame


def query_pairwise_values(frame):
    values = {}

    for query_id, group in frame.groupby(
        "query_id",
        sort=False,
    ):
        cited = group.loc[
            group["target"] == 1,
            "cited_probability",
        ].to_numpy(dtype=float)

        uncited = group.loc[
            group["target"] == 0,
            "cited_probability",
        ].to_numpy(dtype=float)

        if len(cited) == 0 or len(uncited) == 0:
            raise RuntimeError(
                f"Missing class for query {query_id}"
            )

        wins = (
            cited[:, None] > uncited[None, :]
        ).astype(float)

        ties = (
            cited[:, None] == uncited[None, :]
        ).astype(float)

        values[query_id] = float(
            (wins + 0.5 * ties).mean()
        )

    return values


def point_metrics(frame):
    return {
        "roc_auc": float(
            roc_auc_score(
                frame["target"].to_numpy(),
                frame["cited_probability"].to_numpy(),
            )
        ),
        "pairwise_accuracy": float(
            np.mean(
                list(
                    query_pairwise_values(frame).values()
                )
            )
        ),
    }


def build_cells(ensemble):
    cells = {}

    for training in TRAINING_CONDITIONS:
        for evaluation in EVALUATION_CONDITIONS:
            key = (training, evaluation)
            frame = cell_frame(
                ensemble,
                training,
                evaluation,
            )

            cells[key] = {
                "frame": frame,
                "point": point_metrics(frame),
                "pairwise": query_pairwise_values(frame),
            }

    return cells


def bootstrap_cells(cells, repetitions, seed):
    query_sets = [
        set(cell["pairwise"])
        for cell in cells.values()
    ]

    common_queries = set.intersection(*query_sets)

    if len(common_queries) != 1642:
        raise RuntimeError(
            f"Expected 1,642 common queries, "
            f"found {len(common_queries)}"
        )

    queries = np.asarray(
        sorted(common_queries),
        dtype=object,
    )

    query_to_index = {
        query: index
        for index, query in enumerate(queries)
    }

    prepared = {}

    for key, cell in cells.items():
        frame = cell["frame"].copy()

        query_codes = frame["query_id"].map(
            query_to_index
        ).to_numpy(dtype=np.int64)

        pairwise_array = np.asarray(
            [
                cell["pairwise"][query]
                for query in queries
            ],
            dtype=float,
        )

        prepared[key] = {
            "labels": frame["target"].to_numpy(
                dtype=np.int8
            ),
            "scores": frame[
                "cited_probability"
            ].to_numpy(dtype=float),
            "query_codes": query_codes,
            "pairwise": pairwise_array,
        }

    rng = np.random.default_rng(seed)

    bootstrap = {
        key: {
            metric: np.empty(
                repetitions,
                dtype=np.float64,
            )
            for metric in METRICS
        }
        for key in cells
    }

    query_count = len(queries)

    for repetition in range(repetitions):
        sampled = rng.integers(
            0,
            query_count,
            size=query_count,
        )

        multiplicity = np.bincount(
            sampled,
            minlength=query_count,
        ).astype(float)

        for key, data in prepared.items():
            row_weights = multiplicity[
                data["query_codes"]
            ]

            bootstrap[key]["roc_auc"][repetition] = (
                roc_auc_score(
                    data["labels"],
                    data["scores"],
                    sample_weight=row_weights,
                )
            )

            bootstrap[key][
                "pairwise_accuracy"
            ][repetition] = np.average(
                data["pairwise"],
                weights=multiplicity,
            )

    return queries, bootstrap


def comparison_specs():
    specs = []

    # Evaluation effect: Random-U minus Hard-U,
    # separately for each training condition.
    for training in TRAINING_CONDITIONS:
        for metric in METRICS:
            specs.append({
                "comparison": (
                    f"{training}:random_u_minus_hard_u"
                ),
                "metric": metric,
                "a": (training, "hard_u"),
                "b": (training, "random_u"),
            })

    # Training effect: Mixed-U minus Hard-only,
    # separately for each evaluation condition.
    for evaluation in EVALUATION_CONDITIONS:
        for metric in METRICS:
            specs.append({
                "comparison": (
                    f"{evaluation}:mixed_u_minus_hard_only"
                ),
                "metric": metric,
                "a": ("hard_only", evaluation),
                "b": ("mixed_u", evaluation),
            })

    if len(specs) != 8:
        raise AssertionError(len(specs))

    return specs


def evaluate_model(
    artifacts,
    model,
    output,
    repetitions,
    bootstrap_seed,
):
    print(f"\nLoading {model} predictions...")

    ensemble = load_ensemble(
        artifacts,
        model,
    )

    cells = build_cells(ensemble)

    point_rows = []

    for (training, evaluation), cell in cells.items():
        for metric, estimate in cell["point"].items():
            point_rows.append({
                "model": model,
                "training_condition": training,
                "evaluation_condition": evaluation,
                "metric": metric,
                "estimate": estimate,
                "queries": 1642,
                "rows": len(cell["frame"]),
            })

    point_frame = pd.DataFrame(point_rows)

    model_seed = (
        20260922
        if model == "deberta"
        else bootstrap_seed
    )

    queries, bootstrap = bootstrap_cells(
        cells,
        repetitions=repetitions,
        seed=model_seed,
    )

    interval_rows = []

    for key, metrics in bootstrap.items():
        training, evaluation = key

        for metric, values in metrics.items():
            interval_rows.append({
                "model": model,
                "training_condition": training,
                "evaluation_condition": evaluation,
                "metric": metric,
                "estimate": cells[key]["point"][metric],
                "ci_lower": float(
                    np.quantile(values, 0.025)
                ),
                "ci_upper": float(
                    np.quantile(values, 0.975)
                ),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": model_seed,
            })

    interval_frame = pd.DataFrame(interval_rows)

    comparison_rows = []
    replicate_outputs = {}

    for spec in comparison_specs():
        metric = spec["metric"]
        a = spec["a"]
        b = spec["b"]

        point_a = cells[a]["point"][metric]
        point_b = cells[b]["point"][metric]
        difference = point_b - point_a

        distribution = (
            bootstrap[b][metric]
            - bootstrap[a][metric]
        )

        replicate_key = (
            spec["comparison"]
            .replace(":", "__")
            .replace("-", "_")
            + "__"
            + metric
        )

        replicate_outputs[replicate_key] = distribution

        comparison_rows.append({
            "model": model,
            "comparison": spec["comparison"],
            "metric": metric,
            "estimate_a": point_a,
            "estimate_b": point_b,
            "difference_b_minus_a": difference,
            "ci_lower": float(
                np.quantile(distribution, 0.025)
            ),
            "ci_upper": float(
                np.quantile(distribution, 0.975)
            ),
            "raw_p": finite_bootstrap_p(
                distribution
            ),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": model_seed,
        })

    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame["holm_p"] = holm_adjust(
        comparison_frame["raw_p"].to_numpy()
    )
    comparison_frame[
        "significant_after_holm_0.05"
    ] = comparison_frame["holm_p"] < 0.05

    model_output = output / model
    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensemble.to_parquet(
        model_output / "ensemble_predictions.parquet",
        index=False,
        compression="zstd",
    )

    point_frame.to_csv(
        model_output / "point_estimates.csv",
        index=False,
    )

    interval_frame.to_csv(
        model_output / "bootstrap_intervals.csv",
        index=False,
    )

    comparison_frame.to_csv(
        model_output / "paired_comparisons.csv",
        index=False,
    )

    np.savez_compressed(
        model_output / "bootstrap_replicates.npz",
        **replicate_outputs,
    )

    config = {
        "model": model,
        "seeds": SEEDS,
        "ensemble_method":
            "mean cited probability across seeds",
        "matched_queries": len(queries),
        "bootstrap_unit": "query_id",
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": model_seed,
        "confidence_interval":
            "95% percentile bootstrap",
        "p_value":
            "two-sided bootstrap-tail with +1 correction",
        "holm_family_size": 8,
    }

    (
        model_output / "reproduction_config.json"
    ).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"\n{model.upper()} point estimates")
    print(
        point_frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\n{model.upper()} comparisons")
    print(
        comparison_frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return (
        point_frame,
        interval_frame,
        comparison_frame,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_points = []
    all_intervals = []
    all_comparisons = []

    for model in ["minilm", "deberta", "qwen"]:
        points, intervals, comparisons = evaluate_model(
            artifacts=args.artifacts,
            model=model,
            output=args.output,
            repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )

        all_points.append(points)
        all_intervals.append(intervals)
        all_comparisons.append(comparisons)

    pd.concat(
        all_points,
        ignore_index=True,
    ).to_csv(
        args.output / "table4_point_estimates.csv",
        index=False,
    )

    pd.concat(
        all_intervals,
        ignore_index=True,
    ).to_csv(
        args.output / "table4_bootstrap_intervals.csv",
        index=False,
    )

    pd.concat(
        all_comparisons,
        ignore_index=True,
    ).to_csv(
        args.output / "table4_paired_comparisons.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("TABLE 4 REPRODUCTION COMPLETE")
    print("=" * 78)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
