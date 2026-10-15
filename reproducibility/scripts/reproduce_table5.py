#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SEEDS = [13, 42, 77]
CONDITIONS = ["hard_only", "mixed_u"]
SUBSETS = [
    "all",
    "passage_different",
    "document_different",
]
METRICS = [
    "roc_auc",
    "pairwise_accuracy",
]


def normalize(value):
    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip().lower()


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    count = len(p_values)
    order = np.argsort(p_values)
    ordered = p_values[order]

    adjusted_ordered = np.empty(
        count,
        dtype=float,
    )

    running = 0.0

    for rank, value in enumerate(ordered):
        adjusted = min(
            1.0,
            (count - rank) * value,
        )
        running = max(running, adjusted)
        adjusted_ordered[rank] = running

    adjusted = np.empty(
        count,
        dtype=float,
    )
    adjusted[order] = adjusted_ordered

    return adjusted


def finite_bootstrap_p(distribution):
    distribution = np.asarray(
        distribution,
        dtype=float,
    )

    repetitions = len(distribution)

    lower = (
        np.count_nonzero(distribution <= 0) + 1
    ) / (repetitions + 1)

    upper = (
        np.count_nonzero(distribution >= 0) + 1
    ) / (repetitions + 1)

    return min(
        1.0,
        2.0 * min(lower, upper),
    )


def validate_probability(frame, column, path):
    if frame[column].isna().any():
        raise RuntimeError(
            f"NaN probability in {path}"
        )

    if not frame[column].between(0, 1).all():
        raise RuntimeError(
            f"Probability outside [0,1] in {path}"
        )


# ====================================================
# DeBERTa
# ====================================================

def load_deberta(artifacts):
    root = artifacts / "table5/deberta"
    frames = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            path = (
                root
                / (
                    f"deberta_{condition}_seed{seed}_"
                    "dense_hard_predictions.parquet"
                )
            )

            if not path.is_file():
                raise FileNotFoundError(path)

            frame = pd.read_parquet(path)

            required = {
                "source_row_id",
                "query_id",
                "candidate_type",
                "cited_probability",
                "training_condition",
                "seed",
            }

            missing = required - set(frame.columns)

            if missing:
                raise RuntimeError(
                    f"{path} missing columns: "
                    f"{sorted(missing)}"
                )

            if len(frame) != 8232:
                raise RuntimeError(
                    f"Unexpected rows in {path}: "
                    f"{len(frame)}"
                )

            if set(
                frame["training_condition"].unique()
            ) != {condition}:
                raise RuntimeError(
                    f"Condition mismatch in {path}"
                )

            if set(frame["seed"].unique()) != {seed}:
                raise RuntimeError(
                    f"Seed mismatch in {path}"
                )

            validate_probability(
                frame,
                "cited_probability",
                path,
            )

            # Table 5 uses only cited candidates and
            # Dense-Hard U candidates.
            frame = frame[
                frame["candidate_type"].isin(
                    ["cited", "dense_hard_u"]
                )
            ].copy()

            if len(frame) != 4948:
                raise RuntimeError(
                    f"Expected 4,948 relevant rows "
                    f"in {path}; found {len(frame)}"
                )

            frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    keys = [
        "source_row_id",
        "query_id",
        "candidate_type",
        "training_condition",
    ]

    seed_counts = (
        combined.groupby(
            keys,
            dropna=False,
        )["seed"]
        .nunique()
    )

    if not (seed_counts == 3).all():
        raise RuntimeError(
            "DeBERTa seed alignment failed: "
            f"{int((seed_counts != 3).sum())} rows"
        )

    ensemble = (
        combined.groupby(
            keys,
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

    if len(ensemble) != 4948 * 2:
        raise RuntimeError(
            "Unexpected DeBERTa ensemble size: "
            f"{len(ensemble)}"
        )

    ensemble["model"] = "deberta"

    return ensemble


# ====================================================
# Qwen: cited rows from Table 4, Dense-Hard U from
# released Table 5 predictions.
# ====================================================

def load_qwen(artifacts):
    table4_root = (
        artifacts
        / "table4/qwen_control"
    )

    table5_root = (
        artifacts
        / "table5/qwen"
    )

    cited_frames = []
    dense_frames = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            cited_path = (
                table4_root
                / (
                    f"qwen3_reranker_{condition}_seed{seed}_"
                    "ordered_predictions.parquet"
                )
            )

            dense_path = (
                table5_root
                / (
                    f"qwen3_reranker_{condition}_seed{seed}_"
                    "dense_hard_predictions.parquet"
                )
            )

            if not cited_path.is_file():
                raise FileNotFoundError(cited_path)

            if not dense_path.is_file():
                raise FileNotFoundError(dense_path)

            cited = pd.read_parquet(cited_path)
            dense = pd.read_parquet(dense_path)

            cited_required = {
                "source_row_id",
                "query_id",
                "candidate_type",
                "evaluation_condition",
                "training_condition",
                "seed",
                "cited_probability",
            }

            dense_required = {
                "source_row_id",
                "query_id",
                "candidate_type",
                "training_condition",
                "seed",
                "probability",
            }

            missing_cited = (
                cited_required - set(cited.columns)
            )
            missing_dense = (
                dense_required - set(dense.columns)
            )

            if missing_cited:
                raise RuntimeError(
                    f"{cited_path} missing columns: "
                    f"{sorted(missing_cited)}"
                )

            if missing_dense:
                raise RuntimeError(
                    f"{dense_path} missing columns: "
                    f"{sorted(missing_dense)}"
                )

            # The cited rows are duplicated across Hard-U and
            # Random-U evaluation conditions. Retain one copy.
            cited = cited[
                (cited["candidate_type"] == "cited")
                & (
                    cited["evaluation_condition"]
                    == "hard_u"
                )
            ].copy()

            if len(cited) != 3306:
                raise RuntimeError(
                    f"Expected 3,306 cited rows in "
                    f"{cited_path}; found {len(cited)}"
                )

            dense = dense[
                dense["candidate_type"]
                == "dense_hard_u"
            ].copy()

            if len(dense) != 1642:
                raise RuntimeError(
                    f"Expected 1,642 Dense-Hard rows in "
                    f"{dense_path}; found {len(dense)}"
                )

            dense = dense.rename(
                columns={
                    "probability":
                        "cited_probability"
                }
            )

            validate_probability(
                cited,
                "cited_probability",
                cited_path,
            )

            validate_probability(
                dense,
                "cited_probability",
                dense_path,
            )

            cited_frames.append(
                cited[
                    [
                        "source_row_id",
                        "query_id",
                        "candidate_type",
                        "training_condition",
                        "seed",
                        "cited_probability",
                    ]
                ]
            )

            dense_frames.append(
                dense[
                    [
                        "source_row_id",
                        "query_id",
                        "candidate_type",
                        "training_condition",
                        "seed",
                        "cited_probability",
                    ]
                ]
            )

    combined = pd.concat(
        cited_frames + dense_frames,
        ignore_index=True,
    )

    keys = [
        "source_row_id",
        "query_id",
        "candidate_type",
        "training_condition",
    ]

    seed_counts = (
        combined.groupby(
            keys,
            dropna=False,
        )["seed"]
        .nunique()
    )

    if not (seed_counts == 3).all():
        raise RuntimeError(
            "Qwen seed alignment failed: "
            f"{int((seed_counts != 3).sum())} rows"
        )

    ensemble = (
        combined.groupby(
            keys,
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

    if len(ensemble) != 4948 * 2:
        raise RuntimeError(
            "Unexpected Qwen ensemble size: "
            f"{len(ensemble)}"
        )

    ensemble["model"] = "qwen"

    return ensemble


# ====================================================
# Subsets
# ====================================================

def load_subset_map(artifacts):
    path = (
        artifacts
        / "table5/common"
        / "qwen_dense_hard_overlap_alignment.csv"
    )

    frame = pd.read_csv(path)

    required = {
        "query_text_norm",
        "same_passage",
        "same_document",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Overlap file missing columns: "
            f"{sorted(missing)}"
        )

    if len(frame) != 1642:
        raise RuntimeError(
            f"Expected 1,642 overlap rows; "
            f"found {len(frame)}"
        )

    frame["query_key"] = frame[
        "query_text_norm"
    ].map(normalize)

    if frame["query_key"].duplicated().any():
        raise RuntimeError(
            "Duplicate normalized queries in overlap file."
        )

    def as_bool(series):
        if series.dtype == bool:
            return series

        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    frame["same_passage"] = as_bool(
        frame["same_passage"]
    )
    frame["same_document"] = as_bool(
        frame["same_document"]
    )

    subset_map = {}

    for row in frame.itertuples(index=False):
        subset_map[row.query_key] = {
            "all": True,
            "passage_different":
                not row.same_passage,
            "document_different":
                not row.same_document,
        }

    counts = {
        subset: sum(
            flags[subset]
            for flags in subset_map.values()
        )
        for subset in SUBSETS
    }

    expected = {
        "all": 1642,
        "passage_different": 1555,
        "document_different": 1389,
    }

    if counts != expected:
        raise RuntimeError(
            f"Subset counts differ from expected: "
            f"{counts}"
        )

    # Add claim_id aliases for Qwen, whose query_id uses claim_id.
    # DeBERTa continues to use the normalized claim-text keys.
    for row in frame.itertuples(index=False):
        claim_key = normalize(row.claim_id_norm)
        query_key = row.query_key
        flags = subset_map[query_key]

        if claim_key in subset_map and subset_map[claim_key] != flags:
            raise RuntimeError(
                f"Conflicting subset assignment for {claim_key}"
            )

        subset_map[claim_key] = flags

    return subset_map


def add_subset_flags(ensemble, subset_map):
    ensemble = ensemble.copy()

    ensemble["query_key"] = (
        ensemble["query_id"].map(normalize)
    )

    missing_queries = (
        set(ensemble["query_key"])
        - set(subset_map)
    )

    if missing_queries:
        raise RuntimeError(
            f"Queries missing from overlap map: "
            f"{len(missing_queries)}"
        )

    for subset in SUBSETS:
        ensemble[subset] = ensemble[
            "query_key"
        ].map(
            lambda key: subset_map[key][subset]
        )

    return ensemble


# ====================================================
# Metrics
# ====================================================

def prepare_cell(
    ensemble,
    condition,
    subset,
):
    frame = ensemble[
        (ensemble["training_condition"] == condition)
        & ensemble[subset]
    ].copy()

    frame["target"] = (
        frame["candidate_type"] == "cited"
    ).astype(np.int8)

    observed_types = set(
        frame["candidate_type"].unique()
    )

    if observed_types != {
        "cited",
        "dense_hard_u",
    }:
        raise RuntimeError(
            f"Unexpected candidate types: "
            f"{observed_types}"
        )

    expected_queries = {
        "all": 1642,
        "passage_different": 1555,
        "document_different": 1389,
    }[subset]

    if frame["query_key"].nunique() != expected_queries:
        raise RuntimeError(
            f"{condition}/{subset}: expected "
            f"{expected_queries} queries, found "
            f"{frame['query_key'].nunique()}"
        )

    return frame


def query_pairwise(frame):
    values = {}

    for query, group in frame.groupby(
        "query_key",
        sort=False,
    ):
        cited = group.loc[
            group["target"] == 1,
            "cited_probability",
        ].to_numpy(dtype=float)

        dense = group.loc[
            group["target"] == 0,
            "cited_probability",
        ].to_numpy(dtype=float)

        if len(cited) == 0 or len(dense) == 0:
            raise RuntimeError(
                f"Missing class for query {query}"
            )

        wins = (
            cited[:, None] > dense[None, :]
        ).astype(float)

        ties = (
            cited[:, None] == dense[None, :]
        ).astype(float)

        values[query] = float(
            (wins + 0.5 * ties).mean()
        )

    return values


def point_metrics(frame):
    pairwise = query_pairwise(frame)

    return {
        "roc_auc": float(
            roc_auc_score(
                frame["target"],
                frame["cited_probability"],
            )
        ),
        "pairwise_accuracy": float(
            np.mean(list(pairwise.values()))
        ),
        "pairwise_by_query": pairwise,
    }


def evaluate_model(
    ensemble,
    model,
    repetitions,
    seed,
):
    cells = {}

    for subset in SUBSETS:
        for condition in CONDITIONS:
            frame = prepare_cell(
                ensemble,
                condition,
                subset,
            )

            cells[(subset, condition)] = {
                "frame": frame,
                "metrics": point_metrics(frame),
            }

    point_rows = []

    for (subset, condition), cell in cells.items():
        for metric in METRICS:
            point_rows.append({
                "model": model,
                "subset": subset,
                "training_condition": condition,
                "metric": metric,
                "estimate":
                    cell["metrics"][metric],
                "queries":
                    cell["frame"][
                        "query_key"
                    ].nunique(),
                "rows": len(cell["frame"]),
            })

    point_frame = pd.DataFrame(point_rows)

    rng = np.random.default_rng(seed)
    bootstrap = {}

    for subset in SUBSETS:
        hard = cells[
            (subset, "hard_only")
        ]
        mixed = cells[
            (subset, "mixed_u")
        ]

        common_queries = sorted(
            set(
                hard["metrics"][
                    "pairwise_by_query"
                ]
            )
            & set(
                mixed["metrics"][
                    "pairwise_by_query"
                ]
            )
        )

        expected_queries = hard["frame"][
            "query_key"
        ].nunique()

        if len(common_queries) != expected_queries:
            raise RuntimeError(
                f"Query mismatch for {model}/{subset}"
            )

        query_to_index = {
            query: index
            for index, query in enumerate(
                common_queries
            )
        }

        prepared = {}

        for condition in CONDITIONS:
            cell = cells[(subset, condition)]
            frame = cell["frame"]

            prepared[condition] = {
                "labels": frame["target"].to_numpy(
                    dtype=np.int8
                ),
                "scores": frame[
                    "cited_probability"
                ].to_numpy(dtype=float),
                "query_codes": frame[
                    "query_key"
                ].map(
                    query_to_index
                ).to_numpy(dtype=np.int64),
                "pairwise": np.asarray(
                    [
                        cell["metrics"][
                            "pairwise_by_query"
                        ][query]
                        for query in common_queries
                    ],
                    dtype=float,
                ),
            }

        subset_bootstrap = {
            condition: {
                metric: np.empty(
                    repetitions,
                    dtype=float,
                )
                for metric in METRICS
            }
            for condition in CONDITIONS
        }

        query_count = len(common_queries)

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

            for condition in CONDITIONS:
                data = prepared[condition]
                row_weights = multiplicity[
                    data["query_codes"]
                ]

                subset_bootstrap[
                    condition
                ]["roc_auc"][repetition] = (
                    roc_auc_score(
                        data["labels"],
                        data["scores"],
                        sample_weight=row_weights,
                    )
                )

                subset_bootstrap[
                    condition
                ][
                    "pairwise_accuracy"
                ][repetition] = np.average(
                    data["pairwise"],
                    weights=multiplicity,
                )

        bootstrap[subset] = subset_bootstrap

    comparison_rows = []
    replicate_outputs = {}

    for subset in SUBSETS:
        for metric in METRICS:
            hard_point = cells[
                (subset, "hard_only")
            ]["metrics"][metric]

            mixed_point = cells[
                (subset, "mixed_u")
            ]["metrics"][metric]

            distribution = (
                bootstrap[subset][
                    "mixed_u"
                ][metric]
                - bootstrap[subset][
                    "hard_only"
                ][metric]
            )

            replicate_outputs[
                f"{subset}__{metric}"
            ] = distribution

            comparison_rows.append({
                "model": model,
                "subset": subset,
                "metric": metric,
                "hard_only": hard_point,
                "mixed_u": mixed_point,
                "delta_mixed_minus_hard":
                    mixed_point - hard_point,
                "ci_lower": float(
                    np.quantile(
                        distribution,
                        0.025,
                    )
                ),
                "ci_upper": float(
                    np.quantile(
                        distribution,
                        0.975,
                    )
                ),
                "raw_p": finite_bootstrap_p(
                    distribution
                ),
                "queries": cells[
                    (subset, "hard_only")
                ]["frame"][
                    "query_key"
                ].nunique(),
            })

    return (
        point_frame,
        pd.DataFrame(comparison_rows),
        replicate_outputs,
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

    subset_map = load_subset_map(
        args.artifacts
    )

    deberta = add_subset_flags(
        load_deberta(args.artifacts),
        subset_map,
    )

    qwen = add_subset_flags(
        load_qwen(args.artifacts),
        subset_map,
    )

    model_outputs = {
        "deberta": deberta,
        "qwen": qwen,
    }

    all_points = []
    all_comparisons = []

    for model, ensemble in model_outputs.items():
        model_dir = args.output / model
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        points, comparisons, replicates = (
            evaluate_model(
                ensemble=ensemble,
                model=model,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            )
        )

        ensemble.to_parquet(
            model_dir
            / "dense_hard_ensemble_predictions.parquet",
            index=False,
            compression="zstd",
        )

        points.to_csv(
            model_dir / "point_estimates.csv",
            index=False,
        )

        comparisons.to_csv(
            model_dir / "paired_comparisons_raw.csv",
            index=False,
        )

        np.savez_compressed(
            model_dir / "bootstrap_replicates.npz",
            **replicates,
        )

        all_points.append(points)
        all_comparisons.append(comparisons)

        print(
            f"\n{model.upper()} TABLE 5 POINT ESTIMATES"
        )
        print(
            points.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.6f}",
            )
        )

        print(
            f"\n{model.upper()} TABLE 5 COMPARISONS"
        )
        print(
            comparisons.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.6f}",
            )
        )

    all_points_frame = pd.concat(
        all_points,
        ignore_index=True,
    )

    global_comparisons = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    if len(global_comparisons) != 12:
        raise RuntimeError(
            f"Expected 12 global comparisons, "
            f"found {len(global_comparisons)}"
        )

    global_comparisons[
        "global_12_holm_p"
    ] = holm_adjust(
        global_comparisons[
            "raw_p"
        ].to_numpy()
    )

    global_comparisons[
        "significant_after_global_holm_0.05"
    ] = (
        global_comparisons[
            "global_12_holm_p"
        ] < 0.05
    )

    all_points_frame.to_csv(
        args.output
        / "table5_point_estimates.csv",
        index=False,
    )

    global_comparisons.to_csv(
        args.output
        / "table5_global_12_holm.csv",
        index=False,
    )

    config = {
        "models": ["deberta", "qwen"],
        "seeds": SEEDS,
        "subsets": {
            "all": 1642,
            "passage_different": 1555,
            "document_different": 1389,
        },
        "metrics": METRICS,
        "bootstrap_repetitions":
            args.bootstrap_repetitions,
        "bootstrap_seed":
            args.bootstrap_seed,
        "bootstrap_unit": "normalized query",
        "confidence_interval":
            "95% percentile bootstrap",
        "p_value":
            "two-sided bootstrap-tail with +1 correction",
        "multiple_testing":
            "global Holm correction over 12 comparisons",
    }

    (
        args.output / "reproduction_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("TABLE 5 GLOBAL 12-COMPARISON RESULTS")
    print("=" * 78)

    print(
        global_comparisons.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n" + "=" * 78)
    print("TABLE 5 REPRODUCTION COMPLETE")
    print("=" * 78)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
