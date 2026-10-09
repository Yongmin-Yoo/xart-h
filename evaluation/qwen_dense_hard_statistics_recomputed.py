#!/usr/bin/env python3
"""
Recompute canonical Qwen3-Reranker-0.6B Dense-Hard Table 5 statistics.

This script performs no model inference and no training. It consumes:

1. canonical six-adapter Dense-Hard predictions;
2. Table 4 cited-candidate predictions;
3. the Dense-Hard overlap-alignment table; and
4. the existing DeBERTa integrated statistical table, when available.

It calculates:
- three-seed ensembles;
- query-macro cited-vs-Dense-Hard ROC-AUC;
- same-query cited-over-Dense-Hard pairwise accuracy;
- 2,000 paired query bootstrap repetitions;
- primary, passage-disjoint, and document-disjoint sensitivity;
- Holm correction over six Qwen and twelve global Table 5 comparisons.

Usage
-----
Point XARTH_ARTIFACT_ROOT to the artifact tree containing experiments/:

    XARTH_ARTIFACT_ROOT=/path/to/XART-H-artifacts \
      python evaluation/qwen_dense_hard_statistics_recomputed.py

The default root is the repository root. Large experimental inputs may need
to be restored separately because not all training artifacts are versioned.
"""

# ============================================================
# Qwen Dense-Hard canonical Table 5 statistical evaluation
# - No model loading / no inference
# - 3-seed ensemble
# - Query-level bootstrap: 2,000 repetitions
# - Primary + passage/document-disjoint sensitivity
# - Holm correction (Qwen family 6; optional global Table 5 family 12)
# ============================================================

from pathlib import Path
import json
import os
import re
import hashlib
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
ROOT = Path(
    os.environ.get(
        "XARTH_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()

T4 = (
    ROOT
    / "experiments/model_generalization_v1"
    / "qwen3_reranker_0.6b_control"
)

T5_OLD = (
    ROOT
    / "experiments/dense_hard_v1"
    / "qwen_frozen_evaluation"
)

T5_NEW = (
    ROOT
    / "experiments/dense_hard_v1"
    / "qwen_frozen_evaluation_recomputed"
)

NEW_PREDICTIONS = T5_NEW / "qwen_dense_hard_all_seed_predictions.parquet"

OVERLAP_FILE = (
    T5_OLD
    / "statistical_analysis_corrected"
    / "qwen_dense_hard_overlap_alignment.csv"
)

OUT = T5_NEW / "statistical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["hard_only", "mixed_u"]
SEEDS = [13, 42, 77]

EXPECTED_QUERIES = 1642
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

assert NEW_PREDICTIONS.exists(), NEW_PREDICTIONS
assert OVERLAP_FILE.exists(), OVERLAP_FILE
assert T4.exists(), T4

print("New predictions:", NEW_PREDICTIONS)
print("Overlap bridge :", OVERLAP_FILE)
print("Output         :", OUT)


# ------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------
def normalize_key(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {
        "1", "true", "t", "yes", "y"
    }


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, monotonic."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    ordered = p[order]

    adjusted_ordered = np.empty(m, dtype=float)
    running = 0.0

    for rank, raw_p in enumerate(ordered):
        candidate = (m - rank) * raw_p
        running = max(running, candidate)
        adjusted_ordered[rank] = min(1.0, running)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def empirical_two_sided_p(delta_samples):
    """Finite-sample corrected two-sided bootstrap p-value."""
    delta_samples = np.asarray(delta_samples, dtype=float)
    b = len(delta_samples)

    n_nonpositive = int(np.sum(delta_samples <= 0))
    n_nonnegative = int(np.sum(delta_samples >= 0))

    lower = (n_nonpositive + 1) / (b + 1)
    upper = (n_nonnegative + 1) / (b + 1)

    return float(min(1.0, 2.0 * min(lower, upper)))


def ci_bounds(values, level=0.95):
    alpha = 1.0 - level
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


# ------------------------------------------------------------
# 3. Load overlap bridge and define sensitivity subsets
# ------------------------------------------------------------
bridge = pd.read_csv(OVERLAP_FILE)

required_bridge = {
    "claim_id_norm",
    "query_text_norm",
    "same_passage",
    "same_document",
}
missing = required_bridge - set(bridge.columns)
assert not missing, f"Missing overlap columns: {missing}"

bridge["_claim_key"] = bridge["claim_id_norm"].map(normalize_key)
bridge["_query_key"] = bridge["query_text_norm"].map(normalize_key)
bridge["_same_passage"] = bridge["same_passage"].map(parse_bool)
bridge["_same_document"] = bridge["same_document"].map(parse_bool)

assert len(bridge) == EXPECTED_QUERIES, len(bridge)
assert bridge["_claim_key"].is_unique
assert bridge["_query_key"].is_unique
assert (bridge["_claim_key"] != "").all()
assert (bridge["_query_key"] != "").all()

bridge["primary_all_1642"] = True
bridge["passage_disjoint_1555"] = ~bridge["_same_passage"]
bridge["document_disjoint_1389"] = ~bridge["_same_document"]

SUBSETS = {
    "primary_all_1642": set(
        bridge.loc[bridge["primary_all_1642"], "_query_key"]
    ),
    "passage_disjoint_1555": set(
        bridge.loc[bridge["passage_disjoint_1555"], "_query_key"]
    ),
    "document_disjoint_1389": set(
        bridge.loc[bridge["document_disjoint_1389"], "_query_key"]
    ),
}

print("\n=== Subset counts ===")
for name, values in SUBSETS.items():
    print(f"{name:30s}: {len(values)}")

assert len(SUBSETS["primary_all_1642"]) == 1642
assert len(SUBSETS["passage_disjoint_1555"]) == 1555
assert len(SUBSETS["document_disjoint_1389"]) == 1389


# ------------------------------------------------------------
# 4. Load canonical Dense-Hard U predictions and ensemble seeds
# ------------------------------------------------------------
dense = pd.read_parquet(NEW_PREDICTIONS)

required_dense = {
    "training_condition",
    "seed",
    "query_id",
    "probability",
}
missing = required_dense - set(dense.columns)
assert not missing, f"Missing prediction columns: {missing}"

dense["training_condition"] = dense["training_condition"].astype(str)
dense["seed"] = dense["seed"].astype(int)
dense["probability"] = dense["probability"].astype(float)
dense["_dense_query_key"] = dense["query_id"].map(normalize_key)

assert len(dense) == EXPECTED_QUERIES * 6
assert dense["probability"].notna().all()
assert dense["probability"].between(0, 1).all()

# Dense-Hard query_id is normally claim_id_norm. Map it to Table 4 query text.
claim_to_query = dict(
    zip(bridge["_claim_key"], bridge["_query_key"])
)
query_identity = dict(
    zip(bridge["_query_key"], bridge["_query_key"])
)

dense["_query_key"] = dense["_dense_query_key"].map(claim_to_query)

# Fallback in case query_id already contains normalized query text.
fallback_mask = dense["_query_key"].isna()
dense.loc[fallback_mask, "_query_key"] = (
    dense.loc[fallback_mask, "_dense_query_key"].map(query_identity)
)

assert dense["_query_key"].notna().all(), (
    "Some Dense-Hard query_id values could not be mapped through "
    "claim_id_norm/query_text_norm."
)

dense_counts = (
    dense.groupby(["training_condition", "_query_key"])
    ["seed"]
    .nunique()
)

assert (dense_counts == 3).all(), dense_counts.value_counts()

dense_ensemble = (
    dense.groupby(["training_condition", "_query_key"], as_index=False)
    .agg(
        dense_hard_probability=("probability", "mean"),
        seed_probability_std=("probability", "std"),
        number_of_seeds=("seed", "nunique"),
    )
)

assert len(dense_ensemble) == EXPECTED_QUERIES * 2

for condition in CONDITIONS:
    part = dense_ensemble[
        dense_ensemble["training_condition"] == condition
    ]
    assert len(part) == EXPECTED_QUERIES
    assert part["_query_key"].is_unique

print("\nDense-Hard ensemble loaded:", dense_ensemble.shape)


# ------------------------------------------------------------
# 5. Recover Table 4 examiner-cited predictions
# ------------------------------------------------------------
cited_seed_frames = []
candidate_diagnostics = []

for condition in CONDITIONS:
    for seed in SEEDS:
        pred_file = (
            T4
            / f"qwen3_reranker_{condition}_seed{seed}_ordered_predictions.parquet"
        )
        assert pred_file.exists(), pred_file

        df = pd.read_parquet(pred_file)

        required = {
            "query_id",
            "candidate_type",
            "cited_probability",
            "source_row_id",
        }
        missing = required - set(df.columns)
        assert not missing, f"{pred_file.name}: missing {missing}"

        df["_candidate_lower"] = (
            df["candidate_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        candidate_values = sorted(
            df["_candidate_lower"].dropna().unique().tolist()
        )

        # Expected forms include cited, cited_a, cited_x, A, or X.
        cited_mask = (
            df["_candidate_lower"].str.contains("cited", regex=False)
            | df["_candidate_lower"].isin(
                {"a", "x", "positive", "relevant", "examiner_cited"}
            )
        )

        cited = df.loc[cited_mask].copy()

        assert len(cited) > 0, (
            f"Could not identify cited candidates in {pred_file.name}. "
            f"candidate_type values={candidate_values}"
        )

        selected_evaluation_condition = None

        # The Table 4 files may duplicate cited rows for hard-U and random-U
        # evaluation blocks. Dense-Hard uses the hard-U cited block.
        if "evaluation_condition" in cited.columns:
            evaluation_values = sorted(
                cited["evaluation_condition"]
                .astype(str)
                .dropna()
                .unique()
                .tolist()
            )

            hard_values = [
                value for value in evaluation_values
                if "hard" in value.lower()
            ]

            if hard_values:
                selected_evaluation_condition = hard_values[0]
                cited = cited[
                    cited["evaluation_condition"].astype(str)
                    == selected_evaluation_condition
                ].copy()
        else:
            evaluation_values = []

        cited["_query_key"] = cited["query_id"].map(normalize_key)
        cited = cited[
            cited["_query_key"].isin(SUBSETS["primary_all_1642"])
        ].copy()

        cited["cited_probability"] = (
            cited["cited_probability"].astype(float)
        )

        assert cited["cited_probability"].notna().all()
        assert cited["cited_probability"].between(0, 1).all()

        # Candidate identity must be stable across seeds.
        cited["_candidate_key"] = (
            cited["source_row_id"].astype(str)
            + "::"
            + cited["_candidate_lower"].astype(str)
        )

        # Defensive de-duplication if an original output contains a repeated
        # row. Normally this grouping does not alter anything.
        cited = (
            cited.groupby(
                ["_query_key", "_candidate_key"],
                as_index=False
            )
            .agg(cited_probability=("cited_probability", "mean"))
        )

        cited["training_condition"] = condition
        cited["seed"] = seed

        cited_seed_frames.append(cited)

        candidate_diagnostics.append({
            "condition": condition,
            "seed": seed,
            "file": str(pred_file),
            "file_sha256": sha256_file(pred_file),
            "candidate_type_values": " | ".join(candidate_values),
            "evaluation_condition_values": " | ".join(evaluation_values),
            "selected_evaluation_condition": selected_evaluation_condition,
            "selected_cited_rows": int(len(cited)),
            "selected_queries": int(cited["_query_key"].nunique()),
        })

cited_seed = pd.concat(cited_seed_frames, ignore_index=True)

# Every cited candidate should be available for all three seeds.
cited_seed_counts = (
    cited_seed.groupby(
        ["training_condition", "_query_key", "_candidate_key"]
    )["seed"]
    .nunique()
)

assert (cited_seed_counts == 3).all(), (
    "Some cited candidates are not present for all three seeds: "
    f"{cited_seed_counts.value_counts().to_dict()}"
)

cited_ensemble = (
    cited_seed.groupby(
        ["training_condition", "_query_key", "_candidate_key"],
        as_index=False
    )
    .agg(
        cited_probability=("cited_probability", "mean"),
        cited_seed_std=("cited_probability", "std"),
        number_of_seeds=("seed", "nunique"),
    )
)

for condition in CONDITIONS:
    part = cited_ensemble[
        cited_ensemble["training_condition"] == condition
    ]

    assert part["_query_key"].nunique() == EXPECTED_QUERIES
    cited_per_query = part.groupby("_query_key").size()
    assert cited_per_query.min() >= 1

print("Cited ensemble loaded    :", cited_ensemble.shape)

diagnostics_df = pd.DataFrame(candidate_diagnostics)
diagnostics_df.to_csv(
    OUT / "qwen_dense_hard_cited_input_diagnostics.csv",
    index=False,
)


# ------------------------------------------------------------
# 6. Build query-level metric inputs
# ------------------------------------------------------------
condition_data = {}

for condition in CONDITIONS:
    u_part = (
        dense_ensemble[
            dense_ensemble["training_condition"] == condition
        ][["_query_key", "dense_hard_probability"]]
        .set_index("_query_key")
    )

    c_part = cited_ensemble[
        cited_ensemble["training_condition"] == condition
    ].copy()

    cited_lists = (
        c_part.groupby("_query_key")["cited_probability"]
        .apply(lambda x: np.asarray(x, dtype=np.float64))
        .to_dict()
    )

    query_keys = sorted(SUBSETS["primary_all_1642"])

    assert set(query_keys) == set(u_part.index)
    assert set(query_keys) == set(cited_lists)

    u_scores = np.asarray(
        [
            float(u_part.loc[q, "dense_hard_probability"])
            for q in query_keys
        ],
        dtype=np.float64,
    )

    positives = [cited_lists[q] for q in query_keys]

    condition_data[condition] = {
        "query_keys": query_keys,
        "u_scores": u_scores,
        "cited_scores": positives,
        "index": {q: i for i, q in enumerate(query_keys)},
    }


def calculate_metrics(data, selected_query_keys, query_counts=None):
    """
    Query-macro ROC-AUC:
      - each query has total positive weight 1
      - each query has total negative weight 1
      - if a query has multiple cited passages, each cited passage receives
        weight 1 / number_of_cited_passages

    Pairwise:
      mean, over queries, of P(cited score > dense-hard score),
      with ties receiving 0.5.
    """
    indices = np.asarray(
        [data["index"][q] for q in selected_query_keys],
        dtype=np.int64,
    )

    if query_counts is None:
        query_counts = np.ones(len(indices), dtype=np.float64)
    else:
        query_counts = np.asarray(query_counts, dtype=np.float64)

    positive_scores = []
    positive_weights = []
    negative_scores = []
    negative_weights = []
    pairwise_per_query = []

    for local_pos, global_idx in enumerate(indices):
        multiplicity = query_counts[local_pos]
        if multiplicity <= 0:
            pairwise_per_query.append(np.nan)
            continue

        cited_scores = data["cited_scores"][global_idx]
        u_score = data["u_scores"][global_idx]

        n_cited = len(cited_scores)
        assert n_cited >= 1

        positive_scores.extend(cited_scores.tolist())
        positive_weights.extend(
            [multiplicity / n_cited] * n_cited
        )

        negative_scores.append(float(u_score))
        negative_weights.append(float(multiplicity))

        comparison = (
            np.mean(cited_scores > u_score)
            + 0.5 * np.mean(cited_scores == u_score)
        )
        pairwise_per_query.append(float(comparison))

    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    positive_weights = np.asarray(positive_weights, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    negative_weights = np.asarray(negative_weights, dtype=np.float64)

    labels = np.concatenate([
        np.ones(len(positive_scores), dtype=np.int8),
        np.zeros(len(negative_scores), dtype=np.int8),
    ])
    scores = np.concatenate([positive_scores, negative_scores])
    weights = np.concatenate([positive_weights, negative_weights])

    auc = roc_auc_score(
        labels,
        scores,
        sample_weight=weights,
    )

    pairwise_per_query = np.asarray(
        pairwise_per_query,
        dtype=np.float64,
    )
    valid = np.isfinite(pairwise_per_query)

    pairwise = np.average(
        pairwise_per_query[valid],
        weights=query_counts[valid],
    )

    return {
        "roc_auc": float(auc),
        "pairwise_cited_over_dense_hard": float(pairwise),
    }


# ------------------------------------------------------------
# 7. Point estimates and paired query bootstrap
# ------------------------------------------------------------
rng = np.random.default_rng(BOOTSTRAP_SEED)

point_rows = []
bootstrap_rows = []
paired_rows = []
bootstrap_arrays = {}

for subset_name, subset_query_set in SUBSETS.items():
    selected_queries = sorted(subset_query_set)
    n_queries = len(selected_queries)

    point_by_condition = {}

    for condition in CONDITIONS:
        point = calculate_metrics(
            condition_data[condition],
            selected_queries,
        )
        point_by_condition[condition] = point

        for metric, value in point.items():
            point_rows.append({
                "model": "Qwen3-Reranker-0.6B",
                "subset": subset_name,
                "number_of_queries": n_queries,
                "training_condition": condition,
                "metric": metric,
                "point_estimate": value,
            })

    # One shared bootstrap draw per subset gives paired comparisons.
    hard_boot = {
        "roc_auc": np.empty(N_BOOTSTRAP),
        "pairwise_cited_over_dense_hard": np.empty(N_BOOTSTRAP),
    }
    mixed_boot = {
        "roc_auc": np.empty(N_BOOTSTRAP),
        "pairwise_cited_over_dense_hard": np.empty(N_BOOTSTRAP),
    }

    for b in range(N_BOOTSTRAP):
        sampled_indices = rng.integers(
            0, n_queries, size=n_queries
        )
        counts = np.bincount(
            sampled_indices,
            minlength=n_queries,
        ).astype(np.float64)

        hard_metrics = calculate_metrics(
            condition_data["hard_only"],
            selected_queries,
            counts,
        )
        mixed_metrics = calculate_metrics(
            condition_data["mixed_u"],
            selected_queries,
            counts,
        )

        for metric in hard_boot:
            hard_boot[metric][b] = hard_metrics[metric]
            mixed_boot[metric][b] = mixed_metrics[metric]

    bootstrap_arrays[(subset_name, "hard_only")] = hard_boot
    bootstrap_arrays[(subset_name, "mixed_u")] = mixed_boot

    for condition, arrays in [
        ("hard_only", hard_boot),
        ("mixed_u", mixed_boot),
    ]:
        for metric, samples in arrays.items():
            lower, upper = ci_bounds(samples, CI_LEVEL)

            bootstrap_rows.append({
                "model": "Qwen3-Reranker-0.6B",
                "subset": subset_name,
                "number_of_queries": n_queries,
                "training_condition": condition,
                "metric": metric,
                "point_estimate": point_by_condition[condition][metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_mean": float(np.mean(samples)),
                "bootstrap_std": float(np.std(samples, ddof=1)),
                "bootstrap_repetitions": N_BOOTSTRAP,
            })

    for metric in hard_boot:
        delta_samples = mixed_boot[metric] - hard_boot[metric]
        lower, upper = ci_bounds(delta_samples, CI_LEVEL)
        raw_p = empirical_two_sided_p(delta_samples)

        paired_rows.append({
            "model": "Qwen3-Reranker-0.6B",
            "subset": subset_name,
            "number_of_queries": n_queries,
            "metric": metric,
            "hard_only": point_by_condition["hard_only"][metric],
            "mixed_u": point_by_condition["mixed_u"][metric],
            "delta_mixed_minus_hard": (
                point_by_condition["mixed_u"][metric]
                - point_by_condition["hard_only"][metric]
            ),
            "ci_lower": lower,
            "ci_upper": upper,
            "raw_p": raw_p,
            "bootstrap_repetitions": N_BOOTSTRAP,
        })

    print(
        f"Bootstrap completed: {subset_name} "
        f"({n_queries} queries)"
    )

point_df = pd.DataFrame(point_rows)
bootstrap_df = pd.DataFrame(bootstrap_rows)
paired_df = pd.DataFrame(paired_rows)

# Qwen-only family: 3 subsets × 2 metrics = 6 comparisons.
paired_df["qwen_family_6_holm_p"] = holm_adjust(
    paired_df["raw_p"].to_numpy()
)
paired_df["significant_qwen_family_0_05"] = (
    paired_df["qwen_family_6_holm_p"] < 0.05
)


# ------------------------------------------------------------
# 8. Optional global Table 5 Holm correction over 12 comparisons
#    (6 existing DeBERTa + 6 newly recomputed Qwen)
# ------------------------------------------------------------
global_family_df = None

integrated_candidates = [
    T5_OLD
    / "statistical_analysis_corrected"
    / "integrated_dense_hard_query_macro_bootstrap.csv",
    T5_OLD
    / "integrated_dense_hard_query_macro_bootstrap.csv",
]

# Also search narrowly if the expected path differs.
integrated_candidates.extend(
    list(
        (
            ROOT / "experiments/dense_hard_v1"
        ).glob(
            "**/integrated_dense_hard_query_macro_bootstrap.csv"
        )
    )
)

# Preserve order while removing duplicates.
seen_paths = set()
integrated_candidates = [
    p for p in integrated_candidates
    if not (str(p) in seen_paths or seen_paths.add(str(p)))
]

integrated_file = next(
    (p for p in integrated_candidates if p.exists()),
    None,
)

if integrated_file is not None:
    old_integrated = pd.read_csv(integrated_file)
    lower_to_original = {
        c.lower(): c for c in old_integrated.columns
    }

    def find_column(candidates):
        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                return lower_to_original[candidate.lower()]
        return None

    model_col = find_column(["model", "model_name"])
    subset_col = find_column(["subset", "evaluation_subset", "analysis_subset"])
    metric_col = find_column(["metric", "metric_name"])
    p_col = find_column([
        "raw_p",
        "raw_p_value",
        "p_value",
        "unadjusted_p_value",
    ])

    if all(x is not None for x in [
        model_col, subset_col, metric_col, p_col
    ]):
        deberta_old = old_integrated[
            old_integrated[model_col]
            .astype(str)
            .str.contains("deberta", case=False, na=False)
        ].copy()

        # Keep only the six intended comparisons.
        deberta_old = deberta_old[
            deberta_old[metric_col]
            .astype(str)
            .str.lower()
            .str.contains("roc|pairwise", regex=True)
        ].copy()

        if len(deberta_old) >= 6:
            deberta_old = deberta_old.iloc[:6].copy()

            deberta_global = pd.DataFrame({
                "model": deberta_old[model_col].astype(str),
                "subset": deberta_old[subset_col].astype(str),
                "metric": deberta_old[metric_col].astype(str),
                "raw_p": pd.to_numeric(
                    deberta_old[p_col],
                    errors="raise",
                ),
                "source": str(integrated_file),
            })

            qwen_global = paired_df[
                ["model", "subset", "metric", "raw_p"]
            ].copy()
            qwen_global["source"] = "canonical recomputation"

            global_family_df = pd.concat(
                [deberta_global, qwen_global],
                ignore_index=True,
            )

            if len(global_family_df) == 12:
                global_family_df["global_12_holm_p"] = holm_adjust(
                    global_family_df["raw_p"].to_numpy()
                )
                global_family_df["significant_global_12_0_05"] = (
                    global_family_df["global_12_holm_p"] < 0.05
                )

                qwen_holm_map = {
                    (row["subset"], row["metric"]):
                    row["global_12_holm_p"]
                    for _, row in global_family_df[
                        global_family_df["model"]
                        == "Qwen3-Reranker-0.6B"
                    ].iterrows()
                }

                paired_df["global_12_holm_p"] = [
                    qwen_holm_map.get(
                        (row["subset"], row["metric"]),
                        np.nan,
                    )
                    for _, row in paired_df.iterrows()
                ]
                paired_df["significant_global_12_0_05"] = (
                    paired_df["global_12_holm_p"] < 0.05
                )
            else:
                print(
                    "\nWARNING: Global Holm family did not contain "
                    f"12 rows; found {len(global_family_df)}."
                )
                global_family_df = None
        else:
            print(
                "\nWARNING: Fewer than six DeBERTa comparisons "
                f"found in {integrated_file}"
            )
    else:
        print(
            "\nWARNING: Could not identify model/subset/metric/p-value "
            f"columns in {integrated_file}"
        )
else:
    print(
        "\nWARNING: Existing integrated DeBERTa statistics were not found. "
        "Qwen-family Holm correction was still completed."
    )


# ------------------------------------------------------------
# 9. Save auditable metric input
# ------------------------------------------------------------
metric_input_rows = []

for condition in CONDITIONS:
    u_lookup = (
        dense_ensemble[
            dense_ensemble["training_condition"] == condition
        ]
        .set_index("_query_key")["dense_hard_probability"]
        .to_dict()
    )

    c_part = cited_ensemble[
        cited_ensemble["training_condition"] == condition
    ]

    for _, row in c_part.iterrows():
        metric_input_rows.append({
            "training_condition": condition,
            "query_text_norm": row["_query_key"],
            "cited_candidate_key": row["_candidate_key"],
            "cited_probability": row["cited_probability"],
            "dense_hard_probability": u_lookup[row["_query_key"]],
        })

metric_input_df = pd.DataFrame(metric_input_rows)

bridge_export = bridge[
    [
        "_claim_key",
        "_query_key",
        "_same_passage",
        "_same_document",
    ]
].rename(columns={
    "_claim_key": "claim_id_norm",
    "_query_key": "query_text_norm",
    "_same_passage": "same_passage",
    "_same_document": "same_document",
})

metric_input_df = metric_input_df.merge(
    bridge_export,
    on="query_text_norm",
    how="left",
    validate="many_to_one",
)

assert metric_input_df["same_passage"].notna().all()
assert metric_input_df["same_document"].notna().all()


# ------------------------------------------------------------
# 10. Save all outputs
# ------------------------------------------------------------
point_file = OUT / "qwen_dense_hard_point_estimates.csv"
bootstrap_file = OUT / "qwen_dense_hard_bootstrap_ci.csv"
paired_file = OUT / "qwen_dense_hard_paired_tests.csv"
metric_input_file = OUT / "qwen_dense_hard_metric_inputs.parquet"
ensemble_u_file = OUT / "qwen_dense_hard_ensemble_predictions.parquet"
ensemble_cited_file = OUT / "qwen_cited_ensemble_predictions.parquet"
json_file = OUT / "qwen_dense_hard_statistical_results.json"

point_df.to_csv(point_file, index=False)
bootstrap_df.to_csv(bootstrap_file, index=False)
paired_df.to_csv(paired_file, index=False)
metric_input_df.to_parquet(metric_input_file, index=False)
dense_ensemble.to_parquet(ensemble_u_file, index=False)
cited_ensemble.to_parquet(ensemble_cited_file, index=False)

global_file = None
if global_family_df is not None:
    global_file = OUT / "table5_global_12_holm_recomputed.csv"
    global_family_df.to_csv(global_file, index=False)

result_json = {
    "experiment": "Qwen canonical Dense-Hard Table 5 evaluation",
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "conditions": CONDITIONS,
    "seeds": SEEDS,
    "number_of_queries": EXPECTED_QUERIES,
    "subsets": {
        name: len(values)
        for name, values in SUBSETS.items()
    },
    "bootstrap": {
        "unit": "query",
        "repetitions": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CI_LEVEL,
        "paired_between_conditions": True,
    },
    "metric_definition": {
        "roc_auc": (
            "Query-macro weighted ROC-AUC: each query has total cited "
            "weight 1 and Dense-Hard weight 1."
        ),
        "pairwise_cited_over_dense_hard": (
            "Mean same-query cited-over-Dense-Hard comparison; "
            "ties receive 0.5."
        ),
    },
    "multiple_testing": {
        "qwen_family_size": 6,
        "qwen_holm_completed": True,
        "global_table5_family_size": (
            12 if global_family_df is not None else None
        ),
        "global_table5_holm_completed": (
            global_family_df is not None
        ),
    },
    "inputs": {
        "canonical_predictions": str(NEW_PREDICTIONS),
        "canonical_predictions_sha256": sha256_file(
            NEW_PREDICTIONS
        ),
        "overlap_alignment": str(OVERLAP_FILE),
        "overlap_alignment_sha256": sha256_file(
            OVERLAP_FILE
        ),
        "existing_deberta_integrated_file": (
            str(integrated_file)
            if integrated_file is not None else None
        ),
    },
    "point_estimates": point_df.to_dict(orient="records"),
    "bootstrap_confidence_intervals": (
        bootstrap_df.to_dict(orient="records")
    ),
    "paired_tests": paired_df.to_dict(orient="records"),
    "output_files": {
        "point_estimates": str(point_file),
        "bootstrap_ci": str(bootstrap_file),
        "paired_tests": str(paired_file),
        "metric_inputs": str(metric_input_file),
        "ensemble_dense_hard": str(ensemble_u_file),
        "ensemble_cited": str(ensemble_cited_file),
        "global_12_holm": (
            str(global_file) if global_file else None
        ),
    },
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# 11. Final report
# ------------------------------------------------------------
display_columns = [
    "subset",
    "metric",
    "hard_only",
    "mixed_u",
    "delta_mixed_minus_hard",
    "ci_lower",
    "ci_upper",
    "raw_p",
    "qwen_family_6_holm_p",
]

if "global_12_holm_p" in paired_df.columns:
    display_columns.append("global_12_holm_p")

print("\n" + "=" * 90)
print("QWEN CANONICAL DENSE-HARD STATISTICAL EVALUATION COMPLETE")
print("=" * 90)

print("\n=== Point estimates and paired comparisons ===")
print(
    paired_df[display_columns]
    .sort_values(["subset", "metric"])
    .to_string(index=False)
)

print("\n=== Bootstrap confidence intervals ===")
print(
    bootstrap_df[
        [
            "subset",
            "training_condition",
            "metric",
            "point_estimate",
            "ci_lower",
            "ci_upper",
        ]
    ]
    .sort_values(["subset", "metric", "training_condition"])
    .to_string(index=False)
)

print("\n=== Output files ===")
for path in [
    point_file,
    bootstrap_file,
    paired_file,
    metric_input_file,
    ensemble_u_file,
    ensemble_cited_file,
    json_file,
]:
    print(f"{path} ({path.stat().st_size / 1024:.1f} KB)")

if global_file is not None:
    print(
        f"{global_file} "
        f"({global_file.stat().st_size / 1024:.1f} KB)"
    )
    print("\nGLOBAL TABLE 5 HOLM CORRECTION COMPLETED (12 comparisons)")
else:
    print("\nQWEN HOLM CORRECTION COMPLETED (6 comparisons)")
    print("Global 12-comparison Holm requires the existing DeBERTa table.")

print("\n다음으로 위의 paired-comparison 표 전체를 보내주세요.")