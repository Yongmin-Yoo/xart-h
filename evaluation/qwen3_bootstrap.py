"""Reproducibility script exported from the executed XART-H notebook."""

# ============================================================
# XART-H Qwen3-8B
# Query-level bootstrap CI + paired tests + Holm correction
# ============================================================

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    ndcg_score,
    roc_auc_score,
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/qwen3_8b_zero_shot_v1"
)

PREDICTION_FILE = ROOT / "qwen3_8b_test_predictions.parquet"

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(PREDICTION_FILE)

print("Prediction file:", PREDICTION_FILE)
print("Bootstrap repetitions:", N_BOOTSTRAP)

# ------------------------------------------------------------
# 2. Load official test set and saved predictions
# ------------------------------------------------------------
test = load_dataset(
    REPO_ID,
    revision=REVISION,
    split="test",
)

keep_columns = [
    column
    for column in ["_row_id", "text", "label"]
    if column in test.column_names
]

base = test.select_columns(keep_columns).to_pandas()
prediction = pd.read_parquet(PREDICTION_FILE)

required_prediction_columns = {
    "label",
    "direct_p_A",
    "direct_p_X",
    "direct_p_U",
    "hierarchical_p_A",
    "hierarchical_p_X",
    "hierarchical_p_U",
}

missing = required_prediction_columns - set(prediction.columns)

if missing:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing)}"
    )

# Align predictions safely by row ID when available.
if (
    "_row_id" in base.columns
    and "_row_id" in prediction.columns
    and base["_row_id"].is_unique
    and prediction["_row_id"].is_unique
):
    base["_row_id"] = base["_row_id"].astype(str)
    prediction["_row_id"] = prediction["_row_id"].astype(str)

    data = base.merge(
        prediction,
        on="_row_id",
        how="inner",
        suffixes=("_dataset", "_prediction"),
        validate="one_to_one",
    )

    data = data.sort_values(
        "_row_id",
        kind="stable",
    ).reset_index(drop=True)

    y_dataset = data["label_dataset"].astype(int).to_numpy()
    y_prediction = data["label_prediction"].astype(int).to_numpy()

    assert np.array_equal(y_dataset, y_prediction)

    data["label"] = y_dataset

else:
    if len(base) != len(prediction):
        raise ValueError(
            f"Row mismatch: dataset={len(base)}, "
            f"predictions={len(prediction)}"
        )

    base = base.reset_index(drop=True)
    prediction = prediction.reset_index(drop=True)

    if not np.array_equal(
        base["label"].astype(int).to_numpy(),
        prediction["label"].astype(int).to_numpy(),
    ):
        raise ValueError(
            "Prediction order does not match the test set."
        )

    data = pd.concat(
        [
            base,
            prediction.drop(columns=["label"]),
        ],
        axis=1,
    )

assert len(data) == 5316, len(data)

# ------------------------------------------------------------
# 3. Query definition
# ------------------------------------------------------------
def normalize_query(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()


data["query_key"] = data["text"].map(normalize_query)

query_keys = data["query_key"].drop_duplicates().tolist()

query_to_indices = {
    query: group.index.to_numpy(dtype=np.int64)
    for query, group in data.groupby(
        "query_key",
        sort=False,
    )
}

query_groups = [
    query_to_indices[query]
    for query in query_keys
]

assert len(query_groups) == 1808, len(query_groups)

print("Rows:", len(data))
print("Normalized queries:", len(query_groups))

# ------------------------------------------------------------
# 4. Probabilities
# ------------------------------------------------------------
y_true = data["label"].astype(int).to_numpy()

probabilities = {
    "direct": data[
        ["direct_p_A", "direct_p_X", "direct_p_U"]
    ].to_numpy(dtype=np.float64),

    "hierarchical": data[
        [
            "hierarchical_p_A",
            "hierarchical_p_X",
            "hierarchical_p_U",
        ]
    ].to_numpy(dtype=np.float64),
}

for method, probability in probabilities.items():
    if probability.shape != (5316, 3):
        raise ValueError(
            f"{method}: invalid shape {probability.shape}"
        )

    if not np.isfinite(probability).all():
        raise ValueError(
            f"{method}: non-finite probabilities found"
        )

    row_sum = probability.sum(axis=1)

    if not np.allclose(row_sum, 1.0, atol=1e-4):
        probabilities[method] = (
            probability
            / np.maximum(row_sum[:, None], 1e-12)
        )

# ------------------------------------------------------------
# 5. Global classification metrics
# ------------------------------------------------------------
GLOBAL_METRICS = [
    "accuracy",
    "macro_f1",
    "f1_A",
    "f1_X",
    "f1_U",
    "cited_vs_u_roc_auc",
    "cited_vs_u_average_precision",
    "x_vs_a_roc_auc",
    "x_vs_a_average_precision",
]

RANKING_METRICS = [
    "pairwise_cited_over_U",
    "pairwise_X_over_A",
    "mrr_cited",
    "ndcg",
]

ALL_METRICS = GLOBAL_METRICS + RANKING_METRICS


def safe_auc(target, score):
    if len(np.unique(target)) < 2:
        return float("nan")

    return float(
        roc_auc_score(target, score)
    )


def safe_ap(target, score):
    if len(np.unique(target)) < 2:
        return float("nan")

    return float(
        average_precision_score(target, score)
    )


def global_metrics(y, probability):
    prediction_label = probability.argmax(axis=1)

    class_f1 = f1_score(
        y,
        prediction_label,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    cited_target = (y != 2).astype(int)
    cited_score = probability[:, 0] + probability[:, 1]

    xa_mask = y != 2
    xa_target = (y[xa_mask] == 1).astype(int)

    x_score = (
        probability[xa_mask, 1]
        / (
            probability[xa_mask, 0]
            + probability[xa_mask, 1]
            + 1e-12
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y, prediction_label)
        ),
        "macro_f1": float(
            f1_score(
                y,
                prediction_label,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_A": float(class_f1[0]),
        "f1_X": float(class_f1[1]),
        "f1_U": float(class_f1[2]),
        "cited_vs_u_roc_auc": safe_auc(
            cited_target,
            cited_score,
        ),
        "cited_vs_u_average_precision": safe_ap(
            cited_target,
            cited_score,
        ),
        "x_vs_a_roc_auc": safe_auc(
            xa_target,
            x_score,
        ),
        "x_vs_a_average_precision": safe_ap(
            xa_target,
            x_score,
        ),
    }

# ------------------------------------------------------------
# 6. Per-query ranking metrics
# ------------------------------------------------------------
def one_query_ranking_metrics(y, probability):
    cited_score = probability[:, 0] + probability[:, 1]

    x_score = (
        probability[:, 1]
        / (
            probability[:, 0]
            + probability[:, 1]
            + 1e-12
        )
    )

    graded_score = (
        probability[:, 0]
        + 2.0 * probability[:, 1]
    )

    cited_mask = y != 2
    u_mask = y == 2
    x_mask = y == 1
    a_mask = y == 0

    result = {
        "pairwise_cited_over_U": np.nan,
        "pairwise_X_over_A": np.nan,
        "mrr_cited": np.nan,
        "ndcg": np.nan,
    }

    if cited_mask.any() and u_mask.any():
        cited_values = cited_score[cited_mask]
        u_values = cited_score[u_mask]

        wins = (
            cited_values[:, None]
            > u_values[None, :]
        ).astype(float)

        ties = (
            cited_values[:, None]
            == u_values[None, :]
        ).astype(float)

        result["pairwise_cited_over_U"] = float(
            (wins + 0.5 * ties).mean()
        )

        order = np.argsort(
            -cited_score,
            kind="mergesort",
        )

        relevant_positions = np.flatnonzero(
            cited_mask[order]
        )

        if len(relevant_positions):
            result["mrr_cited"] = float(
                1.0 / (relevant_positions[0] + 1)
            )

        relevance = np.where(
            y == 1,
            2.0,
            np.where(y == 0, 1.0, 0.0),
        )

        if len(y) >= 2 and relevance.max() > 0:
            result["ndcg"] = float(
                ndcg_score(
                    relevance.reshape(1, -1),
                    graded_score.reshape(1, -1),
                )
            )

    if x_mask.any() and a_mask.any():
        x_values = x_score[x_mask]
        a_values = x_score[a_mask]

        wins = (
            x_values[:, None]
            > a_values[None, :]
        ).astype(float)

        ties = (
            x_values[:, None]
            == a_values[None, :]
        ).astype(float)

        result["pairwise_X_over_A"] = float(
            (wins + 0.5 * ties).mean()
        )

    return result


def build_query_metric_arrays(probability):
    arrays = {
        metric: np.full(
            len(query_groups),
            np.nan,
            dtype=np.float64,
        )
        for metric in RANKING_METRICS
    }

    for query_index, row_indices in enumerate(query_groups):
        result = one_query_ranking_metrics(
            y_true[row_indices],
            probability[row_indices],
        )

        for metric in RANKING_METRICS:
            arrays[metric][query_index] = result[metric]

    return arrays


query_metric_arrays = {
    method: build_query_metric_arrays(probability)
    for method, probability in probabilities.items()
}


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if not len(values):
        return float("nan")

    return float(values.mean())


def point_metrics(method):
    result = global_metrics(
        y_true,
        probabilities[method],
    )

    for metric in RANKING_METRICS:
        result[metric] = finite_mean(
            query_metric_arrays[method][metric]
        )

    return result


point_results = {
    method: point_metrics(method)
    for method in probabilities
}

print("\nPoint estimates:")
print(json.dumps(point_results, indent=2))

# ------------------------------------------------------------
# 7. Query-level bootstrap
# ------------------------------------------------------------
rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_values = {
    method: {
        metric: np.full(
            N_BOOTSTRAP,
            np.nan,
            dtype=np.float64,
        )
        for metric in ALL_METRICS
    }
    for method in probabilities
}

number_of_queries = len(query_groups)
start_time = time.time()

for bootstrap_index in range(N_BOOTSTRAP):
    sampled_query_indices = rng.integers(
        0,
        number_of_queries,
        size=number_of_queries,
    )

    sampled_rows = np.concatenate([
        query_groups[index]
        for index in sampled_query_indices
    ])

    sampled_y = y_true[sampled_rows]

    for method, probability in probabilities.items():
        sampled_probability = probability[sampled_rows]

        global_result = global_metrics(
            sampled_y,
            sampled_probability,
        )

        for metric in GLOBAL_METRICS:
            bootstrap_values[method][metric][
                bootstrap_index
            ] = global_result[metric]

        for metric in RANKING_METRICS:
            sampled_query_values = (
                query_metric_arrays[method][metric][
                    sampled_query_indices
                ]
            )

            bootstrap_values[method][metric][
                bootstrap_index
            ] = finite_mean(sampled_query_values)

    if (
        (bootstrap_index + 1) % 100 == 0
        or bootstrap_index == 0
    ):
        elapsed = time.time() - start_time
        rate = (bootstrap_index + 1) / max(elapsed, 1e-9)
        remaining = (
            N_BOOTSTRAP - bootstrap_index - 1
        ) / max(rate, 1e-9)

        print(
            f"Bootstrap {bootstrap_index + 1:,}/"
            f"{N_BOOTSTRAP:,} | "
            f"ETA {remaining / 60:.1f} min"
        )

# ------------------------------------------------------------
# 8. Confidence intervals
# ------------------------------------------------------------
alpha = 1.0 - CI_LEVEL
lower_percentile = 100.0 * alpha / 2.0
upper_percentile = 100.0 * (1.0 - alpha / 2.0)


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if not len(values):
        return [None, None]

    return [
        float(np.percentile(values, lower_percentile)),
        float(np.percentile(values, upper_percentile)),
    ]


confidence_intervals = {}

for method in probabilities:
    confidence_intervals[method] = {}

    for metric in ALL_METRICS:
        lower, upper = percentile_interval(
            bootstrap_values[method][metric]
        )

        confidence_intervals[method][metric] = {
            "estimate": point_results[method][metric],
            "ci_lower": lower,
            "ci_upper": upper,
        }

# ------------------------------------------------------------
# 9. Paired bootstrap tests
# Hierarchical minus direct
# ------------------------------------------------------------
paired_results = {}

for metric in ALL_METRICS:
    direct_values = bootstrap_values["direct"][metric]
    hierarchical_values = bootstrap_values[
        "hierarchical"
    ][metric]

    valid = (
        np.isfinite(direct_values)
        & np.isfinite(hierarchical_values)
    )

    difference = (
        hierarchical_values[valid]
        - direct_values[valid]
    )

    point_difference = (
        point_results["hierarchical"][metric]
        - point_results["direct"][metric]
    )

    if len(difference):
        count_nonpositive = int(
            np.sum(difference <= 0)
        )
        count_nonnegative = int(
            np.sum(difference >= 0)
        )

        p_value = min(
            1.0,
            2.0
            * min(
                (count_nonpositive + 1)
                / (len(difference) + 1),
                (count_nonnegative + 1)
                / (len(difference) + 1),
            ),
        )

        delta_ci = percentile_interval(difference)

    else:
        p_value = float("nan")
        delta_ci = [None, None]

    paired_results[metric] = {
        "hierarchical_minus_direct": point_difference,
        "delta_ci_lower": delta_ci[0],
        "delta_ci_upper": delta_ci[1],
        "p_value": p_value,
    }

# ------------------------------------------------------------
# 10. Holm correction
# ------------------------------------------------------------
def holm_adjust(p_values):
    adjusted = {
        key: float("nan")
        for key in p_values
    }

    valid_items = [
        (key, value)
        for key, value in p_values.items()
        if np.isfinite(value)
    ]

    valid_items.sort(key=lambda item: item[1])
    number_of_tests = len(valid_items)

    running_maximum = 0.0

    for rank, (key, p_value) in enumerate(
        valid_items,
        start=1,
    ):
        adjusted_value = min(
            1.0,
            (number_of_tests - rank + 1) * p_value,
        )

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted[key] = running_maximum

    return adjusted


raw_p_values = {
    metric: values["p_value"]
    for metric, values in paired_results.items()
}

holm_p_values = holm_adjust(raw_p_values)

for metric in paired_results:
    paired_results[metric]["holm_p_value"] = (
        holm_p_values[metric]
    )

    paired_results[metric]["significant_after_holm_0.05"] = bool(
        np.isfinite(holm_p_values[metric])
        and holm_p_values[metric] < 0.05
    )

# ------------------------------------------------------------
# 11. Save JSON
# ------------------------------------------------------------
def clean_number(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)

    if isinstance(value, dict):
        return {
            key: clean_number(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_number(item)
            for item in value
        ]

    return value


final_result = {
    "dataset": REPO_ID,
    "revision": REVISION,
    "split": "test",
    "model": "Qwen/Qwen3-8B",
    "bootstrap_unit": (
        "lowercased and whitespace-normalized claim text"
    ),
    "bootstrap_repetitions": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CI_LEVEL,
    "confidence_intervals": confidence_intervals,
    "paired_comparison": {
        "contrast": "hierarchical minus direct",
        "multiple_testing_correction": "Holm",
        "metrics": paired_results,
    },
}

final_result = clean_number(final_result)

json_path = ROOT / "qwen3_8b_bootstrap.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        final_result,
        file,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 12. Save paper-ready CSV files
# ------------------------------------------------------------
ci_rows = []

for method, metrics in confidence_intervals.items():
    for metric, values in metrics.items():
        ci_rows.append({
            "model": "Qwen3-8B",
            "method": method,
            "metric": metric,
            "estimate": values["estimate"],
            "ci_lower": values["ci_lower"],
            "ci_upper": values["ci_upper"],
        })

ci_table = pd.DataFrame(ci_rows)
ci_csv_path = ROOT / "qwen3_8b_bootstrap_ci.csv"
ci_table.to_csv(ci_csv_path, index=False)

paired_rows = []

for metric, values in paired_results.items():
    paired_rows.append({
        "metric": metric,
        **values,
    })

paired_table = pd.DataFrame(paired_rows)
paired_csv_path = ROOT / "qwen3_8b_paired_tests.csv"
paired_table.to_csv(paired_csv_path, index=False)

# ------------------------------------------------------------
# 13. Print concise result
# ------------------------------------------------------------
print("\n" + "=" * 72)
print("QWEN3-8B BOOTSTRAP RESULT")
print("=" * 72)

important_metrics = [
    "accuracy",
    "macro_f1",
    "cited_vs_u_roc_auc",
    "x_vs_a_roc_auc",
    "pairwise_cited_over_U",
    "pairwise_X_over_A",
    "ndcg",
]

for method in ["direct", "hierarchical"]:
    print(f"\n{method.upper()}")

    for metric in important_metrics:
        result = confidence_intervals[method][metric]

        print(
            f"{metric}: "
            f"{result['estimate']:.6f} "
            f"[{result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f}]"
        )

print("\nPAIRED: HIERARCHICAL MINUS DIRECT")

for metric in important_metrics:
    result = paired_results[metric]

    print(
        f"{metric}: "
        f"delta={result['hierarchical_minus_direct']:.6f}, "
        f"95% CI=[{result['delta_ci_lower']:.6f}, "
        f"{result['delta_ci_upper']:.6f}], "
        f"Holm-p={result['holm_p_value']:.6f}"
    )

print("\nSaved:")
print(json_path)
print(ci_csv_path)
print(paired_csv_path)
print("=" * 72)
