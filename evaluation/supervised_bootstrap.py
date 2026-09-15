"""Reproducibility script exported from the executed XART-H notebook."""

# ============================================================
# XART-H supervised models
# Query-level bootstrap CI, paired tests, Holm correction
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
    "XART-H/experiments/supervised_v1"
)

OUTPUT_DIR = ROOT / "bootstrap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["minilm", "deberta"]
TASKS = ["cited_vs_u", "x_vs_a", "direct_3way"]
SEEDS = [13, 42, 77]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

if not ROOT.exists():
    raise FileNotFoundError(ROOT)

# ------------------------------------------------------------
# 2. Load official test set
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

data = test.select_columns(keep_columns).to_pandas()
data["label"] = data["label"].astype(int)

assert len(data) == 5316
assert data["label"].value_counts().sort_index().to_dict() == {
    0: 1816,
    1: 1822,
    2: 1678,
}

y_true = data["label"].to_numpy()

def normalize_query(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()

data["query_key"] = data["text"].map(normalize_query)

query_groups = [
    group.index.to_numpy(dtype=np.int64)
    for _, group in data.groupby(
        "query_key",
        sort=False,
    )
]

assert len(query_groups) == 1808

print("Rows:", len(data))
print("Normalized queries:", len(query_groups))

# ------------------------------------------------------------
# 3. Locate saved NPZ files
# ------------------------------------------------------------
all_npz_files = sorted(ROOT.rglob("*.npz"))

print("\nDiscovered NPZ files:", len(all_npz_files))

for path in all_npz_files:
    print(" -", path.relative_to(ROOT))

def find_npz(model_name, task_name, seed):
    candidates = []

    for path in all_npz_files:
        name = path.name.lower()

        conditions = [
            model_name.lower() in name,
            task_name.lower() in name,
            f"seed{seed}" in name or f"seed_{seed}" in name,
            "test" in name,
            "validation" not in name,
            "val" not in name,
        ]

        if all(conditions):
            candidates.append(path)

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"\nExpected exactly one NPZ file for:\n"
            f"model={model_name}, task={task_name}, seed={seed}\n"
            f"Found: {[str(x) for x in candidates]}"
        )

    return candidates[0]

def extract_logits(path, expected_rows, expected_classes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in archive.files
        }

    preferred_keys = [
        "logits",
        "predictions",
        "pred",
        "scores",
    ]

    selected = None
    selected_key = None

    for key in preferred_keys:
        if key in arrays:
            candidate = np.asarray(arrays[key])

            if candidate.ndim in {1, 2}:
                selected = candidate
                selected_key = key
                break

    if selected is None:
        for key, candidate in arrays.items():
            candidate = np.asarray(candidate)

            if (
                candidate.ndim in {1, 2}
                and candidate.shape[0] == expected_rows
            ):
                selected = candidate
                selected_key = key
                break

    if selected is None:
        shapes = {
            key: np.asarray(value).shape
            for key, value in arrays.items()
        }
        raise ValueError(
            f"No logits found in {path}. Arrays: {shapes}"
        )

    selected = np.asarray(selected, dtype=np.float64)

    if selected.shape[0] != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows, "
            f"found {selected.shape}"
        )

    if selected.ndim == 1:
        if expected_classes != 2:
            raise ValueError(
                f"{path}: one-dimensional logits cannot be "
                f"used for {expected_classes} classes"
            )
        selected = np.column_stack(
            [-selected, selected]
        )

    if selected.shape[1] == 1 and expected_classes == 2:
        selected = np.column_stack(
            [-selected[:, 0], selected[:, 0]]
        )

    if selected.shape[1] != expected_classes:
        raise ValueError(
            f"{path}: expected {expected_classes} classes, "
            f"found {selected.shape}"
        )

    if not np.isfinite(selected).all():
        raise ValueError(
            f"{path}: non-finite logits detected"
        )

    print(
        f"Loaded {path.name}: "
        f"key={selected_key}, shape={selected.shape}"
    )

    return selected

def softmax(logits):
    logits = logits - logits.max(
        axis=1,
        keepdims=True,
    )

    exponential = np.exp(logits)

    return exponential / exponential.sum(
        axis=1,
        keepdims=True,
    )

# ------------------------------------------------------------
# 4. Load all seed predictions
# ------------------------------------------------------------
seed_probabilities = {}

for model_name in MODELS:
    seed_probabilities[model_name] = {}

    for seed in SEEDS:
        cited_path = find_npz(
            model_name,
            "cited_vs_u",
            seed,
        )
        xa_path = find_npz(
            model_name,
            "x_vs_a",
            seed,
        )
        direct_path = find_npz(
            model_name,
            "direct_3way",
            seed,
        )

        cited_logits = extract_logits(
            cited_path,
            expected_rows=len(data),
            expected_classes=2,
        )
        xa_logits = extract_logits(
            xa_path,
            expected_rows=len(data),
            expected_classes=2,
        )
        direct_logits = extract_logits(
            direct_path,
            expected_rows=len(data),
            expected_classes=3,
        )

        # cited_vs_u mapping:
        # class 0 = U, class 1 = cited
        cited_probability = softmax(cited_logits)
        p_u = cited_probability[:, 0]
        p_cited = cited_probability[:, 1]

        # x_vs_a mapping:
        # class 0 = A, class 1 = X
        xa_probability = softmax(xa_logits)
        p_a_given_cited = xa_probability[:, 0]
        p_x_given_cited = xa_probability[:, 1]

        # Direct order:
        # A = 0, X = 1, U = 2
        direct_probability = softmax(direct_logits)

        hierarchical_probability = np.column_stack([
            p_cited * p_a_given_cited,
            p_cited * p_x_given_cited,
            p_u,
        ])

        hierarchical_probability /= (
            hierarchical_probability.sum(
                axis=1,
                keepdims=True,
            )
        )

        seed_probabilities[model_name][seed] = {
            "direct": direct_probability,
            "hierarchical": hierarchical_probability,
        }

# ------------------------------------------------------------
# 5. Average probabilities across three seeds
# ------------------------------------------------------------
probabilities = {}

for model_name in MODELS:
    probabilities[model_name] = {}

    for method in ["direct", "hierarchical"]:
        stacked = np.stack([
            seed_probabilities[model_name][seed][method]
            for seed in SEEDS
        ])

        probabilities[model_name][method] = stacked.mean(
            axis=0
        )

# ------------------------------------------------------------
# 6. Metric functions
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

def calculate_global_metrics(labels, probability):
    prediction = probability.argmax(axis=1)

    class_f1 = f1_score(
        labels,
        prediction,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    cited_target = (labels != 2).astype(int)
    cited_score = probability[:, 0] + probability[:, 1]

    xa_mask = labels != 2
    xa_target = (labels[xa_mask] == 1).astype(int)

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
            accuracy_score(labels, prediction)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
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

def calculate_query_ranking_metrics(
    labels,
    probability,
):
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

    cited_mask = labels != 2
    u_mask = labels == 2
    x_mask = labels == 1
    a_mask = labels == 0

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
            labels == 1,
            2.0,
            np.where(labels == 0, 1.0, 0.0),
        )

        if len(labels) >= 2 and relevance.max() > 0:
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

def finite_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if not len(values):
        return float("nan")

    return float(values.mean())

# ------------------------------------------------------------
# 7. Precompute query-level ranking metrics
# ------------------------------------------------------------
query_metric_arrays = {}

for model_name in MODELS:
    query_metric_arrays[model_name] = {}

    for method in ["direct", "hierarchical"]:
        query_metric_arrays[model_name][method] = {
            metric: np.full(
                len(query_groups),
                np.nan,
                dtype=np.float64,
            )
            for metric in RANKING_METRICS
        }

        probability = probabilities[model_name][method]

        for query_index, row_indices in enumerate(
            query_groups
        ):
            query_result = calculate_query_ranking_metrics(
                y_true[row_indices],
                probability[row_indices],
            )

            for metric in RANKING_METRICS:
                query_metric_arrays[
                    model_name
                ][method][metric][query_index] = (
                    query_result[metric]
                )

# ------------------------------------------------------------
# 8. Point estimates
# ------------------------------------------------------------
def point_metrics(model_name, method):
    probability = probabilities[model_name][method]

    result = calculate_global_metrics(
        y_true,
        probability,
    )

    for metric in RANKING_METRICS:
        result[metric] = finite_mean(
            query_metric_arrays[
                model_name
            ][method][metric]
        )

    return result

point_results = {
    model_name: {
        method: point_metrics(model_name, method)
        for method in ["direct", "hierarchical"]
    }
    for model_name in MODELS
}

print("\nSeed-ensemble point estimates:")
print(json.dumps(point_results, indent=2))

# ------------------------------------------------------------
# 9. Query-level bootstrap
# ------------------------------------------------------------
rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_values = {
    model_name: {
        method: {
            metric: np.full(
                N_BOOTSTRAP,
                np.nan,
                dtype=np.float64,
            )
            for metric in ALL_METRICS
        }
        for method in ["direct", "hierarchical"]
    }
    for model_name in MODELS
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

    sampled_labels = y_true[sampled_rows]

    for model_name in MODELS:
        for method in ["direct", "hierarchical"]:
            sampled_probability = probabilities[
                model_name
            ][method][sampled_rows]

            global_result = calculate_global_metrics(
                sampled_labels,
                sampled_probability,
            )

            for metric in GLOBAL_METRICS:
                bootstrap_values[
                    model_name
                ][method][metric][bootstrap_index] = (
                    global_result[metric]
                )

            for metric in RANKING_METRICS:
                values = query_metric_arrays[
                    model_name
                ][method][metric][sampled_query_indices]

                bootstrap_values[
                    model_name
                ][method][metric][bootstrap_index] = (
                    finite_mean(values)
                )

    if (
        (bootstrap_index + 1) % 100 == 0
        or bootstrap_index == 0
    ):
        elapsed = time.time() - start_time
        rate = (
            bootstrap_index + 1
        ) / max(elapsed, 1e-9)

        remaining = (
            N_BOOTSTRAP - bootstrap_index - 1
        ) / max(rate, 1e-9)

        print(
            f"Bootstrap {bootstrap_index + 1:,}/"
            f"{N_BOOTSTRAP:,} | "
            f"ETA {remaining / 60:.1f} min"
        )

# ------------------------------------------------------------
# 10. Confidence intervals
# ------------------------------------------------------------
alpha = 1.0 - CI_LEVEL

def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if not len(values):
        return [None, None]

    return [
        float(np.percentile(values, 100 * alpha / 2)),
        float(
            np.percentile(
                values,
                100 * (1 - alpha / 2),
            )
        ),
    ]

confidence_intervals = {}

for model_name in MODELS:
    confidence_intervals[model_name] = {}

    for method in ["direct", "hierarchical"]:
        confidence_intervals[model_name][method] = {}

        for metric in ALL_METRICS:
            lower, upper = percentile_interval(
                bootstrap_values[
                    model_name
                ][method][metric]
            )

            confidence_intervals[
                model_name
            ][method][metric] = {
                "estimate": point_results[
                    model_name
                ][method][metric],
                "ci_lower": lower,
                "ci_upper": upper,
            }

# ------------------------------------------------------------
# 11. Paired model comparisons
# ------------------------------------------------------------
COMPARISONS = {
    "minilm_hierarchical_minus_direct": (
        ("minilm", "hierarchical"),
        ("minilm", "direct"),
    ),
    "deberta_hierarchical_minus_direct": (
        ("deberta", "hierarchical"),
        ("deberta", "direct"),
    ),
    "deberta_direct_minus_minilm_direct": (
        ("deberta", "direct"),
        ("minilm", "direct"),
    ),
    "deberta_hierarchical_minus_minilm_hierarchical": (
        ("deberta", "hierarchical"),
        ("minilm", "hierarchical"),
    ),
}

paired_results = {}

for comparison_name, (
    left_specification,
    right_specification,
) in COMPARISONS.items():

    left_model, left_method = left_specification
    right_model, right_method = right_specification

    paired_results[comparison_name] = {}

    for metric in ALL_METRICS:
        left_values = bootstrap_values[
            left_model
        ][left_method][metric]

        right_values = bootstrap_values[
            right_model
        ][right_method][metric]

        valid = (
            np.isfinite(left_values)
            & np.isfinite(right_values)
        )

        difference = (
            left_values[valid]
            - right_values[valid]
        )

        point_difference = (
            point_results[left_model][left_method][metric]
            - point_results[right_model][right_method][metric]
        )

        if len(difference):
            nonpositive = int(
                np.sum(difference <= 0)
            )
            nonnegative = int(
                np.sum(difference >= 0)
            )

            p_value = min(
                1.0,
                2.0 * min(
                    (nonpositive + 1)
                    / (len(difference) + 1),
                    (nonnegative + 1)
                    / (len(difference) + 1),
                ),
            )

            lower, upper = percentile_interval(difference)

        else:
            p_value = float("nan")
            lower, upper = None, None

        paired_results[
            comparison_name
        ][metric] = {
            "difference": point_difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_value,
        }

# ------------------------------------------------------------
# 12. Global Holm correction
# ------------------------------------------------------------
flat_p_values = {}

for comparison_name, metrics in paired_results.items():
    for metric, values in metrics.items():
        flat_p_values[
            f"{comparison_name}::{metric}"
        ] = values["p_value"]

def holm_adjust(p_values):
    adjusted = {
        key: float("nan")
        for key in p_values
    }

    valid = [
        (key, value)
        for key, value in p_values.items()
        if np.isfinite(value)
    ]

    valid.sort(key=lambda item: item[1])
    number_of_tests = len(valid)

    running_maximum = 0.0

    for rank, (key, p_value) in enumerate(
        valid,
        start=1,
    ):
        corrected = min(
            1.0,
            (number_of_tests - rank + 1) * p_value,
        )

        running_maximum = max(
            running_maximum,
            corrected,
        )

        adjusted[key] = running_maximum

    return adjusted

adjusted_p_values = holm_adjust(flat_p_values)

for comparison_name, metrics in paired_results.items():
    for metric, values in metrics.items():
        key = f"{comparison_name}::{metric}"
        corrected = adjusted_p_values[key]

        values["holm_p_value"] = corrected
        values["significant_after_holm_0.05"] = bool(
            np.isfinite(corrected)
            and corrected < 0.05
        )

# ------------------------------------------------------------
# 13. Seed-level point metrics
# ------------------------------------------------------------
seed_level_results = {}

for model_name in MODELS:
    seed_level_results[model_name] = {}

    for seed in SEEDS:
        seed_level_results[model_name][str(seed)] = {}

        for method in ["direct", "hierarchical"]:
            probability = seed_probabilities[
                model_name
            ][seed][method]

            result = calculate_global_metrics(
                y_true,
                probability,
            )

            seed_level_results[
                model_name
            ][str(seed)][method] = result

# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------
def clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)

    if isinstance(value, dict):
        return {
            key: clean_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_value(item)
            for item in value
        ]

    return value

final_result = {
    "dataset": REPO_ID,
    "revision": REVISION,
    "split": "test",
    "models": MODELS,
    "seeds": SEEDS,
    "bootstrap_unit": (
        "lowercased and whitespace-normalized claim text"
    ),
    "bootstrap_repetitions": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CI_LEVEL,
    "bootstrap_prediction": (
        "mean class probabilities across three seeds"
    ),
    "seed_level_results": seed_level_results,
    "confidence_intervals": confidence_intervals,
    "paired_comparisons": paired_results,
    "holm_scope": (
        "all reported paired comparisons and metrics"
    ),
}

final_result = clean_value(final_result)

json_path = (
    OUTPUT_DIR
    / "supervised_bootstrap_results.json"
)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        final_result,
        file,
        indent=2,
        ensure_ascii=False,
    )

# Confidence interval CSV
ci_rows = []

for model_name, methods in confidence_intervals.items():
    for method, metrics in methods.items():
        for metric, values in metrics.items():
            ci_rows.append({
                "model": model_name,
                "method": method,
                "metric": metric,
                **values,
            })

ci_table = pd.DataFrame(ci_rows)

ci_csv_path = (
    OUTPUT_DIR
    / "supervised_bootstrap_ci.csv"
)

ci_table.to_csv(
    ci_csv_path,
    index=False,
)

# Paired comparison CSV
paired_rows = []

for comparison_name, metrics in paired_results.items():
    for metric, values in metrics.items():
        paired_rows.append({
            "comparison": comparison_name,
            "metric": metric,
            **values,
        })

paired_table = pd.DataFrame(paired_rows)

paired_csv_path = (
    OUTPUT_DIR
    / "supervised_paired_tests.csv"
)

paired_table.to_csv(
    paired_csv_path,
    index=False,
)

# ------------------------------------------------------------
# 15. Print concise summary
# ------------------------------------------------------------
IMPORTANT_METRICS = [
    "accuracy",
    "macro_f1",
    "cited_vs_u_roc_auc",
    "x_vs_a_roc_auc",
    "pairwise_cited_over_U",
    "pairwise_X_over_A",
    "ndcg",
]

print("\n" + "=" * 74)
print("SUPERVISED BOOTSTRAP RESULT")
print("=" * 74)

for model_name in MODELS:
    for method in ["direct", "hierarchical"]:
        print(
            f"\n{model_name.upper()} "
            f"{method.upper()}"
        )

        for metric in IMPORTANT_METRICS:
            result = confidence_intervals[
                model_name
            ][method][metric]

            print(
                f"{metric}: "
                f"{result['estimate']:.6f} "
                f"[{result['ci_lower']:.6f}, "
                f"{result['ci_upper']:.6f}]"
            )

print("\nPAIRED COMPARISONS")

for comparison_name, metrics in paired_results.items():
    print(f"\n{comparison_name}")

    for metric in IMPORTANT_METRICS:
        result = metrics[metric]

        print(
            f"{metric}: "
            f"delta={result['difference']:.6f}, "
            f"95% CI=[{result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f}], "
            f"Holm-p={result['holm_p_value']:.6f}"
        )

print("\nSaved:")
print(json_path)
print(ci_csv_path)
print(paired_csv_path)
print("=" * 74)
