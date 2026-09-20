# ============================================================
# XART-H Temporal Robustness Evaluation
# Query-level temporal bins, bootstrap CIs, early vs late tests
# No retraining required
# ============================================================

import json
import math
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

SUPERVISED_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/supervised_v1"
)

QWEN_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/qwen3_8b_zero_shot_v1"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/temporal_robustness_v1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_PREDICTIONS = (
    QWEN_ROOT / "qwen3_8b_test_predictions.parquet"
)

MODELS = ["minilm", "deberta"]
SEEDS = [13, 42, 77]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

TRAIN_END_DATE = pd.Timestamp("2016-09-21", tz="UTC")

METRICS = [
    "accuracy",
    "macro_f1",
    "cited_vs_u_roc_auc",
    "x_vs_a_roc_auc",
    "pairwise_cited_over_U",
    "pairwise_X_over_A",
]

if not SUPERVISED_ROOT.exists():
    raise FileNotFoundError(SUPERVISED_ROOT)

if not QWEN_PREDICTIONS.exists():
    raise FileNotFoundError(QWEN_PREDICTIONS)

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as error:
    print("Drive mount skipped:", error)

# ------------------------------------------------------------
# 2. Load the frozen test set
# Temporal unit: patent application and claim
# ------------------------------------------------------------
test_dataset = load_dataset(
    REPO_ID,
    revision=REVISION,
    split="test",
)

required_columns = {
    "claim_id",
    "patent_application_id",
    "text",
    "label",
}

missing_columns = (
    required_columns
    - set(test_dataset.column_names)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )

# Prefer the normalized split date.
date_candidates = [
    "_date",
    "_parsed_date",
    "date",
]

date_column = next(
    (
        column
        for column in date_candidates
        if column in test_dataset.column_names
    ),
    None,
)

if date_column is None:
    raise ValueError(
        "No temporal split date column was found."
    )

keep_columns = [
    column
    for column in [
        "_row_id",
        "claim_id",
        "patent_application_id",
        "text",
        "label",
        date_column,
    ]
    if column in test_dataset.column_names
]

data = test_dataset.select_columns(
    keep_columns
).to_pandas()

data["label"] = data["label"].astype(int)
data["target_date"] = pd.to_datetime(
    data[date_column],
    errors="coerce",
    utc=True,
)

if data["target_date"].isna().any():
    raise ValueError(
        "Unparseable target dates: "
        f"{int(data['target_date'].isna().sum())}"
    )

assert len(data) == 5316
assert data["label"].value_counts().sort_index().to_dict() == {
    0: 1816,
    1: 1822,
    2: 1678,
}

data["_original_order"] = np.arange(len(data))

# An identical normalized claim can occur in multiple applications
# with different dates. Temporal evaluation therefore uses a unique
# application-claim unit rather than claim text alone.
data["query_key"] = (
    data["patent_application_id"]
    .fillna("")
    .astype(str)
    .str.strip()
    + "::"
    + data["claim_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if (data["query_key"] == "::").any():
    raise ValueError(
        "Missing application and claim identifiers."
    )

query_groups = []
query_records = []

for query_index, (query_key, group) in enumerate(
    data.groupby("query_key", sort=False)
):
    row_indices = group.index.to_numpy(dtype=np.int64)
    unique_dates = group["target_date"].drop_duplicates()

    if len(unique_dates) != 1:
        raise ValueError(
            "Application-claim unit has multiple dates: "
            f"{query_key}, dates="
            f"{unique_dates.astype(str).tolist()}"
        )

    query_date = unique_dates.iloc[0]

    query_groups.append(row_indices)
    query_records.append({
        "query_index": query_index,
        "query_key": query_key,
        "query_hash": hashlib.sha256(
            query_key.encode("utf-8")
        ).hexdigest(),
        "target_date": query_date,
        "days_since_train_end": int(
            (query_date - TRAIN_END_DATE).days
        ),
        "rows": int(len(row_indices)),
    })

query_table = pd.DataFrame(query_records)

if len(query_table) == 0:
    raise RuntimeError(
        "No application-claim query units were created."
    )

print("Rows:", len(data))
print("Application-claim units:", len(query_table))
print("Date column:", date_column)
print(
    "Test date range:",
    query_table["target_date"].min().date(),
    "to",
    query_table["target_date"].max().date(),
)

# ------------------------------------------------------------
# 3. Create chronological tertiles
# Identical dates remain in the same temporal bin
# ------------------------------------------------------------
date_rank = query_table["target_date"].rank(
    method="average",
    pct=True,
)

query_table["temporal_bin"] = pd.cut(
    date_rank,
    bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    labels=["early", "middle", "late"],
    include_lowest=True,
).astype(str)

if query_table["temporal_bin"].isna().any():
    raise RuntimeError(
        "Temporal bin assignment failed."
    )

BIN_ORDER = ["early", "middle", "late"]

bin_query_indices = {
    bin_name: query_table.loc[
        query_table["temporal_bin"] == bin_name,
        "query_index",
    ].to_numpy(dtype=np.int64)
    for bin_name in BIN_ORDER
}

bin_summary = []

for bin_name in BIN_ORDER:
    subset = query_table[
        query_table["temporal_bin"] == bin_name
    ]

    if len(subset) == 0:
        raise RuntimeError(
            f"Empty temporal bin: {bin_name}"
        )

    bin_summary.append({
        "temporal_bin": bin_name,
        "queries": int(len(subset)),
        "start_date": str(
            subset["target_date"].min().date()
        ),
        "end_date": str(
            subset["target_date"].max().date()
        ),
        "median_days_since_train_end": float(
            subset["days_since_train_end"].median()
        ),
    })

bin_summary_table = pd.DataFrame(bin_summary)

print("\nTemporal bins:")
print(bin_summary_table.to_string(index=False))

# ------------------------------------------------------------
# 3. Create chronological tertiles
# Identical dates remain in the same temporal bin
# ------------------------------------------------------------
date_rank = query_table["target_date"].rank(
    method="average",
    pct=True,
)

query_table["temporal_bin"] = pd.cut(
    date_rank,
    bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    labels=["early", "middle", "late"],
    include_lowest=True,
).astype(str)

if query_table["temporal_bin"].isna().any():
    raise RuntimeError("Temporal bin assignment failed.")

BIN_ORDER = ["early", "middle", "late"]

bin_query_indices = {
    bin_name: query_table.loc[
        query_table["temporal_bin"] == bin_name,
        "query_index",
    ].to_numpy(dtype=np.int64)
    for bin_name in BIN_ORDER
}

bin_summary = []

for bin_name in BIN_ORDER:
    subset = query_table[
        query_table["temporal_bin"] == bin_name
    ]

    bin_summary.append({
        "temporal_bin": bin_name,
        "queries": int(len(subset)),
        "start_date": str(
            subset["target_date"].min().date()
        ),
        "end_date": str(
            subset["target_date"].max().date()
        ),
        "median_days_since_train_end": float(
            subset["days_since_train_end"].median()
        ),
    })

bin_summary_table = pd.DataFrame(bin_summary)

print("\nTemporal bins:")
print(bin_summary_table.to_string(index=False))

# ------------------------------------------------------------
# 4. Prediction loading helpers
# ------------------------------------------------------------
def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)

    return exponentials / exponentials.sum(
        axis=1,
        keepdims=True,
    )


def normalize_probabilities(probability):
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    if not np.isfinite(probability).all():
        raise ValueError("Non-finite probabilities detected.")

    row_sum = probability.sum(axis=1, keepdims=True)

    return probability / np.maximum(row_sum, 1e-12)


all_npz_files = sorted(
    SUPERVISED_ROOT.rglob("*.npz")
)

print("\nDiscovered supervised NPZ files:", len(all_npz_files))


def find_npz(model_name, task_name, seed):
    candidates = []

    for path in all_npz_files:
        name = path.name.lower()

        conditions = [
            model_name.lower() in name,
            task_name.lower() in name,
            (
                f"seed{seed}" in name
                or f"seed_{seed}" in name
            ),
            "test" in name,
            "validation" not in name,
            "val" not in name,
        ]

        if all(conditions):
            candidates.append(path)

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one NPZ file for "
            f"{model_name}/{task_name}/seed{seed}. "
            f"Found: {[str(path) for path in candidates]}"
        )

    return candidates[0]


def extract_logits(path, expected_rows, expected_classes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in archive.files
        }

    selected = None

    for key in [
        "logits",
        "predictions",
        "pred",
        "scores",
    ]:
        if key in arrays:
            candidate = np.asarray(arrays[key])

            if (
                candidate.ndim in {1, 2}
                and candidate.shape[0] == expected_rows
            ):
                selected = candidate
                break

    if selected is None:
        for candidate in arrays.values():
            candidate = np.asarray(candidate)

            if (
                candidate.ndim in {1, 2}
                and candidate.shape[0] == expected_rows
            ):
                selected = candidate
                break

    if selected is None:
        raise ValueError(f"No usable logits in {path}")

    selected = np.asarray(selected, dtype=np.float64)

    if selected.ndim == 1:
        if expected_classes != 2:
            raise ValueError(
                f"Invalid one-dimensional logits: {path}"
            )

        selected = np.column_stack(
            [-selected, selected]
        )

    if (
        selected.ndim == 2
        and selected.shape[1] == 1
        and expected_classes == 2
    ):
        selected = np.column_stack([
            -selected[:, 0],
            selected[:, 0],
        ])

    if selected.shape != (
        expected_rows,
        expected_classes,
    ):
        raise ValueError(
            f"{path}: expected "
            f"({expected_rows}, {expected_classes}), "
            f"found {selected.shape}"
        )

    return selected

# ------------------------------------------------------------
# 5. Load supervised seed ensembles
# ------------------------------------------------------------
systems = {}

for model_name in MODELS:
    direct_seed_probabilities = []
    hierarchical_seed_probabilities = []

    for seed in SEEDS:
        cited_logits = extract_logits(
            find_npz(
                model_name,
                "cited_vs_u",
                seed,
            ),
            len(data),
            2,
        )

        xa_logits = extract_logits(
            find_npz(
                model_name,
                "x_vs_a",
                seed,
            ),
            len(data),
            2,
        )

        direct_logits = extract_logits(
            find_npz(
                model_name,
                "direct_3way",
                seed,
            ),
            len(data),
            3,
        )

        cited_probability = softmax(cited_logits)
        xa_probability = softmax(xa_logits)
        direct_probability = softmax(direct_logits)

        # Binary mappings:
        # cited_vs_u: class 0 = U, class 1 = cited
        # x_vs_a: class 0 = A, class 1 = X
        p_u = cited_probability[:, 0]
        p_cited = cited_probability[:, 1]

        p_a_given_cited = xa_probability[:, 0]
        p_x_given_cited = xa_probability[:, 1]

        hierarchical_probability = np.column_stack([
            p_cited * p_a_given_cited,
            p_cited * p_x_given_cited,
            p_u,
        ])

        hierarchical_probability = normalize_probabilities(
            hierarchical_probability
        )

        direct_seed_probabilities.append(
            direct_probability
        )
        hierarchical_seed_probabilities.append(
            hierarchical_probability
        )

    systems[f"{model_name}_direct"] = (
        np.stack(
            direct_seed_probabilities,
            axis=0,
        ).mean(axis=0)
    )

    systems[f"{model_name}_hierarchical"] = (
        np.stack(
            hierarchical_seed_probabilities,
            axis=0,
        ).mean(axis=0)
    )

# ------------------------------------------------------------
# 6. Load and align Qwen3-8B predictions
# ------------------------------------------------------------
qwen = pd.read_parquet(QWEN_PREDICTIONS)

required_qwen_columns = {
    "label",
    "direct_p_A",
    "direct_p_X",
    "direct_p_U",
    "hierarchical_p_A",
    "hierarchical_p_X",
    "hierarchical_p_U",
}

missing_qwen = required_qwen_columns - set(qwen.columns)

if missing_qwen:
    raise ValueError(
        f"Missing Qwen columns: {sorted(missing_qwen)}"
    )

if (
    "_row_id" in data.columns
    and "_row_id" in qwen.columns
    and data["_row_id"].is_unique
    and qwen["_row_id"].is_unique
):
    base_ids = data["_row_id"].astype(str)
    qwen = qwen.copy()
    qwen["_row_id"] = qwen["_row_id"].astype(str)
    qwen = qwen.set_index("_row_id").reindex(base_ids)

    if qwen.index.isna().any() or qwen["label"].isna().any():
        raise ValueError("Qwen row-ID alignment failed.")

    qwen = qwen.reset_index(drop=True)

else:
    qwen = qwen.reset_index(drop=True)

    if len(qwen) != len(data):
        raise ValueError(
            f"Qwen rows={len(qwen)}, dataset rows={len(data)}"
        )

if not np.array_equal(
    qwen["label"].astype(int).to_numpy(),
    data["label"].to_numpy(),
):
    raise ValueError(
        "Qwen labels do not match the official test set."
    )

systems["qwen3_8b_direct"] = normalize_probabilities(
    qwen[
        ["direct_p_A", "direct_p_X", "direct_p_U"]
    ].to_numpy(dtype=np.float64)
)

systems["qwen3_8b_hierarchical"] = normalize_probabilities(
    qwen[
        [
            "hierarchical_p_A",
            "hierarchical_p_X",
            "hierarchical_p_U",
        ]
    ].to_numpy(dtype=np.float64)
)

for system_name, probability in systems.items():
    if probability.shape != (5316, 3):
        raise ValueError(
            f"{system_name}: invalid shape {probability.shape}"
        )

print("\nLoaded systems:")
for system_name in systems:
    print(" -", system_name)

# ------------------------------------------------------------
# 7. Metric functions
# ------------------------------------------------------------
y_true = data["label"].to_numpy()


def safe_auc(target, score):
    target = np.asarray(target)
    score = np.asarray(score)

    if len(np.unique(target)) < 2:
        return float("nan")

    return float(roc_auc_score(target, score))


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(values.mean())


def query_ranking_metrics(labels, probability):
    cited_score = (
        probability[:, 0]
        + probability[:, 1]
    )

    x_score = (
        probability[:, 1]
        / (
            probability[:, 0]
            + probability[:, 1]
            + 1e-12
        )
    )

    cited_mask = labels != 2
    u_mask = labels == 2
    x_mask = labels == 1
    a_mask = labels == 0

    cited_over_u = np.nan
    x_over_a = np.nan

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

        cited_over_u = float(
            (wins + 0.5 * ties).mean()
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

        x_over_a = float(
            (wins + 0.5 * ties).mean()
        )

    return {
        "pairwise_cited_over_U": cited_over_u,
        "pairwise_X_over_A": x_over_a,
    }


def build_query_metric_arrays(probability):
    arrays = {
        "pairwise_cited_over_U": np.full(
            len(query_groups),
            np.nan,
            dtype=np.float64,
        ),
        "pairwise_X_over_A": np.full(
            len(query_groups),
            np.nan,
            dtype=np.float64,
        ),
    }

    for query_index, rows in enumerate(query_groups):
        result = query_ranking_metrics(
            y_true[rows],
            probability[rows],
        )

        for metric in arrays:
            arrays[metric][query_index] = result[metric]

    return arrays


query_metric_arrays = {
    system_name: build_query_metric_arrays(probability)
    for system_name, probability in systems.items()
}


def calculate_metrics(
    probability,
    query_metric_array,
    sampled_query_indices,
):
    sampled_query_indices = np.asarray(
        sampled_query_indices,
        dtype=np.int64,
    )

    sampled_rows = np.concatenate([
        query_groups[index]
        for index in sampled_query_indices
    ])

    labels = y_true[sampled_rows]
    sampled_probability = probability[sampled_rows]
    predictions = sampled_probability.argmax(axis=1)

    cited_target = (labels != 2).astype(int)
    cited_score = (
        sampled_probability[:, 0]
        + sampled_probability[:, 1]
    )

    xa_mask = labels != 2
    xa_target = (
        labels[xa_mask] == 1
    ).astype(int)

    x_score = (
        sampled_probability[xa_mask, 1]
        / (
            sampled_probability[xa_mask, 0]
            + sampled_probability[xa_mask, 1]
            + 1e-12
        )
    )

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "cited_vs_u_roc_auc": safe_auc(
            cited_target,
            cited_score,
        ),
        "x_vs_a_roc_auc": safe_auc(
            xa_target,
            x_score,
        ),
        "pairwise_cited_over_U": finite_mean(
            query_metric_array[
                "pairwise_cited_over_U"
            ][sampled_query_indices]
        ),
        "pairwise_X_over_A": finite_mean(
            query_metric_array[
                "pairwise_X_over_A"
            ][sampled_query_indices]
        ),
    }

# ------------------------------------------------------------
# 8. Point estimates and bootstrap distributions
# ------------------------------------------------------------
rng = np.random.default_rng(BOOTSTRAP_SEED)
alpha = 1.0 - CI_LEVEL

point_results = {}
bootstrap_values = {}

start_time = time.time()

for system_number, (
    system_name,
    probability,
) in enumerate(systems.items(), start=1):

    print(
        f"\nSystem {system_number}/{len(systems)}:",
        system_name,
    )

    point_results[system_name] = {}
    bootstrap_values[system_name] = {}

    for bin_name in BIN_ORDER:
        query_indices = bin_query_indices[bin_name]

        point_results[system_name][bin_name] = (
            calculate_metrics(
                probability,
                query_metric_arrays[system_name],
                query_indices,
            )
        )

        bootstrap_values[system_name][bin_name] = {
            metric: np.full(
                N_BOOTSTRAP,
                np.nan,
                dtype=np.float64,
            )
            for metric in METRICS
        }

        for bootstrap_index in range(N_BOOTSTRAP):
            sampled_queries = rng.choice(
                query_indices,
                size=len(query_indices),
                replace=True,
            )

            result = calculate_metrics(
                probability,
                query_metric_arrays[system_name],
                sampled_queries,
            )

            for metric in METRICS:
                bootstrap_values[
                    system_name
                ][bin_name][metric][bootstrap_index] = (
                    result[metric]
                )

        print(
            f"  {bin_name}: "
            f"{len(query_indices)} queries complete"
        )

    elapsed = time.time() - start_time
    print(f"  elapsed: {elapsed / 60:.1f} min")

# ------------------------------------------------------------
# 9. Confidence intervals
# ------------------------------------------------------------
def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [None, None]

    return [
        float(
            np.percentile(
                values,
                100.0 * alpha / 2.0,
            )
        ),
        float(
            np.percentile(
                values,
                100.0 * (1.0 - alpha / 2.0),
            )
        ),
    ]


confidence_intervals = {}

for system_name in systems:
    confidence_intervals[system_name] = {}

    for bin_name in BIN_ORDER:
        confidence_intervals[system_name][bin_name] = {}

        for metric in METRICS:
            lower, upper = percentile_interval(
                bootstrap_values[
                    system_name
                ][bin_name][metric]
            )

            confidence_intervals[
                system_name
            ][bin_name][metric] = {
                "estimate": point_results[
                    system_name
                ][bin_name][metric],
                "ci_lower": lower,
                "ci_upper": upper,
            }

# ------------------------------------------------------------
# 10. Early versus late bootstrap tests
# ------------------------------------------------------------
def empirical_two_sided_p(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    nonpositive = (
        np.sum(values <= 0) + 1
    ) / (len(values) + 1)

    nonnegative = (
        np.sum(values >= 0) + 1
    ) / (len(values) + 1)

    return float(
        min(1.0, 2.0 * min(nonpositive, nonnegative))
    )


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


early_late_tests = {}

for system_name in systems:
    early_late_tests[system_name] = {}
    raw_p_values = {}

    for metric in METRICS:
        delta_distribution = (
            bootstrap_values[
                system_name
            ]["late"][metric]
            - bootstrap_values[
                system_name
            ]["early"][metric]
        )

        lower, upper = percentile_interval(
            delta_distribution
        )

        point_delta = (
            point_results[system_name]["late"][metric]
            - point_results[system_name]["early"][metric]
        )

        p_value = empirical_two_sided_p(
            delta_distribution
        )

        early_late_tests[system_name][metric] = {
            "contrast": "late minus early",
            "difference": point_delta,
            "ci_lower": lower,
            "ci_upper": upper,
            "raw_p_value": p_value,
        }

        raw_p_values[metric] = p_value

    adjusted = holm_adjust(raw_p_values)

    for metric in METRICS:
        corrected = adjusted[metric]

        early_late_tests[
            system_name
        ][metric]["holm_p_value"] = corrected

        early_late_tests[
            system_name
        ][metric]["significant_after_holm_0.05"] = bool(
            np.isfinite(corrected)
            and corrected < 0.05
        )

# ------------------------------------------------------------
# 11. Save output files
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
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": REPO_ID,
    "revision": REVISION,
    "split": "test",
    "date_column": date_column,
    "train_end_date": str(
        TRAIN_END_DATE.date()
    ),
    "temporal_binning": (
        "query-level chronological tertiles based on "
        "average percentile ranks; identical dates remain "
        "in the same bin"
    ),
    "systems": list(systems.keys()),
    "supervised_prediction": (
        "mean class probabilities across seeds 13, 42, and 77"
    ),
    "bootstrap_unit": (
        "patent application and claim identifier"
    ),
    "bootstrap_repetitions": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CI_LEVEL,
    "multiple_testing": (
        "Holm correction across six temporal metrics "
        "within each model and prediction method"
    ),
    "bin_summary": bin_summary,
    "confidence_intervals": confidence_intervals,
    "early_late_tests": early_late_tests,
}

final_result = clean_value(final_result)

json_path = (
    OUTPUT_DIR / "temporal_robustness_results.json"
)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        final_result,
        file,
        ensure_ascii=False,
        indent=2,
    )

# Bin-level result CSV
bin_rows = []

for system_name in systems:
    for bin_name in BIN_ORDER:
        for metric in METRICS:
            result = confidence_intervals[
                system_name
            ][bin_name][metric]

            bin_rows.append({
                "system": system_name,
                "temporal_bin": bin_name,
                "metric": metric,
                **result,
            })

bin_results_path = (
    OUTPUT_DIR / "temporal_bin_results.csv"
)

pd.DataFrame(bin_rows).to_csv(
    bin_results_path,
    index=False,
)

# Early versus late test CSV
test_rows = []

for system_name, metric_results in (
    early_late_tests.items()
):
    for metric, result in metric_results.items():
        test_rows.append({
            "system": system_name,
            "metric": metric,
            **result,
        })

test_results_path = (
    OUTPUT_DIR / "temporal_early_late_tests.csv"
)

pd.DataFrame(test_rows).to_csv(
    test_results_path,
    index=False,
)

# Save only hashed query identifiers
assignment_path = (
    OUTPUT_DIR / "temporal_query_assignments.csv"
)

query_table[
    [
        "query_hash",
        "target_date",
        "days_since_train_end",
        "temporal_bin",
        "rows",
    ]
].to_csv(
    assignment_path,
    index=False,
)

bin_summary_path = (
    OUTPUT_DIR / "temporal_bin_summary.csv"
)

bin_summary_table.to_csv(
    bin_summary_path,
    index=False,
)

# ------------------------------------------------------------
# 12. Print concise result
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("TEMPORAL ROBUSTNESS RESULT")
print("=" * 78)

print("\nTEMPORAL BINS")
print(bin_summary_table.to_string(index=False))

important_print_metrics = [
    "macro_f1",
    "cited_vs_u_roc_auc",
    "x_vs_a_roc_auc",
    "pairwise_cited_over_U",
]

for system_name in systems:
    print(f"\n{system_name.upper()}")

    for metric in important_print_metrics:
        values = []

        for bin_name in BIN_ORDER:
            result = confidence_intervals[
                system_name
            ][bin_name][metric]

            values.append(
                f"{bin_name}="
                f"{result['estimate']:.4f} "
                f"[{result['ci_lower']:.4f},"
                f"{result['ci_upper']:.4f}]"
            )

        test_result = early_late_tests[
            system_name
        ][metric]

        print(
            f"{metric}: "
            + " | ".join(values)
            + f" | late-early="
            f"{test_result['difference']:.4f} "
            f"[{test_result['ci_lower']:.4f},"
            f"{test_result['ci_upper']:.4f}] "
            f"Holm-p="
            f"{test_result['holm_p_value']:.4f}"
        )

print("\nSaved:")
print(json_path)
print(bin_results_path)
print(test_results_path)
print(assignment_path)
print(bin_summary_path)
print("=" * 78)
