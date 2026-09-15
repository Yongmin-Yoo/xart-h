"""Reproducibility script exported from the executed XART-H notebook."""

# ============================================================
# Correct Holm scope: adjust within each paired comparison
# No bootstrap rerun required
# ============================================================

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/supervised_v1/bootstrap"
)

INPUT_PATH = ROOT / "supervised_bootstrap_results.json"
OUTPUT_PATH = ROOT / "supervised_bootstrap_family_holm.json"
CSV_PATH = ROOT / "supervised_paired_tests_family_holm.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)

paired = results["paired_comparisons"]

def holm_within_family(metric_results):
    valid = []

    for metric, values in metric_results.items():
        p_value = values.get("p_value")

        if p_value is not None and np.isfinite(p_value):
            valid.append((metric, float(p_value)))

    valid.sort(key=lambda item: item[1])

    number_of_tests = len(valid)
    adjusted = {}
    running_maximum = 0.0

    for rank, (metric, p_value) in enumerate(
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

        adjusted[metric] = running_maximum

    return adjusted

rows = []

for comparison, metric_results in paired.items():
    adjusted = holm_within_family(metric_results)

    for metric, values in metric_results.items():
        corrected = adjusted.get(metric)

        values["family_holm_p_value"] = corrected
        values["family_significant_0.05"] = bool(
            corrected is not None
            and corrected < 0.05
        )

        rows.append({
            "comparison": comparison,
            "metric": metric,
            "difference": values.get("difference"),
            "ci_lower": values.get("ci_lower"),
            "ci_upper": values.get("ci_upper"),
            "raw_p_value": values.get("p_value"),
            "global_holm_p_value": values.get(
                "holm_p_value"
            ),
            "family_holm_p_value": corrected,
            "family_significant_0.05": values[
                "family_significant_0.05"
            ],
        })

results["holm_scope"] = (
    "Holm correction applied separately within each "
    "paired comparison across its reported metrics"
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=2,
    )

table = pd.DataFrame(rows)
table.to_csv(CSV_PATH, index=False)

important_metrics = [
    "accuracy",
    "macro_f1",
    "cited_vs_u_roc_auc",
    "x_vs_a_roc_auc",
    "pairwise_cited_over_U",
    "pairwise_X_over_A",
    "ndcg",
]

print("=" * 72)
print("FAMILY-WISE HOLM RESULTS")
print("=" * 72)

for comparison, metric_results in paired.items():
    print(f"\n{comparison}")

    for metric in important_metrics:
        values = metric_results[metric]

        print(
            f"{metric}: "
            f"delta={values['difference']:.6f}, "
            f"95% CI=[{values['ci_lower']:.6f}, "
            f"{values['ci_upper']:.6f}], "
            f"raw-p={values['p_value']:.6f}, "
            f"family-Holm-p="
            f"{values['family_holm_p_value']:.6f}"
        )

print("\nSaved:")
print(OUTPUT_PATH)
print(CSV_PATH)
