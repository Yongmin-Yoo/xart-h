"""Validation-defined lexical-matching diagnostic.

This script reproduces the Stage 2B matching analysis. Matching uses
lexical and length features only; model predictions and error outcomes
are not used to select pairs. The q50 validation-defined caliper is the
primary analysis, with q25 and q75 as sensitivity analyses.

This diagnostic is exploratory and post-hoc. It does not establish that
all candidate-construction differences have been removed.
"""

# =====================================================================
# XART-H CONSTRUCTION-BIAS DIAGNOSTIC: STAGE 2B
# Validation-defined lexical and length matching
# CPU only
# =====================================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

STAGE2_START = time.time()

STAGE2_OUTPUT = (
    EXPERIMENT_ROOT
    / "construction_bias_v1"
    / "lexical_matching"
)
STAGE2_OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 42

# ---------------------------------------------------------------------
# 1. Verify required Stage 1 objects and columns
# ---------------------------------------------------------------------

assert "hard_data" in globals(), (
    "hard_data is not available. "
    "Run the Stage 1 construction-bias cell first."
)

required_splits = {
    "train",
    "validation",
    "test",
}

assert required_splits.issubset(
    hard_data.keys()
)

required_columns = {
    "query_key",
    "label",
    "is_cited",
    "bm25_score",
    "bm25_original",
    "bm25_direction_selected",
    "claim_token_count",
    "passage_token_count",
    "passage_claim_ratio",
    "token_jaccard",
    "claim_coverage",
    "length_lr",
    "lexical_lr",
    "combined_lr",
}

for split in required_splits:
    missing = (
        required_columns
        - set(hard_data[split].columns)
    )

    assert not missing, (
        f"{split}: missing columns {missing}"
    )

for split in ["validation", "test"]:
    assert (
        "deberta_ensemble"
        in hard_data[split].columns
    ), f"{split}: missing DeBERTa ensemble score"

print("=" * 80)
print("STAGE 2B: VALIDATION-DEFINED LEXICAL MATCHING")
print("=" * 80)


# ---------------------------------------------------------------------
# 2. Define matching features
# ---------------------------------------------------------------------

MATCH_FEATURES = [
    "log1p_bm25",
    "token_jaccard",
    "claim_coverage",
    "log1p_passage_length",
    "log1p_passage_claim_ratio",
]

RAW_FEATURE_NAMES = {
    "log1p_bm25": "bm25_score",
    "token_jaccard": "token_jaccard",
    "claim_coverage": "claim_coverage",
    "log1p_passage_length":
        "passage_token_count",
    "log1p_passage_claim_ratio":
        "passage_claim_ratio",
}


def add_matching_features(dataframe):
    result = dataframe.copy()

    result["log1p_bm25"] = np.log1p(
        np.maximum(
            result["bm25_score"]
            .to_numpy(dtype=np.float64),
            0.0,
        )
    )

    result["log1p_passage_length"] = np.log1p(
        np.maximum(
            result["passage_token_count"]
            .to_numpy(dtype=np.float64),
            0.0,
        )
    )

    result[
        "log1p_passage_claim_ratio"
    ] = np.log1p(
        np.maximum(
            result["passage_claim_ratio"]
            .to_numpy(dtype=np.float64),
            0.0,
        )
    )

    return result


matching_data = {
    split: add_matching_features(
        dataframe
    ).reset_index(drop=True)
    for split, dataframe in hard_data.items()
}

# Fit feature standardization using train only.
train_feature_matrix = (
    matching_data["train"][
        MATCH_FEATURES
    ]
    .to_numpy(dtype=np.float64)
)

feature_means = np.nanmean(
    train_feature_matrix,
    axis=0,
)

feature_stds = np.nanstd(
    train_feature_matrix,
    axis=0,
    ddof=0,
)

feature_stds = np.where(
    feature_stds > 1e-12,
    feature_stds,
    1.0,
)

print("\nMatching features and train scaling:")

for feature, mean, std in zip(
    MATCH_FEATURES,
    feature_means,
    feature_stds,
):
    print(
        f"{feature:30s} "
        f"mean={mean:.6f}, std={std:.6f}"
    )


def standardized_feature_matrix(dataframe):
    matrix = dataframe[
        MATCH_FEATURES
    ].to_numpy(dtype=np.float64)

    matrix = np.where(
        np.isfinite(matrix),
        matrix,
        feature_means,
    )

    return (
        matrix - feature_means
    ) / feature_stds


standardized_matrices = {
    split: standardized_feature_matrix(
        dataframe
    )
    for split, dataframe
    in matching_data.items()
}


# ---------------------------------------------------------------------
# 3. Select one feature-nearest cited candidate for each U
# ---------------------------------------------------------------------

def construct_nearest_pairs(
    dataframe,
    standardized_matrix,
):
    records = []

    for query_key, group in dataframe.groupby(
        "query_key",
        sort=True,
    ):
        cited_indices = group.index[
            group["is_cited"] == 1
        ].to_numpy(dtype=np.int64)

        u_indices = group.index[
            group["is_cited"] == 0
        ].to_numpy(dtype=np.int64)

        if (
            len(cited_indices) == 0
            or len(u_indices) == 0
        ):
            continue

        if len(u_indices) != 1:
            raise ValueError(
                f"Query {query_key} has "
                f"{len(u_indices)} U rows."
            )

        u_index = int(u_indices[0])

        absolute_differences = np.abs(
            standardized_matrix[
                cited_indices
            ]
            - standardized_matrix[
                u_index
            ][None, :]
        )

        # L-infinity distance ensures that no single
        # observed feature differs by more than the
        # selected validation-defined caliper.
        candidate_distances = (
            absolute_differences.max(axis=1)
        )

        minimum_distance = (
            candidate_distances.min()
        )

        tied_positions = np.flatnonzero(
            np.isclose(
                candidate_distances,
                minimum_distance,
                rtol=0.0,
                atol=1e-12,
            )
        )

        # Deterministic tie handling based only on
        # original row order, never model predictions.
        selected_position = int(
            tied_positions[0]
        )

        cited_index = int(
            cited_indices[selected_position]
        )

        record = {
            "query_key": query_key,
            "cited_index": cited_index,
            "u_index": u_index,
            "matching_distance":
                float(minimum_distance),
            "number_of_cited_candidates":
                int(len(cited_indices)),
            "matching_tie":
                bool(len(tied_positions) > 1),
        }

        for feature_index, feature in enumerate(
            MATCH_FEATURES
        ):
            signed_difference = (
                standardized_matrix[
                    cited_index,
                    feature_index,
                ]
                - standardized_matrix[
                    u_index,
                    feature_index,
                ]
            )

            record[
                f"z_diff_{feature}"
            ] = float(signed_difference)

            record[
                f"abs_z_diff_{feature}"
            ] = float(
                abs(signed_difference)
            )

        records.append(record)

    return pd.DataFrame(records)


nearest_pairs = {
    split: construct_nearest_pairs(
        matching_data[split],
        standardized_matrices[split],
    )
    for split in [
        "validation",
        "test",
    ]
}

for split, pairs in nearest_pairs.items():
    print(f"\n[{split} nearest-pair pool]")
    print("Queries:", len(pairs))
    print(
        "Distance median:",
        pairs[
            "matching_distance"
        ].median(),
    )
    print(
        "Distance 25/50/75:",
        pairs[
            "matching_distance"
        ].quantile(
            [0.25, 0.50, 0.75]
        ).to_dict(),
    )
    print(
        "Matching ties:",
        int(pairs["matching_tie"].sum()),
    )


# ---------------------------------------------------------------------
# 4. Fix calipers using validation only
# ---------------------------------------------------------------------

validation_distances = nearest_pairs[
    "validation"
]["matching_distance"]

CALIPERS = {
    "q25": float(
        validation_distances.quantile(0.25)
    ),
    "q50_primary": float(
        validation_distances.quantile(0.50)
    ),
    "q75": float(
        validation_distances.quantile(0.75)
    ),
}

print("\nValidation-defined calipers:")

for name, value in CALIPERS.items():
    print(f"{name:12s}: {value:.6f}")

with open(
    STAGE2_OUTPUT
    / "matching_calipers.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "distance": (
                "maximum absolute standardized "
                "feature difference"
            ),
            "standardization_fit": (
                "official training rows only"
            ),
            "caliper_selection": (
                "validation nearest-pair "
                "distance quantiles"
            ),
            "primary_caliper": "q50_primary",
            "features": MATCH_FEATURES,
            "feature_means": {
                feature: float(mean)
                for feature, mean in zip(
                    MATCH_FEATURES,
                    feature_means,
                )
            },
            "feature_stds": {
                feature: float(std)
                for feature, std in zip(
                    MATCH_FEATURES,
                    feature_stds,
                )
            },
            "calipers": CALIPERS,
        },
        file,
        indent=2,
    )


# ---------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------

MODEL_COLUMNS_MATCHING = {
    "BM25":
        "bm25_original",
    "Direction-selected BM25":
        "bm25_direction_selected",
    "Length LR":
        "length_lr",
    "Lexical LR":
        "lexical_lr",
    "Combined LR":
        "combined_lr",
    "DeBERTa ensemble":
        "deberta_ensemble",
}


def extract_pair_scores(
    dataframe,
    pairs,
    score_column,
):
    cited_indices = pairs[
        "cited_index"
    ].to_numpy(dtype=np.int64)

    u_indices = pairs[
        "u_index"
    ].to_numpy(dtype=np.int64)

    all_scores = dataframe[
        score_column
    ].to_numpy(dtype=np.float64)

    cited_scores = all_scores[
        cited_indices
    ]

    u_scores = all_scores[
        u_indices
    ]

    return cited_scores, u_scores


def evaluate_selected_pairs(
    dataframe,
    pairs,
    score_column,
):
    cited_scores, u_scores = (
        extract_pair_scores(
            dataframe,
            pairs,
            score_column,
        )
    )

    labels = np.concatenate([
        np.ones(
            len(cited_scores),
            dtype=np.int64,
        ),
        np.zeros(
            len(u_scores),
            dtype=np.int64,
        ),
    ])

    scores = np.concatenate([
        cited_scores,
        u_scores,
    ])

    auc = roc_auc_score(
        labels,
        scores,
    )

    differences = (
        cited_scores - u_scores
    )

    pairwise_credits = (
        (differences > 0).astype(float)
        + 0.5
        * (differences == 0).astype(float)
    )

    return {
        "roc_auc": float(auc),
        "pairwise_accuracy": float(
            pairwise_credits.mean()
        ),
        "pairwise_credits":
            pairwise_credits,
        "cited_scores": cited_scores,
        "u_scores": u_scores,
    }


# ---------------------------------------------------------------------
# 6. Evaluate all validation-defined calipers
# ---------------------------------------------------------------------

matching_result_rows = []
balance_rows = []
assignment_frames = []

for split in [
    "validation",
    "test",
]:
    dataframe = matching_data[split]
    all_pairs = nearest_pairs[split]

    scopes = {
        "all_nearest": all_pairs,
    }

    for caliper_name, threshold in (
        CALIPERS.items()
    ):
        scopes[caliper_name] = (
            all_pairs[
                all_pairs[
                    "matching_distance"
                ] <= threshold
            ]
            .copy()
            .reset_index(drop=True)
        )

    for scope_name, selected_pairs in (
        scopes.items()
    ):
        retained_queries = len(
            selected_pairs
        )

        total_queries = len(
            all_pairs
        )

        retention_rate = (
            retained_queries
            / total_queries
            if total_queries > 0
            else np.nan
        )

        assignment = (
            selected_pairs.copy()
        )
        assignment["split"] = split
        assignment["scope"] = scope_name
        assignment[
            "retention_rate"
        ] = retention_rate

        assignment_frames.append(
            assignment
        )

        for feature in MATCH_FEATURES:
            signed_values = (
                selected_pairs[
                    f"z_diff_{feature}"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            absolute_values = np.abs(
                signed_values
            )

            balance_rows.append({
                "split": split,
                "scope": scope_name,
                "feature": feature,
                "queries":
                    retained_queries,
                "retention_rate":
                    retention_rate,
                "mean_signed_z_difference":
                    float(
                        np.mean(
                            signed_values
                        )
                    ),
                "mean_absolute_z_difference":
                    float(
                        np.mean(
                            absolute_values
                        )
                    ),
                "median_absolute_z_difference":
                    float(
                        np.median(
                            absolute_values
                        )
                    ),
                "maximum_absolute_z_difference":
                    float(
                        np.max(
                            absolute_values
                        )
                    ),
            })

        for model_name, score_column in (
            MODEL_COLUMNS_MATCHING.items()
        ):
            result = evaluate_selected_pairs(
                dataframe,
                selected_pairs,
                score_column,
            )

            matching_result_rows.append({
                "split": split,
                "scope": scope_name,
                "model": model_name,
                "queries":
                    retained_queries,
                "rows":
                    2 * retained_queries,
                "retention_rate":
                    retention_rate,
                "roc_auc":
                    result["roc_auc"],
                "pairwise_accuracy":
                    result[
                        "pairwise_accuracy"
                    ],
            })

matching_results = pd.DataFrame(
    matching_result_rows
)

balance_results = pd.DataFrame(
    balance_rows
)

matching_assignments = pd.concat(
    assignment_frames,
    ignore_index=True,
)

matching_results.to_csv(
    STAGE2_OUTPUT
    / "lexical_matching_results.csv",
    index=False,
)

balance_results.to_csv(
    STAGE2_OUTPUT
    / "lexical_matching_balance.csv",
    index=False,
)

# Row-level assignments remain local and should
# not be committed to the public repository.
matching_assignments.to_parquet(
    STAGE2_OUTPUT
    / "lexical_matching_assignments.parquet",
    index=False,
)


# ---------------------------------------------------------------------
# 7. Primary q50 paired query bootstrap
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PRIMARY Q50 PAIRED BOOTSTRAP")
print("=" * 80)

test_dataframe = matching_data[
    "test"
]

primary_pairs = (
    nearest_pairs["test"][
        nearest_pairs["test"][
            "matching_distance"
        ] <= CALIPERS[
            "q50_primary"
        ]
    ]
    .copy()
    .reset_index(drop=True)
)

number_of_primary_queries = len(
    primary_pairs
)

assert number_of_primary_queries > 1

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_samples = rng.integers(
    low=0,
    high=number_of_primary_queries,
    size=(
        BOOTSTRAP_REPETITIONS,
        number_of_primary_queries,
    ),
    dtype=np.int32,
)

model_pair_data = {}

for model_name, score_column in (
    MODEL_COLUMNS_MATCHING.items()
):
    model_pair_data[model_name] = (
        evaluate_selected_pairs(
            test_dataframe,
            primary_pairs,
            score_column,
        )
    )

primary_bootstrap_rows = []

for model_name, result in (
    model_pair_data.items()
):
    auc_samples = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    pairwise_samples = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    cited_scores = result[
        "cited_scores"
    ]
    u_scores = result[
        "u_scores"
    ]
    pairwise_credits = result[
        "pairwise_credits"
    ]

    for bootstrap_index in range(
        BOOTSTRAP_REPETITIONS
    ):
        sampled = bootstrap_samples[
            bootstrap_index
        ]

        sampled_cited = cited_scores[
            sampled
        ]
        sampled_u = u_scores[
            sampled
        ]

        labels = np.concatenate([
            np.ones(
                len(sampled),
                dtype=np.int64,
            ),
            np.zeros(
                len(sampled),
                dtype=np.int64,
            ),
        ])

        scores = np.concatenate([
            sampled_cited,
            sampled_u,
        ])

        auc_samples[
            bootstrap_index
        ] = roc_auc_score(
            labels,
            scores,
        )

        pairwise_samples[
            bootstrap_index
        ] = pairwise_credits[
            sampled
        ].mean()

    auc_ci = np.quantile(
        auc_samples,
        [0.025, 0.975],
    )

    pairwise_ci = np.quantile(
        pairwise_samples,
        [0.025, 0.975],
    )

    primary_bootstrap_rows.extend([
        {
            "model": model_name,
            "metric": "roc_auc",
            "estimate":
                result["roc_auc"],
            "ci_lower": auc_ci[0],
            "ci_upper": auc_ci[1],
            "queries":
                number_of_primary_queries,
        },
        {
            "model": model_name,
            "metric":
                "pairwise_accuracy",
            "estimate":
                result[
                    "pairwise_accuracy"
                ],
            "ci_lower":
                pairwise_ci[0],
            "ci_upper":
                pairwise_ci[1],
            "queries":
                number_of_primary_queries,
        },
    ])

primary_bootstrap_df = pd.DataFrame(
    primary_bootstrap_rows
)

primary_bootstrap_df.to_csv(
    STAGE2_OUTPUT
    / "primary_q50_model_intervals.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 8. DeBERTa minus baselines on primary q50 subset
# ---------------------------------------------------------------------

def bootstrap_two_sided_p(
    differences,
):
    return min(
        1.0,
        2.0
        * min(
            (
                np.sum(
                    differences <= 0
                )
                + 1
            )
            / (
                len(differences) + 1
            ),
            (
                np.sum(
                    differences >= 0
                )
                + 1
            )
            / (
                len(differences) + 1
            ),
        ),
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    order = np.argsort(p_values)
    number = len(p_values)
    adjusted = np.empty(number)
    running_maximum = 0.0

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (number - rank)
            * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )
        adjusted[index] = (
            running_maximum
        )

    return adjusted


deberta_data = model_pair_data[
    "DeBERTa ensemble"
]

comparison_rows = []

for baseline_name in [
    "BM25",
    "Direction-selected BM25",
    "Length LR",
    "Lexical LR",
    "Combined LR",
]:
    baseline_data = model_pair_data[
        baseline_name
    ]

    auc_differences = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    pairwise_differences = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        BOOTSTRAP_REPETITIONS
    ):
        sampled = bootstrap_samples[
            bootstrap_index
        ]

        labels = np.concatenate([
            np.ones(
                len(sampled),
                dtype=np.int64,
            ),
            np.zeros(
                len(sampled),
                dtype=np.int64,
            ),
        ])

        deberta_sample_scores = (
            np.concatenate([
                deberta_data[
                    "cited_scores"
                ][sampled],
                deberta_data[
                    "u_scores"
                ][sampled],
            ])
        )

        baseline_sample_scores = (
            np.concatenate([
                baseline_data[
                    "cited_scores"
                ][sampled],
                baseline_data[
                    "u_scores"
                ][sampled],
            ])
        )

        auc_differences[
            bootstrap_index
        ] = (
            roc_auc_score(
                labels,
                deberta_sample_scores,
            )
            - roc_auc_score(
                labels,
                baseline_sample_scores,
            )
        )

        pairwise_differences[
            bootstrap_index
        ] = np.mean(
            deberta_data[
                "pairwise_credits"
            ][sampled]
            - baseline_data[
                "pairwise_credits"
            ][sampled]
        )

    observed_auc_difference = (
        deberta_data["roc_auc"]
        - baseline_data["roc_auc"]
    )

    observed_pairwise_difference = (
        deberta_data[
            "pairwise_accuracy"
        ]
        - baseline_data[
            "pairwise_accuracy"
        ]
    )

    auc_ci = np.quantile(
        auc_differences,
        [0.025, 0.975],
    )

    pairwise_ci = np.quantile(
        pairwise_differences,
        [0.025, 0.975],
    )

    comparison_rows.extend([
        {
            "comparison":
                f"DeBERTa ensemble - {baseline_name}",
            "metric": "roc_auc",
            "difference":
                observed_auc_difference,
            "ci_lower": auc_ci[0],
            "ci_upper": auc_ci[1],
            "raw_p":
                bootstrap_two_sided_p(
                    auc_differences
                ),
            "queries":
                number_of_primary_queries,
        },
        {
            "comparison":
                f"DeBERTa ensemble - {baseline_name}",
            "metric":
                "pairwise_accuracy",
            "difference":
                observed_pairwise_difference,
            "ci_lower":
                pairwise_ci[0],
            "ci_upper":
                pairwise_ci[1],
            "raw_p":
                bootstrap_two_sided_p(
                    pairwise_differences
                ),
            "queries":
                number_of_primary_queries,
        },
    ])

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df["holm_p"] = (
    holm_adjust(
        comparison_df[
            "raw_p"
        ].to_numpy()
    )
)

comparison_df.to_csv(
    STAGE2_OUTPUT
    / "primary_q50_deberta_comparisons.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 9. Save configuration
# ---------------------------------------------------------------------

stage2_configuration = {
    "analysis":
        "validation-defined lexical matching",
    "status":
        "post-hoc diagnostic",
    "matching_unit":
        "one cited candidate and one existing Hard U per normalized query",
    "candidate_selection":
        "feature-nearest cited candidate to each Hard U",
    "distance":
        "maximum absolute standardized feature difference",
    "standardization_fit":
        "official train split only",
    "features": MATCH_FEATURES,
    "caliper_selection":
        "validation nearest-pair distance quantiles",
    "primary_caliper":
        "validation q50",
    "sensitivity_calipers": [
        "validation q25",
        "validation q75",
    ],
    "model_predictions_used_for_matching":
        False,
    "bootstrap": {
        "unit":
            "normalized query",
        "repetitions":
            BOOTSTRAP_REPETITIONS,
        "seed":
            BOOTSTRAP_SEED,
        "confidence_level":
            0.95,
        "holm_family":
            "five baseline comparisons across AUC and pairwise accuracy",
    },
}

with open(
    STAGE2_OUTPUT
    / "lexical_matching_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        stage2_configuration,
        file,
        indent=2,
    )


# ---------------------------------------------------------------------
# 10. Final output
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("LEXICAL MATCHING RESULTS")
print("=" * 80)

test_results = matching_results[
    matching_results["split"]
    == "test"
].copy()

for scope in [
    "all_nearest",
    "q25",
    "q50_primary",
    "q75",
]:
    scope_rows = test_results[
        test_results["scope"]
        == scope
    ]

    if len(scope_rows) == 0:
        continue

    print(f"\n[{scope}]")
    print(
        scope_rows[
            [
                "model",
                "queries",
                "retention_rate",
                "roc_auc",
                "pairwise_accuracy",
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 80)
print("PRIMARY Q50 CONFIDENCE INTERVALS")
print("=" * 80)

print(
    primary_bootstrap_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("PRIMARY Q50 DEBERTA COMPARISONS")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("PRIMARY Q50 FEATURE BALANCE")
print("=" * 80)

primary_balance = balance_results[
    (
        balance_results["split"]
        == "test"
    )
    & (
        balance_results["scope"]
        == "q50_primary"
    )
]

print(
    primary_balance[
        [
            "feature",
            "queries",
            "retention_rate",
            "mean_signed_z_difference",
            "mean_absolute_z_difference",
            "maximum_absolute_z_difference",
        ]
    ].to_string(index=False)
)

elapsed_minutes = (
    time.time() - STAGE2_START
) / 60

print(
    f"\nElapsed time: "
    f"{elapsed_minutes:.1f} minutes"
)

print("\nSaved files:")

for path in sorted(
    STAGE2_OUTPUT.iterdir()
):
    print(" -", path)

print("\nSTAGE 2B LEXICAL MATCHING COMPLETE")
