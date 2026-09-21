"""Post-hoc Hard-U construction-bias diagnostic.

This script reproduces the Stage 1 surface-feature baseline analysis.
It evaluates BM25, validation-selected score direction, length and
lexical logistic-regression baselines, and existing DeBERTa predictions.

The diagnostic is exploratory and post-hoc. It does not modify the
frozen XART-H dataset or overwrite official benchmark results.
"""

# =====================================================================
# XART-H CONSTRUCTION-BIAS DIAGNOSTIC: STAGE 1
# CPU only
# =====================================================================

import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

START_TIME = time.time()
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 42
SEED = 42

OUTPUT_ROOT = (
    EXPERIMENT_ROOT
    / "construction_bias_v1"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

HARD_FINAL_ROOT = (
    MYDRIVE
    / "PatentSearchBench"
    / "external"
    / "PatentMatch"
    / "xart_hard_u_final"
)

HARD_FILES = {
    "train": HARD_FINAL_ROOT / "train.csv",
    "validation": HARD_FINAL_ROOT / "dev.csv",
    "test": HARD_FINAL_ROOT / "test.csv",
}

RANDOM_ROOT = (
    EXPERIMENT_ROOT
    / "random_u_v1"
)

MATCHED_RANDOM_PATH = (
    RANDOM_ROOT
    / "hard_vs_random"
    / "random_vs_hard_scored_pairs.parquet"
)

BM25_K1 = 1.5
BM25_B = 0.75
BM25_MAX_FEATURES = 200_000

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


# =====================================================================
# 1. Helpers
# =====================================================================

def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def tokenize_text(value):
    return TOKEN_PATTERN.findall(
        normalize_text(value)
    )


def stable_softmax(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )
    logits = (
        logits
        - logits.max(axis=1, keepdims=True)
    )

    values = np.exp(logits)

    return (
        values
        / values.sum(axis=1, keepdims=True)
    )


def evaluate_auc(dataframe, scores):
    labels = (
        dataframe["is_cited"]
        .to_numpy(dtype=np.int64)
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    return float(
        roc_auc_score(labels, scores)
    )


def query_pairwise_credits(
    dataframe,
    scores,
):
    work = dataframe[
        ["query_key", "is_cited"]
    ].copy()

    work["score"] = np.asarray(
        scores,
        dtype=np.float64,
    )

    records = []

    for query_key, group in work.groupby(
        "query_key",
        sort=True,
    ):
        cited_scores = group.loc[
            group["is_cited"] == 1,
            "score",
        ].to_numpy(dtype=np.float64)

        u_scores = group.loc[
            group["is_cited"] == 0,
            "score",
        ].to_numpy(dtype=np.float64)

        if len(cited_scores) == 0 or len(u_scores) == 0:
            continue

        comparisons = (
            cited_scores[:, None]
            - u_scores[None, :]
        )

        credit = np.mean(
            (comparisons > 0).astype(float)
            + 0.5
            * (comparisons == 0).astype(float)
        )

        records.append({
            "query_key": query_key,
            "pairwise_credit": float(credit),
            "cited_rows": int(len(cited_scores)),
            "u_rows": int(len(u_scores)),
        })

    result = pd.DataFrame(records)

    return result


def pairwise_accuracy(dataframe, scores):
    credits = query_pairwise_credits(
        dataframe,
        scores,
    )

    if len(credits) == 0:
        return np.nan

    return float(
        credits["pairwise_credit"].mean()
    )


def select_u_eligible(dataframe):
    u_queries = set(
        dataframe.loc[
            dataframe["is_cited"] == 0,
            "query_key",
        ]
    )

    return (
        dataframe[
            dataframe["query_key"].isin(u_queries)
        ]
        .copy()
        .reset_index(drop=True)
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    number = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(number, dtype=float)

    running_maximum = 0.0

    for rank, index in enumerate(order):
        multiplier = number - rank

        candidate = min(
            1.0,
            multiplier * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[index] = running_maximum

    return adjusted


# =====================================================================
# 2. Load official Hard-U splits
# =====================================================================

print("=" * 80)
print("LOADING OFFICIAL HARD-U DATA")
print("=" * 80)

hard_data = {}

for split, path in HARD_FILES.items():
    assert path.exists(), path

    dataframe = pd.read_csv(path)
    dataframe = dataframe.reset_index(drop=True)

    dataframe["label"] = (
        dataframe["label"].astype(int)
    )

    dataframe["query_key"] = (
        dataframe["text"].map(normalize_text)
    )

    dataframe["is_cited"] = (
        dataframe["label"].isin([0, 1])
    ).astype(int)

    hard_data[split] = dataframe

    print(
        f"{split:12s} | "
        f"rows={len(dataframe):,} | "
        f"queries={dataframe['query_key'].nunique():,} | "
        f"cited={(dataframe['is_cited'] == 1).sum():,} | "
        f"U={(dataframe['is_cited'] == 0).sum():,}"
    )


# =====================================================================
# 3. Fit the official BM25 representation on training passages
# =====================================================================

print("\n" + "=" * 80)
print("FITTING OFFICIAL BM25 REPRESENTATION")
print("=" * 80)

bm25_vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    max_features=BM25_MAX_FEATURES,
    dtype=np.float32,
)

training_passage_matrix = (
    bm25_vectorizer.fit_transform(
        hard_data["train"]["text_b"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    .tocsr()
)

number_of_documents = (
    training_passage_matrix.shape[0]
)

document_frequency = np.asarray(
    (
        training_passage_matrix > 0
    ).sum(axis=0)
).ravel()

inverse_document_frequency = np.log(
    1.0
    + (
        number_of_documents
        - document_frequency
        + 0.5
    )
    / (
        document_frequency
        + 0.5
    )
)

training_lengths = np.asarray(
    training_passage_matrix.sum(axis=1)
).ravel()

average_document_length = max(
    float(training_lengths.mean()),
    1e-12,
)

print(
    "Training passages:",
    number_of_documents,
)
print(
    "BM25 vocabulary:",
    len(bm25_vectorizer.vocabulary_),
)
print(
    "Average document length:",
    average_document_length,
)


def calculate_bm25_scores(dataframe):
    query_counts = (
        bm25_vectorizer.transform(
            dataframe["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        .tocsr()
    )

    passage_counts = (
        bm25_vectorizer.transform(
            dataframe["text_b"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        .tocsr()
    )

    query_binary = query_counts.copy()
    query_binary.data = np.ones_like(
        query_binary.data,
        dtype=np.float32,
    )

    passage_lengths = np.asarray(
        passage_counts.sum(axis=1)
    ).ravel()

    weighted_passages = (
        passage_counts
        .copy()
        .astype(np.float64)
    )

    row_indices = np.repeat(
        np.arange(
            weighted_passages.shape[0]
        ),
        np.diff(weighted_passages.indptr),
    )

    term_indices = weighted_passages.indices
    term_frequencies = weighted_passages.data.copy()

    length_normalizers = (
        BM25_K1
        * (
            1.0
            - BM25_B
            + BM25_B
            * passage_lengths
            / average_document_length
        )
    )

    weighted_passages.data = (
        inverse_document_frequency[
            term_indices
        ]
        * (
            term_frequencies
            * (BM25_K1 + 1.0)
        )
        / (
            term_frequencies
            + length_normalizers[
                row_indices
            ]
        )
    )

    scores = np.asarray(
        query_binary.multiply(
            weighted_passages
        ).sum(axis=1)
    ).ravel()

    return scores.astype(np.float64)


for split, dataframe in hard_data.items():
    print("Scoring BM25:", split)

    dataframe["bm25_score"] = (
        calculate_bm25_scores(dataframe)
    )


# =====================================================================
# 4. Surface-feature extraction
# =====================================================================

print("\n" + "=" * 80)
print("EXTRACTING LENGTH AND LEXICAL FEATURES")
print("=" * 80)


def extract_surface_features(dataframe):
    records = []

    for claim, passage in zip(
        dataframe["text"],
        dataframe["text_b"],
    ):
        claim_tokens = tokenize_text(claim)
        passage_tokens = tokenize_text(passage)

        claim_set = set(claim_tokens)
        passage_set = set(passage_tokens)

        intersection_size = len(
            claim_set & passage_set
        )
        union_size = len(
            claim_set | passage_set
        )

        claim_length = len(claim_tokens)
        passage_length = len(passage_tokens)

        jaccard = (
            intersection_size / union_size
            if union_size > 0
            else 0.0
        )

        claim_coverage = (
            intersection_size
            / len(claim_set)
            if len(claim_set) > 0
            else 0.0
        )

        passage_claim_ratio = (
            passage_length
            / max(claim_length, 1)
        )

        records.append({
            "claim_token_count": claim_length,
            "passage_token_count": passage_length,
            "passage_claim_ratio": passage_claim_ratio,
            "token_jaccard": jaccard,
            "claim_coverage": claim_coverage,
        })

    return pd.DataFrame(records)


for split, dataframe in hard_data.items():
    print("Extracting features:", split)

    surface = extract_surface_features(
        dataframe
    )

    for column in surface.columns:
        dataframe[column] = surface[
            column
        ].to_numpy()


# =====================================================================
# 5. Verify BM25 and select its direction using validation only
# =====================================================================

print("\n" + "=" * 80)
print("BM25 DIRECTION SELECTION")
print("=" * 80)

validation_bm25_auc = evaluate_auc(
    hard_data["validation"],
    hard_data["validation"]["bm25_score"],
)

validation_inverse_auc = evaluate_auc(
    hard_data["validation"],
    -hard_data["validation"]["bm25_score"],
)

if validation_bm25_auc >= validation_inverse_auc:
    bm25_direction = 1.0
    bm25_direction_name = "original"
else:
    bm25_direction = -1.0
    bm25_direction_name = "reversed"

print(
    "Validation original BM25 AUC:",
    f"{validation_bm25_auc:.6f}",
)
print(
    "Validation reversed BM25 AUC:",
    f"{validation_inverse_auc:.6f}",
)
print(
    "Selected direction:",
    bm25_direction_name,
)

for split, dataframe in hard_data.items():
    dataframe["bm25_original"] = (
        dataframe["bm25_score"]
    )

    dataframe["bm25_direction_selected"] = (
        bm25_direction
        * dataframe["bm25_score"]
    )


# =====================================================================
# 6. Train logistic-regression surface baselines
# =====================================================================

FEATURE_GROUPS = {
    "length_lr": [
        "claim_token_count",
        "passage_token_count",
        "passage_claim_ratio",
    ],
    "lexical_lr": [
        "bm25_score",
        "token_jaccard",
        "claim_coverage",
    ],
    "combined_lr": [
        "claim_token_count",
        "passage_token_count",
        "passage_claim_ratio",
        "bm25_score",
        "token_jaccard",
        "claim_coverage",
    ],
}

CANDIDATE_C = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

trained_models = {}
selected_hyperparameters = []

print("\n" + "=" * 80)
print("TRAINING LOGISTIC-REGRESSION BASELINES")
print("=" * 80)

for model_name, features in FEATURE_GROUPS.items():
    best_auc = -np.inf
    best_c = None
    best_pipeline = None

    x_train = hard_data["train"][
        features
    ].to_numpy(dtype=np.float64)

    y_train = hard_data["train"][
        "is_cited"
    ].to_numpy(dtype=np.int64)

    x_validation = hard_data[
        "validation"
    ][features].to_numpy(dtype=np.float64)

    y_validation = hard_data[
        "validation"
    ]["is_cited"].to_numpy(dtype=np.int64)

    for c_value in CANDIDATE_C:
        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ])

        pipeline.fit(
            x_train,
            y_train,
        )

        validation_scores = (
            pipeline.predict_proba(
                x_validation
            )[:, 1]
        )

        validation_auc = roc_auc_score(
            y_validation,
            validation_scores,
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            best_c = c_value
            best_pipeline = pipeline

    trained_models[model_name] = {
        "pipeline": best_pipeline,
        "features": features,
        "selected_c": best_c,
        "validation_auc": best_auc,
    }

    selected_hyperparameters.append({
        "model": model_name,
        "features": "|".join(features),
        "selected_c": best_c,
        "validation_auc": best_auc,
    })

    print(
        f"{model_name:15s} | "
        f"C={best_c:<7g} | "
        f"Validation AUC={best_auc:.6f}"
    )

    for split, dataframe in hard_data.items():
        matrix = dataframe[
            features
        ].to_numpy(dtype=np.float64)

        dataframe[model_name] = (
            best_pipeline.predict_proba(
                matrix
            )[:, 1]
        )

pd.DataFrame(
    selected_hyperparameters
).to_csv(
    OUTPUT_ROOT
    / "selected_lr_hyperparameters.csv",
    index=False,
)


# =====================================================================
# 7. Load DeBERTa seed predictions
# =====================================================================

print("\n" + "=" * 80)
print("LOADING DEBERTA PREDICTIONS")
print("=" * 80)

deberta_seed_metrics = []

for split in [
    "validation",
    "test",
]:
    dataframe = hard_data[split]
    y = dataframe[
        "is_cited"
    ].to_numpy(dtype=np.int64)

    seed_scores = []

    for seed in [13, 42, 77]:
        path = (
            SUPERVISED_ROOT
            / (
                f"deberta_cited_vs_u_seed"
                f"{seed}_{split}.npz"
            )
        )

        assert path.exists(), path

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            logits = data[
                "logits"
            ].astype(np.float64)

        assert logits.shape == (
            len(dataframe),
            2,
        )

        probabilities = stable_softmax(
            logits
        )

        # Verified in the previous preflight:
        # column 1 is the cited score.
        scores = probabilities[:, 1]
        seed_scores.append(scores)

        seed_auc = roc_auc_score(
            y,
            scores,
        )

        eligible = select_u_eligible(
            dataframe
        )
        eligible_indices = (
            dataframe["query_key"].isin(
                set(eligible["query_key"])
            )
        ).to_numpy()

        seed_pairwise = pairwise_accuracy(
            dataframe.loc[
                eligible_indices
            ].reset_index(drop=True),
            scores[eligible_indices],
        )

        deberta_seed_metrics.append({
            "split": split,
            "seed": seed,
            "roc_auc": seed_auc,
            "pairwise_accuracy":
                seed_pairwise,
        })

        print(
            f"{split:12s} seed={seed} | "
            f"AUC={seed_auc:.6f} | "
            f"Pairwise={seed_pairwise:.6f}"
        )

    ensemble_score = np.mean(
        np.stack(seed_scores, axis=0),
        axis=0,
    )

    dataframe[
        "deberta_ensemble"
    ] = ensemble_score

deberta_seed_metrics_df = pd.DataFrame(
    deberta_seed_metrics
)

deberta_seed_metrics_df.to_csv(
    OUTPUT_ROOT
    / "deberta_seed_metrics.csv",
    index=False,
)


# =====================================================================
# 8. Evaluate full and U-eligible Hard-U scopes
# =====================================================================

MODEL_COLUMNS = {
    "BM25": "bm25_original",
    "Direction-selected BM25":
        "bm25_direction_selected",
    "Length LR": "length_lr",
    "Lexical LR": "lexical_lr",
    "Combined LR": "combined_lr",
    "DeBERTa ensemble":
        "deberta_ensemble",
}

evaluation_rows = []

print("\n" + "=" * 80)
print("HARD-U BASELINE RESULTS")
print("=" * 80)

for split in [
    "validation",
    "test",
]:
    dataframe = hard_data[split]

    eligible_queries = set(
        dataframe.loc[
            dataframe["is_cited"] == 0,
            "query_key",
        ]
    )

    eligible_mask = (
        dataframe["query_key"]
        .isin(eligible_queries)
        .to_numpy()
    )

    eligible_dataframe = (
        dataframe.loc[
            eligible_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    for model_name, score_column in (
        MODEL_COLUMNS.items()
    ):
        full_scores = dataframe[
            score_column
        ].to_numpy(dtype=np.float64)

        eligible_scores = full_scores[
            eligible_mask
        ]

        full_auc = evaluate_auc(
            dataframe,
            full_scores,
        )

        eligible_auc = evaluate_auc(
            eligible_dataframe,
            eligible_scores,
        )

        pairwise = pairwise_accuracy(
            eligible_dataframe,
            eligible_scores,
        )

        evaluation_rows.append({
            "split": split,
            "scope": "full",
            "model": model_name,
            "rows": len(dataframe),
            "queries":
                dataframe[
                    "query_key"
                ].nunique(),
            "roc_auc": full_auc,
            "pairwise_accuracy": np.nan,
        })

        evaluation_rows.append({
            "split": split,
            "scope": "u_eligible",
            "model": model_name,
            "rows": len(
                eligible_dataframe
            ),
            "queries":
                eligible_dataframe[
                    "query_key"
                ].nunique(),
            "roc_auc": eligible_auc,
            "pairwise_accuracy": pairwise,
        })

        print(
            f"{split:10s} | "
            f"{model_name:24s} | "
            f"Full AUC={full_auc:.6f} | "
            f"Eligible AUC={eligible_auc:.6f} | "
            f"Pairwise={pairwise:.6f}"
        )

evaluation_df = pd.DataFrame(
    evaluation_rows
)

evaluation_df.to_csv(
    OUTPUT_ROOT
    / "hard_u_baseline_results.csv",
    index=False,
)


# =====================================================================
# 9. DeBERTa seed mean versus ensemble
# =====================================================================

seed_summary_rows = []

for split in [
    "validation",
    "test",
]:
    subset = deberta_seed_metrics_df[
        deberta_seed_metrics_df[
            "split"
        ] == split
    ]

    ensemble_row = evaluation_df[
        (
            evaluation_df["split"]
            == split
        )
        & (
            evaluation_df["scope"]
            == "u_eligible"
        )
        & (
            evaluation_df["model"]
            == "DeBERTa ensemble"
        )
    ].iloc[0]

    seed_summary_rows.append({
        "split": split,
        "seed_mean_auc":
            subset["roc_auc"].mean(),
        "seed_sd_auc":
            subset["roc_auc"].std(
                ddof=1
            ),
        "seed_mean_pairwise":
            subset[
                "pairwise_accuracy"
            ].mean(),
        "seed_sd_pairwise":
            subset[
                "pairwise_accuracy"
            ].std(ddof=1),
        "ensemble_u_eligible_auc":
            ensemble_row["roc_auc"],
        "ensemble_pairwise":
            ensemble_row[
                "pairwise_accuracy"
            ],
    })

seed_summary_df = pd.DataFrame(
    seed_summary_rows
)

seed_summary_df.to_csv(
    OUTPUT_ROOT
    / "deberta_seed_mean_vs_ensemble.csv",
    index=False,
)


# =====================================================================
# 10. Paired query-level bootstrap on test U-eligible scope
# =====================================================================

print("\n" + "=" * 80)
print("PAIRED QUERY-LEVEL BOOTSTRAP")
print("=" * 80)

test_df = hard_data["test"]

test_u_queries = sorted(
    set(
        test_df.loc[
            test_df["is_cited"] == 0,
            "query_key",
        ]
    )
)

test_common = (
    test_df[
        test_df["query_key"].isin(
            test_u_queries
        )
    ]
    .copy()
    .reset_index(drop=True)
)

query_to_index = {
    query: index
    for index, query
    in enumerate(test_u_queries)
}

row_query_indices = (
    test_common["query_key"]
    .map(query_to_index)
    .to_numpy(dtype=np.int64)
)

number_of_queries = len(
    test_u_queries
)

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_samples = rng.integers(
    low=0,
    high=number_of_queries,
    size=(
        BOOTSTRAP_REPETITIONS,
        number_of_queries,
    ),
    dtype=np.int32,
)

y_test = test_common[
    "is_cited"
].to_numpy(dtype=np.int64)

deberta_scores = test_common[
    "deberta_ensemble"
].to_numpy(dtype=np.float64)

deberta_pairwise_df = (
    query_pairwise_credits(
        test_common,
        deberta_scores,
    )
    .set_index("query_key")
    .loc[test_u_queries]
)

deberta_query_credits = (
    deberta_pairwise_df[
        "pairwise_credit"
    ].to_numpy(dtype=np.float64)
)

comparison_models = {
    key: value
    for key, value in MODEL_COLUMNS.items()
    if key != "DeBERTa ensemble"
}

bootstrap_results = []

for model_name, score_column in (
    comparison_models.items()
):
    baseline_scores = test_common[
        score_column
    ].to_numpy(dtype=np.float64)

    baseline_pairwise_df = (
        query_pairwise_credits(
            test_common,
            baseline_scores,
        )
        .set_index("query_key")
        .loc[test_u_queries]
    )

    baseline_query_credits = (
        baseline_pairwise_df[
            "pairwise_credit"
        ].to_numpy(dtype=np.float64)
    )

    observed_auc_difference = (
        roc_auc_score(
            y_test,
            deberta_scores,
        )
        - roc_auc_score(
            y_test,
            baseline_scores,
        )
    )

    observed_pairwise_difference = (
        deberta_query_credits.mean()
        - baseline_query_credits.mean()
    )

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
        sampled_queries = (
            bootstrap_samples[
                bootstrap_index
            ]
        )

        query_counts = np.bincount(
            sampled_queries,
            minlength=number_of_queries,
        ).astype(np.float64)

        row_weights = query_counts[
            row_query_indices
        ]

        deberta_auc = roc_auc_score(
            y_test,
            deberta_scores,
            sample_weight=row_weights,
        )

        baseline_auc = roc_auc_score(
            y_test,
            baseline_scores,
            sample_weight=row_weights,
        )

        auc_differences[
            bootstrap_index
        ] = (
            deberta_auc
            - baseline_auc
        )

        pairwise_differences[
            bootstrap_index
        ] = np.average(
            deberta_query_credits
            - baseline_query_credits,
            weights=query_counts,
        )

    auc_ci = np.quantile(
        auc_differences,
        [0.025, 0.975],
    )

    pairwise_ci = np.quantile(
        pairwise_differences,
        [0.025, 0.975],
    )

    auc_p = min(
        1.0,
        2.0
        * min(
            (
                np.sum(
                    auc_differences <= 0
                )
                + 1
            )
            / (
                BOOTSTRAP_REPETITIONS
                + 1
            ),
            (
                np.sum(
                    auc_differences >= 0
                )
                + 1
            )
            / (
                BOOTSTRAP_REPETITIONS
                + 1
            ),
        ),
    )

    pairwise_p = min(
        1.0,
        2.0
        * min(
            (
                np.sum(
                    pairwise_differences <= 0
                )
                + 1
            )
            / (
                BOOTSTRAP_REPETITIONS
                + 1
            ),
            (
                np.sum(
                    pairwise_differences >= 0
                )
                + 1
            )
            / (
                BOOTSTRAP_REPETITIONS
                + 1
            ),
        ),
    )

    bootstrap_results.append({
        "comparison":
            f"DeBERTa ensemble - {model_name}",
        "metric": "roc_auc",
        "difference":
            observed_auc_difference,
        "ci_lower": auc_ci[0],
        "ci_upper": auc_ci[1],
        "raw_p": auc_p,
    })

    bootstrap_results.append({
        "comparison":
            f"DeBERTa ensemble - {model_name}",
        "metric": "pairwise_accuracy",
        "difference":
            observed_pairwise_difference,
        "ci_lower": pairwise_ci[0],
        "ci_upper": pairwise_ci[1],
        "raw_p": pairwise_p,
    })

    print(
        f"{model_name:24s} | "
        f"AUC difference="
        f"{observed_auc_difference:+.6f} "
        f"[{auc_ci[0]:+.6f}, "
        f"{auc_ci[1]:+.6f}] | "
        f"Pairwise difference="
        f"{observed_pairwise_difference:+.6f} "
        f"[{pairwise_ci[0]:+.6f}, "
        f"{pairwise_ci[1]:+.6f}]"
    )

bootstrap_df = pd.DataFrame(
    bootstrap_results
)

bootstrap_df["holm_p"] = (
    holm_adjust(
        bootstrap_df["raw_p"]
        .to_numpy()
    )
)

bootstrap_df.to_csv(
    OUTPUT_ROOT
    / "deberta_vs_surface_bootstrap.csv",
    index=False,
)


# =====================================================================
# 11. Distribution diagnostics
# =====================================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION DIAGNOSTICS")
print("=" * 80)

distribution_features = [
    "bm25_score",
    "token_jaccard",
    "claim_coverage",
    "claim_token_count",
    "passage_token_count",
    "passage_claim_ratio",
]

distribution_rows = []

for split, dataframe in hard_data.items():
    groups = {
        "C": dataframe[
            dataframe["label"].isin([0, 1])
        ],
        "A": dataframe[
            dataframe["label"] == 0
        ],
        "X": dataframe[
            dataframe["label"] == 1
        ],
        "U": dataframe[
            dataframe["label"] == 2
        ],
    }

    for group_name, group in groups.items():
        for feature in distribution_features:
            values = group[
                feature
            ].to_numpy(dtype=np.float64)

            distribution_rows.append({
                "split": split,
                "label_group": group_name,
                "feature": feature,
                "rows": len(group),
                "queries":
                    group[
                        "query_key"
                    ].nunique(),
                "mean": np.mean(values),
                "std": np.std(
                    values,
                    ddof=1,
                ),
                "q25": np.quantile(
                    values,
                    0.25,
                ),
                "median": np.median(values),
                "q75": np.quantile(
                    values,
                    0.75,
                ),
                "minimum": np.min(values),
                "maximum": np.max(values),
            })

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    OUTPUT_ROOT
    / "feature_distributions.csv",
    index=False,
)


# =====================================================================
# 12. Evaluate fixed BM25/LR models on matched Hard/Random-U test set
# =====================================================================

print("\n" + "=" * 80)
print("FIXED BASELINES ON MATCHED HARD-U AND RANDOM-U")
print("=" * 80)

assert MATCHED_RANDOM_PATH.exists(), (
    MATCHED_RANDOM_PATH
)

matched = pd.read_parquet(
    MATCHED_RANDOM_PATH
).reset_index(drop=True)

matched["query_key"] = (
    matched["query_key"]
    .map(normalize_text)
)

matched["is_cited"] = (
    matched["candidate_type"]
    == "cited"
).astype(int)

matched_surface = extract_surface_features(
    matched
)

for column in matched_surface.columns:
    matched[column] = (
        matched_surface[column]
        .to_numpy()
    )

matched[
    "bm25_recomputed"
] = calculate_bm25_scores(
    matched
)

if "score_bm25" in matched.columns:
    maximum_bm25_difference = float(
        np.max(
            np.abs(
                matched[
                    "bm25_recomputed"
                ].to_numpy()
                - matched[
                    "score_bm25"
                ].to_numpy()
            )
        )
    )

    bm25_correlation = float(
        np.corrcoef(
            matched[
                "bm25_recomputed"
            ],
            matched[
                "score_bm25"
            ],
        )[0, 1]
    )

    print(
        "Maximum cached/recomputed BM25 difference:",
        maximum_bm25_difference,
    )
    print(
        "Cached/recomputed BM25 correlation:",
        bm25_correlation,
    )

    # Preserve the exact cached score from the
    # original matched Random-U experiment.
    matched["bm25_score"] = (
        matched["score_bm25"]
        .astype(float)
    )
else:
    matched["bm25_score"] = (
        matched["bm25_recomputed"]
    )

matched["bm25_original"] = (
    matched["bm25_score"]
)

matched[
    "bm25_direction_selected"
] = (
    bm25_direction
    * matched["bm25_score"]
)

for model_name, model_info in (
    trained_models.items()
):
    features = model_info["features"]
    pipeline = model_info["pipeline"]

    matrix = matched[
        features
    ].to_numpy(dtype=np.float64)

    matched[model_name] = (
        pipeline.predict_proba(
            matrix
        )[:, 1]
    )

random_evaluation_rows = []

for condition in [
    "hard_u",
    "random_u",
]:
    condition_df = matched[
        matched["candidate_type"].isin(
            ["cited", condition]
        )
    ].copy().reset_index(drop=True)

    for model_name, score_column in {
        "BM25": "bm25_original",
        "Direction-selected BM25":
            "bm25_direction_selected",
        "Length LR": "length_lr",
        "Lexical LR": "lexical_lr",
        "Combined LR": "combined_lr",
    }.items():
        scores = condition_df[
            score_column
        ].to_numpy(dtype=np.float64)

        auc = evaluate_auc(
            condition_df,
            scores,
        )

        pairwise = pairwise_accuracy(
            condition_df,
            scores,
        )

        random_evaluation_rows.append({
            "condition": condition,
            "model": model_name,
            "rows": len(condition_df),
            "queries":
                condition_df[
                    "query_key"
                ].nunique(),
            "roc_auc": auc,
            "pairwise_accuracy":
                pairwise,
        })

        print(
            f"{condition:10s} | "
            f"{model_name:24s} | "
            f"AUC={auc:.6f} | "
            f"Pairwise={pairwise:.6f}"
        )

random_evaluation_df = pd.DataFrame(
    random_evaluation_rows
)

random_evaluation_df.to_csv(
    OUTPUT_ROOT
    / "fixed_baselines_hard_vs_random.csv",
    index=False,
)


# =====================================================================
# 13. Save scored rows and configuration
# =====================================================================

save_columns = [
    column
    for column in [
        "_row_id",
        "claim_id",
        "patent_application_id",
        "cited_document_id",
        "_component_id",
        "_supercomponent_id",
        "_priority_component_id",
        "_xau_component_id",
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
        "deberta_ensemble",
    ]
    if column in hard_data["test"].columns
]

hard_data["test"][
    save_columns
].to_parquet(
    OUTPUT_ROOT
    / "test_surface_and_deberta_scores.parquet",
    index=False,
)

matched_save_columns = [
    column
    for column in [
        "query_key",
        "candidate_type",
        "gold_label",
        "is_cited",
        "bm25_score",
        "bm25_direction_selected",
        "claim_token_count",
        "passage_token_count",
        "passage_claim_ratio",
        "token_jaccard",
        "claim_coverage",
        "length_lr",
        "lexical_lr",
        "combined_lr",
    ]
    if column in matched.columns
]

matched[
    matched_save_columns
].to_parquet(
    OUTPUT_ROOT
    / "matched_hard_random_surface_scores.parquet",
    index=False,
)

configuration = {
    "dataset": "XART-H v1.0.1",
    "positive_class": "C = labels 0 or 1",
    "negative_class": "U = label 2",
    "bm25": {
        "vectorizer": "CountVectorizer",
        "lowercase": True,
        "ngram_range": [1, 1],
        "min_df": 2,
        "max_features": BM25_MAX_FEATURES,
        "k1": BM25_K1,
        "b": BM25_B,
        "fit_data":
            "official training passages",
        "query_terms": "binary presence",
        "selected_direction":
            bm25_direction_name,
        "validation_original_auc":
            validation_bm25_auc,
        "validation_reversed_auc":
            validation_inverse_auc,
    },
    "tokenization_for_surface_features":
        r"lowercase regex (?u)\b\w\w+\b",
    "logistic_regression": {
        "fit_split": "train",
        "selection_split": "validation",
        "candidate_C": CANDIDATE_C,
        "penalty": "L2",
        "solver": "liblinear",
    },
    "bootstrap": {
        "unit":
            "lowercased whitespace-normalized claim",
        "repetitions":
            BOOTSTRAP_REPETITIONS,
        "seed":
            BOOTSTRAP_SEED,
        "confidence_level":
            0.95,
        "multiple_testing":
            "Holm across reported DeBERTa comparisons",
    },
    "deberta": {
        "score":
            "mean seed probability for class column 1",
        "seeds": [13, 42, 77],
        "note":
            "Seed-mean and ensemble metrics are reported separately",
    },
}

with open(
    OUTPUT_ROOT
    / "construction_bias_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        configuration,
        file,
        indent=2,
    )

elapsed_minutes = (
    time.time() - START_TIME
) / 60

print("\n" + "=" * 80)
print("STAGE 1 CONSTRUCTION-BIAS ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nSelected BM25 direction:",
    bm25_direction_name,
)

print("\nHard-U test results:")
print(
    evaluation_df[
        (
            evaluation_df["split"]
            == "test"
        )
        & (
            evaluation_df["scope"]
            == "u_eligible"
        )
    ][
        [
            "model",
            "roc_auc",
            "pairwise_accuracy",
        ]
    ].to_string(index=False)
)

print("\nDeBERTa comparisons:")
print(
    bootstrap_df.to_string(
        index=False
    )
)

print("\nMatched Hard/Random results:")
print(
    random_evaluation_df.to_string(
        index=False
    )
)

print(
    f"\nElapsed time: "
    f"{elapsed_minutes:.1f} minutes"
)

print("\nSaved files:")

for path in sorted(
    OUTPUT_ROOT.iterdir()
):
    if path.is_file():
        print(" -", path)

print("\nNO MODEL TRAINING OR NEURAL INFERENCE WAS PERFORMED.")
