"""
Reproducibility script recovered from the executed XART-H
Random U experiment notebook.

The XART-H v1.0.1 release is not modified by this script.
"""

# ============================================================
# XART-H: Matched Random U versus Hard U ablation
# Models: TF-IDF, BM25, MiniLM, PatentSBERTa
# Evaluation: similarity, ROC-AUC, pairwise accuracy,
# query-level paired bootstrap, Holm correction
# ============================================================

import gc
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from datasets import load_dataset
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import roc_auc_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "sentence-transformers>=3.4,<6",
    ])
    from sentence_transformers import SentenceTransformer

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as error:
    print("Drive mount skipped:", error)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/random_u_v1"
)

RANDOM_U_PATH = ROOT / "test_random_u.parquet"
GENERATION_REPORT = (
    ROOT / "random_u_generation_report.json"
)

OUTPUT_DIR = ROOT / "hard_vs_random"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

TFIDF_MAX_FEATURES = 200_000
BM25_MAX_FEATURES = 200_000
BM25_K1 = 1.5
BM25_B = 0.75

DENSE_MODELS = {
    "minilm_dense": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 256,
    },
    "patentsberta": {
        "id": "AI-Growth-Lab/PatentSBERTa",
        "batch_size": 128,
    },
}

if not RANDOM_U_PATH.exists():
    raise FileNotFoundError(RANDOM_U_PATH)

if not GENERATION_REPORT.exists():
    raise FileNotFoundError(GENERATION_REPORT)

print("Random U:", RANDOM_U_PATH)
print("Output:", OUTPUT_DIR)

# ------------------------------------------------------------
# 2. Verify Random U generation report
# ------------------------------------------------------------
with open(
    GENERATION_REPORT,
    "r",
    encoding="utf-8",
) as file:
    generation_report = json.load(file)

test_report = generation_report["splits"]["test"]

if test_report["random_u_generated"] != 1642:
    raise ValueError(
        "Unexpected Random U test count: "
        f"{test_report['random_u_generated']}"
    )

if any(
    value != 0
    for value
    in test_report["violations"].values()
):
    raise ValueError(
        "Random U audit violations detected: "
        f"{test_report['violations']}"
    )

if any(
    value != 0
    for value
    in generation_report[
        "cross_split_violations"
    ].values()
):
    raise ValueError(
        "Random U cross-split violation detected."
    )

print("Random U audit: PASSED")

# ------------------------------------------------------------
# 3. Load the frozen benchmark
# ------------------------------------------------------------
dataset = load_dataset(
    REPO_ID,
    revision=REVISION,
)

train = dataset["train"].select_columns([
    "text",
    "text_b",
]).to_pandas()

test = dataset["test"].select_columns([
    "_row_id",
    "text",
    "text_b",
    "label",
    "cited_document_id",
]).to_pandas()

random_u = pd.read_parquet(
    RANDOM_U_PATH
)

test["label"] = test["label"].astype(int)
random_u["label"] = random_u["label"].astype(int)

def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()

test["query_key"] = test["text"].map(
    normalize_text
)

random_u["query_key"] = random_u["text"].map(
    normalize_text
)

hard_u = test[
    test["label"] == 2
].copy()

cited = test[
    test["label"].isin([0, 1])
].copy()

if hard_u["query_key"].duplicated().any():
    raise ValueError(
        "More than one Hard U found for a query."
    )

if random_u["query_key"].duplicated().any():
    raise ValueError(
        "More than one Random U found for a query."
    )

hard_queries = set(hard_u["query_key"])
random_queries = set(random_u["query_key"])
cited_queries = set(cited["query_key"])

common_queries = sorted(
    hard_queries
    & random_queries
    & cited_queries
)

if len(common_queries) != 1642:
    raise ValueError(
        f"Expected 1,642 common queries, "
        f"found {len(common_queries)}"
    )

common_query_set = set(common_queries)

hard_u = hard_u[
    hard_u["query_key"].isin(common_query_set)
].copy()

random_u = random_u[
    random_u["query_key"].isin(common_query_set)
].copy()

cited = cited[
    cited["query_key"].isin(common_query_set)
].copy()

# Use exactly the same claim string for cited, Hard U, and Random U.
canonical_claim = dict(
    zip(
        hard_u["query_key"],
        hard_u["text"].astype(str),
    )
)

hard_u["text"] = hard_u[
    "query_key"
].map(canonical_claim)

random_u["text"] = random_u[
    "query_key"
].map(canonical_claim)

cited["text"] = cited[
    "query_key"
].map(canonical_claim)

hard_u["candidate_type"] = "hard_u"
random_u["candidate_type"] = "random_u"
cited["candidate_type"] = "cited"

# Preserve original X/A category for cited rows.
hard_u["gold_label"] = 2
random_u["gold_label"] = 2
cited["gold_label"] = cited["label"].astype(int)

columns = [
    "query_key",
    "text",
    "text_b",
    "candidate_type",
    "gold_label",
]

pairs = pd.concat(
    [
        cited[columns],
        hard_u[columns],
        random_u[columns],
    ],
    ignore_index=True,
)

pairs["text"] = pairs["text"].fillna("").astype(str)
pairs["text_b"] = pairs["text_b"].fillna("").astype(str)

assert len(hard_u) == 1642
assert len(random_u) == 1642
assert pairs["text_b"].map(
    normalize_text
).ne("").all()

hard_map = dict(
    zip(
        hard_u["query_key"],
        hard_u["text_b"].map(normalize_text),
    )
)

random_map = dict(
    zip(
        random_u["query_key"],
        random_u["text_b"].map(normalize_text),
    )
)

same_random_hard = sum(
    hard_map[query] == random_map[query]
    for query in common_queries
)

assert same_random_hard == 0

print("\nMatched evaluation set")
print("Queries:", len(common_queries))
print("Cited pairs:", len(cited))
print("Hard U pairs:", len(hard_u))
print("Random U pairs:", len(random_u))
print("Total scored pairs:", len(pairs))

# ------------------------------------------------------------
# 4. Dataset fingerprint for score caches
# ------------------------------------------------------------
fingerprint_hasher = hashlib.sha256()

for row in pairs[
    ["query_key", "candidate_type", "text_b"]
].itertuples(index=False):
    fingerprint_hasher.update(
        (
            f"{row.query_key}\t"
            f"{row.candidate_type}\t"
            f"{normalize_text(row.text_b)}\n"
        ).encode("utf-8")
    )

PAIR_FINGERPRINT = fingerprint_hasher.hexdigest()

print("Evaluation fingerprint:", PAIR_FINGERPRINT)

# ------------------------------------------------------------
# 5. Cache helpers
# ------------------------------------------------------------
def score_cache_path(model_name):
    return (
        OUTPUT_DIR
        / (
            f"{model_name}_scores_"
            f"{PAIR_FINGERPRINT[:12]}.npy"
        )
    )

def load_score_cache(model_name):
    path = score_cache_path(model_name)

    if not path.exists():
        return None

    scores = np.load(path)

    if scores.shape != (len(pairs),):
        print(
            "Ignoring invalid score cache:",
            path,
            scores.shape,
        )
        return None

    if not np.isfinite(scores).all():
        print(
            "Ignoring non-finite score cache:",
            path,
        )
        return None

    print("Loaded cache:", path)
    return scores.astype(np.float64)

def save_score_cache(model_name, scores):
    path = score_cache_path(model_name)

    np.save(
        path,
        np.asarray(scores, dtype=np.float32),
    )

    print("Saved cache:", path)

# ------------------------------------------------------------
# 6. TF-IDF score
# Fit only on the official training split.
# ------------------------------------------------------------
all_scores = {}

cached = load_score_cache("tfidf")

if cached is not None:
    all_scores["tfidf"] = cached

else:
    print("\nFitting TF-IDF...")

    training_texts = pd.unique(
        pd.concat(
            [
                train["text"].fillna("").astype(str),
                train["text_b"].fillna("").astype(str),
            ],
            ignore_index=True,
        )
    ).tolist()

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    tfidf.fit(training_texts)

    query_matrix = tfidf.transform(
        pairs["text"].tolist()
    )

    passage_matrix = tfidf.transform(
        pairs["text_b"].tolist()
    )

    tfidf_scores = np.asarray(
        query_matrix.multiply(
            passage_matrix
        ).sum(axis=1)
    ).ravel()

    all_scores["tfidf"] = tfidf_scores
    save_score_cache("tfidf", tfidf_scores)

    del tfidf
    del query_matrix
    del passage_matrix
    gc.collect()

# ------------------------------------------------------------
# 7. BM25 score
# Fit vocabulary and IDF only on training passages.
# ------------------------------------------------------------
cached = load_score_cache("bm25")

if cached is not None:
    all_scores["bm25"] = cached

else:
    print("\nFitting BM25...")

    count_vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_features=BM25_MAX_FEATURES,
        dtype=np.float32,
    )

    training_passage_matrix = (
        count_vectorizer.fit_transform(
            train["text_b"]
            .fillna("")
            .astype(str)
            .tolist()
        ).tocsr()
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

    query_counts = count_vectorizer.transform(
        pairs["text"].tolist()
    ).tocsr()

    passage_counts = count_vectorizer.transform(
        pairs["text_b"].tolist()
    ).tocsr()

    # Binary query term presence.
    query_binary = query_counts.copy()
    query_binary.data = np.ones_like(
        query_binary.data,
        dtype=np.float32,
    )

    passage_lengths = np.asarray(
        passage_counts.sum(axis=1)
    ).ravel()

    weighted_passages = passage_counts.copy().astype(
        np.float64
    )

    for row_index in range(
        weighted_passages.shape[0]
    ):
        start = weighted_passages.indptr[row_index]
        end = weighted_passages.indptr[row_index + 1]

        if start == end:
            continue

        term_indices = weighted_passages.indices[
            start:end
        ]

        term_frequencies = weighted_passages.data[
            start:end
        ]

        length_normalizer = (
            BM25_K1
            * (
                1.0
                - BM25_B
                + BM25_B
                * passage_lengths[row_index]
                / average_document_length
            )
        )

        weighted_passages.data[start:end] = (
            inverse_document_frequency[term_indices]
            * (
                term_frequencies
                * (BM25_K1 + 1.0)
            )
            / (
                term_frequencies
                + length_normalizer
            )
        )

    bm25_scores = np.asarray(
        query_binary.multiply(
            weighted_passages
        ).sum(axis=1)
    ).ravel()

    all_scores["bm25"] = bm25_scores
    save_score_cache("bm25", bm25_scores)

    del count_vectorizer
    del training_passage_matrix
    del query_counts
    del query_binary
    del passage_counts
    del weighted_passages
    gc.collect()

# ------------------------------------------------------------
# 8. Dense similarity
# ------------------------------------------------------------
def encode_with_oom_recovery(
    model,
    texts,
    initial_batch_size,
):
    batch_size = initial_batch_size

    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return embeddings.astype(np.float32)

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()

            if batch_size <= 8:
                raise

            batch_size = max(
                8,
                batch_size // 2,
            )

            print(
                "OOM: reducing dense batch size to",
                batch_size,
            )

def dense_pair_scores(
    model_name,
    model_id,
    batch_size,
):
    cached_scores = load_score_cache(model_name)

    if cached_scores is not None:
        return cached_scores

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU required for {model_name}"
        )

    print("\nLoading:", model_id)

    model = SentenceTransformer(
        model_id,
        device="cuda",
    )

    model.max_seq_length = 384

    try:
        model.half()
    except Exception:
        pass

    unique_texts = pd.unique(
        pd.concat(
            [
                pairs["text"],
                pairs["text_b"],
            ],
            ignore_index=True,
        )
    ).tolist()

    print(
        model_name,
        "unique texts:",
        len(unique_texts),
    )

    embeddings = encode_with_oom_recovery(
        model,
        unique_texts,
        initial_batch_size=batch_size,
    )

    text_to_index = {
        text: index
        for index, text in enumerate(unique_texts)
    }

    query_indices = np.asarray([
        text_to_index[text]
        for text in pairs["text"]
    ])

    passage_indices = np.asarray([
        text_to_index[text]
        for text in pairs["text_b"]
    ])

    scores = np.sum(
        embeddings[query_indices]
        * embeddings[passage_indices],
        axis=1,
    )

    save_score_cache(model_name, scores)

    del embeddings
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores.astype(np.float64)

for model_name, config in DENSE_MODELS.items():
    all_scores[model_name] = dense_pair_scores(
        model_name=model_name,
        model_id=config["id"],
        batch_size=config["batch_size"],
    )

# ------------------------------------------------------------
# 9. Attach and save scores
# ------------------------------------------------------------
for model_name, scores in all_scores.items():
    pairs[f"score_{model_name}"] = scores

scored_pairs_path = (
    OUTPUT_DIR
    / "random_vs_hard_scored_pairs.parquet"
)

pairs.to_parquet(
    scored_pairs_path,
    index=False,
    compression="zstd",
)

print("\nSaved scored pairs:", scored_pairs_path)

# ------------------------------------------------------------
# 10. Evaluation helpers
# ------------------------------------------------------------
query_positions = {
    query: position
    for position, query
    in enumerate(common_queries)
}

grouped = {
    query: group
    for query, group
    in pairs.groupby(
        "query_key",
        sort=False,
    )
}

def auc_for_condition(
    query_indices,
    model_name,
    condition,
):
    labels = []
    scores = []

    score_column = f"score_{model_name}"

    for query_index in query_indices:
        query = common_queries[query_index]
        group = grouped[query]

        cited_scores = group.loc[
            group["candidate_type"] == "cited",
            score_column,
        ].to_numpy(dtype=np.float64)

        u_scores = group.loc[
            group["candidate_type"] == condition,
            score_column,
        ].to_numpy(dtype=np.float64)

        labels.extend(
            np.ones(
                len(cited_scores),
                dtype=np.int8,
            )
        )

        scores.extend(cited_scores)

        labels.extend(
            np.zeros(
                len(u_scores),
                dtype=np.int8,
            )
        )

        scores.extend(u_scores)

    return float(
        roc_auc_score(
            np.asarray(labels),
            np.asarray(scores),
        )
    )

def precompute_query_metrics(model_name):
    score_column = f"score_{model_name}"

    output = {
        "hard_u_score": np.full(
            len(common_queries),
            np.nan,
        ),
        "random_u_score": np.full(
            len(common_queries),
            np.nan,
        ),
        "hard_pairwise": np.full(
            len(common_queries),
            np.nan,
        ),
        "random_pairwise": np.full(
            len(common_queries),
            np.nan,
        ),
    }

    for query_index, query in enumerate(
        common_queries
    ):
        group = grouped[query]

        cited_scores = group.loc[
            group["candidate_type"] == "cited",
            score_column,
        ].to_numpy(dtype=np.float64)

        hard_score = group.loc[
            group["candidate_type"] == "hard_u",
            score_column,
        ].to_numpy(dtype=np.float64)

        random_score = group.loc[
            group["candidate_type"] == "random_u",
            score_column,
        ].to_numpy(dtype=np.float64)

        if (
            len(hard_score) != 1
            or len(random_score) != 1
            or len(cited_scores) < 1
        ):
            raise ValueError(
                f"Invalid candidate structure for query: "
                f"{query[:80]}"
            )

        hard_score = float(hard_score[0])
        random_score = float(random_score[0])

        output["hard_u_score"][query_index] = hard_score
        output["random_u_score"][query_index] = random_score

        output["hard_pairwise"][query_index] = float(
            np.mean(
                (cited_scores > hard_score)
                + 0.5
                * (cited_scores == hard_score)
            )
        )

        output["random_pairwise"][query_index] = float(
            np.mean(
                (cited_scores > random_score)
                + 0.5
                * (cited_scores == random_score)
            )
        )

    return output

query_metrics = {
    model_name: precompute_query_metrics(
        model_name
    )
    for model_name in all_scores
}

all_query_indices = np.arange(
    len(common_queries),
    dtype=np.int64,
)

# ------------------------------------------------------------
# 11. Point estimates
# ------------------------------------------------------------
point_results = {}

for model_name, metrics in query_metrics.items():
    hard_auc = auc_for_condition(
        all_query_indices,
        model_name,
        "hard_u",
    )

    random_auc = auc_for_condition(
        all_query_indices,
        model_name,
        "random_u",
    )

    hard_pairwise = float(
        np.mean(metrics["hard_pairwise"])
    )

    random_pairwise = float(
        np.mean(metrics["random_pairwise"])
    )

    hard_similarity = float(
        np.mean(metrics["hard_u_score"])
    )

    random_similarity = float(
        np.mean(metrics["random_u_score"])
    )

    point_results[model_name] = {
        "hard_u": {
            "mean_similarity": hard_similarity,
            "cited_vs_u_auc": hard_auc,
            "pairwise_cited_over_u": hard_pairwise,
        },
        "random_u": {
            "mean_similarity": random_similarity,
            "cited_vs_u_auc": random_auc,
            "pairwise_cited_over_u": random_pairwise,
        },
        "hardness_effect": {
            "similarity_hard_minus_random": (
                hard_similarity
                - random_similarity
            ),
            "auc_random_minus_hard": (
                random_auc - hard_auc
            ),
            "pairwise_random_minus_hard": (
                random_pairwise
                - hard_pairwise
            ),
        },
    }

# ------------------------------------------------------------
# 12. Query-level paired bootstrap
# ------------------------------------------------------------
rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

BOOTSTRAP_METRICS = [
    "hard_mean_similarity",
    "random_mean_similarity",
    "similarity_hard_minus_random",
    "hard_cited_vs_u_auc",
    "random_cited_vs_u_auc",
    "auc_random_minus_hard",
    "hard_pairwise",
    "random_pairwise",
    "pairwise_random_minus_hard",
]

bootstrap_values = {
    model_name: {
        metric: np.full(
            N_BOOTSTRAP,
            np.nan,
            dtype=np.float64,
        )
        for metric in BOOTSTRAP_METRICS
    }
    for model_name in all_scores
}

start_time = time.time()

for bootstrap_index in range(N_BOOTSTRAP):
    sampled_queries = rng.integers(
        0,
        len(common_queries),
        size=len(common_queries),
    )

    for model_name, metrics in query_metrics.items():
        hard_similarity = float(
            np.mean(
                metrics["hard_u_score"][
                    sampled_queries
                ]
            )
        )

        random_similarity = float(
            np.mean(
                metrics["random_u_score"][
                    sampled_queries
                ]
            )
        )

        hard_pairwise = float(
            np.mean(
                metrics["hard_pairwise"][
                    sampled_queries
                ]
            )
        )

        random_pairwise = float(
            np.mean(
                metrics["random_pairwise"][
                    sampled_queries
                ]
            )
        )

        hard_auc = auc_for_condition(
            sampled_queries,
            model_name,
            "hard_u",
        )

        random_auc = auc_for_condition(
            sampled_queries,
            model_name,
            "random_u",
        )

        values = bootstrap_values[model_name]

        values["hard_mean_similarity"][
            bootstrap_index
        ] = hard_similarity

        values["random_mean_similarity"][
            bootstrap_index
        ] = random_similarity

        values["similarity_hard_minus_random"][
            bootstrap_index
        ] = hard_similarity - random_similarity

        values["hard_cited_vs_u_auc"][
            bootstrap_index
        ] = hard_auc

        values["random_cited_vs_u_auc"][
            bootstrap_index
        ] = random_auc

        values["auc_random_minus_hard"][
            bootstrap_index
        ] = random_auc - hard_auc

        values["hard_pairwise"][
            bootstrap_index
        ] = hard_pairwise

        values["random_pairwise"][
            bootstrap_index
        ] = random_pairwise

        values["pairwise_random_minus_hard"][
            bootstrap_index
        ] = random_pairwise - hard_pairwise

    if (
        (bootstrap_index + 1) % 100 == 0
        or bootstrap_index == 0
    ):
        elapsed = time.time() - start_time
        rate = (
            bootstrap_index + 1
        ) / max(elapsed, 1e-9)

        remaining = (
            N_BOOTSTRAP
            - bootstrap_index
            - 1
        ) / max(rate, 1e-9)

        print(
            f"Bootstrap {bootstrap_index + 1:,}/"
            f"{N_BOOTSTRAP:,} | "
            f"ETA {remaining / 60:.1f} min"
        )

# ------------------------------------------------------------
# 13. Confidence intervals and significance
# ------------------------------------------------------------
alpha = 1.0 - CI_LEVEL

def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    return [
        float(
            np.percentile(
                values,
                100 * alpha / 2,
            )
        ),
        float(
            np.percentile(
                values,
                100 * (1 - alpha / 2),
            )
        ),
    ]

def two_sided_bootstrap_p(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    nonpositive = (
        np.sum(values <= 0) + 1
    ) / (len(values) + 1)

    nonnegative = (
        np.sum(values >= 0) + 1
    ) / (len(values) + 1)

    return float(
        min(
            1.0,
            2.0 * min(
                nonpositive,
                nonnegative,
            ),
        )
    )

def holm_adjust(p_values):
    ordered = sorted(
        p_values.items(),
        key=lambda item: item[1],
    )

    number_of_tests = len(ordered)
    adjusted = {}
    running_maximum = 0.0

    for rank, (metric, p_value) in enumerate(
        ordered,
        start=1,
    ):
        corrected = min(
            1.0,
            (
                number_of_tests
                - rank
                + 1
            )
            * p_value,
        )

        running_maximum = max(
            running_maximum,
            corrected,
        )

        adjusted[metric] = running_maximum

    return adjusted

final_results = {}

PRIMARY_DIFFERENCES = [
    "similarity_hard_minus_random",
    "auc_random_minus_hard",
    "pairwise_random_minus_hard",
]

for model_name in all_scores:
    metric_results = {}

    for metric in BOOTSTRAP_METRICS:
        values = bootstrap_values[
            model_name
        ][metric]

        lower, upper = percentile_interval(
            values
        )

        metric_results[metric] = {
            "bootstrap_mean": float(
                np.mean(values)
            ),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    raw_p_values = {
        metric: two_sided_bootstrap_p(
            bootstrap_values[
                model_name
            ][metric]
        )
        for metric in PRIMARY_DIFFERENCES
    }

    adjusted_p_values = holm_adjust(
        raw_p_values
    )

    for metric in PRIMARY_DIFFERENCES:
        metric_results[metric][
            "raw_p_value"
        ] = raw_p_values[metric]

        metric_results[metric][
            "holm_p_value"
        ] = adjusted_p_values[metric]

        metric_results[metric][
            "significant_after_holm_0.05"
        ] = bool(
            adjusted_p_values[metric] < 0.05
        )

    final_results[model_name] = {
        "point_estimates": (
            point_results[model_name]
        ),
        "bootstrap": metric_results,
    }

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

output = {
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": REPO_ID,
    "revision": REVISION,
    "split": "test",
    "matched_queries": len(common_queries),
    "hard_u_count": len(hard_u),
    "random_u_count": len(random_u),
    "cited_pair_count": len(cited),
    "random_u_generation_seed": 42,
    "bootstrap_repetitions": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_unit": (
        "lowercased and whitespace-normalized claim text"
    ),
    "tfidf_fit_data": (
        "official XART-H v1.0.1 training claims and passages"
    ),
    "bm25_fit_data": (
        "official XART-H v1.0.1 training passages"
    ),
    "dense_models": {
        key: value["id"]
        for key, value in DENSE_MODELS.items()
    },
    "results": final_results,
}

output = clean_value(output)

result_path = (
    OUTPUT_DIR
    / "random_vs_hard_results.json"
)

with open(
    result_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=2,
    )

summary_rows = []

for model_name, model_result in final_results.items():
    point = model_result["point_estimates"]
    bootstrap = model_result["bootstrap"]

    summary_rows.append({
        "model": model_name,
        "hard_mean_similarity": (
            point["hard_u"]["mean_similarity"]
        ),
        "random_mean_similarity": (
            point["random_u"]["mean_similarity"]
        ),
        "similarity_hard_minus_random": (
            point["hardness_effect"][
                "similarity_hard_minus_random"
            ]
        ),
        "hard_auc": (
            point["hard_u"]["cited_vs_u_auc"]
        ),
        "random_auc": (
            point["random_u"]["cited_vs_u_auc"]
        ),
        "auc_random_minus_hard": (
            point["hardness_effect"][
                "auc_random_minus_hard"
            ]
        ),
        "hard_pairwise": (
            point["hard_u"][
                "pairwise_cited_over_u"
            ]
        ),
        "random_pairwise": (
            point["random_u"][
                "pairwise_cited_over_u"
            ]
        ),
        "pairwise_random_minus_hard": (
            point["hardness_effect"][
                "pairwise_random_minus_hard"
            ]
        ),
        "similarity_holm_p": (
            bootstrap[
                "similarity_hard_minus_random"
            ]["holm_p_value"]
        ),
        "auc_holm_p": (
            bootstrap[
                "auc_random_minus_hard"
            ]["holm_p_value"]
        ),
        "pairwise_holm_p": (
            bootstrap[
                "pairwise_random_minus_hard"
            ]["holm_p_value"]
        ),
    })

summary = pd.DataFrame(summary_rows)

summary_path = (
    OUTPUT_DIR
    / "random_vs_hard_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

bootstrap_rows = []

for model_name, model_result in final_results.items():
    for metric, values in (
        model_result["bootstrap"].items()
    ):
        bootstrap_rows.append({
            "model": model_name,
            "metric": metric,
            **values,
        })

bootstrap_table = pd.DataFrame(
    bootstrap_rows
)

bootstrap_path = (
    OUTPUT_DIR
    / "random_vs_hard_bootstrap.csv"
)

bootstrap_table.to_csv(
    bootstrap_path,
    index=False,
)

# ------------------------------------------------------------
# 15. Print concise result
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("RANDOM U VERSUS HARD U RESULT")
print("=" * 80)
print("Matched queries:", len(common_queries))

for row in summary_rows:
    print(f"\n{row['model']}")

    print(
        "  Mean similarity, Hard / Random:",
        f"{row['hard_mean_similarity']:.6f}",
        "/",
        f"{row['random_mean_similarity']:.6f}",
    )

    print(
        "  Cited/U AUC, Hard / Random:",
        f"{row['hard_auc']:.6f}",
        "/",
        f"{row['random_auc']:.6f}",
    )

    print(
        "  Pairwise cited>U, Hard / Random:",
        f"{row['hard_pairwise']:.6f}",
        "/",
        f"{row['random_pairwise']:.6f}",
    )

    print(
        "  Hardness deltas:",
        f"similarity={row['similarity_hard_minus_random']:.6f},",
        f"AUC={row['auc_random_minus_hard']:.6f},",
        f"pairwise={row['pairwise_random_minus_hard']:.6f}",
    )

    print(
        "  Holm p-values:",
        f"similarity={row['similarity_holm_p']:.6f},",
        f"AUC={row['auc_holm_p']:.6f},",
        f"pairwise={row['pairwise_holm_p']:.6f}",
    )

print("\nSaved:")
print(result_path)
print(summary_path)
print(bootstrap_path)
print(scored_pairs_path)
print("=" * 80)
