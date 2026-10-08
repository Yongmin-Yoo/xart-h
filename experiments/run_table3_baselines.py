import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer, CrossEncoder

REPO_ID = "yongminyoo91/xart-h"
DATASET_REVISION = "4a3ad685a395252c66932c49e98a00728089b614"

MAX_LENGTH = 384
SEED = 42
K1 = 1.5
B = 0.75
MAX_FEATURES = 200_000

MODELS = {
    "minilm_bi": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "batch": 256,
    },
    "patentsberta": {
        "id": "AI-Growth-Lab/PatentSBERTa",
        "revision": "3ff1d553c861d8f5bfd902333d97fc95eb6b4c8f",
        "batch": 128,
    },
}

CROSS_ENCODER = {
    "id": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "batch": 256,
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/table3_recomputed"
OUT.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def evaluate(scores, labels, queries):
    eligible_queries = {
        q for q, y in zip(queries, labels) if y == 2
    }
    cu_mask = np.asarray([
        q in eligible_queries for q in queries
    ])
    xa_mask = labels != 2

    cu_labels = (labels[cu_mask] != 2).astype(int)
    xa_labels = (labels[xa_mask] == 1).astype(int)

    return {
        "cited_vs_u_roc_auc": float(
            roc_auc_score(cu_labels, scores[cu_mask])
        ),
        "x_vs_a_roc_auc": float(
            roc_auc_score(xa_labels, scores[xa_mask])
        ),
        "cited_vs_u_rows": int(cu_mask.sum()),
        "x_vs_a_rows": int(xa_mask.sum()),
    }


print("Loading public dataset...")
dataset = load_dataset(
    REPO_ID,
    revision=DATASET_REVISION,
)

train = dataset["train"].select_columns([
    "text", "text_b"
]).to_pandas()

test = dataset["test"].select_columns([
    "_row_id", "text", "text_b", "label"
]).to_pandas()

test["text"] = test["text"].fillna("").astype(str)
test["text_b"] = test["text_b"].fillna("").astype(str)

labels = test["label"].to_numpy(dtype=np.int64)
queries = np.asarray([
    normalize(x) for x in test["text"]
])

scores = {}

# Random
rng = np.random.default_rng(SEED)
scores["random"] = rng.random(len(test))

# TF-IDF
print("Fitting TF-IDF...")
training_texts = pd.unique(pd.concat([
    train["text"].fillna("").astype(str),
    train["text_b"].fillna("").astype(str),
], ignore_index=True)).tolist()

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32,
)
tfidf.fit(training_texts)

query_matrix = tfidf.transform(test["text"])
passage_matrix = tfidf.transform(test["text_b"])

scores["tfidf"] = np.asarray(
    query_matrix.multiply(passage_matrix).sum(axis=1)
).ravel()

del tfidf, query_matrix, passage_matrix
gc.collect()

# BM25
print("Fitting BM25...")
vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    max_features=MAX_FEATURES,
    dtype=np.float32,
)

train_passages = vectorizer.fit_transform(
    train["text_b"].fillna("").astype(str)
).tocsr()

n_documents = train_passages.shape[0]
df = np.asarray((train_passages > 0).sum(axis=0)).ravel()
idf = np.log(
    1.0 + (n_documents - df + 0.5) / (df + 0.5)
)

average_length = max(
    float(np.asarray(
        train_passages.sum(axis=1)
    ).ravel().mean()),
    1e-12,
)

query_counts = vectorizer.transform(test["text"]).tocsr()
passage_counts = vectorizer.transform(test["text_b"]).tocsr()

query_binary = query_counts.copy()
query_binary.data = np.ones_like(
    query_binary.data,
    dtype=np.float32,
)

passage_lengths = np.asarray(
    passage_counts.sum(axis=1)
).ravel()

weighted = passage_counts.copy().astype(np.float64)

for row in range(weighted.shape[0]):
    start, end = weighted.indptr[row], weighted.indptr[row + 1]
    if start == end:
        continue

    term_ids = weighted.indices[start:end]
    tf = weighted.data[start:end]
    normalizer = K1 * (
        1.0 - B + B * passage_lengths[row] / average_length
    )

    weighted.data[start:end] = (
        idf[term_ids]
        * (tf * (K1 + 1.0))
        / (tf + normalizer)
    )

scores["bm25"] = np.asarray(
    query_binary.multiply(weighted).sum(axis=1)
).ravel()

del (
    vectorizer, train_passages, query_counts,
    passage_counts, query_binary, weighted
)
gc.collect()

# Dense bi-encoders
unique_texts = pd.unique(pd.concat([
    test["text"], test["text_b"]
], ignore_index=True)).tolist()

text_to_index = {
    text: index for index, text in enumerate(unique_texts)
}
query_indices = np.asarray([
    text_to_index[x] for x in test["text"]
])
passage_indices = np.asarray([
    text_to_index[x] for x in test["text_b"]
])

for name, config in MODELS.items():
    print("Loading:", name)
    model = SentenceTransformer(
        config["id"],
        revision=config["revision"],
        device="cuda",
    )
    model.max_seq_length = MAX_LENGTH
    model.half()

    embeddings = model.encode(
        unique_texts,
        batch_size=config["batch"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores[name] = np.sum(
        embeddings[query_indices]
        * embeddings[passage_indices],
        axis=1,
    )

    del model, embeddings
    gc.collect()
    torch.cuda.empty_cache()

# MiniLM cross-encoder
print("Loading: minilm_cross_encoder")
cross_encoder = CrossEncoder(
    CROSS_ENCODER["id"],
    revision=CROSS_ENCODER["revision"],
    max_length=MAX_LENGTH,
    device="cuda",
)

pairs = list(zip(
    test["text"].tolist(),
    test["text_b"].tolist(),
))

scores["minilm_cross_encoder"] = np.asarray(
    cross_encoder.predict(
        pairs,
        batch_size=CROSS_ENCODER["batch"],
        show_progress_bar=True,
    )
).reshape(-1)

results = {
    name: evaluate(value, labels, queries)
    for name, value in scores.items()
}

for name, result in results.items():
    print(
        name,
        "C/U =", round(result["cited_vs_u_roc_auc"], 6),
        "X/A =", round(result["x_vs_a_roc_auc"], 6),
    )

scored = test.copy()
for name, value in scores.items():
    scored[f"score_{name}"] = value

scored.to_parquet(
    OUT / "table3_baseline_scores.parquet",
    index=False,
    compression="zstd",
)

with open(
    OUT / "table3_baseline_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump({
        "dataset": REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "models": MODELS,
        "cross_encoder": CROSS_ENCODER,
        "results": results,
    }, file, indent=2)

print("Saved:", OUT)
