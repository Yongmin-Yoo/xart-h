import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sentence_transformers import CrossEncoder

REPO_ID = "yongminyoo91/xart-h"
DATASET_REVISION = "4a3ad685a395252c66932c49e98a00728089b614"

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

MAX_LENGTH = 384
BATCH_SIZE = 16

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/table3_recomputed"
OUT.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


test = load_dataset(
    REPO_ID,
    revision=DATASET_REVISION,
    split="test",
).to_pandas()

test["text"] = test["text"].fillna("").astype(str)
test["text_b"] = test["text_b"].fillna("").astype(str)

labels = test["label"].to_numpy(dtype=np.int64)
queries = np.asarray([normalize(x) for x in test["text"]])

eligible_queries = {
    q for q, y in zip(queries, labels) if y == 2
}
cu_mask = np.asarray([
    q in eligible_queries for q in queries
])
xa_mask = labels != 2

print("Loading:", MODEL_ID)

model = CrossEncoder(
    MODEL_ID,
    revision=MODEL_REVISION,
    max_length=MAX_LENGTH,
    device="cuda",
)

try:
    model.model.half()
except Exception:
    pass

pairs = list(zip(
    test["text"].tolist(),
    test["text_b"].tolist(),
))

scores = np.asarray(
    model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
).reshape(-1)

results = {
    "cited_vs_u_roc_auc": float(roc_auc_score(
        labels[cu_mask] != 2,
        scores[cu_mask],
    )),
    "x_vs_a_roc_auc": float(roc_auc_score(
        labels[xa_mask] == 1,
        scores[xa_mask],
    )),
    "cited_vs_u_rows": int(cu_mask.sum()),
    "x_vs_a_rows": int(xa_mask.sum()),
}

print(json.dumps(results, indent=2))

pd.DataFrame({
    "_row_id": test["_row_id"],
    "score_qwen3_reranker": scores,
}).to_parquet(
    OUT / "table3_qwen3_reranker_scores.parquet",
    index=False,
    compression="zstd",
)

with open(
    OUT / "table3_qwen3_reranker_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump({
        "dataset": REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "maximum_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "prompt": "SentenceTransformers model-card default query prompt",
        "results": results,
    }, file, indent=2)

print("Saved:", OUT)
