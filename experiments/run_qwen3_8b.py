"""Reproducibility script exported from the executed XART-H notebook."""

# ============================================================
# XART-H: Qwen3-8B zero-shot direct vs hierarchical evaluation
# GPU: NVIDIA L4 22.5GB
# Dataset: yongminyoo91/xart-h, revision v1.0.1
# ============================================================

import os
import sys
import gc
import json
import math
import time
import re
import subprocess
import importlib.metadata
from pathlib import Path

# ------------------------------------------------------------
# 0. Install only missing/incompatible packages
# ------------------------------------------------------------
def version_tuple(v):
    nums = re.findall(r"\d+", v)
    return tuple(map(int, nums[:3]))

required = {
    "transformers": ("4.51.0", "transformers>=4.51,<5"),
    "accelerate": ("1.2.0", "accelerate>=1.2,<2"),
    "bitsandbytes": ("0.45.0", "bitsandbytes>=0.45"),
    "datasets": ("3.0.0", "datasets>=3,<5"),
}

to_install = []

for pkg, (minimum, specification) in required.items():
    try:
        current = importlib.metadata.version(pkg)
        if version_tuple(current) < version_tuple(minimum):
            to_install.append(specification)
    except importlib.metadata.PackageNotFoundError:
        to_install.append(specification)

if to_install:
    print("Installing:", to_install)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + to_install
    )

# ------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ndcg_score,
)

# ------------------------------------------------------------
# 2. Reproducibility and paths
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("GPU가 필요합니다. Colab에서 L4 GPU를 선택하세요.")

GPU_NAME = torch.cuda.get_device_name(0)
print("GPU:", GPU_NAME)
print("BF16 supported:", torch.cuda.is_bf16_supported())

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as e:
    print("Drive mount skipped:", e)

OUT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/qwen3_8b_zero_shot_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"
MODEL_ID = "Qwen/Qwen3-8B"

# L4-safe defaults
MAX_LENGTH = 1280
TEXT_TOKENS_PER_SIDE = 500
INITIAL_BATCH_SIZE = 8
SAVE_EVERY_BATCHES = 25

print("Output:", OUT)

# ------------------------------------------------------------
# 3. Load XART-H test split
# ------------------------------------------------------------
dataset = load_dataset(
    REPO_ID,
    revision=REVISION,
)

test = dataset["test"]

required_columns = {"text", "text_b", "label"}
missing = required_columns - set(test.column_names)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

keep_columns = [
    c for c in ["_row_id", "text", "text_b", "label"]
    if c in test.column_names
]

df = test.select_columns(keep_columns).to_pandas()
df["label"] = df["label"].astype(int)

expected_counts = {0: 1816, 1: 1822, 2: 1678}
actual_counts = df["label"].value_counts().sort_index().to_dict()

assert len(df) == 5316, len(df)
assert actual_counts == expected_counts, actual_counts

print("Test rows:", len(df))
print("Label counts:", actual_counts)

# ------------------------------------------------------------
# 4. Load Qwen3-8B in 4-bit NF4
# ------------------------------------------------------------
compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
tokenizer.truncation_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True,
)

model.eval()
model.config.use_cache = False

print(
    "Model memory footprint:",
    round(model.get_memory_footprint() / 1024**3, 2),
    "GB",
)

# ------------------------------------------------------------
# 5. Prepare texts
# ------------------------------------------------------------
def normalize_text(value):
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value).strip()


def clip_text(value, max_tokens=TEXT_TOKENS_PER_SIDE):
    value = normalize_text(value)

    token_ids = tokenizer.encode(
        value,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


print("Clipping claim and passage texts...")

claims = [
    clip_text(x)
    for x in tqdm(df["text"].tolist(), desc="Claims")
]

passages = [
    clip_text(x)
    for x in tqdm(df["text_b"].tolist(), desc="Passages")
]

# ------------------------------------------------------------
# 6. Prompt templates
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "You evaluate patent claim and prior-art passage pairs. "
    "Use only the supplied text. Do not provide an explanation. "
    "Return exactly one permitted capital letter."
)


def build_user_prompt(mode, claim, passage):
    pair = (
        f"PATENT CLAIM:\n{claim}\n\n"
        f"CANDIDATE PASSAGE:\n{passage}\n\n"
    )

    if mode == "direct":
        instruction = (
            "Choose exactly one label:\n"
            "X: examiner-cited prior art that is particularly relevant "
            "when considered alone.\n"
            "A: examiner-cited technological background or general "
            "state of the art.\n"
            "U: a temporally eligible and topically similar hard candidate "
            "absent from the observed citation record.\n"
            "Answer with only A, X, or U."
        )

    elif mode == "stage1":
        instruction = (
            "Choose exactly one label:\n"
            "C: the passage belongs to an examiner-cited category, X or A.\n"
            "U: the passage is a hard uncited candidate absent from the "
            "observed citation record.\n"
            "Answer with only C or U."
        )

    elif mode == "stage2":
        instruction = (
            "Assume that the passage is examiner-cited. "
            "Choose exactly one label:\n"
            "X: particularly relevant prior art when considered alone.\n"
            "A: technological background or general state of the art.\n"
            "Answer with only A or X."
        )

    else:
        raise ValueError(mode)

    return pair + instruction


def render_prompt(mode, claim, passage):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(mode, claim, passage),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# ------------------------------------------------------------
# 7. Verbalizer token IDs
# ------------------------------------------------------------
MODE_LABELS = {
    "direct": ["A", "X", "U"],
    "stage1": ["C", "U"],
    "stage2": ["A", "X"],
}


def get_single_token_id(label):
    ids = tokenizer.encode(label, add_special_tokens=False)

    if len(ids) != 1:
        raise ValueError(
            f"Verbalizer {label!r} is not a single token: {ids}"
        )

    return ids[0]


MODE_TOKEN_IDS = {
    mode: [get_single_token_id(label) for label in labels]
    for mode, labels in MODE_LABELS.items()
}

print("Verbalizers:")

for mode, labels in MODE_LABELS.items():
    decoded = [
        tokenizer.decode([token_id])
        for token_id in MODE_TOKEN_IDS[mode]
    ]
    print(mode, list(zip(labels, MODE_TOKEN_IDS[mode], decoded)))

# ------------------------------------------------------------
# 8. Cached zero-shot inference
# ------------------------------------------------------------
@torch.inference_mode()
def infer_batch(prompts, token_ids):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    output = model(
        **encoded,
        use_cache=False,
        return_dict=True,
    )

    next_token_logits = output.logits[:, -1, :]
    selected_logits = next_token_logits[:, token_ids]

    return selected_logits.float().cpu().numpy()


def run_mode(mode):
    labels = MODE_LABELS[mode]
    token_ids = MODE_TOKEN_IDS[mode]

    cache_path = OUT / f"{mode}_logits.npz"
    n = len(df)
    k = len(labels)

    if cache_path.exists():
        cache = np.load(cache_path)
        logits = cache["logits"]

        if logits.shape != (n, k):
            print("Invalid cache shape; rebuilding:", logits.shape)
            logits = np.full((n, k), np.nan, dtype=np.float32)
    else:
        logits = np.full((n, k), np.nan, dtype=np.float32)

    missing_indices = np.where(np.isnan(logits).any(axis=1))[0]

    if len(missing_indices) == 0:
        print(f"SKIP {mode}: cache complete")
        return logits

    print(
        f"\nRUN {mode}: "
        f"{len(missing_indices):,}/{n:,} rows remaining"
    )

    batch_size = INITIAL_BATCH_SIZE
    position = 0
    completed_batches = 0

    progress = tqdm(
        total=len(missing_indices),
        desc=mode,
        unit="pair",
    )

    while position < len(missing_indices):
        current_indices = missing_indices[
            position:position + batch_size
        ]

        prompts = [
            render_prompt(mode, claims[i], passages[i])
            for i in current_indices
        ]

        try:
            batch_logits = infer_batch(prompts, token_ids)
            logits[current_indices] = batch_logits

            position += len(current_indices)
            progress.update(len(current_indices))
            completed_batches += 1

            if completed_batches % SAVE_EVERY_BATCHES == 0:
                np.savez_compressed(
                    cache_path,
                    logits=logits,
                    labels=np.array(labels),
                )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()

            if batch_size <= 1:
                raise

            batch_size = max(1, batch_size // 2)
            print(f"\nOOM: reducing {mode} batch size to {batch_size}")

    progress.close()

    np.savez_compressed(
        cache_path,
        logits=logits,
        labels=np.array(labels),
    )

    assert not np.isnan(logits).any()
    print("Saved:", cache_path)

    return logits


start_time = time.time()

direct_logits = run_mode("direct")
stage1_logits = run_mode("stage1")
stage2_logits = run_mode("stage2")

elapsed_hours = (time.time() - start_time) / 3600
print("Inference elapsed hours:", round(elapsed_hours, 3))

# ------------------------------------------------------------
# 9. Convert logits to probabilities
# ------------------------------------------------------------
def softmax_numpy(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Direct order: A, X, U
direct_prob = softmax_numpy(direct_logits)

# Stage 1 order: C, U
stage1_prob = softmax_numpy(stage1_logits)

# Stage 2 order: A, X
stage2_prob = softmax_numpy(stage2_logits)

p_cited = stage1_prob[:, 0]
p_u = stage1_prob[:, 1]
p_a_given_cited = stage2_prob[:, 0]
p_x_given_cited = stage2_prob[:, 1]

# Hierarchical order: A, X, U
hierarchical_prob = np.column_stack([
    p_cited * p_a_given_cited,
    p_cited * p_x_given_cited,
    p_u,
])

hierarchical_prob /= hierarchical_prob.sum(
    axis=1,
    keepdims=True,
)

# ------------------------------------------------------------
# 10. Evaluation helpers
# ------------------------------------------------------------
y_true = df["label"].to_numpy()
query_keys = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .to_numpy()
)


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def classification_metrics(probabilities):
    prediction = probabilities.argmax(axis=1)

    return {
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1_A_X_U": [
            float(x)
            for x in f1_score(
                y_true,
                prediction,
                labels=[0, 1, 2],
                average=None,
                zero_division=0,
            )
        ],
    }


def binary_metrics(probabilities):
    # Cited = A or X
    cited_target = (y_true != 2).astype(int)
    cited_score = probabilities[:, 0] + probabilities[:, 1]

    # X vs A among examiner-cited pairs
    xa_mask = y_true != 2
    xa_target = (y_true[xa_mask] == 1).astype(int)

    xa_denominator = (
        probabilities[xa_mask, 0]
        + probabilities[xa_mask, 1]
        + 1e-12
    )
    x_score = probabilities[xa_mask, 1] / xa_denominator

    return {
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


def ranking_metrics(probabilities):
    cited_score = probabilities[:, 0] + probabilities[:, 1]

    conditional_x_score = (
        probabilities[:, 1]
        / (
            probabilities[:, 0]
            + probabilities[:, 1]
            + 1e-12
        )
    )

    graded_score = (
        probabilities[:, 0]
        + 2.0 * probabilities[:, 1]
    )

    work = pd.DataFrame({
        "query": query_keys,
        "label": y_true,
        "cited_score": cited_score,
        "x_score": conditional_x_score,
        "graded_score": graded_score,
    })

    cited_u_values = []
    x_a_values = []
    reciprocal_ranks = []
    ndcg_values = []

    for _, group in work.groupby("query", sort=False):
        labels = group["label"].to_numpy()

        cited_mask = labels != 2
        u_mask = labels == 2
        x_mask = labels == 1
        a_mask = labels == 0

        # Ranking metrics involving U are evaluated only on
        # U-eligible queries.
        if cited_mask.any() and u_mask.any():
            cited_scores = group.loc[
                cited_mask, "cited_score"
            ].to_numpy()

            u_scores = group.loc[
                u_mask, "cited_score"
            ].to_numpy()

            comparisons = (
                cited_scores[:, None]
                > u_scores[None, :]
            ).astype(float)

            ties = (
                cited_scores[:, None]
                == u_scores[None, :]
            ).astype(float)

            cited_u_values.append(
                float((comparisons + 0.5 * ties).mean())
            )

            order = np.argsort(
                -group["cited_score"].to_numpy(),
                kind="mergesort",
            )

            ranked_cited = cited_mask[order]
            first_relevant = np.flatnonzero(ranked_cited)

            if len(first_relevant):
                reciprocal_ranks.append(
                    1.0 / (first_relevant[0] + 1)
                )

            relevance = np.where(
                labels == 1,
                2.0,
                np.where(labels == 0, 1.0, 0.0),
            )

            if len(labels) >= 2 and relevance.max() > 0:
                ndcg_values.append(
                    float(
                        ndcg_score(
                            relevance.reshape(1, -1),
                            group["graded_score"]
                            .to_numpy()
                            .reshape(1, -1),
                        )
                    )
                )

        if x_mask.any() and a_mask.any():
            x_scores = group.loc[
                x_mask, "x_score"
            ].to_numpy()

            a_scores = group.loc[
                a_mask, "x_score"
            ].to_numpy()

            comparisons = (
                x_scores[:, None]
                > a_scores[None, :]
            ).astype(float)

            ties = (
                x_scores[:, None]
                == a_scores[None, :]
            ).astype(float)

            x_a_values.append(
                float((comparisons + 0.5 * ties).mean())
            )

    return {
        "pairwise_cited_over_U": float(
            np.mean(cited_u_values)
        ),
        "pairwise_X_over_A": float(
            np.mean(x_a_values)
        ),
        "mrr_cited": float(
            np.mean(reciprocal_ranks)
        ),
        "ndcg": float(
            np.mean(ndcg_values)
        ),
        "u_eligible_queries": int(
            len(cited_u_values)
        ),
    }


def evaluate(probabilities):
    result = {}
    result.update(classification_metrics(probabilities))
    result.update(binary_metrics(probabilities))
    result.update(ranking_metrics(probabilities))
    return result

# ------------------------------------------------------------
# 11. Final results
# ------------------------------------------------------------
results = {
    "dataset": REPO_ID,
    "revision": REVISION,
    "split": "test",
    "rows": int(len(df)),
    "model": MODEL_ID,
    "quantization": "4-bit NF4 double quantization",
    "compute_dtype": str(compute_dtype),
    "max_length": MAX_LENGTH,
    "text_tokens_per_side": TEXT_TOKENS_PER_SIDE,
    "seed": SEED,
    "input_columns": ["text", "text_b"],
    "direct_3way": evaluate(direct_prob),
    "hierarchical_3way": evaluate(hierarchical_prob),
    "elapsed_hours_current_session": elapsed_hours,
}

# Round metrics for readable JSON
def round_nested(value):
    if isinstance(value, dict):
        return {
            k: round_nested(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [round_nested(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    return value


results = round_nested(results)

results_path = OUT / "qwen3_8b_results.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )

# ------------------------------------------------------------
# 12. Save per-example predictions
# ------------------------------------------------------------
predictions = pd.DataFrame({
    "_row_id": (
        df["_row_id"]
        if "_row_id" in df.columns
        else np.arange(len(df))
    ),
    "label": y_true,
    "direct_p_A": direct_prob[:, 0],
    "direct_p_X": direct_prob[:, 1],
    "direct_p_U": direct_prob[:, 2],
    "direct_prediction": direct_prob.argmax(axis=1),
    "hierarchical_p_A": hierarchical_prob[:, 0],
    "hierarchical_p_X": hierarchical_prob[:, 1],
    "hierarchical_p_U": hierarchical_prob[:, 2],
    "hierarchical_prediction": hierarchical_prob.argmax(axis=1),
    "stage1_p_cited": p_cited,
    "stage1_p_U": p_u,
    "stage2_p_A_given_cited": p_a_given_cited,
    "stage2_p_X_given_cited": p_x_given_cited,
})

prediction_path = OUT / "qwen3_8b_test_predictions.parquet"
predictions.to_parquet(
    prediction_path,
    index=False,
    compression="zstd",
)

print("\n" + "=" * 70)
print("QWEN3-8B RESULT")
print("=" * 70)
print(json.dumps(results, ensure_ascii=False, indent=2))
print("\nSaved results:", results_path)
print("Saved predictions:", prediction_path)
print("=" * 70)
