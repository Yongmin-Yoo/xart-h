#!/usr/bin/env python3
"""
Canonical Qwen3-Reranker-0.6B Dense-Hard inference.

This script evaluates six existing LoRA adapters:

- hard_only: seeds 13, 42, 77
- mixed_u: seeds 13, 42, 77

It performs inference only and does not train or modify an adapter.
Each completed adapter result is saved independently, allowing safe resume.

Required artifact root
----------------------
XARTH_ARTIFACT_ROOT must contain:

- experiments/model_generalization_v1/qwen3_reranker_0.6b_control/
- experiments/dense_hard_v1/test_dense_hard_u.parquet
- experiments/dense_hard_v1/qwen_frozen_evaluation/

Usage
-----
XARTH_ARTIFACT_ROOT=/path/to/XART-H-artifacts \
  python experiments/run_qwen_dense_hard_recomputed.py

Canonical environment
---------------------
- NVIDIA L4
- torch 2.11.0+cu128
- CUDA 12.8
- transformers 5.18.0
- peft 0.21.1
- BF16, SDPA, TF32
- maximum length 384
- evaluation batch size 36
"""

# ============================================================
# Canonical Qwen Dense-Hard inference
# - 1,642 Dense-Hard candidates
# - hard_only / mixed_u × seeds 13, 42, 77
# - patent-specific instruction
# - raw text/text_b
# - no training
# ============================================================

import gc
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import peft

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Canonical Qwen yes/no answer-token IDs
FALSE_TOKEN_ID = 2152
TRUE_TOKEN_ID = 9693
# ------------------------------------------------------------

assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()

# ------------------------------------------------------------
# 2. Paths and immutable configuration
# ------------------------------------------------------------
ROOT = Path(
    os.environ.get(
        "XARTH_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()

T4 = (
    ROOT
    / "experiments"
    / "model_generalization_v1"
    / "qwen3_reranker_0.6b_control"
)

T5 = (
    ROOT
    / "experiments"
    / "dense_hard_v1"
    / "qwen_frozen_evaluation"
)

DENSE_MANIFEST = (
    ROOT
    / "experiments"
    / "dense_hard_v1"
    / "test_dense_hard_u.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "dense_hard_v1"
    / "qwen_frozen_evaluation_recomputed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DATASET_REVISION = "4a3ad685a395252c66932c49e98a00728089b614"

PATENT_INSTRUCTION = (
    "Given a patent claim, determine whether the prior-art passage "
    "is examiner-cited evidence for that claim"
)

CONDITIONS = ["hard_only", "mixed_u"]
SEEDS = [13, 42, 77]

EXPECTED_ROWS = 1642
MAX_LENGTH = 384
EVAL_BATCH_SIZE = 36

assert DENSE_MANIFEST.exists(), DENSE_MANIFEST
assert T4.exists(), T4
assert T5.exists(), T5

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("Output:", OUTPUT_DIR)

# ------------------------------------------------------------
# 3. Hash helpers
# ------------------------------------------------------------
def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()

def adapter_hash(adapter_dir):
    digest = hashlib.sha256()

    files = sorted([
        path
        for path in adapter_dir.rglob("*")
        if path.is_file()
        and path.name in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        }
    ])

    assert files, f"No adapter files found: {adapter_dir}"

    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))

    return digest.hexdigest()

# ------------------------------------------------------------
# 4. Load Dense-Hard raw data
# ------------------------------------------------------------
dense = pd.read_parquet(DENSE_MANIFEST)

required_dense_columns = {
    "_row_id",
    "text",
    "text_b",
    "claim_id",
    "cited_document_id",
}

missing_columns = required_dense_columns - set(dense.columns)
assert not missing_columns, (
    f"Dense manifest missing: {sorted(missing_columns)}"
)

assert len(dense) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, found {len(dense)}"
)

assert dense["_row_id"].is_unique
assert dense["text"].notna().all()
assert dense["text_b"].notna().all()

dense["_source_key"] = dense["_row_id"].astype(str)

print("\nDense-Hard manifest:")
print("Rows:", len(dense))
print("Unique claims:", dense["claim_id"].astype(str).nunique())
print("Manifest SHA256:", sha256_file(DENSE_MANIFEST))

# ------------------------------------------------------------
# 5. Load old prediction templates only for row identity/order
# ------------------------------------------------------------

# 6. Exact tokenizer and native Qwen reranker prompt
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    str(LOCAL_TOKENIZER),
    local_files_only=True,
    trust_remote_code=True,
)

tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n'
    '<|im_start|>user\n'
)

SUFFIX = (
    '<|im_end|>\n'
    '<|im_start|>assistant\n'
    '<think>\n\n'
    '</think>\n\n'
)

def enc(text):
    return tokenizer.encode(
        str(text),
        add_special_tokens=False,
    )

prefix_ids = enc(PREFIX)
suffix_ids = enc(SUFFIX)

fixed_1 = enc(
    f"<Instruct>: {PATENT_INSTRUCTION}\n"
    f"<Query>: "
)

fixed_2 = enc("\n<Document>: ")

print("\nToken IDs:")
print("false:", tokenizer.convert_ids_to_tokens(FALSE_TOKEN_ID))
print("true :", tokenizer.convert_ids_to_tokens(TRUE_TOKEN_ID))
print("prefix tokens:", len(prefix_ids))
print("suffix tokens:", len(suffix_ids))

# ------------------------------------------------------------
# 7. Balanced truncation
# ------------------------------------------------------------
def balanced_truncate(q_ids, d_ids, available):
    q_ids = list(q_ids)
    d_ids = list(d_ids)

    if len(q_ids) + len(d_ids) <= available:
        return q_ids, d_ids

    q_cap = min(len(q_ids), available // 2)
    d_cap = min(len(d_ids), available - available // 2)

    remaining = available - q_cap - d_cap

    if remaining > 0:
        add_q = min(remaining, len(q_ids) - q_cap)
        q_cap += add_q
        remaining -= add_q

    if remaining > 0:
        add_d = min(remaining, len(d_ids) - d_cap)
        d_cap += add_d

    return q_ids[:q_cap], d_ids[:d_cap]

def build_sequence(query, document):
    q_ids = enc(query)
    d_ids = enc(document)

    available = (
        MAX_LENGTH
        - len(prefix_ids)
        - len(fixed_1)
        - len(fixed_2)
        - len(suffix_ids)
    )

    q_ids, d_ids = balanced_truncate(
        q_ids,
        d_ids,
        available,
    )

    ids = (
        prefix_ids
        + fixed_1
        + q_ids
        + fixed_2
        + d_ids
        + suffix_ids
    )

    assert len(ids) <= MAX_LENGTH
    return ids

# All adapters use the same 87 raw inputs.
canonical = reference_frames[("hard_only", 13)]

sequences = [
    build_sequence(query, document)
    for query, document in zip(
        canonical["text"].astype(str),
        canonical["text_b"].astype(str),
    )
]

lengths = np.asarray([len(ids) for ids in sequences])

print("\nSequence lengths:")
print(pd.Series(lengths).describe().to_string())
print("Rows at 384:", int((lengths == MAX_LENGTH).sum()))

# ------------------------------------------------------------
# 8. Load pinned base model
# ------------------------------------------------------------
gc.collect()
torch.cuda.empty_cache()

print("\nLoading pinned base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

base_model.eval()

# ------------------------------------------------------------
# 9. Load all six adapters
# ------------------------------------------------------------
adapter_paths = {
    (condition, seed):
        T4 / "adapters" / f"{condition}_seed{seed}"
    for condition in CONDITIONS
    for seed in SEEDS
}

for path in adapter_paths.values():
    assert (path / "adapter_config.json").exists(), path

first_key = ("hard_only", 13)
first_adapter_name = "hard_only_seed13"

model = PeftModel.from_pretrained(
    base_model,
    str(adapter_paths[first_key]),
    adapter_name=first_adapter_name,
    is_trainable=False,
)

for condition in CONDITIONS:
    for seed in SEEDS:
        if (condition, seed) == first_key:
            continue

        adapter_name = f"{condition}_seed{seed}"

        model.load_adapter(
            str(adapter_paths[(condition, seed)]),
            adapter_name=adapter_name,
            is_trainable=False,
        )

model.eval()

print("Loaded adapters:", sorted(model.peft_config.keys()))

# ------------------------------------------------------------

# ------------------------------------------------------------
# 5. Load old prediction templates only for row identity/order
# ------------------------------------------------------------
# source_row_id identifies the reused Dense-Hard passage and is therefore
# intentionally non-unique: 1,642 queries share 541 source passages.
# The one-to-one query alignment key is legacy query_id <-> manifest claim_id.

def normalize_query_key_canonical(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


dense = dense.copy()

dense["_query_key"] = dense[
    "claim_id"
].map(normalize_query_key_canonical)

assert len(dense) == EXPECTED_ROWS
assert dense["_query_key"].is_unique
assert dense["_query_key"].ne("").all()
assert dense["text"].notna().all()
assert dense["text_b"].notna().all()

dense_query_mapping = dense[
    [
        "_query_key",
        "_row_id",
        "text",
        "text_b",
        "claim_id",
        "cited_document_id",
    ]
].copy()

dense_query_mapping = dense_query_mapping.rename(columns={
    "_row_id": "_manifest_row_id",
    "claim_id": "_manifest_claim_id",
    "cited_document_id": "_manifest_cited_document_id",
})

templates = {}

for condition in CONDITIONS:
    for seed in SEEDS:
        old_file = (
            T5
            / f"qwen3_reranker_{condition}_seed{seed}_"
              "dense_hard_predictions.parquet"
        )

        assert old_file.exists(), old_file

        old = pd.read_parquet(
            old_file
        ).reset_index(drop=True)

        required_old_columns = {
            "training_condition",
            "seed",
            "query_id",
            "source_row_id",
            "candidate_type",
            "probability",
            "source_cited_document_id",
        }

        missing_old = (
            required_old_columns - set(old.columns)
        )

        assert not missing_old, (
            f"{old_file.name}: missing "
            f"{sorted(missing_old)}"
        )

        assert len(old) == EXPECTED_ROWS

        old["_query_key"] = old[
            "query_id"
        ].map(normalize_query_key_canonical)

        assert old["_query_key"].is_unique
        assert old["_query_key"].ne("").all()

        # Preserve legacy row order exactly.
        old["_template_order"] = np.arange(
            len(old),
            dtype=np.int64,
        )

        merged = old.merge(
            dense_query_mapping,
            on="_query_key",
            how="left",
            validate="one_to_one",
            sort=False,
        )

        merged = (
            merged
            .sort_values("_template_order")
            .reset_index(drop=True)
        )

        assert len(merged) == EXPECTED_ROWS
        assert merged["text"].notna().all()
        assert merged["text_b"].notna().all()
        assert merged["_manifest_row_id"].notna().all()

        assert (
            merged["training_condition"]
            .astype(str)
            .str.lower()
            .eq(condition)
            .all()
        )

        assert (
            pd.to_numeric(merged["seed"])
            .astype(int)
            .eq(seed)
            .all()
        )

        # Retain the original field used by the later order verification.
        merged["_source_key"] = (
            merged["source_row_id"].astype(str)
        )

        unique_queries = merged[
            "_query_key"
        ].nunique()

        unique_source_rows = merged[
            "_source_key"
        ].nunique()

        assert unique_queries == EXPECTED_ROWS
        assert unique_source_rows == 541

        templates[(condition, seed)] = {
            "old_file": old_file,
            "frame": merged,
        }

        print(
            f"Template verified: "
            f"{condition:9s} seed={seed} "
            f"rows={len(merged)} "
            f"unique_queries={unique_queries} "
            f"unique_source_rows={unique_source_rows}"
        )

assert len(templates) == 6

print(
    "\nAll six templates aligned by "
    "query_id ↔ claim_id."
)

# Verify that all six files use the same source order.
canonical_source_order = (
    templates[("hard_only", 13)]["frame"]["_source_key"]
    .tolist()
)

for key, item in templates.items():
    current_order = item["frame"]["_source_key"].tolist()

    assert current_order == canonical_source_order, (
        f"Dense row order differs for {key}"
    )

# ------------------------------------------------------------
# 6. Build canonical token sequences once
# ------------------------------------------------------------
canonical_frame = templates[
    ("hard_only", 13)
]["frame"]

print("\nBuilding canonical Dense-Hard token sequences...")

dense_sequences = [
    build_sequence(query, document)
    for query, document in zip(
        canonical_frame["text"].astype(str),
        canonical_frame["text_b"].astype(str),
    )
]

assert len(dense_sequences) == EXPECTED_ROWS

sequence_lengths = np.asarray([
    len(ids) for ids in dense_sequences
])

assert sequence_lengths.max() <= MAX_LENGTH

print(pd.Series(sequence_lengths).describe().to_string())
print("Rows at max length:", int((sequence_lengths == MAX_LENGTH).sum()))

# ------------------------------------------------------------
# 7. Scoring helper
# ------------------------------------------------------------
def make_dense_batch(sequences):
    features = [
        {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        for ids in sequences
    ]

    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )

    return {
        key: value.to("cuda")
        for key, value in batch.items()
    }

@torch.inference_mode()
def score_dense_adapter(adapter_name):
    model.set_adapter(adapter_name)
    model.eval()

    all_scores = []

    number_of_batches = int(
        np.ceil(len(dense_sequences) / EVAL_BATCH_SIZE)
    )

    started = time.time()

    for batch_index, start in enumerate(
        range(
            0,
            len(dense_sequences),
            EVAL_BATCH_SIZE,
        ),
        start=1,
    ):
        batch_sequences = dense_sequences[
            start:start + EVAL_BATCH_SIZE
        ]

        batch = make_dense_batch(batch_sequences)

        output = model(
            **batch,
            use_cache=False,
            return_dict=True,
        )

        final_logits = output.logits[:, -1, :]

        binary_logits = torch.stack(
            [
                final_logits[:, FALSE_TOKEN_ID],
                final_logits[:, TRUE_TOKEN_ID],
            ],
            dim=1,
        )

        probabilities = torch.softmax(
            binary_logits.float(),
            dim=1,
        )[:, 1]

        all_scores.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        del batch, output, final_logits
        del binary_logits, probabilities

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == number_of_batches
        ):
            elapsed = time.time() - started
            print(
                f"  batch {batch_index:02d}/{number_of_batches:02d} "
                f"rows={min(start + EVAL_BATCH_SIZE, EXPECTED_ROWS)}/"
                f"{EXPECTED_ROWS} "
                f"elapsed={elapsed / 60:.1f} min"
            )

    scores = np.asarray(
        all_scores,
        dtype=np.float64,
    )

    assert len(scores) == EXPECTED_ROWS
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()

    return scores, time.time() - started

# ------------------------------------------------------------
# 8. Resumable six-adapter inference
# ------------------------------------------------------------
summary_rows = []
combined_frames = []

overall_started = time.time()

for condition in CONDITIONS:
    for seed in SEEDS:
        adapter_name = f"{condition}_seed{seed}"

        adapter_dir = (
            T4
            / "adapters"
            / f"{condition}_seed{seed}"
        )

        assert adapter_name in model.peft_config, (
            f"Adapter not loaded: {adapter_name}"
        )

        output_file = (
            OUTPUT_DIR
            / f"qwen3_reranker_{condition}_seed{seed}_"
              "dense_hard_predictions.parquet"
        )

        metadata_file = (
            OUTPUT_DIR
            / f"qwen3_reranker_{condition}_seed{seed}_"
              "dense_hard_summary.json"
        )

        # Resume behavior: only skip a complete, valid output.
        if output_file.exists() and metadata_file.exists():
            existing = pd.read_parquet(output_file)

            valid_existing = (
                len(existing) == EXPECTED_ROWS
                and "probability" in existing.columns
                and existing["probability"].notna().all()
            )

            if valid_existing:
                print(
                    f"\nSKIP complete output: "
                    f"{condition} seed {seed}"
                )

                existing["training_condition"] = condition
                existing["seed"] = seed

                combined_frames.append(existing)

                metadata = json.loads(
                    metadata_file.read_text(encoding="utf-8")
                )

                summary_rows.append({
                    "condition": condition,
                    "seed": seed,
                    "status": "reused",
                    "rows": len(existing),
                    "mean_probability":
                        float(existing["probability"].mean()),
                    "standard_deviation":
                        float(existing["probability"].std()),
                    "minimum_probability":
                        float(existing["probability"].min()),
                    "maximum_probability":
                        float(existing["probability"].max()),
                    "old_new_mae":
                        metadata.get("old_new_mae"),
                    "old_new_correlation":
                        metadata.get("old_new_correlation"),
                    "elapsed_seconds":
                        metadata.get("elapsed_seconds"),
                    "prediction_sha256":
                        sha256_file(output_file),
                })

                continue

        print("\n============================================================")
        print(f"INFERENCE: {condition} seed {seed}")
        print("============================================================")

        scores, elapsed = score_dense_adapter(adapter_name)

        template = templates[
            (condition, seed)
        ]["frame"]

        old_probability = template[
            "probability"
        ].to_numpy(dtype=np.float64)

        difference = scores - old_probability

        # Preserve the old public schema.
        output = pd.DataFrame({
            "training_condition":
                template["training_condition"].astype(str),
            "seed":
                pd.to_numeric(template["seed"]).astype(int),
            "query_id":
                template["query_id"],
            "source_row_id":
                template["source_row_id"],
            "candidate_type":
                template["candidate_type"],
            "probability":
                scores,
            "source_cited_document_id":
                template["source_cited_document_id"],
        })

        assert len(output) == EXPECTED_ROWS
        assert output["source_row_id"].astype(str).is_unique
        assert output["probability"].notna().all()

        # Atomic-style write.
        temporary_file = output_file.with_suffix(
            ".parquet.tmp"
        )

        output.to_parquet(
            temporary_file,
            index=False,
        )

        os.replace(
            temporary_file,
            output_file,
        )

        correlation = float(
            np.corrcoef(
                old_probability,
                scores,
            )[0, 1]
        )

        old_new_mae = float(
            np.mean(np.abs(difference))
        )

        metadata = {
            "experiment":
                "Qwen Dense-Hard canonical recomputation",
            "training_condition": condition,
            "seed": seed,
            "rows": EXPECTED_ROWS,
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "adapter_directory": str(adapter_dir),
            "adapter_sha256": adapter_hash(adapter_dir),
            "dense_manifest": str(DENSE_MANIFEST),
            "dense_manifest_sha256":
                sha256_file(DENSE_MANIFEST),
            "instruction": PATENT_INSTRUCTION,
            "prompt_construction":
                "native Qwen reranker prefix/suffix with "
                "token-ID concatenation",
            "truncation":
                "balanced claim-passage truncation after "
                "reserving native reranker prompt tokens",
            "input_columns": ["text", "text_b"],
            "maximum_length": MAX_LENGTH,
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "precision": "bfloat16",
            "attention_implementation": "sdpa",
            "tf32": True,
            "false_token_id": FALSE_TOKEN_ID,
            "true_token_id": TRUE_TOKEN_ID,
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
            "probability_summary": {
                "mean": float(scores.mean()),
                "standard_deviation":
                    float(scores.std(ddof=1)),
                "minimum": float(scores.min()),
                "maximum": float(scores.max()),
            },
            "legacy_web_prompt_comparison": {
                "mean_absolute_difference":
                    old_new_mae,
                "correlation": correlation,
                "mean_signed_difference":
                    float(difference.mean()),
                "maximum_absolute_difference":
                    float(np.max(np.abs(difference))),
            },
            "old_new_mae": old_new_mae,
            "old_new_correlation": correlation,
            "elapsed_seconds": elapsed,
            "prediction_file": str(output_file),
            "prediction_sha256": sha256_file(output_file),
            "completed_at_utc":
                datetime.now(timezone.utc).isoformat(),
        }

        metadata_file.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        combined_frames.append(output)

        summary_rows.append({
            "condition": condition,
            "seed": seed,
            "status": "computed",
            "rows": len(output),
            "mean_probability": float(scores.mean()),
            "standard_deviation":
                float(scores.std(ddof=1)),
            "minimum_probability": float(scores.min()),
            "maximum_probability": float(scores.max()),
            "old_new_mae": old_new_mae,
            "old_new_correlation": correlation,
            "elapsed_seconds": elapsed,
            "prediction_sha256":
                sha256_file(output_file),
        })

        print("Saved:", output_file)
        print("Mean probability:", float(scores.mean()))
        print("Old vs new MAE:", old_new_mae)
        print("Old vs new correlation:", correlation)
        print("Elapsed minutes:", elapsed / 60)

        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# 9. Save combined predictions and summary
# ------------------------------------------------------------
summary = pd.DataFrame(summary_rows).sort_values(
    ["condition", "seed"]
).reset_index(drop=True)

combined = pd.concat(
    combined_frames,
    ignore_index=True,
)

assert len(summary) == 6
assert len(combined) == EXPECTED_ROWS * 6

combined_file = (
    OUTPUT_DIR
    / "qwen_dense_hard_all_seed_predictions.parquet"
)

summary_file = (
    OUTPUT_DIR
    / "qwen_dense_hard_inference_summary.csv"
)

config_file = (
    OUTPUT_DIR
    / "qwen_dense_hard_canonical_config.json"
)

combined.to_parquet(
    combined_file,
    index=False,
)

summary.to_csv(
    summary_file,
    index=False,
)

config = {
    "experiment":
        "Qwen Dense-Hard canonical recomputation",
    "conditions": CONDITIONS,
    "seeds": SEEDS,
    "rows_per_run": EXPECTED_ROWS,
    "total_predictions": len(combined),
    "model": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "instruction": PATENT_INSTRUCTION,
    "input_columns": ["text", "text_b"],
    "prompt_construction":
        "native Qwen reranker prefix/suffix with "
        "token-ID concatenation",
    "truncation":
        "balanced claim-passage truncation after "
        "reserving native reranker prompt tokens",
    "maximum_length": MAX_LENGTH,
    "evaluation_batch_size": EVAL_BATCH_SIZE,
    "precision": "bfloat16",
    "attention_implementation": "sdpa",
    "tf32": True,
    "false_token_id": FALSE_TOKEN_ID,
    "true_token_id": TRUE_TOKEN_ID,
    "runtime": {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "calibration_note": (
        "The canonical public pipeline was reconstructed from the "
        "archived Table 4 configuration. Legacy predictions are not "
        "used as final Dense-Hard results."
    ),
    "combined_prediction_file": str(combined_file),
    "combined_prediction_sha256":
        sha256_file(combined_file),
    "total_elapsed_seconds":
        time.time() - overall_started,
    "completed_at_utc":
        datetime.now(timezone.utc).isoformat(),
}

config_file.write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# ------------------------------------------------------------
# 10. Final verification
# ------------------------------------------------------------
print("\n============================================================")
print("QWEN DENSE-HARD INFERENCE COMPLETE")
print("============================================================")

print(summary.to_string(index=False))

print("\nCombined rows:", len(combined))
print("Expected rows:", EXPECTED_ROWS * 6)
print("NaN probabilities:",
      int(combined["probability"].isna().sum()))

print("\n=== Output files ===")
for path in [
    combined_file,
    summary_file,
    config_file,
]:
    print(
        path,
        f"({path.stat().st_size / 1024:.1f} KB)"
    )

print("\nTotal elapsed minutes:",
      (time.time() - overall_started) / 60)

assert len(combined) == EXPECTED_ROWS * 6
assert combined["probability"].notna().all()
assert ((combined["probability"] >= 0)
        & (combined["probability"] <= 1)).all()

print("\nAll six canonical Dense-Hard evaluations are saved.")
print("Next: recompute ensemble metrics, bootstrap CIs, and Holm tests.")