#!/usr/bin/env python3

"""Generate PatScout Table 4 ordered predictions from a checkpoint.

Supports:
- MiniLM full sequence-classification checkpoint
- DeBERTa full sequence-classification checkpoint
- Qwen3-Reranker LoRA adapter

The output schema is accepted by reproduce_table4.py.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


MODELS = {
    "minilm": {
        "base_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": None,
        "default_batch_size": 128,
    },
    "deberta": {
        "base_model": "microsoft/deberta-v3-base",
        "revision": None,
        "default_batch_size": 64,
    },
    "qwen": {
        "base_model": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "default_batch_size": 16,
    },
}

SEEDS = {13, 42, 77}
CONDITIONS = {"hard_only", "mixed_u"}
MAX_LENGTH = 384

QWEN_INSTRUCTION = (
    "Given a patent claim, determine whether the prior-art passage "
    "is examiner-cited evidence for that claim"
)

QWEN_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only '
    'be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)

QWEN_SUFFIX = (
    '<|im_end|>\n'
    '<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n'
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EncoderDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.tokenizer(
            str(row["text"]),
            str(row["text_b"]),
            truncation="longest_first",
            max_length=MAX_LENGTH,
        )


class QwenDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.prefix = tokenizer.encode(
            QWEN_PREFIX,
            add_special_tokens=False,
        )
        self.suffix = tokenizer.encode(
            QWEN_SUFFIX,
            add_special_tokens=False,
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        body = (
            f"<Instruct>: {QWEN_INSTRUCTION}\n"
            f"<Query>: {row['text']}\n"
            f"<Document>: {row['text_b']}"
        )

        available = MAX_LENGTH - len(self.prefix) - len(self.suffix)

        if available <= 0:
            raise RuntimeError(
                "Qwen prompt overhead exceeds maximum length."
            )

        body_tokens = self.tokenizer.encode(
            body,
            add_special_tokens=False,
            truncation=True,
            max_length=available,
        )

        input_ids = self.prefix + body_tokens + self.suffix

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


def validate_evaluation_manifest(frame):
    required = {
        "source_row_id",
        "query_id",
        "candidate_type",
        "evaluation_condition",
        "prediction_row_id",
        "text",
        "text_b",
        "label",
    }

    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Evaluation manifest is missing: {sorted(missing)}"
        )

    if len(frame) != 9896:
        raise RuntimeError(
            f"Expected 9,896 rows, found {len(frame)}."
        )

    if frame["query_id"].astype(str).nunique() != 1642:
        raise RuntimeError("Expected 1,642 queries.")

    evaluation_counts = (
        frame["evaluation_condition"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    if evaluation_counts != {
        "hard_u": 4948,
        "random_u": 4948,
    }:
        raise RuntimeError(
            f"Unexpected evaluation counts: {evaluation_counts}"
        )

    candidate_counts = (
        frame["candidate_type"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    if candidate_counts != {
        "cited": 6612,
        "hard_u": 1642,
        "random_u": 1642,
    }:
        raise RuntimeError(
            f"Unexpected candidate counts: {candidate_counts}"
        )

    if frame["text"].isna().any():
        raise RuntimeError("Null query text.")
    if frame["text_b"].isna().any():
        raise RuntimeError("Null passage text.")


def encoder_probabilities(
    frame,
    checkpoint,
    model_name,
    batch_size,
    device,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint
    )
    model.to(device)
    model.eval()

    dataset = EncoderDataset(frame, tokenizer)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    probabilities = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            logits = outputs.logits

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    f"{model_name}: expected [batch,2] logits, "
                    f"found {tuple(logits.shape)}"
                )

            cited = torch.softmax(
                logits.float(),
                dim=1,
            )[:, 1]

            probabilities.extend(
                cited.cpu().numpy().astype(float).tolist()
            )

    return probabilities


def qwen_probabilities(
    frame,
    checkpoint,
    batch_size,
    device,
):
    from peft import PeftModel

    config = MODELS["qwen"]

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    false_token_id = tokenizer.convert_tokens_to_ids("no")
    true_token_id = tokenizer.convert_tokens_to_ids("yes")

    if false_token_id != 2152 or true_token_id != 9693:
        raise RuntimeError(
            "Canonical Qwen token IDs do not match: "
            f"no={false_token_id}, yes={true_token_id}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model.to(device)
    model.eval()

    dataset = QwenDataset(frame, tokenizer)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    probabilities = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(**batch).logits[:, -1, :]

            binary_logits = torch.stack(
                [
                    logits[:, false_token_id],
                    logits[:, true_token_id],
                ],
                dim=1,
            )

            cited = torch.softmax(
                binary_logits.float(),
                dim=1,
            )[:, 1]

            probabilities.extend(
                cited.cpu().numpy().astype(float).tolist()
            )

    return probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        required=True,
    )
    parser.add_argument(
        "--condition",
        choices=sorted(CONDITIONS),
        required=True,
    )
    parser.add_argument(
        "--seed",
        choices=sorted(SEEDS),
        type=int,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.evaluation_manifest.is_file():
        raise FileNotFoundError(args.evaluation_manifest)

    frame = pd.read_parquet(args.evaluation_manifest)
    validate_evaluation_manifest(frame)

    config = MODELS[args.model]
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config["default_batch_size"]
    )

    run_config = {
        "model": args.model,
        "base_model": config["base_model"],
        "base_model_revision": config["revision"],
        "condition": args.condition,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "evaluation_manifest": str(args.evaluation_manifest),
        "evaluation_manifest_sha256": sha256_file(
            args.evaluation_manifest
        ),
        "rows": len(frame),
        "queries": int(
            frame["query_id"].astype(str).nunique()
        ),
        "batch_size": batch_size,
        "maximum_length": MAX_LENGTH,
        "output": str(args.output),
    }

    print(json.dumps(run_config, indent=2, sort_keys=True))

    if args.dry_run:
        print(
            "TABLE 4 PREDICTION DRY RUN PASSED: "
            "no checkpoint was loaded."
        )
        return

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference requested but CUDA is unavailable."
        )

    device = torch.device("cuda")

    if args.model in {"minilm", "deberta"}:
        probabilities = encoder_probabilities(
            frame=frame,
            checkpoint=args.checkpoint,
            model_name=args.model,
            batch_size=batch_size,
            device=device,
        )
    else:
        probabilities = qwen_probabilities(
            frame=frame,
            checkpoint=args.checkpoint,
            batch_size=batch_size,
            device=device,
        )

    if len(probabilities) != len(frame):
        raise RuntimeError(
            "Prediction count does not match evaluation rows."
        )

    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities generated.")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("Probabilities outside [0,1].")

    output = frame[
        [
            "source_row_id",
            "query_id",
            "candidate_type",
            "evaluation_condition",
        ]
    ].copy()

    output["training_condition"] = args.condition
    output["seed"] = args.seed
    output["cited_probability"] = probabilities
    output["prediction_row_id"] = [
        (
            f"{args.model}:{args.condition}:"
            f"seed{args.seed}:{index:06d}"
        )
        for index in range(len(output))
    ]

    ordered_columns = [
        "source_row_id",
        "query_id",
        "candidate_type",
        "evaluation_condition",
        "training_condition",
        "seed",
        "cited_probability",
        "prediction_row_id",
    ]
    output = output[ordered_columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(
        args.output,
        index=False,
        compression="zstd",
    )

    config_path = args.output.with_suffix(
        ".prediction_config.json"
    )

    run_config["completed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    run_config["output_sha256"] = sha256_file(args.output)
    run_config["probability_min"] = float(
        probabilities.min()
    )
    run_config["probability_max"] = float(
        probabilities.max()
    )
    run_config["probability_mean"] = float(
        probabilities.mean()
    )

    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("TABLE 4 ORDERED PREDICTION COMPLETE")
    print("Rows:", len(output))
    print("Output:", args.output)
    print("Config:", config_path)


if __name__ == "__main__":
    main()
