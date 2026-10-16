#!/usr/bin/env python3

"""Reconstructed PatScout Table 4 controlled-training entry point.

This executable reconstruction consumes the released frozen manifests.
It was reconstructed from the public manifests, configurations, training
metadata, logs, Qwen adapters, and ordered predictions. It is not
represented as the byte-identical original notebook or script.

Models:
- minilm: cross-encoder/ms-marco-MiniLM-L6-v2
- deberta: microsoft/deberta-v3-base
- qwen: Qwen/Qwen3-Reranker-0.6B with LoRA

Conditions:
- hard_only
- mixed_u

Seeds:
- 13
- 42
- 77
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODELS = {
    "minilm": {
        "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": None,
        "learning_rate": 2e-5,
        "epochs": 2,
        "batch_size": 64,
        "gradient_accumulation": 4,
        "precision": "fp16",
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
    },
    "deberta": {
        "model_id": "microsoft/deberta-v3-base",
        "revision": None,
        "learning_rate": 1e-5,
        "epochs": 2,
        "batch_size": 32,
        "gradient_accumulation": 1,
        "precision": "bf16",
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
    },
    "qwen": {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "learning_rate": 1e-4,
        "epochs": 2,
        "batch_size": 12,
        "gradient_accumulation": 21,
        "precision": "bf16",
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
    },
}

SEEDS = {13, 42, 77}
CONDITIONS = {"hard_only", "mixed_u"}
MAX_LENGTH = 384
EXPECTED_ROWS = 24863
EXPECTED_QUERIES = 8246
EXPECTED_CITED = 16617
EXPECTED_U = 8246

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


def norm(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def resolve(columns, candidates, required=True):
    lookup = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    if required:
        raise RuntimeError(
            f"Could not resolve {candidates}; columns={list(columns)}"
        )
    return None


def resolve_schema(frame):
    columns = frame.columns
    schema = {
        "query_id": resolve(
            columns,
            ["query_id", "claim_id", "normalized_query_id", "query_key"],
        ),
        "query_text": resolve(
            columns,
            [
                "query_text", "claim_text", "claim",
                "query", "normalized_claim_text", "text",
            ],
        ),
        "passage_text": resolve(
            columns,
            [
                "text_b", "passage_text", "candidate_text", "document_text",
                "passage", "document", "text",
            ],
        ),
        "label": resolve(
            columns,
            ["label", "binary_label", "target", "is_cited", "y"],
            required=False,
        ),
        "candidate_type": resolve(
            columns,
            [
                "candidate_type", "row_type",
                "evidence_type", "source_type",
            ],
            required=False,
        ),
    }
    if schema["label"] is None and schema["candidate_type"] is None:
        raise RuntimeError("No label or candidate-type column.")
    return schema


def derive_labels(frame, schema):
    if schema["label"] is not None:
        raw = frame[schema["label"]]

        if pd.api.types.is_bool_dtype(raw):
            labels = raw.astype(int)
        elif pd.api.types.is_numeric_dtype(raw):
            numeric = pd.to_numeric(raw, errors="raise").astype(int)
            unique = set(int(x) for x in numeric.unique())
            if unique <= {0, 1}:
                labels = numeric
            elif unique <= {0, 1, 2}:
                # 0=A, 1=X -> CITED; 2=U -> uncited.
                labels = (numeric != 2).astype(int)
            else:
                raise RuntimeError(
                    f"Unexpected numeric labels: {sorted(unique)}"
                )
        else:
            positive = {
                "1", "true", "cited", "citation",
                "positive", "x", "a", "examiner_cited",
            }
            negative = {
                "0", "false", "u", "hard_u",
                "random_u", "uncited", "negative",
            }
            normalized = raw.map(norm)
            unknown = set(normalized.unique()) - positive - negative
            if unknown:
                raise RuntimeError(
                    f"Unexpected labels: {sorted(unknown)}"
                )
            labels = normalized.map(
                lambda x: 1 if x in positive else 0
            ).astype(int)
    else:
        negatives = {
            "u", "hard_u", "random_u",
            "dense_hard_u", "uncited", "negative",
        }
        labels = frame[schema["candidate_type"]].map(
            lambda x: 0 if norm(x) in negatives else 1
        ).astype(int)

    if not set(labels.unique()) <= {0, 1}:
        raise RuntimeError("Training labels are not binary.")

    return labels


def validate_frame(frame, schema):
    labels = derive_labels(frame, schema)
    queries = (
        frame[schema["query_id"]]
        .astype(str)
        .str.strip()
        .nunique()
    )
    counts = labels.value_counts().to_dict()

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, found {len(frame)}."
        )
    if queries != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries, found {queries}."
        )
    if counts.get(1, 0) != EXPECTED_CITED:
        raise RuntimeError(
            f"Expected {EXPECTED_CITED} cited rows; found {counts}."
        )
    if counts.get(0, 0) != EXPECTED_U:
        raise RuntimeError(
            f"Expected {EXPECTED_U} U rows; found {counts}."
        )
    if frame[schema["query_text"]].isna().any():
        raise RuntimeError("Null query text found.")
    if frame[schema["passage_text"]].isna().any():
        raise RuntimeError("Null passage text found.")

    return labels


class EncoderPairDataset(Dataset):
    def __init__(self, frame, schema, labels, tokenizer):
        self.queries = frame[schema["query_text"]].astype(str).tolist()
        self.passages = frame[schema["passage_text"]].astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.queries[index],
            self.passages[index],
            truncation="longest_first",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = self.labels[index]
        return encoded



def balanced_truncate(query_ids, document_ids, available):
    """Apply the balanced truncation used by the canonical Qwen run."""
    query_ids = list(query_ids)
    document_ids = list(document_ids)

    if available < 0:
        raise RuntimeError("Negative Qwen content-token budget.")

    if len(query_ids) + len(document_ids) <= available:
        return query_ids, document_ids

    query_cap = min(len(query_ids), available // 2)
    document_cap = min(
        len(document_ids),
        available - (available // 2),
    )

    remaining = available - query_cap - document_cap

    if remaining > 0:
        add_query = min(
            remaining,
            len(query_ids) - query_cap,
        )
        query_cap += add_query
        remaining -= add_query

    if remaining > 0:
        add_document = min(
            remaining,
            len(document_ids) - document_cap,
        )
        document_cap += add_document

    return (
        query_ids[:query_cap],
        document_ids[:document_cap],
    )


class QwenPairDataset(Dataset):
    def __init__(self, frame, schema, labels, tokenizer):
        self.queries = (
            frame[schema["query_text"]].astype(str).tolist()
        )
        self.passages = (
            frame[schema["passage_text"]].astype(str).tolist()
        )
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer

        self.prefix_tokens = tokenizer.encode(
            QWEN_PREFIX,
            add_special_tokens=False,
        )
        self.suffix_tokens = tokenizer.encode(
            QWEN_SUFFIX,
            add_special_tokens=False,
        )
        self.fixed_1_tokens = tokenizer.encode(
            f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: ",
            add_special_tokens=False,
        )
        self.fixed_2_tokens = tokenizer.encode(
            "\n<Document>: ",
            add_special_tokens=False,
        )

        self.available = (
            MAX_LENGTH
            - len(self.prefix_tokens)
            - len(self.fixed_1_tokens)
            - len(self.fixed_2_tokens)
            - len(self.suffix_tokens)
        )

        if self.available <= 0:
            raise RuntimeError(
                "Qwen prompt overhead exceeds maximum length."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        query_ids = self.tokenizer.encode(
            self.queries[index],
            add_special_tokens=False,
        )
        document_ids = self.tokenizer.encode(
            self.passages[index],
            add_special_tokens=False,
        )

        query_ids, document_ids = balanced_truncate(
            query_ids,
            document_ids,
            self.available,
        )

        input_ids = (
            self.prefix_tokens
            + self.fixed_1_tokens
            + query_ids
            + self.fixed_2_tokens
            + document_ids
            + self.suffix_tokens
        )

        if len(input_ids) > MAX_LENGTH:
            raise RuntimeError(
                f"Qwen sequence exceeds {MAX_LENGTH} tokens."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": self.labels[index],
        }


class MiniLMBinaryTrainer(Trainer):
    """Train the native one-logit MiniLM cross-encoder with BCE."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError(
                "MiniLM must produce [batch,1] logits; "
                f"found {tuple(logits.shape)}"
            )

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0].float(),
            labels,
        )

        if return_outputs:
            return loss, outputs
        return loss

class QwenBinaryTrainer(Trainer):
    def __init__(
        self,
        *args,
        false_token_id,
        true_token_id,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.false_token_id = int(false_token_id)
        self.true_token_id = int(true_token_id)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        final_logits = outputs.logits[:, -1, :]

        binary_logits = torch.stack(
            [
                final_logits[:, self.false_token_id],
                final_logits[:, self.true_token_id],
            ],
            dim=1,
        )

        loss = torch.nn.functional.cross_entropy(
            binary_logits.float(),
            labels,
        )

        if return_outputs:
            return loss, {
                "logits": binary_logits,
                "model_outputs": outputs,
            }
        return loss


def add_qwen_lora(model):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Qwen training requires peft. Install the published "
            "table4_qwen environment."
        ) from exc

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


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
        type=int,
        choices=sorted(SEEDS),
        required=True,
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved run without training.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
    )
    args = parser.parse_args()

    config = MODELS[args.model]
    manifest = (
        args.manifests
        / f"{args.condition}_train_manifest.parquet"
    )

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    output = (
        args.output_root
        / args.model
        / args.condition
        / f"seed{args.seed}"
    )

    if output.exists() and any(output.iterdir()):
        if not args.overwrite_output:
            raise RuntimeError(
                f"Output already exists and is not empty: {output}"
            )

    frame = pd.read_parquet(manifest)
    schema = resolve_schema(frame)
    labels = validate_frame(frame, schema)

    resolved = {
        "model": args.model,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "condition": args.condition,
        "seed": args.seed,
        "manifest": str(manifest),
        "manifest_bytes": manifest.stat().st_size,
        "manifest_sha256": sha256_file(manifest),
        "rows": len(frame),
        "queries": int(
            frame[schema["query_id"]]
            .astype(str)
            .str.strip()
            .nunique()
        ),
        "label_counts": {
            "cited": int((labels == 1).sum()),
            "u": int((labels == 0).sum()),
        },
        "schema": schema,
        "maximum_length": MAX_LENGTH,
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "per_device_train_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config[
            "gradient_accumulation"
        ],
        "effective_batch_size": (
            config["batch_size"]
            * config["gradient_accumulation"]
        ),
        "precision": config["precision"],
        "warmup_ratio": config["warmup_ratio"],
        "weight_decay": config["weight_decay"],
        "output": str(output),
        "provenance_status": (
            "reconstructed_executable_from_frozen_public_artifacts"
        ),
    }

    print(json.dumps(resolved, indent=2, sort_keys=True))

    if args.dry_run:
        print("DRY RUN COMPLETE: no model was loaded or trained.")
        return

    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    tokenizer_kwargs = {}
    if args.model == "qwen":
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model == "qwen":
        tokenizer.padding_side = "left"

        false_token_id = tokenizer.convert_tokens_to_ids("no")
        true_token_id = tokenizer.convert_tokens_to_ids("yes")

        if false_token_id != 2152 or true_token_id != 9693:
            raise RuntimeError(
                "Qwen canonical token IDs do not match. "
                f"no={false_token_id}, yes={true_token_id}"
            )

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = add_qwen_lora(model)
        dataset = QwenPairDataset(
            frame,
            schema,
            labels,
            tokenizer,
        )
    elif args.model == "minilm":
        # The released cross-encoder has a native one-logit head.
        # Keep that architecture and train it with BCEWithLogitsLoss.
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            num_labels=1,
        )
        dataset = EncoderPairDataset(
            frame,
            schema,
            labels,
            tokenizer,
        )
    else:
        id2label = {0: "U", 1: "CITED"}
        label2id = {"U": 0, "CITED": 1}

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        dataset = EncoderPairDataset(
            frame,
            schema,
            labels,
            tokenizer,
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        output_dir=str(output),
        overwrite_output_dir=args.overwrite_output,
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        fp16=(config["precision"] == "fp16"),
        bf16=(config["precision"] == "bf16"),
        tf32=True,
        group_by_length=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=25,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": collator,
    }

    if args.model == "qwen":
        trainer = QwenBinaryTrainer(
            **trainer_kwargs,
            false_token_id=false_token_id,
            true_token_id=true_token_id,
        )
    elif args.model == "minilm":
        trainer = MiniLMBinaryTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    resolved["completed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    resolved["train_metrics"] = train_result.metrics
    resolved["software"] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    try:
        import transformers
        resolved["software"]["transformers"] = (
            transformers.__version__
        )
    except Exception:
        pass

    try:
        import peft
        resolved["software"]["peft"] = peft.__version__
    except Exception:
        pass

    (output / "controlled_training_run.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("CONTROLLED TRAINING COMPLETE")
    print("Output:", output)


if __name__ == "__main__":
    main()
