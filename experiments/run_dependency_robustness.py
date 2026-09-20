# ============================================================
# XART-H Dependency Robustness Experiment
# L4-optimized DeBERTa X/A classification
#
# Conditions:
#   1. official dependency-controlled training split
#   2. relaxed row-level temporal training split
#
# Evaluation:
#   frozen official X/A test set
#
# Models:
#   microsoft/deberta-v3-base
#
# Seeds:
#   13, 42, 77
# ============================================================

import os
import re
import gc
import json
import time
import math
import random
import hashlib
import shutil
import warnings
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import transformers
import accelerate
import datasets

from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from google.colab import drive

warnings.filterwarnings("ignore")

# ============================================================
# 0. Environment verification
# ============================================================

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)
print("torchvision spec:", importlib.util.find_spec("torchvision"))

assert transformers.__version__ == "4.51.3"
assert importlib.util.find_spec("torchvision") is None, (
    "torchvision is still installed. Remove it and restart the runtime."
)
assert torch.cuda.is_available(), "Enable an L4 GPU runtime."

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)

print("GPU:", GPU_NAME)
print(f"GPU memory: {GPU_MEMORY_GB:.1f} GB")
print("CUDA:", torch.version.cuda)

assert "L4" in GPU_NAME.upper(), (
    f"This configuration was prepared for L4, but detected: {GPU_NAME}"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================================================
# 1. Mount Drive and configuration
# ============================================================

drive.mount("/content/drive")

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/XART-H"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "dependency_robustness_v1"
)

OUTPUT_DIR = AUDIT_DIR / "model_comparison_l4"
CACHE_DIR = OUTPUT_DIR / "cache"
RUN_DIR = OUTPUT_DIR / "runs"

for directory in [AUDIT_DIR, OUTPUT_DIR, CACHE_DIR, RUN_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"
MODEL_ID = "microsoft/deberta-v3-base"

SEEDS = [13, 42, 77]

MAX_LENGTH = 384
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

# Safe L4 configuration
TRAIN_BATCH_SIZE = 48
EVAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
EFFECTIVE_BATCH_SIZE = (
    TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95

# Cutoffs used by the row-level temporal reconstruction audit
TRAIN_END_DATE = pd.Timestamp("2016-09-25")
VALIDATION_END_DATE = pd.Timestamp("2017-05-13")

EXPECTED_COUNTS = {
    "official_train": 16912,
    "official_validation": 3626,
    "official_test": 3638,
    "quarantine": 1100,
    "relaxed_train": 17432,
    "relaxed_validation": 3894,
    "relaxed_test": 3950,
}

# If automatic discovery fails, enter the quarantine or master
# Parquet path here.
QUARANTINE_OR_MASTER_PATH = (
    "/content/drive/MyDrive/PatentSearchBench/XART-H/"
    "experiments/dependency_robustness_v1/"
    "dependency_quarantine_1100.parquet"
)

print("\nConfiguration")
print("Model:", MODEL_ID)
print("Seeds:", SEEDS)
print("Maximum length:", MAX_LENGTH)
print("Training batch:", TRAIN_BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
print("Effective batch:", EFFECTIVE_BATCH_SIZE)
print("Evaluation batch:", EVAL_BATCH_SIZE)
print("Precision: BF16")
print("Output:", OUTPUT_DIR)

# ============================================================
# 2. Utility functions
# ============================================================

def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_label_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype(int)

    mapping = {
        "a": 0,
        "x": 1,
        "u": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        bad = series[converted.isna()].astype(str).unique()[:10]
        raise ValueError(f"Unrecognized labels: {bad}")

    return converted.astype(int)


def choose_date_column(frame):
    for column in ["_date", "_parsed_date", "date"]:
        if column in frame.columns:
            return column

    raise ValueError(
        "No temporal column found. Expected _date, "
        "_parsed_date, or date."
    )


def parse_dates(series):
    text = series.astype(str).str.strip()

    parsed_compact = pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce",
    )

    parsed_general = pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    parsed = parsed_compact.fillna(parsed_general)

    if parsed.isna().any():
        examples = text[parsed.isna()].head(10).tolist()
        raise ValueError(f"Unparseable dates: {examples}")

    return parsed


def sha256_frame(frame, columns):
    usable = [column for column in columns if column in frame.columns]

    values = pd.util.hash_pandas_object(
        frame[usable].reset_index(drop=True),
        index=True,
    ).values

    return hashlib.sha256(values.tobytes()).hexdigest()


def safe_json_value(value):
    if isinstance(value, dict):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return [safe_json_value(item) for item in value.tolist()]

    return value


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "f1_a": float(
            f1_score(
                labels == 0,
                predictions == 0,
                zero_division=0,
            )
        ),
        "f1_x": float(
            f1_score(
                labels == 1,
                predictions == 1,
                zero_division=0,
            )
        ),
    }


def trainer_metrics(eval_prediction):
    logits, labels = eval_prediction
    probabilities = softmax_numpy(logits)[:, 1]
    return compute_binary_metrics(labels, probabilities)


def percentile_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    alpha = 1.0 - confidence

    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1.0 - alpha / 2)),
    ]


def empirical_two_sided_pvalue(differences):
    differences = np.asarray(differences, dtype=np.float64)
    n = len(differences)

    less_equal_zero = (
        np.sum(differences <= 0) + 1
    ) / (n + 1)

    greater_equal_zero = (
        np.sum(differences >= 0) + 1
    ) / (n + 1)

    return float(min(1.0, 2.0 * min(
        less_equal_zero,
        greater_equal_zero,
    )))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    number = len(pvalues)
    order = np.argsort(pvalues)

    adjusted = np.empty(number, dtype=np.float64)
    running_maximum = 0.0

    for rank, original_index in enumerate(order):
        candidate = (number - rank) * pvalues[original_index]
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)

    return adjusted


# ============================================================
# 3. Load frozen official XART-H data
# ============================================================

print("\nLoading frozen XART-H dataset...")

raw_dataset = load_dataset(
    REPO_ID,
    revision=REVISION,
)

official_frames = {}

for split in ["train", "validation", "test"]:
    frame = raw_dataset[split].to_pandas()
    frame["label"] = normalize_label_series(frame["label"])

    # Dependency comparison is restricted to X/A rows.
    frame = frame[frame["label"].isin([0, 1])].copy()
    frame["_official_split"] = split
    frame = frame.reset_index(drop=True)

    official_frames[split] = frame

official_train = official_frames["train"]
official_validation = official_frames["validation"]
official_test = official_frames["test"]

print("\nOfficial X/A counts")
print("train:", len(official_train))
print("validation:", len(official_validation))
print("test:", len(official_test))

assert len(official_train) == EXPECTED_COUNTS["official_train"]
assert len(official_validation) == EXPECTED_COUNTS["official_validation"]
assert len(official_test) == EXPECTED_COUNTS["official_test"]

for split_name, frame in official_frames.items():
    print(
        split_name,
        frame["label"].value_counts().sort_index().to_dict(),
    )

official_all = pd.concat(
    [
        official_train,
        official_validation,
        official_test,
    ],
    ignore_index=True,
)

assert "_row_id" in official_all.columns
official_row_ids = set(
    official_all["_row_id"].astype(str).tolist()
)

# ============================================================
# 4. Recover the 1,100 quarantined rows
# ============================================================

REQUIRED_SOURCE_COLUMNS = {
    "text",
    "text_b",
    "label",
}


def candidate_source_files(root):
    patterns = [
        "**/*quarantine*.parquet",
        "**/*master*.parquet",
        "**/*controlled*.parquet",
        "**/*dependency*.parquet",
        "**/*split*.parquet",
        "**/*.parquet",
    ]

    found = []
    seen = set()

    excluded_terms = [
        "model_comparison_l4",
        "/cache/",
        "/runs/",
        "random_u",
        "temporal_robustness_v1",
    ]

    for pattern in patterns:
        for path in root.glob(pattern):
            path_string = str(path).lower().replace("\\", "/")

            if any(term in path_string for term in excluded_terms):
                continue

            if path in seen or not path.is_file():
                continue

            seen.add(path)
            found.append(path)

    return found


def inspect_parquet(path):
    try:
        schema = pq.read_schema(path)
        names = set(schema.names)

        if not REQUIRED_SOURCE_COLUMNS.issubset(names):
            return None

        metadata = pq.ParquetFile(path).metadata
        return {
            "path": path,
            "rows": metadata.num_rows,
            "columns": names,
        }

    except Exception:
        return None


def extract_quarantine(frame, source_path):
    frame = frame.copy()

    if not REQUIRED_SOURCE_COLUMNS.issubset(frame.columns):
        return None

    frame["label"] = normalize_label_series(frame["label"])
    frame = frame[frame["label"].isin([0, 1])].copy()

    # Preferred method: explicit quarantine marker.
    for split_column in [
        "_benchmark_split",
        "benchmark_split",
        "split",
    ]:
        if split_column in frame.columns:
            mask = (
                frame[split_column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("quarantine")
            )

            selected = frame[mask].copy()

            if len(selected) == EXPECTED_COUNTS["quarantine"]:
                return selected

    # A master table can be identified by rows absent from
    # the frozen official dataset.
    if "_row_id" in frame.columns:
        row_ids = frame["_row_id"].astype(str)
        selected = frame[
            ~row_ids.isin(official_row_ids)
        ].copy()

        selected = selected.drop_duplicates(
            subset=["_row_id"],
            keep="first",
        )

        if len(selected) == EXPECTED_COUNTS["quarantine"]:
            return selected

    # A dedicated quarantine file may contain exactly 1,100 rows.
    if (
        "quarantine" in source_path.name.lower()
        and len(frame) == EXPECTED_COUNTS["quarantine"]
    ):
        return frame.copy()

    return None


quarantine = None
quarantine_source = None

if QUARANTINE_OR_MASTER_PATH is not None:
    manual_path = Path(QUARANTINE_OR_MASTER_PATH)

    if not manual_path.exists():
        raise FileNotFoundError(manual_path)

    manual_frame = pd.read_parquet(manual_path)
    quarantine = extract_quarantine(manual_frame, manual_path)
    quarantine_source = manual_path

else:
    possible_files = candidate_source_files(PROJECT_ROOT)
    inspected_files = []

    print(
        f"\nSearching {len(possible_files)} local Parquet files "
        "for the quarantined rows..."
    )

    for position, path in enumerate(possible_files, start=1):
        information = inspect_parquet(path)

        if information is None:
            continue

        inspected_files.append(
            (
                str(path),
                information["rows"],
            )
        )

        try:
            candidate_frame = pd.read_parquet(path)
            selected = extract_quarantine(candidate_frame, path)

            if selected is not None:
                quarantine = selected
                quarantine_source = path
                break

        except Exception as error:
            print("Skipped:", path, type(error).__name__)

if quarantine is None:
    print("\nParquet candidates containing text/text_b/label:")
    for path, rows in inspected_files[:50]:
        print(f"{rows:>8}  {path}")

    raise FileNotFoundError(
        "\nCould not automatically locate the 1,100 quarantined "
        "X/A rows. Set QUARANTINE_OR_MASTER_PATH near the top of "
        "the script to the quarantine Parquet or complete master "
        "Parquet used by the dependency audit."
    )

quarantine = quarantine.reset_index(drop=True)
quarantine["label"] = normalize_label_series(quarantine["label"])

assert len(quarantine) == EXPECTED_COUNTS["quarantine"]
assert set(quarantine["label"].unique()).issubset({0, 1})

print("\nQuarantine source:", quarantine_source)
print("Quarantine rows:", len(quarantine))
print(
    "Quarantine labels:",
    quarantine["label"].value_counts().sort_index().to_dict(),
)

# ============================================================
# 5. Reconstruct relaxed row-level temporal partitions
# ============================================================

common_columns = sorted(
    set(official_all.columns).union(quarantine.columns)
)

source_all = pd.concat(
    [
        official_all.reindex(columns=common_columns),
        quarantine.reindex(columns=common_columns),
    ],
    ignore_index=True,
)

if "_row_id" in source_all.columns:
    duplicate_count = source_all["_row_id"].astype(str).duplicated().sum()

    if duplicate_count:
        print("Removing duplicated row IDs:", duplicate_count)

        source_all = source_all.drop_duplicates(
            subset=["_row_id"],
            keep="first",
        )

source_all["label"] = normalize_label_series(source_all["label"])
source_all = source_all[
    source_all["label"].isin([0, 1])
].copy()

date_column = choose_date_column(source_all)
source_all["_experiment_date"] = parse_dates(
    source_all[date_column]
)

source_all["_relaxed_split"] = np.select(
    [
        source_all["_experiment_date"] <= TRAIN_END_DATE,
        source_all["_experiment_date"] <= VALIDATION_END_DATE,
    ],
    [
        "train",
        "validation",
    ],
    default="test",
)

relaxed_train = source_all[
    source_all["_relaxed_split"] == "train"
].copy().reset_index(drop=True)

relaxed_validation = source_all[
    source_all["_relaxed_split"] == "validation"
].copy().reset_index(drop=True)

relaxed_test = source_all[
    source_all["_relaxed_split"] == "test"
].copy().reset_index(drop=True)

print("\nRelaxed row-level temporal counts")
print("train:", len(relaxed_train))
print("validation:", len(relaxed_validation))
print("test:", len(relaxed_test))
print("total:", len(source_all))
print("date column:", date_column)
print(
    "date range:",
    source_all["_experiment_date"].min(),
    "to",
    source_all["_experiment_date"].max(),
)

assert len(relaxed_train) == EXPECTED_COUNTS["relaxed_train"], (
    f"Expected {EXPECTED_COUNTS['relaxed_train']} relaxed train rows, "
    f"found {len(relaxed_train)}."
)

assert len(relaxed_validation) == EXPECTED_COUNTS["relaxed_validation"], (
    f"Expected {EXPECTED_COUNTS['relaxed_validation']} relaxed "
    f"validation rows, found {len(relaxed_validation)}."
)

assert len(relaxed_test) == EXPECTED_COUNTS["relaxed_test"], (
    f"Expected {EXPECTED_COUNTS['relaxed_test']} relaxed test rows, "
    f"found {len(relaxed_test)}."
)

# Both conditions are evaluated on exactly the same frozen official test.
conditions = {
    "official": {
        "train": official_train.copy(),
        "validation": official_validation.copy(),
    },
    "relaxed": {
        "train": relaxed_train.copy(),
        "validation": relaxed_validation.copy(),
    },
}

evaluation_frame = official_test.copy().reset_index(drop=True)

for condition_name, partitions in conditions.items():
    for split_name, frame in partitions.items():
        assert frame["label"].isin([0, 1]).all()
        assert frame["text"].notna().all()
        assert frame["text_b"].notna().all()

        print(
            condition_name,
            split_name,
            len(frame),
            frame["label"].value_counts().sort_index().to_dict(),
        )

print(
    "Official evaluation:",
    len(evaluation_frame),
    evaluation_frame["label"].value_counts().sort_index().to_dict(),
)

# ============================================================
# 6. Fingerprints and experiment metadata
# ============================================================

fingerprint_columns = [
    "_row_id",
    "text",
    "text_b",
    "label",
]

fingerprints = {
    "official_train": sha256_frame(
        official_train,
        fingerprint_columns,
    ),
    "official_validation": sha256_frame(
        official_validation,
        fingerprint_columns,
    ),
    "official_test": sha256_frame(
        evaluation_frame,
        fingerprint_columns,
    ),
    "relaxed_train": sha256_frame(
        relaxed_train,
        fingerprint_columns,
    ),
    "relaxed_validation": sha256_frame(
        relaxed_validation,
        fingerprint_columns,
    ),
    "relaxed_test": sha256_frame(
        relaxed_test,
        fingerprint_columns,
    ),
    "quarantine": sha256_frame(
        quarantine,
        fingerprint_columns,
    ),
}

experiment_config = {
    "experiment": "dependency relaxation model comparison",
    "dataset": REPO_ID,
    "revision": REVISION,
    "model": MODEL_ID,
    "task": "X versus A",
    "conditions": ["official", "relaxed"],
    "evaluation_split": "frozen official X/A test",
    "seeds": SEEDS,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "maximum_length": MAX_LENGTH,
    "training_batch_size": TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "evaluation_batch_size": EVAL_BATCH_SIZE,
    "precision": "bfloat16",
    "gradient_checkpointing": True,
    "gpu": GPU_NAME,
    "gpu_memory_gb": GPU_MEMORY_GB,
    "train_end_date": str(TRAIN_END_DATE.date()),
    "validation_end_date": str(VALIDATION_END_DATE.date()),
    "bootstrap_repetitions": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "bootstrap_unit": (
        "lowercased and whitespace-normalized claim text"
    ),
    "quarantine_source": str(quarantine_source),
    "counts": {
        "official_train": len(official_train),
        "official_validation": len(official_validation),
        "official_test": len(evaluation_frame),
        "quarantine": len(quarantine),
        "relaxed_train": len(relaxed_train),
        "relaxed_validation": len(relaxed_validation),
        "relaxed_test": len(relaxed_test),
    },
    "fingerprints": fingerprints,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

with open(
    OUTPUT_DIR / "dependency_model_config.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        safe_json_value(experiment_config),
        handle,
        indent=2,
        ensure_ascii=False,
    )

# ============================================================
# 7. Tokenizer and tokenized datasets
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


def make_hf_dataset(frame):
    prepared = pd.DataFrame({
        "text": frame["text"].astype(str),
        "text_b": frame["text_b"].astype(str),
        "labels": frame["label"].astype(int),
    })

    dataset = Dataset.from_pandas(
        prepared,
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            batch["text_b"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=512,
        num_proc=2,
        remove_columns=["text", "text_b"],
        desc="Tokenizing",
    )

    return dataset


print("\nTokenizing datasets...")

tokenized = {
    "official": {
        "train": make_hf_dataset(official_train),
        "validation": make_hf_dataset(official_validation),
    },
    "relaxed": {
        "train": make_hf_dataset(relaxed_train),
        "validation": make_hf_dataset(relaxed_validation),
    },
    "test": make_hf_dataset(evaluation_frame),
}

# ============================================================
# 8. Train and predict
# ============================================================

seed_records = []
prediction_files = {}


def run_cache_path(condition, seed):
    return CACHE_DIR / f"{condition}_seed_{seed}_predictions.npz"


def validate_cached_prediction(cache, condition, seed):
    required = {
        "probabilities",
        "labels",
        "condition",
        "seed",
        "test_fingerprint",
    }

    if not required.issubset(cache.files):
        return False

    if str(cache["condition"].item()) != condition:
        return False

    if int(cache["seed"].item()) != seed:
        return False

    if (
        str(cache["test_fingerprint"].item())
        != fingerprints["official_test"]
    ):
        return False

    if len(cache["probabilities"]) != len(evaluation_frame):
        return False

    return True


def train_and_predict(condition, seed):
    cache_path = run_cache_path(condition, seed)

    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)

            if validate_cached_prediction(
                cached,
                condition,
                seed,
            ):
                probabilities = cached[
                    "probabilities"
                ].astype(np.float64)

                labels = cached["labels"].astype(np.int64)
                metrics = compute_binary_metrics(
                    labels,
                    probabilities,
                )

                elapsed = float(
                    cached["elapsed_seconds"].item()
                )

                print(
                    f"\nUsing valid cache: {condition}, seed {seed}"
                )

                return probabilities, metrics, elapsed

        except Exception as error:
            print("Invalid cache removed:", cache_path, error)
            cache_path.unlink(missing_ok=True)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_output = RUN_DIR / f"{condition}_seed_{seed}"
    run_output.mkdir(parents=True, exist_ok=True)

    clear_memory()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        problem_type="single_label_classification",
    )

    model.config.id2label = {
        0: "A",
        1: "X",
    }
    model.config.label2id = {
        "A": 0,
        "X": 1,
    }
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    training_arguments = TrainingArguments(
        output_dir=str(run_output),
        overwrite_output_dir=True,

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        bf16=True,
        fp16=False,
        tf32=True,

        gradient_checkpointing=True,
        optim="adamw_torch_fused",

        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        report_to=[],
        disable_tqdm=False,

        seed=seed,
        data_seed=seed,

        remove_unused_columns=True,
        prediction_loss_only=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized[condition]["train"],
        eval_dataset=tokenized[condition]["validation"],
        data_collator=data_collator,
        compute_metrics=trainer_metrics,
    )

    print("\n" + "=" * 70)
    print("TRAINING")
    print("Condition:", condition)
    print("Seed:", seed)
    print("Train rows:", len(tokenized[condition]["train"]))
    print(
        "Validation rows:",
        len(tokenized[condition]["validation"]),
    )
    print("=" * 70)

    start_time = time.time()

    trainer.train()

    prediction_output = trainer.predict(
        tokenized["test"]
    )

    elapsed_seconds = time.time() - start_time

    probabilities = softmax_numpy(
        prediction_output.predictions
    )[:, 1]

    labels = np.asarray(
        prediction_output.label_ids,
        dtype=np.int64,
    )

    metrics = compute_binary_metrics(
        labels,
        probabilities,
    )

    np.savez_compressed(
        cache_path,
        probabilities=probabilities.astype(np.float32),
        labels=labels.astype(np.int8),
        condition=np.asarray(condition),
        seed=np.asarray(seed),
        elapsed_seconds=np.asarray(elapsed_seconds),
        test_fingerprint=np.asarray(
            fingerprints["official_test"]
        ),
    )

    print("Test metrics:", metrics)
    print(
        f"Elapsed: {elapsed_seconds / 60:.1f} minutes"
    )
    print("Saved cache:", cache_path)

    del trainer
    del model
    del prediction_output

    clear_memory()

    # Checkpoints are not required because predictions are cached.
    if run_output.exists():
        shutil.rmtree(run_output, ignore_errors=True)

    return probabilities, metrics, elapsed_seconds


experiment_start = time.time()

for condition in ["official", "relaxed"]:
    prediction_files[condition] = []

    for seed in SEEDS:
        probabilities, metrics, elapsed_seconds = (
            train_and_predict(condition, seed)
        )

        prediction_files[condition].append(probabilities)

        record = {
            "condition": condition,
            "seed": seed,
            "test_rows": len(evaluation_frame),
            "elapsed_seconds": elapsed_seconds,
            **metrics,
        }

        seed_records.append(record)

seed_results = pd.DataFrame(seed_records)

seed_results.to_csv(
    OUTPUT_DIR / "dependency_seed_results.csv",
    index=False,
)

# ============================================================
# 9. Aggregate seed results
# ============================================================

metric_names = [
    "accuracy",
    "macro_f1",
    "roc_auc",
    "average_precision",
]

summary_rows = []

for condition in ["official", "relaxed"]:
    condition_rows = seed_results[
        seed_results["condition"] == condition
    ]

    for metric in metric_names:
        values = condition_rows[metric].to_numpy(
            dtype=np.float64
        )

        summary_rows.append({
            "condition": condition,
            "metric": metric,
            "mean": float(values.mean()),
            "standard_deviation": float(
                values.std(ddof=1)
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "num_seeds": len(values),
        })

seed_summary = pd.DataFrame(summary_rows)

seed_summary.to_csv(
    OUTPUT_DIR / "dependency_seed_summary.csv",
    index=False,
)

official_ensemble = np.mean(
    np.stack(prediction_files["official"]),
    axis=0,
)

relaxed_ensemble = np.mean(
    np.stack(prediction_files["relaxed"]),
    axis=0,
)

gold_labels = evaluation_frame["label"].to_numpy(
    dtype=np.int64
)

official_ensemble_metrics = compute_binary_metrics(
    gold_labels,
    official_ensemble,
)

relaxed_ensemble_metrics = compute_binary_metrics(
    gold_labels,
    relaxed_ensemble,
)

# ============================================================
# 10. Paired query-level bootstrap
# ============================================================

query_keys = (
    evaluation_frame["text"]
    .map(normalize_text)
    .to_numpy(dtype=object)
)

unique_queries = np.unique(query_keys)

query_indices = {
    query: np.flatnonzero(query_keys == query)
    for query in unique_queries
}

print("\nBootstrap queries:", len(unique_queries))
print("Bootstrap repetitions:", N_BOOTSTRAP)

rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_values = {
    metric: {
        "official": [],
        "relaxed": [],
        "difference": [],
    }
    for metric in metric_names
}

bootstrap_start = time.time()

for repetition in range(N_BOOTSTRAP):
    sampled_queries = rng.choice(
        unique_queries,
        size=len(unique_queries),
        replace=True,
    )

    sampled_indices = np.concatenate([
        query_indices[query]
        for query in sampled_queries
    ])

    sampled_labels = gold_labels[sampled_indices]

    # Extremely unlikely safeguard.
    if len(np.unique(sampled_labels)) < 2:
        continue

    official_metrics = compute_binary_metrics(
        sampled_labels,
        official_ensemble[sampled_indices],
    )

    relaxed_metrics = compute_binary_metrics(
        sampled_labels,
        relaxed_ensemble[sampled_indices],
    )

    for metric in metric_names:
        official_value = official_metrics[metric]
        relaxed_value = relaxed_metrics[metric]

        bootstrap_values[metric]["official"].append(
            official_value
        )
        bootstrap_values[metric]["relaxed"].append(
            relaxed_value
        )
        bootstrap_values[metric]["difference"].append(
            relaxed_value - official_value
        )

    if (
        (repetition + 1) % 250 == 0
        or repetition == 0
    ):
        print(
            f"Bootstrap {repetition + 1}/{N_BOOTSTRAP}"
        )

raw_pvalues = []

for metric in metric_names:
    raw_pvalues.append(
        empirical_two_sided_pvalue(
            bootstrap_values[metric]["difference"]
        )
    )

holm_pvalues = holm_adjust(raw_pvalues)

paired_rows = []

for metric_index, metric in enumerate(metric_names):
    official_bootstrap = np.asarray(
        bootstrap_values[metric]["official"]
    )

    relaxed_bootstrap = np.asarray(
        bootstrap_values[metric]["relaxed"]
    )

    difference_bootstrap = np.asarray(
        bootstrap_values[metric]["difference"]
    )

    point_official = official_ensemble_metrics[metric]
    point_relaxed = relaxed_ensemble_metrics[metric]
    point_difference = point_relaxed - point_official

    difference_ci = percentile_interval(
        difference_bootstrap,
        CONFIDENCE_LEVEL,
    )

    paired_rows.append({
        "metric": metric,
        "official_estimate": point_official,
        "official_ci_low": percentile_interval(
            official_bootstrap,
            CONFIDENCE_LEVEL,
        )[0],
        "official_ci_high": percentile_interval(
            official_bootstrap,
            CONFIDENCE_LEVEL,
        )[1],
        "relaxed_estimate": point_relaxed,
        "relaxed_ci_low": percentile_interval(
            relaxed_bootstrap,
            CONFIDENCE_LEVEL,
        )[0],
        "relaxed_ci_high": percentile_interval(
            relaxed_bootstrap,
            CONFIDENCE_LEVEL,
        )[1],
        "delta_relaxed_minus_official": point_difference,
        "delta_ci_low": difference_ci[0],
        "delta_ci_high": difference_ci[1],
        "raw_p_value": raw_pvalues[metric_index],
        "holm_adjusted_p_value": holm_pvalues[metric_index],
        "significant_after_holm": bool(
            holm_pvalues[metric_index] < 0.05
        ),
    })

paired_tests = pd.DataFrame(paired_rows)

paired_tests.to_csv(
    OUTPUT_DIR / "dependency_paired_tests.csv",
    index=False,
)

# Save row-level predictions for reproducibility.
prediction_table = pd.DataFrame({
    "row_index": np.arange(len(evaluation_frame)),
    "row_id": evaluation_frame["_row_id"].astype(str),
    "normalized_query": query_keys,
    "label": gold_labels,
    "official_ensemble_probability_x": official_ensemble,
    "relaxed_ensemble_probability_x": relaxed_ensemble,
})

prediction_table.to_parquet(
    OUTPUT_DIR / "dependency_test_predictions.parquet",
    index=False,
)

# ============================================================
# 11. Save complete JSON
# ============================================================

total_elapsed = time.time() - experiment_start
bootstrap_elapsed = time.time() - bootstrap_start

complete_results = {
    "experiment": experiment_config,
    "seed_results": seed_records,
    "seed_summary": seed_summary.to_dict(
        orient="records"
    ),
    "ensemble_results": {
        "official": official_ensemble_metrics,
        "relaxed": relaxed_ensemble_metrics,
    },
    "paired_query_bootstrap": {
        "unit": (
            "lowercased and whitespace-normalized claim text"
        ),
        "number_of_queries": len(unique_queries),
        "repetitions": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "multiple_testing": (
            "Holm correction across four metrics"
        ),
        "results": paired_rows,
    },
    "timing": {
        "total_elapsed_seconds": total_elapsed,
        "bootstrap_elapsed_seconds": bootstrap_elapsed,
    },
    "output_files": [
        "dependency_model_config.json",
        "dependency_seed_results.csv",
        "dependency_seed_summary.csv",
        "dependency_paired_tests.csv",
        "dependency_test_predictions.parquet",
        "dependency_model_results.json",
    ],
    "completed_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

with open(
    OUTPUT_DIR / "dependency_model_results.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        safe_json_value(complete_results),
        handle,
        indent=2,
        ensure_ascii=False,
    )

# ============================================================
# 12. Final output
# ============================================================

print("\n")
print("=" * 78)
print("DEPENDENCY RELAXATION MODEL RESULT")
print("=" * 78)

print("\nDataset")
print(f"Official train:      {len(official_train):,}")
print(f"Official validation: {len(official_validation):,}")
print(f"Official test:       {len(evaluation_frame):,}")
print(f"Quarantine:          {len(quarantine):,}")
print(f"Relaxed train:       {len(relaxed_train):,}")
print(f"Relaxed validation:  {len(relaxed_validation):,}")
print(f"Relaxed test:        {len(relaxed_test):,}")

print("\nSeed-level official-test results")
print(
    seed_results[
        [
            "condition",
            "seed",
            "accuracy",
            "macro_f1",
            "roc_auc",
            "average_precision",
            "elapsed_seconds",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nMean ± SD across seeds")

for condition in ["official", "relaxed"]:
    print(f"\n{condition.upper()}")

    condition_summary = seed_summary[
        seed_summary["condition"] == condition
    ]

    for _, row in condition_summary.iterrows():
        print(
            f"{row['metric']:>18}: "
            f"{row['mean']:.6f} ± "
            f"{row['standard_deviation']:.6f}"
        )

print("\nSeed-ensemble paired query-level bootstrap")
print(
    paired_tests.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nInterpretation direction")
print(
    "Positive delta means that dependency-relaxed training "
    "performed better than dependency-controlled training."
)

print("\nSaved files")

for filename in complete_results["output_files"]:
    print(" •", OUTPUT_DIR / filename)

print(
    f"\nTotal elapsed time: {total_elapsed / 60:.1f} minutes"
)
print("=" * 78)
