"""Reproducibility script exported from the executed XART-H notebook."""

import gc, re, json, math, random, inspect, torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
from google.colab import drive

# =========================================================
# 0. Clean previous failed run
# =========================================================
for variable in ["trainer", "model", "train_data"]:
    if variable in globals():
        del globals()[variable]

gc.collect()
torch.cuda.empty_cache()

drive.mount("/content/drive")

REPO = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

OUT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/supervised_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "GPU runtime을 활성화하세요."

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================================================
# 1. Configuration
# =========================================================
SEEDS = [13, 42, 77]

TASKS = {
    "cited_vs_u": 2,
    "x_vs_a": 2,
    "direct_3way": 3
}

MODELS = {
    "minilm": {
        "id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "batch": 256,
        "eval_batch": 512,
        "epochs": 2,
        "lr": 2e-5
    },
    "deberta": {
        "id": "microsoft/deberta-v3-base",
        "batch": 32,
        "eval_batch": 64,
        "epochs": 2,
        "lr": 1e-5
    }
}

MAX_LENGTH = 384

run_config = {
    "dataset": REPO,
    "revision": REVISION,
    "seeds": SEEDS,
    "tasks": TASKS,
    "models": MODELS,
    "max_length": MAX_LENGTH,
    "gpu": torch.cuda.get_device_name(0)
}

with open(OUT / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

# =========================================================
# 2. Load dataset
# =========================================================
def norm(text):
    return re.sub(
        r"\s+", " ", str(text)
    ).strip().lower()

raw = load_dataset(
    REPO,
    revision=REVISION
)

gold = {}
eligible = {}

for split in ["train", "validation", "test"]:
    labels = np.asarray(
        raw[split]["label"],
        dtype=np.int64
    )

    queries = np.asarray(
        [norm(x) for x in raw[split]["text"]],
        dtype=object
    )

    eligible_queries = {
        query
        for query, label in zip(queries, labels)
        if label == 2
    }

    gold[split] = {
        "labels": labels,
        "queries": queries
    }

    eligible[split] = np.asarray(
        [
            query in eligible_queries
            for query in queries
        ],
        dtype=bool
    )

print(raw)

# =========================================================
# 3. Task construction
# =========================================================
def task_indices(split, task):
    labels = gold[split]["labels"]

    if task == "cited_vs_u":
        indices = np.flatnonzero(
            eligible[split]
        )
        task_labels = (
            labels[indices] != 2
        ).astype(np.int64)

    elif task == "x_vs_a":
        indices = np.flatnonzero(
            labels != 2
        )
        task_labels = labels[
            indices
        ].astype(np.int64)

    elif task == "direct_3way":
        indices = np.arange(len(labels))
        task_labels = labels.astype(
            np.int64
        )

    else:
        raise ValueError(task)

    return indices, task_labels

# =========================================================
# 4. Metric functions
# =========================================================
def softmax(logits):
    logits = logits - logits.max(
        axis=1,
        keepdims=True
    )
    exp = np.exp(logits)
    return exp / exp.sum(
        axis=1,
        keepdims=True
    )

def classification_metrics(
    labels,
    predictions,
    num_classes
):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    accuracy = float(
        np.mean(labels == predictions)
    )

    per_class_f1 = []

    for class_id in range(num_classes):
        tp = np.sum(
            (predictions == class_id)
            & (labels == class_id)
        )
        fp = np.sum(
            (predictions == class_id)
            & (labels != class_id)
        )
        fn = np.sum(
            (predictions != class_id)
            & (labels == class_id)
        )

        denominator = (
            2 * tp + fp + fn
        )

        f1 = (
            float(2 * tp / denominator)
            if denominator
            else 0.0
        )
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(
            np.mean(per_class_f1)
        ),
        "per_class_f1": per_class_f1
    }

def binary_auc(labels, scores):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    num_positive = int(labels.sum())
    num_negative = (
        len(labels) - num_positive
    )

    order = np.argsort(
        scores,
        kind="mergesort"
    )
    sorted_scores = scores[order]

    ranks = np.empty(
        len(scores),
        dtype=np.float64
    )

    start = 0

    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end]
            == sorted_scores[start]
        ):
            end += 1

        ranks[order[start:end]] = (
            start + 1 + end
        ) / 2

        start = end

    auc = (
        ranks[labels == 1].sum()
        - num_positive
        * (num_positive + 1)
        / 2
    ) / (
        num_positive
        * num_negative
    )

    return float(auc)

def average_precision(labels, scores):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )
    scores = np.asarray(scores)

    order = np.argsort(-scores)
    ranked_labels = labels[order]
    num_positive = ranked_labels.sum()

    if num_positive == 0:
        return 0.0

    precision = (
        np.cumsum(ranked_labels)
        / np.arange(
            1,
            len(ranked_labels) + 1
        )
    )

    return float(
        (
            precision
            * ranked_labels
        ).sum()
        / num_positive
    )

def ranking_metrics(
    labels,
    queries,
    probabilities
):
    groups = {}

    for label, query, probability in zip(
        labels,
        queries,
        probabilities
    ):
        groups.setdefault(query, []).append(
            (
                int(label),
                np.asarray(
                    probability,
                    dtype=np.float64
                )
            )
        )

    pair_cited_u = []
    pair_x_a = []
    reciprocal_ranks = []
    ndcg_scores = []

    for rows in groups.values():
        labels_array = np.asarray(
            [row[0] for row in rows]
        )

        probability_array = np.stack(
            [row[1] for row in rows]
        )

        cited_score = (
            probability_array[:, 0]
            + probability_array[:, 1]
        )

        x_score = (
            probability_array[:, 1]
            / (
                probability_array[:, 0]
                + probability_array[:, 1]
                + 1e-12
            )
        )

        graded_score = (
            probability_array[:, 0]
            + 2.0 * probability_array[:, 1]
        )

        cited_mask = labels_array != 2
        u_mask = labels_array == 2
        x_mask = labels_array == 1
        a_mask = labels_array == 0

        if cited_mask.any() and u_mask.any():
            cited_values = cited_score[cited_mask]
            u_values = cited_score[u_mask]

            wins = (
                cited_values[:, None]
                > u_values[None, :]
            ).astype(float)

            ties = (
                cited_values[:, None]
                == u_values[None, :]
            ).astype(float)

            pair_cited_u.append(
                float((wins + 0.5 * ties).mean())
            )

            order = np.argsort(
                -cited_score,
                kind="mergesort"
            )

            positions = np.flatnonzero(
                cited_mask[order]
            )

            reciprocal_ranks.append(
                1.0 / (positions[0] + 1)
                if len(positions)
                else 0.0
            )

            relevance = np.where(
                labels_array == 1,
                2,
                np.where(labels_array == 0, 1, 0)
            )

            ranked_relevance = relevance[
                np.argsort(
                    -graded_score,
                    kind="mergesort"
                )
            ]

            ideal_relevance = np.sort(
                relevance
            )[::-1]

            discounts = np.log2(
                np.arange(
                    2,
                    len(relevance) + 2
                )
            )

            dcg = np.sum(
                (2 ** ranked_relevance - 1)
                / discounts
            )

            idcg = np.sum(
                (2 ** ideal_relevance - 1)
                / discounts
            )

            ndcg_scores.append(
                float(dcg / idcg)
                if idcg
                else 0.0
            )

        if x_mask.any() and a_mask.any():
            x_values = x_score[x_mask]
            a_values = x_score[a_mask]

            wins = (
                x_values[:, None]
                > a_values[None, :]
            ).astype(float)

            ties = (
                x_values[:, None]
                == a_values[None, :]
            ).astype(float)

            pair_x_a.append(
                float((wins + 0.5 * ties).mean())
            )

    return {
        "pairwise_cited_over_U": float(
            np.mean(pair_cited_u)
        ),
        "pairwise_X_over_A": float(
            np.mean(pair_x_a)
        ),
        "mrr_cited": float(
            np.mean(reciprocal_ranks)
        ),
        "ndcg": float(
            np.mean(ndcg_scores)
        )
    }

def evaluate_binary(
    labels,
    probabilities
):
    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    result = classification_metrics(
        labels,
        predictions,
        2
    )

    result["roc_auc"] = binary_auc(
        labels,
        probabilities
    )

    result["average_precision"] = (
        average_precision(
            labels,
            probabilities
        )
    )

    return result

# =========================================================
# 5. Transformers compatibility helpers
# =========================================================
training_argument_parameters = set(
    inspect.signature(
        TrainingArguments.__init__
    ).parameters
)

trainer_parameters = set(
    inspect.signature(
        Trainer.__init__
    ).parameters
)

def create_training_arguments(
    prefix,
    config,
    seed,
    train_size
):
    steps_per_epoch = math.ceil(
        train_size
        / config["batch"]
    )

    warmup_steps = max(
        1,
        int(
            steps_per_epoch
            * config["epochs"]
            * 0.06
        )
    )

    values = {
        "output_dir": f"/content/{prefix}",
        "num_train_epochs": config["epochs"],
        "learning_rate": config["lr"],
        "per_device_train_batch_size": config["batch"],
        "per_device_eval_batch_size": config["eval_batch"],
        "gradient_accumulation_steps": 1,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "fp16": False,
        "bf16": torch.cuda.is_bf16_supported(),
        "tf32": True,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "linear",
        "logging_steps": 100,
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "group_by_length": True,
        "remove_unused_columns": True
    }

    if (
        "eval_strategy"
        in training_argument_parameters
    ):
        values["eval_strategy"] = "no"

    elif (
        "evaluation_strategy"
        in training_argument_parameters
    ):
        values["evaluation_strategy"] = "no"

    supported_values = {
        key: value
        for key, value in values.items()
        if key in training_argument_parameters
    }

    ignored = sorted(
        set(values)
        - set(supported_values)
    )

    if ignored:
        print(
            "Unsupported arguments ignored:",
            ignored
        )

    return TrainingArguments(
        **supported_values
    )

def create_trainer(
    model,
    arguments,
    train_dataset,
    collator,
    tokenizer
):
    values = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "data_collator": collator
    }

    if "processing_class" in trainer_parameters:
        values["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        values["tokenizer"] = tokenizer

    return Trainer(**values)

def safe_predict(
    trainer,
    dataset
):
    while True:
        try:
            output = trainer.predict(
                dataset
            ).predictions

            if isinstance(output, tuple):
                output = output[0]

            return output

        except torch.cuda.OutOfMemoryError:
            current_batch = (
                trainer.args
                .per_device_eval_batch_size
            )

            next_batch = max(
                1,
                current_batch // 2
            )

            if next_batch == current_batch:
                raise

            trainer.args.per_device_eval_batch_size = (
                next_batch
            )

            gc.collect()
            torch.cuda.empty_cache()

            print(
                "Evaluation OOM. Retrying with batch:",
                next_batch
            )

# =========================================================
# 6. Training
# =========================================================
for model_name, config in MODELS.items():
    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        config["id"],
        use_fast=True
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            batch["text_b"],
            truncation="longest_first",
            max_length=MAX_LENGTH
        )

    print("Tokenizing once...")

    tokenized = raw.map(
        tokenize,
        batched=True,
        desc=f"Tokenizing {model_name}"
    )

    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(
            "label",
            "_gold_label"
        )

    collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8
    )

    for task, num_labels in TASKS.items():
        for seed in SEEDS:
            prefix = (
                f"{model_name}_"
                f"{task}_seed{seed}"
            )

            required_files = [
                OUT
                / f"{prefix}_validation.npz",
                OUT
                / f"{prefix}_test.npz"
            ]

            if all(
                path.exists()
                for path in required_files
            ):
                print("Cached:", prefix)
                continue

            print("\nTRAIN:", prefix)

            random.seed(seed)
            np.random.seed(seed)
            set_seed(seed)

            train_indices, train_labels = (
                task_indices(
                    "train",
                    task
                )
            )

            train_data = (
                tokenized["train"]
                .select(
                    train_indices.tolist()
                )
                .add_column(
                    "labels",
                    train_labels.tolist()
                )
            )

            gc.collect()
            torch.cuda.empty_cache()

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    config["id"],
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True
                )
            )
            model = model.float()


            arguments = (
                create_training_arguments(
                    prefix=prefix,
                    config=config,
                    seed=seed,
                    train_size=len(train_data)
                )
            )

            trainer = create_trainer(
                model=model,
                arguments=arguments,
                train_dataset=train_data,
                collator=collator,
                tokenizer=tokenizer
            )

            trainer.train()

            for split in [
                "validation",
                "test"
            ]:
                logits = safe_predict(
                    trainer,
                    tokenized[split]
                )

                output_path = (
                    OUT
                    / f"{prefix}_{split}.npz"
                )

                np.savez_compressed(
                    output_path,
                    logits=logits.astype(
                        np.float16
                    )
                )

                print(
                    "Saved:",
                    output_path.name
                )

            del trainer
            del model
            del train_data

            gc.collect()
            torch.cuda.empty_cache()

    del tokenized
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

# =========================================================
# 7. Aggregate direct and hierarchical results
# =========================================================
all_results = {}

for model_name in MODELS:
    all_results[model_name] = {}

    for seed in SEEDS:
        seed_key = f"seed{seed}"
        all_results[
            model_name
        ][seed_key] = {}

        for split in [
            "validation",
            "test"
        ]:
            labels = gold[split]["labels"]
            queries = gold[split]["queries"]

            u_mask = eligible[split]
            xa_mask = labels != 2

            cited_logits = np.load(
                OUT
                / (
                    f"{model_name}_"
                    f"cited_vs_u_seed{seed}_"
                    f"{split}.npz"
                )
            )["logits"].astype(
                np.float32
            )

            xa_logits = np.load(
                OUT
                / (
                    f"{model_name}_"
                    f"x_vs_a_seed{seed}_"
                    f"{split}.npz"
                )
            )["logits"].astype(
                np.float32
            )

            direct_logits = np.load(
                OUT
                / (
                    f"{model_name}_"
                    f"direct_3way_seed{seed}_"
                    f"{split}.npz"
                )
            )["logits"].astype(
                np.float32
            )

            cited_probability = softmax(
                cited_logits
            )[:, 1]

            x_given_cited = softmax(
                xa_logits
            )[:, 1]

            direct_probability = softmax(
                direct_logits
            )

            # Binary cited vs U
            cited_metrics = evaluate_binary(
                (
                    labels[u_mask] != 2
                ).astype(np.int64),
                cited_probability[u_mask]
            )

            # Binary X vs A
            xa_metrics = evaluate_binary(
                (
                    labels[xa_mask] == 1
                ).astype(np.int64),
                x_given_cited[xa_mask]
            )

            # Direct 3-way
            direct_predictions = (
                direct_probability.argmax(
                    axis=1
                )
            )

            direct_metrics = (
                classification_metrics(
                    labels[u_mask],
                    direct_predictions[u_mask],
                    3
                )
            )


            direct_metrics.update(
                ranking_metrics(
                    labels[u_mask],
                    queries[u_mask],
                    direct_probability[u_mask]
                )
            )

            # Hierarchical A/X/U probabilities
            hierarchical_probability = (
                np.column_stack([
                    cited_probability
                    * (1 - x_given_cited),

                    cited_probability
                    * x_given_cited,

                    1 - cited_probability
                ])
            )

            hierarchical_predictions = (
                hierarchical_probability.argmax(
                    axis=1
                )
            )

            hierarchical_metrics = (
                classification_metrics(
                    labels[u_mask],
                    hierarchical_predictions[
                        u_mask
                    ],
                    3
                )
            )


            hierarchical_metrics.update(
                ranking_metrics(
                    labels[u_mask],
                    queries[u_mask],
                    hierarchical_probability[u_mask]
                )
            )

            all_results[
                model_name
            ][seed_key][split] = {
                "cited_vs_u": cited_metrics,
                "x_vs_a": xa_metrics,
                "direct_3way": direct_metrics,
                "hierarchical_3way": (
                    hierarchical_metrics
                )
            }

# =========================================================
# 8. Mean and standard deviation
# =========================================================
summary = {}

for model_name in MODELS:
    summary[model_name] = {}

    for split in [
        "validation",
        "test"
    ]:
        summary[
            model_name
        ][split] = {}

        for task in [
            "cited_vs_u",
            "x_vs_a",
            "direct_3way",
            "hierarchical_3way"
        ]:
            reference_metrics = (
                all_results[
                    model_name
                ]["seed13"][split][task]
            )

            summary[
                model_name
            ][split][task] = {}

            for metric in reference_metrics:
                if metric == "per_class_f1":
                    values = np.asarray([
                        all_results[
                            model_name
                        ][f"seed{seed}"][
                            split
                        ][task][metric]
                        for seed in SEEDS
                    ])

                    summary[
                        model_name
                    ][split][task][metric] = {
                        "mean": (
                            np.mean(
                                values,
                                axis=0
                            )
                            .round(6)
                            .tolist()
                        ),
                        "std": (
                            np.std(
                                values,
                                axis=0
                            )
                            .round(6)
                            .tolist()
                        )
                    }

                else:
                    values = [
                        all_results[
                            model_name
                        ][f"seed{seed}"][
                            split
                        ][task][metric]
                        for seed in SEEDS
                    ]

                    summary[
                        model_name
                    ][split][task][metric] = {
                        "mean": round(
                            float(
                                np.mean(values)
                            ),
                            6
                        ),
                        "std": round(
                            float(
                                np.std(values)
                            ),
                            6
                        )
                    }

with open(
    OUT / "all_seed_results.json",
    "w"
) as f:
    json.dump(
        all_results,
        f,
        indent=2
    )

with open(
    OUT / "summary.json",
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSUMMARY")
print(
    json.dumps(
        summary,
        indent=2
    )
)
print("Saved:", OUT)
