#!/usr/bin/env python3

"""Build the PatScout Table 4 evaluation manifest from public data.

Sources:
1. Hugging Face frozen PatScout test Parquet at revision v1.0.1
2. Public repository Random-U test Parquet
3. A canonical released Table 4 ordered prediction file used only as
   the exact row-ID and ordering template

No probability from the canonical prediction file is copied.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re

import pandas as pd


HF_REPO_ID = "yongminyoo91/xart-h"
HF_REPO_TYPE = "dataset"
HF_REVISION = "v1.0.1"
HF_TEST_FILE = "parquet/test.parquet"

EXPECTED_TEMPLATE_ROWS = 9896
EXPECTED_QUERIES = 1642
EXPECTED_EVALUATION_ROWS = {
    "hard_u": 4948,
    "random_u": 4948,
}
EXPECTED_CANDIDATE_COUNTS = {
    "cited": 6612,
    "hard_u": 1642,
    "random_u": 1642,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def resolve(columns, candidates, required=True):
    lookup = {norm(column): column for column in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    if required:
        raise RuntimeError(
            f"Could not resolve {candidates}; columns={list(columns)}"
        )
    return None


def choose_id_column(frame, needed_ids, source_name):
    candidates = [
        "source_row_id",
        "_row_id",
        "row_id",
        "training_row_id",
        "_dense_source_row_id",
        "id",
    ]
    available = [
        column for column in candidates if column in frame.columns
    ]

    scores = {}
    needed = set(str(value) for value in needed_ids)

    for column in available:
        values = set(frame[column].dropna().astype(str))
        scores[column] = len(needed & values)

    if not scores:
        raise RuntimeError(
            f"{source_name}: no candidate row-ID column found; "
            f"columns={list(frame.columns)}"
        )

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        raise RuntimeError(
            f"{source_name}: no row-ID intersection with template; "
            f"scores={scores}"
        )

    return best, scores


def prepare_source(frame, needed_ids, source_name):
    id_column, scores = choose_id_column(
        frame,
        needed_ids,
        source_name,
    )

    query_text = resolve(
        frame.columns,
        [
            "query_text",
            "claim_text",
            "claim",
            "query",
            "normalized_claim_text",
            "text",
        ],
    )
    passage_text = resolve(
        frame.columns,
        [
            "text_b",
            "passage_text",
            "candidate_text",
            "document_text",
            "passage",
            "document",
        ],
    )

    source = frame[
        [id_column, query_text, passage_text]
    ].copy()

    source.columns = [
        "source_row_id",
        "text",
        "text_b",
    ]

    source["source_row_id"] = (
        source["source_row_id"].astype(str)
    )

    if source["source_row_id"].duplicated().any():
        duplicated = int(
            source["source_row_id"].duplicated().sum()
        )
        raise RuntimeError(
            f"{source_name}: duplicate row IDs: {duplicated}"
        )

    if source["text"].isna().any():
        raise RuntimeError(
            f"{source_name}: null query text found."
        )
    if source["text_b"].isna().any():
        raise RuntimeError(
            f"{source_name}: null passage text found."
        )

    return source, {
        "id_column": id_column,
        "intersection_scores": scores,
        "query_text_column": query_text,
        "passage_text_column": passage_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--random-u",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not args.template.is_file():
        raise FileNotFoundError(args.template)
    if not args.random_u.is_file():
        raise FileNotFoundError(args.random_u)

    template = pd.read_parquet(args.template)

    required_template_columns = {
        "source_row_id",
        "query_id",
        "candidate_type",
        "evaluation_condition",
        "training_condition",
        "seed",
        "cited_probability",
        "prediction_row_id",
    }

    missing = required_template_columns - set(template.columns)
    if missing:
        raise RuntimeError(
            f"Template is missing columns: {sorted(missing)}"
        )

    if len(template) != EXPECTED_TEMPLATE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEMPLATE_ROWS} template rows, "
            f"found {len(template)}."
        )

    if template["query_id"].astype(str).nunique() != EXPECTED_QUERIES:
        raise RuntimeError(
            "Template query count is not 1,642."
        )

    evaluation_counts = (
        template["evaluation_condition"]
        .map(norm)
        .value_counts()
        .to_dict()
    )
    candidate_counts = (
        template["candidate_type"]
        .map(norm)
        .value_counts()
        .to_dict()
    )

    if evaluation_counts != EXPECTED_EVALUATION_ROWS:
        raise RuntimeError(
            f"Unexpected evaluation counts: {evaluation_counts}"
        )

    if candidate_counts != EXPECTED_CANDIDATE_COUNTS:
        raise RuntimeError(
            f"Unexpected candidate counts: {candidate_counts}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub before building the public "
            "evaluation manifest."
        ) from exc

    hf_test_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_TEST_FILE,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
        )
    )

    hf_test = pd.read_parquet(hf_test_path)
    random_u = pd.read_parquet(args.random_u)

    template_types = template["candidate_type"].map(norm)
    template_ids = template["source_row_id"].astype(str)

    benchmark_needed = set(
        template_ids[template_types != "random_u"]
    )
    random_needed = set(
        template_ids[template_types == "random_u"]
    )

    benchmark_source, benchmark_schema = prepare_source(
        hf_test,
        benchmark_needed,
        "HF frozen test",
    )
    random_source, random_schema = prepare_source(
        random_u,
        random_needed,
        "public Random-U test",
    )

    benchmark_lookup = benchmark_source.set_index(
        "source_row_id"
    )
    random_lookup = random_source.set_index(
        "source_row_id"
    )

    missing_benchmark = sorted(
        benchmark_needed - set(benchmark_lookup.index)
    )
    missing_random = sorted(
        random_needed - set(random_lookup.index)
    )

    if missing_benchmark:
        raise RuntimeError(
            f"HF test is missing {len(missing_benchmark)} "
            "required row IDs."
        )

    if missing_random:
        raise RuntimeError(
            f"Random-U test is missing {len(missing_random)} "
            "required row IDs."
        )

    output = template[
        [
            "source_row_id",
            "query_id",
            "candidate_type",
            "evaluation_condition",
            "prediction_row_id",
        ]
    ].copy()

    output["source_row_id"] = output["source_row_id"].astype(str)
    output["candidate_type"] = output["candidate_type"].map(norm)
    output["evaluation_condition"] = (
        output["evaluation_condition"].map(norm)
    )

    query_texts = []
    passage_texts = []

    for row in output.itertuples(index=False):
        lookup = (
            random_lookup
            if row.candidate_type == "random_u"
            else benchmark_lookup
        )
        source_row = lookup.loc[row.source_row_id]
        query_texts.append(str(source_row["text"]))
        passage_texts.append(str(source_row["text_b"]))

    output["text"] = query_texts
    output["text_b"] = passage_texts
    output["label"] = (
        output["candidate_type"] != "random_u"
    ).astype(int)

    # Hard-U is also an uncited candidate.
    output.loc[
        output["candidate_type"] == "hard_u",
        "label",
    ] = 0

    output["label"] = output["label"].astype(int)

    expected_labels = {
        "cited": 1,
        "hard_u": 0,
        "random_u": 0,
    }

    for candidate_type, expected_label in expected_labels.items():
        values = set(
            output.loc[
                output["candidate_type"] == candidate_type,
                "label",
            ].unique()
        )
        if values != {expected_label}:
            raise RuntimeError(
                f"Label mismatch for {candidate_type}: {values}"
            )

    if output["text"].isna().any():
        raise RuntimeError("Null query text in output.")
    if output["text_b"].isna().any():
        raise RuntimeError("Null passage text in output.")

    if len(output) != EXPECTED_TEMPLATE_ROWS:
        raise RuntimeError("Output row count changed.")

    if output["query_id"].astype(str).nunique() != EXPECTED_QUERIES:
        raise RuntimeError("Output query count changed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    output.to_parquet(
        args.output,
        index=False,
        compression="zstd",
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PUBLIC_INPUT_ALIGNMENT_VERIFIED",
        "hf_dataset": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "hf_test_file": HF_TEST_FILE,
        "hf_test_sha256": sha256_file(hf_test_path),
        "random_u_path": str(args.random_u),
        "random_u_sha256": sha256_file(args.random_u),
        "template_path": str(args.template),
        "template_sha256": sha256_file(args.template),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(output),
        "queries": int(
            output["query_id"].astype(str).nunique()
        ),
        "evaluation_condition_counts": evaluation_counts,
        "candidate_type_counts": candidate_counts,
        "benchmark_source_schema": benchmark_schema,
        "random_u_source_schema": random_schema,
        "probabilities_copied_from_template": False,
    }

    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("TABLE 4 PUBLIC EVALUATION MANIFEST COMPLETE")
    print("Rows:", len(output))
    print("Queries:", output["query_id"].astype(str).nunique())
    print("Output:", args.output)
    print("Report:", args.report)


if __name__ == "__main__":
    main()
