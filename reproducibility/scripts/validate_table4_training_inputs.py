#!/usr/bin/env python3

"""Validate PatScout Table 4 frozen controlled-training inputs.

This validator performs no training and writes no model files.
"""

from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd
import pyarrow.parquet as pq


EXPECTED_ROWS = 24863
EXPECTED_QUERIES = 8246
EXPECTED_CITED = 16617
EXPECTED_U = 8246


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
            f"Could not resolve {candidates}; columns={columns}"
        )
    return None


def resolve_schema(path):
    columns = pq.read_schema(path).names
    schema = {
        "query_id": resolve(
            columns,
            ["query_id", "claim_id", "normalized_query_id", "query_key"],
        ),
        "query_text": resolve(
            columns,
            [
                "query_text", "claim_text", "claim", "query",
                "normalized_claim_text", "text",
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
        if pd.api.types.is_numeric_dtype(raw):
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
        elif pd.api.types.is_bool_dtype(raw):
            labels = raw.astype(int)
        else:
            positive = {
                "1", "true", "cited", "citation",
                "positive", "x", "a", "examiner_cited",
            }
            negative = {
                "0", "false", "u", "hard_u",
                "random_u", "uncited", "negative",
            }
            values = raw.map(norm)
            unknown = set(values.unique()) - positive - negative
            if unknown:
                raise RuntimeError(f"Unknown labels: {sorted(unknown)}")
            labels = values.map(
                lambda value: 1 if value in positive else 0
            ).astype(int)
    else:
        negatives = {
            "u", "hard_u", "random_u",
            "dense_hard_u", "uncited", "negative",
        }
        labels = frame[schema["candidate_type"]].map(
            lambda value: 0 if norm(value) in negatives else 1
        ).astype(int)

    if not set(labels.unique()) <= {0, 1}:
        raise RuntimeError("Labels are not binary.")
    return labels


def validate(path):
    frame = pd.read_parquet(path)
    schema = resolve_schema(path)
    labels = derive_labels(frame, schema)

    queries = (
        frame[schema["query_id"]]
        .astype(str)
        .str.strip()
        .nunique()
    )
    counts = labels.value_counts().to_dict()

    assert len(frame) == EXPECTED_ROWS, (path, len(frame))
    assert queries == EXPECTED_QUERIES, (path, queries)
    assert counts.get(1, 0) == EXPECTED_CITED, counts
    assert counts.get(0, 0) == EXPECTED_U, counts
    assert not frame[schema["query_text"]].isna().any()
    assert not frame[schema["passage_text"]].isna().any()

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(frame),
        "queries": queries,
        "cited_rows": int(counts.get(1, 0)),
        "u_rows": int(counts.get(0, 0)),
        "schema": schema,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for condition in ["hard_only", "mixed_u"]:
        path = args.manifests / f"{condition}_train_manifest.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        results[condition] = validate(path)

    print(json.dumps(results, indent=2, sort_keys=True))
    print("TABLE 4 CONTROLLED-TRAINING INPUT VALIDATION PASSED")


if __name__ == "__main__":
    main()
