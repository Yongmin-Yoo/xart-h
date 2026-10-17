#!/usr/bin/env python3
"""
Validated reconstruction of the PatScout Dense-Hard candidate-generation
pipeline from the released inputs, Random-U eligibility rules, generation
report, and canonical Dense-Hard output.

The script reconstructs the candidate pool, applies the released
eligibility constraints, computes PatentSBERTa CLS embeddings, applies
L2 normalization, selects the highest-cosine eligible candidate for each
query, and saves the generated Dense-Hard dataset.

It does not copy canonical candidate selections.
"""

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO_ID = "yongminyoo91/xart-h"
DATASET_REVISION = "v1.0.1"

RETRIEVER = "AI-Growth-Lab/PatentSBERTa"
RETRIEVER_REVISION = "3ff1d553c861d8f5bfd902333d97fc95eb6b4c8f"

MAXIMUM_LENGTH = 384
NEAR_TIE_MAXIMUM_RANK = 3
NEAR_TIE_MAXIMUM_GAP = 0.002

EXPECTED_TARGET_QUERIES = 1642
EXPECTED_CANDIDATE_PASSAGES = 2079
EXPECTED_ELIGIBLE_PAIRS = 208123
EXPECTED_POOL_MINIMUM = 1
EXPECTED_POOL_MEDIAN = 107.0
EXPECTED_POOL_MAXIMUM = 393


def normalize_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def normalize_id(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper(),
    )


def fallback_base_id(value):
    value = normalize_id(value)

    match = re.match(
        r"^([A-Z]{2}\d+)(?:[A-Z]\d?)?$",
        value,
    )

    if match:
        return match.group(1)

    return value


def normalize_component(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return value


def parse_date(value):
    return pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )


def parse_collection(value):
    if value is None:
        return set()

    if isinstance(value, (list, tuple, set, np.ndarray)):
        items = list(value)
    else:
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if text.lower() in {
            "",
            "nan",
            "none",
            "null",
            "[]",
            "{}",
        }:
            return set()

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            else:
                items = [parsed]
        except Exception:
            items = re.split(
                r"[;,|\s]+",
                text,
            )

    result = set()

    for item in items:
        item = str(item).strip()

        if (
            item
            and item.lower()
            not in {"nan", "none", "null"}
        ):
            result.add(item)

    return result


def parse_cpc(value):
    raw_codes = parse_collection(value)
    normalized = set()

    for code in raw_codes:
        code = re.sub(
            r"[^A-Z0-9/]",
            "",
            str(code).upper(),
        )

        if not code:
            continue

        match = re.match(
            r"^([A-HY]\d{2}[A-Z])",
            code,
        )

        if match:
            normalized.add(match.group(1))
        else:
            normalized.add(code)

    return normalized


def parse_references(value):
    return {
        fallback_base_id(item)
        for item in parse_collection(value)
        if fallback_base_id(item)
    }


def choose_target_date(row):
    for column in [
        "_date",
        "_parsed_date",
        "date",
    ]:
        if column in row.index:
            date = parse_date(row[column])

            if not pd.isna(date):
                return date, column

    return pd.NaT, None


def passage_hash(normalized_passage):
    return hashlib.sha256(
        normalized_passage.encode("utf-8")
    ).hexdigest()


def build_metadata(metadata_path):
    metadata = pd.read_csv(
        metadata_path,
        dtype=str,
        low_memory=False,
    )

    required = {
        "input_id",
        "base_id",
        "publication_date",
        "cpc_subclasses",
        "reference_bases",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    metadata_records = {}
    alias_to_base = {}

    for _, row in metadata.iterrows():
        input_id = normalize_id(row["input_id"])
        base_id = normalize_id(row["base_id"])

        if not base_id:
            base_id = fallback_base_id(input_id)

        record = {
            "input_id": input_id,
            "base_id": base_id,
            "publication_date": parse_date(
                row["publication_date"]
            ),
            "cpc": parse_cpc(
                row["cpc_subclasses"]
            ),
            "references": parse_references(
                row["reference_bases"]
            ),
            "status": str(
                row.get("status", "")
            ).strip(),
        }

        if base_id:
            metadata_records[base_id] = record
            alias_to_base[base_id] = base_id

        if input_id:
            alias_to_base[input_id] = base_id

    def resolve_base(value):
        normalized = normalize_id(value)

        if normalized in alias_to_base:
            return alias_to_base[normalized]

        fallback = fallback_base_id(normalized)

        if fallback in alias_to_base:
            return alias_to_base[fallback]

        return fallback

    def get_metadata(value):
        return metadata_records.get(
            resolve_base(value)
        )

    return (
        metadata_records,
        alias_to_base,
        resolve_base,
        get_metadata,
    )


def reconstruct_pool(metadata_path, random_u_path):
    (
        metadata_records,
        alias_to_base,
        resolve_base,
        get_metadata,
    ) = build_metadata(metadata_path)

    public_dataset = load_dataset(
        REPO_ID,
        revision=DATASET_REVISION,
    )

    public_frame = public_dataset["test"].to_pandas()
    public_frame["label"] = pd.to_numeric(
        public_frame["label"],
        errors="coerce",
    )

    source = public_frame[
        public_frame["label"].isin([0, 1])
    ].copy()

    source["_normalized_query"] = source[
        "text"
    ].map(normalize_text)

    source["_normalized_passage"] = source[
        "text_b"
    ].map(normalize_text)

    application_citations = defaultdict(set)
    query_citations = defaultdict(set)
    query_passages = defaultdict(set)

    for _, row in source.iterrows():
        application = resolve_base(
            row["patent_application_id"]
        )
        candidate_base = resolve_base(
            row["cited_document_id"]
        )
        query = row["_normalized_query"]
        passage = row["_normalized_passage"]

        if application and candidate_base:
            application_citations[
                application
            ].add(candidate_base)

        if query and candidate_base:
            query_citations[
                query
            ].add(candidate_base)

        if query and passage:
            query_passages[
                query
            ].add(passage)

    document_components = defaultdict(set)

    for _, row in source.iterrows():
        candidate_base = resolve_base(
            row["cited_document_id"]
        )

        component = normalize_component(
            row.get("_priority_component_id")
        )

        if candidate_base and component:
            document_components[
                candidate_base
            ].add(component)

    candidate_rows = []
    seen_candidates = set()

    for source_index, row in source.iterrows():
        candidate_base = resolve_base(
            row["cited_document_id"]
        )
        passage = normalize_text(
            row["text_b"]
        )

        if not candidate_base or not passage:
            continue

        key = (candidate_base, passage)

        if key in seen_candidates:
            continue

        seen_candidates.add(key)

        candidate_metadata = get_metadata(
            candidate_base
        )

        if candidate_metadata is None:
            continue

        publication_date = candidate_metadata[
            "publication_date"
        ]
        cpc = candidate_metadata["cpc"]

        if pd.isna(publication_date) or not cpc:
            continue

        candidate_rows.append({
            "source_row_id": str(source_index),
            "document_id": str(
                row["cited_document_id"]
            ),
            "base_id": candidate_base,
            "passage": str(row["text_b"]),
            "normalized_passage": passage,
            "passage_hash": passage_hash(passage),
            "publication_date": publication_date,
            "cpc": cpc,
            "components": document_components.get(
                candidate_base,
                set(),
            ),
        })

    cpc_to_candidate_indices = defaultdict(set)

    for candidate_index, candidate in enumerate(
        candidate_rows
    ):
        for cpc_code in candidate["cpc"]:
            cpc_to_candidate_indices[
                cpc_code
            ].add(candidate_index)

    random_u = pd.read_parquet(
        random_u_path
    ).reset_index(drop=True)

    normalized_queries = random_u[
        "text"
    ].map(normalize_text)

    if normalized_queries.duplicated().any():
        duplicates = int(
            normalized_queries.duplicated().sum()
        )
        raise ValueError(
            f"Duplicate target queries: {duplicates}"
        )

    target_records = []
    pool_sizes = []
    eligible_pair_count = 0
    failure_reasons = defaultdict(int)

    for target_number, row in random_u.iterrows():
        query = normalize_text(row["text"])
        application = resolve_base(
            row["patent_application_id"]
        )

        target_metadata = get_metadata(
            application
        )

        if target_metadata is None:
            failure_reasons[
                "missing_target_metadata"
            ] += 1
            continue

        target_cpc = target_metadata["cpc"]

        if not target_cpc:
            failure_reasons[
                "missing_target_cpc"
            ] += 1
            continue

        target_date, target_date_column = (
            choose_target_date(row)
        )

        if pd.isna(target_date):
            failure_reasons[
                "missing_target_date"
            ] += 1
            continue

        target_component = normalize_component(
            row.get(
                "_u_target_component",
                row.get(
                    "_priority_component_id",
                    "",
                ),
            )
        )

        excluded_citations = set()
        excluded_citations.update(
            application_citations.get(
                application,
                set(),
            )
        )
        excluded_citations.update(
            query_citations.get(
                query,
                set(),
            )
        )
        excluded_citations.update(
            target_metadata["references"]
        )

        hard_candidate_base = resolve_base(
            row.get(
                "_hard_u_candidate_base",
                "",
            )
        )
        hard_passage_hash = str(
            row.get(
                "_hard_u_passage_hash",
                "",
            )
        ).strip()

        candidate_index_pool = set()

        for cpc_code in target_cpc:
            candidate_index_pool.update(
                cpc_to_candidate_indices.get(
                    cpc_code,
                    set(),
                )
            )

        eligible_indices = []

        for candidate_index in candidate_index_pool:
            candidate = candidate_rows[
                candidate_index
            ]

            if candidate["base_id"] == application:
                continue

            if candidate["base_id"] in excluded_citations:
                continue

            if (
                hard_candidate_base
                and candidate["base_id"]
                == hard_candidate_base
            ):
                continue

            if not (
                candidate["publication_date"]
                < target_date
            ):
                continue

            if not (
                candidate["cpc"] & target_cpc
            ):
                continue

            if (
                target_component
                and target_component
                in candidate["components"]
            ):
                continue

            if (
                candidate["normalized_passage"]
                in query_passages.get(
                    query,
                    set(),
                )
            ):
                continue

            if (
                hard_passage_hash
                and candidate["passage_hash"]
                == hard_passage_hash
            ):
                continue

            eligible_indices.append(
                candidate_index
            )

        eligible_indices = sorted(
            eligible_indices
        )

        if not eligible_indices:
            failure_reasons[
                "no_remaining_candidate"
            ] += 1
            continue

        eligible_pair_count += len(
            eligible_indices
        )
        pool_sizes.append(
            len(eligible_indices)
        )

        target_records.append({
            "target_number": int(target_number),
            "row": row,
            "query": str(row["text"]),
            "normalized_query": query,
            "application": application,
            "target_date": target_date,
            "target_date_column": target_date_column,
            "target_component": target_component,
            "hard_candidate_base": hard_candidate_base,
            "hard_passage_hash": hard_passage_hash,
            "eligible_indices": eligible_indices,
        })

    summary = {
        "target_queries": int(
            len(target_records)
        ),
        "candidate_passages": int(
            len(candidate_rows)
        ),
        "eligible_pairs": int(
            eligible_pair_count
        ),
        "mean_pool_size": float(
            np.mean(pool_sizes)
        ) if pool_sizes else 0.0,
        "median_pool_size": float(
            np.median(pool_sizes)
        ) if pool_sizes else 0.0,
        "minimum_pool_size": int(
            np.min(pool_sizes)
        ) if pool_sizes else 0,
        "maximum_pool_size": int(
            np.max(pool_sizes)
        ) if pool_sizes else 0,
        "failure_reasons": {
            key: int(value)
            for key, value
            in failure_reasons.items()
        },
    }

    return (
        candidate_rows,
        target_records,
        summary,
        resolve_base,
    )


def validate_pool_summary(summary):
    expected = {
        "target_queries": EXPECTED_TARGET_QUERIES,
        "candidate_passages": EXPECTED_CANDIDATE_PASSAGES,
        "eligible_pairs": EXPECTED_ELIGIBLE_PAIRS,
        "minimum_pool_size": EXPECTED_POOL_MINIMUM,
        "median_pool_size": EXPECTED_POOL_MEDIAN,
        "maximum_pool_size": EXPECTED_POOL_MAXIMUM,
    }

    mismatches = {}

    for key, expected_value in expected.items():
        actual = summary[key]

        if actual != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual,
            }

    if mismatches:
        raise RuntimeError(
            "Eligibility-pool reconstruction differs "
            "from the released audit:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )


def encode_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size,
):
    import torch
    import torch.nn.functional as F

    embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAXIMUM_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            output = model(**encoded)
            cls = output.last_hidden_state[
                :,
                0,
                :,
            ].float()
            cls = F.normalize(
                cls,
                p=2,
                dim=1,
            )

        embeddings.append(
            cls.cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        )

    return np.concatenate(
        embeddings,
        axis=0,
    )


def canonical_lookup(
    canonical,
    resolve_base,
):
    result = {}

    for _, row in canonical.iterrows():
        query = normalize_text(row["text"])
        base_id = resolve_base(
            row.get(
                "_u_candidate_base",
                row.get(
                    "cited_document_id",
                    "",
                ),
            )
        )
        passage = normalize_text(
            row["text_b"]
        )

        result[query] = {
            "base_id": base_id,
            "passage": passage,
            "stored_score": row.get(
                "_dense_similarity",
                np.nan,
            ),
        }

    return result


def run_generation(args):
    (
        candidate_rows,
        target_records,
        summary,
        resolve_base,
    ) = reconstruct_pool(
        args.metadata,
        args.random_u,
    )

    validate_pool_summary(summary)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool_report_path = (
        args.output
        / "dense_hard_pool_audit.json"
    )

    pool_report = {
        "status": "DENSE_HARD_POOL_AUDIT_PASSED",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "date_criterion": (
            "candidate_publication_date < target_date"
        ),
        "target_date_resolution": [
            "_date",
            "_parsed_date",
            "date",
        ],
        **summary,
    }

    pool_report_path.write_text(
        json.dumps(
            pool_report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            pool_report,
            indent=2,
        )
    )

    if args.pool_only:
        return

    if args.canonical is None:
        raise ValueError(
            "--canonical is required for full generation"
        )

    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for full generation"
        )

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        RETRIEVER,
        revision=RETRIEVER_REVISION,
    )

    model = AutoModel.from_pretrained(
        RETRIEVER,
        revision=RETRIEVER_REVISION,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model.eval()

    candidate_texts = [
        candidate["passage"]
        for candidate in candidate_rows
    ]
    query_texts = [
        target["query"]
        for target in target_records
    ]

    candidate_embeddings = encode_texts(
        candidate_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
    )
    query_embeddings = encode_texts(
        query_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
    )

    np.save(
        args.output
        / "candidate_embeddings.npy",
        candidate_embeddings,
    )
    np.save(
        args.output
        / "query_embeddings.npy",
        query_embeddings,
    )

    canonical = pd.read_parquet(
        args.canonical
    )
    canonical_by_query = canonical_lookup(
        canonical,
        resolve_base,
    )

    generated_rows = []
    audit_rows = []

    passage_mismatches = 0
    document_mismatches = 0
    canonical_missing = 0
    canonical_ineligible = 0
    canonical_ranks = []
    cosine_gaps = []
    stored_score_differences = []
    selected_scores = []

    for query_position, target in enumerate(
        target_records
    ):
        eligible = np.asarray(
            target["eligible_indices"],
            dtype=np.int64,
        )

        scores = (
            candidate_embeddings[eligible]
            @ query_embeddings[
                query_position
            ]
        )

        local_position = int(
            np.argmax(scores)
        )
        selected_index = int(
            eligible[local_position]
        )
        selected_score = float(
            scores[local_position]
        )
        selected_scores.append(
            selected_score
        )

        selected = candidate_rows[
            selected_index
        ]

        tie_count = int(
            np.isclose(
                scores,
                selected_score,
                rtol=0.0,
                atol=1e-8,
            ).sum()
        )

        output_row = target[
            "row"
        ].to_dict()

        output_row.pop(
            "_random_seed",
            None,
        )
        output_row.pop(
            "_random_pool_size",
            None,
        )

        output_row[
            "cited_document_id"
        ] = selected["document_id"]
        output_row[
            "text_b"
        ] = selected["passage"]
        output_row["label"] = 2
        output_row["label_name"] = "U"
        output_row[
            "_u_method"
        ] = "DENSE_HARD_COSINE"
        output_row[
            "_u_bm25_score"
        ] = np.nan
        output_row[
            "_u_candidate_base"
        ] = selected["base_id"]
        output_row[
            "_u_candidate_publication_date"
        ] = selected[
            "publication_date"
        ].strftime("%Y-%m-%d")
        output_row[
            "_u_target_component"
        ] = target["target_component"]
        output_row[
            "_u_candidate_component"
        ] = "|".join(
            sorted(
                selected["components"]
            )
        )
        output_row[
            "_dense_retriever"
        ] = RETRIEVER
        output_row[
            "_dense_retriever_revision"
        ] = RETRIEVER_REVISION
        output_row[
            "_dense_pooling"
        ] = "CLS"
        output_row[
            "_dense_similarity"
        ] = selected_score
        output_row[
            "_dense_pool_size"
        ] = len(eligible)
        output_row[
            "_dense_tie_count"
        ] = tie_count
        output_row[
            "_dense_source_row_id"
        ] = selected["source_row_id"]
        output_row[
            "_dense_embedding_max_length"
        ] = MAXIMUM_LENGTH
        output_row[
            "_target_date_source"
        ] = target[
            "target_date_column"
        ]
        output_row[
            "_hard_u_candidate_base"
        ] = target[
            "hard_candidate_base"
        ]
        output_row[
            "_hard_u_passage_hash"
        ] = target[
            "hard_passage_hash"
        ]
        output_row[
            "_row_id"
        ] = (
            f"dense_hard_test_"
            f"{query_position:06d}"
        )

        generated_rows.append(
            output_row
        )

        canonical_item = canonical_by_query.get(
            target["normalized_query"]
        )

        audit = {
            "query_position": query_position,
            "claim_id": output_row.get(
                "claim_id"
            ),
            "selected_base": selected[
                "base_id"
            ],
            "selected_score": selected_score,
            "pool_size": len(eligible),
        }

        if canonical_item is None:
            canonical_missing += 1
            audit[
                "canonical_status"
            ] = "missing"
            audit_rows.append(audit)
            continue

        canonical_base = canonical_item[
            "base_id"
        ]
        canonical_passage = canonical_item[
            "passage"
        ]

        exact_passage = (
            selected["normalized_passage"]
            == canonical_passage
        )
        exact_document = (
            selected["base_id"]
            == canonical_base
        )

        if not exact_passage:
            passage_mismatches += 1

        if not exact_document:
            document_mismatches += 1

        canonical_indices = [
            index
            for index in eligible.tolist()
            if (
                candidate_rows[index][
                    "base_id"
                ] == canonical_base
                and candidate_rows[index][
                    "normalized_passage"
                ] == canonical_passage
            )
        ]

        audit.update({
            "canonical_base": canonical_base,
            "exact_passage": bool(
                exact_passage
            ),
            "exact_document": bool(
                exact_document
            ),
        })

        if not canonical_indices:
            canonical_ineligible += 1
            audit[
                "canonical_status"
            ] = "not_in_eligible_pool"
            audit_rows.append(audit)
            continue

        canonical_index = canonical_indices[0]
        canonical_local = int(
            np.where(
                eligible == canonical_index
            )[0][0]
        )
        canonical_score = float(
            scores[canonical_local]
        )

        rank = int(
            1
            + np.sum(
                scores > canonical_score
            )
        )
        gap = float(
            selected_score
            - canonical_score
        )

        canonical_ranks.append(rank)
        cosine_gaps.append(gap)

        stored_score = canonical_item[
            "stored_score"
        ]

        stored_difference = np.nan

        if not pd.isna(stored_score):
            stored_difference = abs(
                float(stored_score)
                - canonical_score
            )
            stored_score_differences.append(
                stored_difference
            )

        audit.update({
            "canonical_status": "eligible",
            "canonical_rank": rank,
            "canonical_recomputed_score": (
                canonical_score
            ),
            "top_minus_canonical_score": gap,
            "canonical_stored_score": (
                stored_score
            ),
            "stored_score_abs_difference": (
                stored_difference
            ),
        })

        audit_rows.append(audit)

    generated = pd.DataFrame(
        generated_rows
    )

    parquet_path = (
        args.output
        / "test_dense_hard_u.parquet"
    )
    csv_path = (
        args.output
        / "test_dense_hard_u.csv"
    )
    audit_path = (
        args.output
        / "canonical_near_tie_audit.csv"
    )

    generated.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )
    generated.to_csv(
        csv_path,
        index=False,
    )
    pd.DataFrame(
        audit_rows
    ).to_csv(
        audit_path,
        index=False,
    )

    maximum_rank = (
        int(max(canonical_ranks))
        if canonical_ranks
        else 0
    )
    maximum_gap = (
        float(max(cosine_gaps))
        if cosine_gaps
        else 0.0
    )

    near_tie_passed = bool(
        canonical_missing == 0
        and canonical_ineligible == 0
        and maximum_rank
        <= NEAR_TIE_MAXIMUM_RANK
        and maximum_gap
        <= NEAR_TIE_MAXIMUM_GAP
    )

    report = {
        "status": "DENSE_HARD_GENERATED",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": (
            "Top-1 cosine similarity under the released "
            "Random-U eligibility constraints"
        ),
        "date_criterion": (
            "candidate_publication_date < target_date"
        ),
        "target_date_resolution": [
            "_date",
            "_parsed_date",
            "date",
        ],
        "retriever": RETRIEVER,
        "retriever_revision": (
            RETRIEVER_REVISION
        ),
        "pooling": "CLS token",
        "embedding_normalization": "L2",
        "similarity": "cosine",
        "embedding_precision": (
            "BF16 model inference with float32 "
            "normalized output"
        ),
        "maximum_length": MAXIMUM_LENGTH,
        **summary,
        "generated_rows": int(
            len(generated)
        ),
        "mean_selected_cosine": float(
            np.mean(selected_scores)
        ),
        "median_selected_cosine": float(
            np.median(selected_scores)
        ),
        "minimum_selected_cosine": float(
            np.min(selected_scores)
        ),
        "maximum_selected_cosine": float(
            np.max(selected_scores)
        ),
        "canonical_comparison": {
            "passage_mismatches": int(
                passage_mismatches
            ),
            "document_mismatches": int(
                document_mismatches
            ),
            "exact_passage_matches": int(
                len(generated)
                - passage_mismatches
            ),
            "exact_document_matches": int(
                len(generated)
                - document_mismatches
            ),
            "canonical_candidate_missing": int(
                canonical_missing
            ),
            "canonical_candidate_ineligible": int(
                canonical_ineligible
            ),
            "maximum_canonical_rank": (
                maximum_rank
            ),
            "maximum_cosine_gap": (
                maximum_gap
            ),
            "maximum_stored_score_abs_difference": (
                float(
                    max(
                        stored_score_differences
                    )
                )
                if stored_score_differences
                else 0.0
            ),
            "near_tie_validation_passed": (
                near_tie_passed
            ),
        },
        "output_files": {
            "parquet": str(
                parquet_path
            ),
            "csv": str(
                csv_path
            ),
            "candidate_embeddings": str(
                args.output
                / "candidate_embeddings.npy"
            ),
            "query_embeddings": str(
                args.output
                / "query_embeddings.npy"
            ),
            "near_tie_audit": str(
                audit_path
            ),
        },
    }

    report_path = (
        args.output
        / "dense_hard_generation_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    if not near_tie_passed:
        raise RuntimeError(
            "Canonical near-tie validation failed:\n"
            + json.dumps(
                report[
                    "canonical_comparison"
                ],
                indent=2,
            )
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--random-u",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--canonical",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--pool-only",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_generation(parse_args())
