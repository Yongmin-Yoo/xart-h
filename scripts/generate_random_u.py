"""
Reproducibility script recovered from the executed XART-H
Random U experiment notebook.

The XART-H v1.0.1 release is not modified by this script.
"""

# ============================================================
# XART-H Matched Random U construction and audit
# CPU only, deterministic, read-only source data
# ============================================================

import ast
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as error:
    print("Drive mount skipped:", error)

# ------------------------------------------------------------
# 1. Paths and configuration
# ------------------------------------------------------------
ROOT = Path(
    "/content/drive/MyDrive/"
    "PatentSearchBench/external/PatentMatch"
)

SOURCE_ROOT = ROOT / "priority_augmented_temporal_split"
HARD_ROOT = ROOT / "xart_hard_u_v1"
METADATA_PATH = (
    ROOT
    / "hard_u_metadata"
    / "patent_metadata.csv"
)

OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/random_u_v1"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42

SPLIT_FILES = {
    "train": {
        "source": SOURCE_ROOT / "train.csv",
        "hard_u": HARD_ROOT / "train_hard_u.csv",
    },
    "validation": {
        "source": SOURCE_ROOT / "dev.csv",
        "hard_u": HARD_ROOT / "dev_hard_u.csv",
    },
    "test": {
        "source": SOURCE_ROOT / "test.csv",
        "hard_u": HARD_ROOT / "test_hard_u.csv",
    },
}

for split, paths in SPLIT_FILES.items():
    for role, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{split}/{role}: {path}"
            )

if not METADATA_PATH.exists():
    raise FileNotFoundError(METADATA_PATH)

print("Metadata:", METADATA_PATH)
print("Output:", OUTPUT_ROOT)

# ------------------------------------------------------------
# 2. Normalization helpers
# ------------------------------------------------------------
def normalize_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def normalize_id(value):
    if pd.isna(value):
        return ""

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
    if pd.isna(value):
        return ""

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
    if value is None or pd.isna(value):
        return set()

    if isinstance(value, (list, tuple, set)):
        items = list(value)

    else:
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
            code.upper(),
        )

        if not code:
            continue

        # Use CPC subclass, such as B41J.
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


def deterministic_choice(
    candidate_indices,
    split,
    query,
    application,
):
    ordered = sorted(candidate_indices)

    digest = hashlib.sha256(
        (
            f"{SEED}|{split}|"
            f"{application}|{query}"
        ).encode("utf-8")
    ).hexdigest()

    position = int(digest[:16], 16) % len(ordered)
    return ordered[position]

# ------------------------------------------------------------
# 3. Load and index metadata
# ------------------------------------------------------------
metadata = pd.read_csv(
    METADATA_PATH,
    dtype=str,
    low_memory=False,
)

required_metadata = {
    "input_id",
    "base_id",
    "publication_date",
    "cpc_subclasses",
    "reference_bases",
}

missing_metadata = (
    required_metadata - set(metadata.columns)
)

if missing_metadata:
    raise ValueError(
        f"Missing metadata columns: "
        f"{sorted(missing_metadata)}"
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
        "status": (
            str(row.get("status", "")).strip()
        ),
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

print("Metadata records:", len(metadata_records))

# ------------------------------------------------------------
# 4. Generate one matched Random U per Hard-U query
# ------------------------------------------------------------
all_random_rows = []
split_reports = {}

for split, paths in SPLIT_FILES.items():
    print("\n" + "=" * 70)
    print("SPLIT:", split)
    print("=" * 70)

    source = pd.read_csv(
        paths["source"],
        low_memory=False,
    )

    hard_u = pd.read_csv(
        paths["hard_u"],
        low_memory=False,
    )

    source["label"] = pd.to_numeric(
        source["label"],
        errors="coerce",
    )

    hard_u["label"] = pd.to_numeric(
        hard_u["label"],
        errors="coerce",
    )

    source = source[
        source["label"].isin([0, 1])
    ].copy()

    hard_u = hard_u[
        hard_u["label"] == 2
    ].copy()

    source["_normalized_query"] = source[
        "text"
    ].map(normalize_text)

    source["_normalized_passage"] = source[
        "text_b"
    ].map(normalize_text)

    hard_u["_normalized_query"] = hard_u[
        "text"
    ].map(normalize_text)

    hard_u["_normalized_passage"] = hard_u[
        "text_b"
    ].map(normalize_text)

    # One target row per normalized Hard-U query.
    duplicate_hard_queries = int(
        hard_u["_normalized_query"].duplicated().sum()
    )

    if duplicate_hard_queries:
        raise ValueError(
            f"{split}: duplicate Hard-U queries: "
            f"{duplicate_hard_queries}"
        )

    # Existing citations by target application and query.
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

    # Candidate document component membership.
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

    # Candidate passage table.
    candidate_rows = []
    seen_candidates = set()

    for _, row in source.iterrows():
        candidate_base = resolve_base(
            row["cited_document_id"]
        )
        passage = normalize_text(row["text_b"])

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
            "document_id": str(
                row["cited_document_id"]
            ),
            "base_id": candidate_base,
            "passage": str(row["text_b"]),
            "normalized_passage": passage,
            "publication_date": publication_date,
            "cpc": cpc,
            "components": document_components.get(
                candidate_base,
                set(),
            ),
        })

    print("Unique eligible metadata candidates:", len(candidate_rows))

    # CPC inverted index for efficient sampling.
    cpc_to_candidate_indices = defaultdict(set)

    for candidate_index, candidate in enumerate(
        candidate_rows
    ):
        for cpc_code in candidate["cpc"]:
            cpc_to_candidate_indices[
                cpc_code
            ].add(candidate_index)

    random_rows = []
    failure_reasons = defaultdict(int)
    eligible_pool_sizes = []

    for row_number, (_, hard_row) in enumerate(
        hard_u.iterrows()
    ):
        query = hard_row["_normalized_query"]
        application = resolve_base(
            hard_row["patent_application_id"]
        )

        target_metadata = get_metadata(application)

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
            choose_target_date(hard_row)
        )

        if pd.isna(target_date):
            failure_reasons[
                "missing_target_date"
            ] += 1
            continue

        target_component = normalize_component(
            hard_row.get(
                "_u_target_component",
                hard_row.get(
                    "_priority_component_id",
                    "",
                ),
            )
        )

        target_references = target_metadata[
            "references"
        ]

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
            target_references
        )

        hard_candidate_base = resolve_base(
            hard_row.get(
                "_u_candidate_base",
                hard_row.get(
                    "cited_document_id",
                    "",
                ),
            )
        )

        hard_passage = hard_row[
            "_normalized_passage"
        ]

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

            # Same application or document.
            if candidate["base_id"] == application:
                continue

            # Existing citation or reference.
            if candidate["base_id"] in excluded_citations:
                continue

            # Ensure the random candidate differs from Hard U.
            if candidate["base_id"] == hard_candidate_base:
                continue

            # Strictly earlier publication.
            if not (
                candidate["publication_date"]
                < target_date
            ):
                continue

            # CPC overlap.
            if not (
                candidate["cpc"] & target_cpc
            ):
                continue

            # Observed priority-component exclusion.
            if (
                target_component
                and target_component
                in candidate["components"]
            ):
                continue

            # Exclude existing or duplicated passage text.
            if (
                candidate["normalized_passage"]
                in query_passages.get(
                    query,
                    set(),
                )
            ):
                continue

            if (
                candidate["normalized_passage"]
                == hard_passage
            ):
                continue

            eligible_indices.append(candidate_index)

        if not eligible_indices:
            failure_reasons[
                "no_remaining_candidate"
            ] += 1
            continue

        selected_index = deterministic_choice(
            eligible_indices,
            split,
            query,
            application,
        )

        selected = candidate_rows[
            selected_index
        ]

        output_row = hard_row.drop(
            labels=[
                "_normalized_query",
                "_normalized_passage",
            ],
            errors="ignore",
        ).to_dict()

        output_row["cited_document_id"] = (
            selected["document_id"]
        )
        output_row["text_b"] = selected["passage"]
        output_row["label"] = 2
        output_row["label_name"] = "U"
        output_row["_u_method"] = "RANDOM_CPC"
        output_row["_u_bm25_score"] = np.nan
        output_row["_u_candidate_base"] = (
            selected["base_id"]
        )
        output_row[
            "_u_candidate_publication_date"
        ] = selected[
            "publication_date"
        ].strftime("%Y-%m-%d")
        output_row["_u_target_component"] = (
            target_component
        )
        output_row["_u_candidate_component"] = (
            "|".join(
                sorted(selected["components"])
            )
        )
        output_row["_random_seed"] = SEED
        output_row["_random_pool_size"] = len(
            eligible_indices
        )
        output_row["_target_date_source"] = (
            target_date_column
        )
        output_row["_hard_u_candidate_base"] = (
            hard_candidate_base
        )
        output_row["_hard_u_passage_hash"] = (
            hashlib.sha256(
                hard_passage.encode("utf-8")
            ).hexdigest()
        )
        output_row["_row_id"] = (
            f"random_u_{split}_{row_number:06d}"
        )

        random_rows.append(output_row)
        eligible_pool_sizes.append(
            len(eligible_indices)
        )

    random_u = pd.DataFrame(random_rows)

    # --------------------------------------------------------
    # 5. Audit generated Random U
    # --------------------------------------------------------
    violations = {
        "duplicate_queries": 0,
        "duplicate_query_passage_pairs": 0,
        "same_as_hard_u_passage": 0,
        "same_as_hard_u_document": 0,
        "existing_citation_or_reference": 0,
        "time_violation": 0,
        "cpc_violation": 0,
        "same_component_violation": 0,
    }

    if len(random_u):
        random_u["_audit_query"] = random_u[
            "text"
        ].map(normalize_text)

        random_u["_audit_passage"] = random_u[
            "text_b"
        ].map(normalize_text)

        violations["duplicate_queries"] = int(
            random_u["_audit_query"]
            .duplicated()
            .sum()
        )

        violations[
            "duplicate_query_passage_pairs"
        ] = int(
            random_u[
                ["_audit_query", "_audit_passage"]
            ].duplicated().sum()
        )

        hard_passage_by_query = dict(
            zip(
                hard_u["_normalized_query"],
                hard_u["_normalized_passage"],
            )
        )

        hard_base_by_query = {
            row["_normalized_query"]: resolve_base(
                row.get(
                    "_u_candidate_base",
                    row.get(
                        "cited_document_id",
                        "",
                    ),
                )
            )
            for _, row in hard_u.iterrows()
        }

        for _, row in random_u.iterrows():
            query = row["_audit_query"]
            application = resolve_base(
                row["patent_application_id"]
            )
            candidate_base = resolve_base(
                row["_u_candidate_base"]
            )
            candidate_metadata = get_metadata(
                candidate_base
            )
            target_metadata = get_metadata(
                application
            )
            target_date, _ = choose_target_date(row)

            if (
                row["_audit_passage"]
                == hard_passage_by_query.get(
                    query,
                    "",
                )
            ):
                violations[
                    "same_as_hard_u_passage"
                ] += 1

            if (
                candidate_base
                == hard_base_by_query.get(query)
            ):
                violations[
                    "same_as_hard_u_document"
                ] += 1

            excluded = set()
            excluded.update(
                application_citations.get(
                    application,
                    set(),
                )
            )
            excluded.update(
                query_citations.get(
                    query,
                    set(),
                )
            )

            if target_metadata:
                excluded.update(
                    target_metadata["references"]
                )

            if candidate_base in excluded:
                violations[
                    "existing_citation_or_reference"
                ] += 1

            if (
                candidate_metadata is None
                or pd.isna(
                    candidate_metadata[
                        "publication_date"
                    ]
                )
                or not (
                    candidate_metadata[
                        "publication_date"
                    ]
                    < target_date
                )
            ):
                violations[
                    "time_violation"
                ] += 1

            target_cpc = (
                target_metadata["cpc"]
                if target_metadata
                else set()
            )
            candidate_cpc = (
                candidate_metadata["cpc"]
                if candidate_metadata
                else set()
            )

            if not (target_cpc & candidate_cpc):
                violations[
                    "cpc_violation"
                ] += 1

            target_component = normalize_component(
                row.get("_u_target_component")
            )

            candidate_components = (
                document_components.get(
                    candidate_base,
                    set(),
                )
            )

            if (
                target_component
                and target_component
                in candidate_components
            ):
                violations[
                    "same_component_violation"
                ] += 1

        random_u = random_u.drop(
            columns=[
                "_audit_query",
                "_audit_passage",
            ],
            errors="ignore",
        )

    # --------------------------------------------------------
    # 6. Save split
    # --------------------------------------------------------
    csv_path = (
        OUTPUT_ROOT
        / f"{split}_random_u.csv"
    )

    parquet_path = (
        OUTPUT_ROOT
        / f"{split}_random_u.parquet"
    )

    random_u.to_csv(
        csv_path,
        index=False,
    )

    random_u.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )

    split_report = {
        "hard_u_queries": int(len(hard_u)),
        "random_u_generated": int(len(random_u)),
        "coverage_against_hard_u": (
            float(len(random_u) / len(hard_u))
            if len(hard_u)
            else 0.0
        ),
        "candidate_passages": int(
            len(candidate_rows)
        ),
        "mean_eligible_pool_size": (
            float(np.mean(eligible_pool_sizes))
            if eligible_pool_sizes
            else 0.0
        ),
        "median_eligible_pool_size": (
            float(np.median(eligible_pool_sizes))
            if eligible_pool_sizes
            else 0.0
        ),
        "minimum_eligible_pool_size": (
            int(np.min(eligible_pool_sizes))
            if eligible_pool_sizes
            else 0
        ),
        "maximum_eligible_pool_size": (
            int(np.max(eligible_pool_sizes))
            if eligible_pool_sizes
            else 0
        ),
        "failure_reasons": {
            key: int(value)
            for key, value
            in failure_reasons.items()
        },
        "violations": violations,
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
    }

    split_reports[split] = split_report
    all_random_rows.append(random_u)

    print(
        f"{split}: generated "
        f"{len(random_u):,}/{len(hard_u):,} "
        f"({split_report['coverage_against_hard_u']:.3%})"
    )
    print("Failures:", dict(failure_reasons))
    print("Violations:", violations)

# ------------------------------------------------------------
# 7. Combined output and cross-split audit
# ------------------------------------------------------------
combined = pd.concat(
    all_random_rows,
    ignore_index=True,
)

combined_path = OUTPUT_ROOT / "all_random_u.parquet"
combined.to_parquet(
    combined_path,
    index=False,
    compression="zstd",
)

cross_split_violations = {}

split_frames = {
    split: pd.read_parquet(
        OUTPUT_ROOT / f"{split}_random_u.parquet"
    )
    for split in SPLIT_FILES
}

for split, frame in split_frames.items():
    frame["_normalized_query"] = frame[
        "text"
    ].map(normalize_text)

    frame["_normalized_passage"] = frame[
        "text_b"
    ].map(normalize_text)

for left, right in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    left_frame = split_frames[left]
    right_frame = split_frames[right]

    cross_split_violations[
        f"{left}_{right}_query_overlap"
    ] = len(
        set(left_frame["_normalized_query"])
        & set(right_frame["_normalized_query"])
    )

    cross_split_violations[
        f"{left}_{right}_passage_overlap"
    ] = len(
        set(left_frame["_normalized_passage"])
        & set(right_frame["_normalized_passage"])
    )

    cross_split_violations[
        f"{left}_{right}_document_overlap"
    ] = len(
        {
            resolve_base(value)
            for value in left_frame[
                "_u_candidate_base"
            ]
        }
        & {
            resolve_base(value)
            for value in right_frame[
                "_u_candidate_base"
            ]
        }
    )

report = {
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "method": (
        "Uniform deterministic sampling from candidates "
        "satisfying the Hard-U eligibility constraints, "
        "excluding the selected Hard-U document and passage"
    ),
    "seed": SEED,
    "source_split": (
        "priority_augmented_temporal_split"
    ),
    "metadata": str(METADATA_PATH),
    "splits": split_reports,
    "cross_split_violations": (
        cross_split_violations
    ),
    "combined_rows": int(len(combined)),
    "combined_path": str(combined_path),
}

report_path = (
    OUTPUT_ROOT
    / "random_u_generation_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=2,
    )

# ------------------------------------------------------------
# 8. Final assertions and output
# ------------------------------------------------------------
for split, split_report in split_reports.items():
    assert all(
        value == 0
        for value
        in split_report["violations"].values()
    ), (
        f"{split}: eligibility audit failed: "
        f"{split_report['violations']}"
    )

assert all(
    value == 0
    for value in cross_split_violations.values()
), (
    "Cross-split audit failed: "
    f"{cross_split_violations}"
)

print("\n" + "=" * 78)
print("MATCHED RANDOM U GENERATION RESULT")
print("=" * 78)

for split, split_report in split_reports.items():
    print(
        f"{split}: "
        f"{split_report['random_u_generated']:,}/"
        f"{split_report['hard_u_queries']:,} "
        f"coverage="
        f"{split_report['coverage_against_hard_u']:.3%}, "
        f"mean_pool="
        f"{split_report['mean_eligible_pool_size']:.1f}, "
        f"median_pool="
        f"{split_report['median_eligible_pool_size']:.1f}"
    )

print("\nCross-split violations:")
print(
    json.dumps(
        cross_split_violations,
        indent=2,
    )
)

print("\nSaved:")
print(report_path)
print(combined_path)
print("=" * 78)
