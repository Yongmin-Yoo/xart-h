#!/usr/bin/env python3
"""Reconstruct and audit dependency-relaxed XART-H partitions."""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO_ID = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

DEFAULT_QUARANTINE_URL = (
    "https://raw.githubusercontent.com/Yongmin-Yoo/xart-h/"
    "Yongmin-Yoo/data/dependency/dependency_quarantine_xa.csv"
)

TRAIN_END_DATE = pd.Timestamp("2016-09-25")
VALIDATION_END_DATE = pd.Timestamp("2017-05-13")

EXPECTED = {
    "official_train": 16912,
    "official_validation": 3626,
    "official_test": 3638,
    "quarantine": 1100,
    "relaxed_train": 17432,
    "relaxed_validation": 3894,
    "relaxed_test": 3950,
}


def normalize(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_labels(series):
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

    output = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if output.isna().any():
        raise ValueError("Unrecognized labels detected.")

    return output.astype(int)


def parse_dates(series):
    text = series.astype(str).str.strip()

    compact = pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce",
    )

    general = pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    output = compact.fillna(general)

    if output.isna().any():
        raise ValueError("Unparseable dates detected.")

    return output


def cross_split_statistics(frame, relation, split_column):
    usable = frame[
        frame[relation].notna()
        & frame[relation].astype(str).ne("")
    ].copy()

    split_counts = usable.groupby(
        relation
    )[split_column].nunique()

    shared_values = set(
        split_counts[split_counts > 1].index
    )

    affected = usable[
        usable[relation].isin(shared_values)
    ]

    train_values = set(
        usable.loc[
            usable[split_column] == "train",
            relation,
        ]
    )

    test_values = set(
        usable.loc[
            usable[split_column] == "test",
            relation,
        ]
    )

    train_test_values = train_values & test_values

    train_test_rows = usable[
        usable[relation].isin(train_test_values)
        & usable[split_column].isin(["train", "test"])
    ]

    test_rows = usable[
        usable[split_column] == "test"
    ]

    contaminated_test_rows = test_rows[
        test_rows[relation].isin(train_values)
    ]

    contamination = (
        len(contaminated_test_rows) / len(test_rows)
        if len(test_rows)
        else 0.0
    )

    return {
        "relation": relation,
        "shared_values": len(shared_values),
        "rows_affected": len(affected),
        "train_test_shared_values": len(train_test_values),
        "train_test_rows": len(train_test_rows),
        "test_contaminated_rows": len(
            contaminated_test_rows
        ),
        "test_rows": len(test_rows),
        "test_contamination_fraction": contamination,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quarantine",
        default=DEFAULT_QUARANTINE_URL,
    )

    parser.add_argument(
        "--output-dir",
        default="results/dependency/reconstructed",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(
        REPO_ID,
        revision=REVISION,
    )

    official_frames = []

    for split in ["train", "validation", "test"]:
        frame = raw[split].to_pandas()
        frame["label"] = normalize_labels(frame["label"])
        frame = frame[frame["label"].isin([0, 1])].copy()
        frame["_official_split"] = split
        official_frames.append(frame)

    official = pd.concat(
        official_frames,
        ignore_index=True,
    )

    quarantine = pd.read_csv(
        args.quarantine,
        low_memory=False,
    )

    quarantine["label"] = normalize_labels(
        quarantine["label"]
    )

    quarantine = quarantine[
        quarantine["label"].isin([0, 1])
    ].copy()

    quarantine["_official_split"] = "quarantine"

    assert len(quarantine) == EXPECTED["quarantine"]

    common_columns = sorted(
        set(official.columns).union(quarantine.columns)
    )

    combined = pd.concat(
        [
            official.reindex(columns=common_columns),
            quarantine.reindex(columns=common_columns),
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["_row_id"],
        keep="first",
    )

    combined["_parsed_experiment_date"] = parse_dates(
        combined["_date"]
    )

    combined["_relaxed_split"] = np.select(
        [
            combined["_parsed_experiment_date"]
            <= TRAIN_END_DATE,
            combined["_parsed_experiment_date"]
            <= VALIDATION_END_DATE,
        ],
        [
            "train",
            "validation",
        ],
        default="test",
    )

    official_counts = (
        combined["_official_split"]
        .value_counts()
        .to_dict()
    )

    relaxed_counts = (
        combined["_relaxed_split"]
        .value_counts()
        .to_dict()
    )

    assert official_counts["train"] == EXPECTED[
        "official_train"
    ]
    assert official_counts["validation"] == EXPECTED[
        "official_validation"
    ]
    assert official_counts["test"] == EXPECTED[
        "official_test"
    ]
    assert official_counts["quarantine"] == EXPECTED[
        "quarantine"
    ]

    assert relaxed_counts["train"] == EXPECTED[
        "relaxed_train"
    ]
    assert relaxed_counts["validation"] == EXPECTED[
        "relaxed_validation"
    ]
    assert relaxed_counts["test"] == EXPECTED[
        "relaxed_test"
    ]

    combined["normalized_claim"] = combined[
        "text"
    ].map(normalize)

    combined["normalized_passage"] = combined[
        "text_b"
    ].map(normalize)

    relation_columns = {
        "normalized_claim": "normalized_claim",
        "normalized_passage": "normalized_passage",
        "claim_id": "claim_id",
        "patent_application": "patent_application_id",
        "cited_document": "cited_document_id",
        "dependency_component": "_component_id",
        "supercomponent": "_supercomponent_id",
        "priority_component": "_priority_component_id",
    }

    official_audit_frame = combined[
        combined["_official_split"] != "quarantine"
    ].copy()

    official_statistics = []
    relaxed_statistics = []

    for relation_name, column in relation_columns.items():
        official_result = cross_split_statistics(
            official_audit_frame,
            column,
            "_official_split",
        )
        official_result["relation"] = relation_name
        official_statistics.append(official_result)

        relaxed_result = cross_split_statistics(
            combined,
            column,
            "_relaxed_split",
        )
        relaxed_result["relation"] = relation_name
        relaxed_statistics.append(relaxed_result)

    official_summary = pd.DataFrame(
        official_statistics
    )

    relaxed_summary = pd.DataFrame(
        relaxed_statistics
    )

    official_summary.insert(
        0,
        "condition",
        "official",
    )

    relaxed_summary.insert(
        0,
        "condition",
        "relaxed",
    )

    overlap_summary = pd.concat(
        [
            official_summary,
            relaxed_summary,
        ],
        ignore_index=True,
    )

    contamination = relaxed_summary[
        [
            "relation",
            "test_contaminated_rows",
            "test_rows",
            "test_contamination_fraction",
        ]
    ].copy()

    split_counts = pd.DataFrame([
        {
            "condition": "official",
            **{
                key: official_counts.get(key, 0)
                for key in [
                    "train",
                    "validation",
                    "test",
                    "quarantine",
                ]
            },
        },
        {
            "condition": "relaxed",
            **{
                key: relaxed_counts.get(key, 0)
                for key in [
                    "train",
                    "validation",
                    "test",
                ]
            },
            "quarantine": 0,
        },
    ])

    quarantine_destinations = (
        combined[
            combined["_official_split"] == "quarantine"
        ]["_relaxed_split"]
        .value_counts()
        .to_dict()
    )

    report = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": REPO_ID,
        "revision": REVISION,
        "quarantine_source": args.quarantine,
        "train_cutoff": str(TRAIN_END_DATE.date()),
        "validation_cutoff": str(
            VALIDATION_END_DATE.date()
        ),
        "official_method": (
            "dependency-controlled temporal split with "
            "cross-boundary components quarantined"
        ),
        "relaxed_method": (
            "row-level chronological assignment without "
            "dependency-component quarantine"
        ),
        "total_xa_rows": len(combined),
        "official_counts": official_counts,
        "relaxed_counts": relaxed_counts,
        "quarantine_destinations": (
            quarantine_destinations
        ),
        "audited_relations": list(
            relation_columns.keys()
        ),
        "official_total_cross_split_values": int(
            official_summary["shared_values"].sum()
        ),
        "relaxed_total_cross_split_values": int(
            relaxed_summary["shared_values"].sum()
        ),
        "reconstruction_valid_for_comparison": True,
    }

    overlap_summary.to_csv(
        output_dir
        / "dependency_overlap_summary.csv",
        index=False,
    )

    contamination.to_csv(
        output_dir
        / "dependency_test_contamination.csv",
        index=False,
    )

    split_counts.to_csv(
        output_dir
        / "dependency_split_counts.csv",
        index=False,
    )

    with open(
        output_dir
        / "dependency_reconstruction_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("DEPENDENCY RELAXATION AUDIT RESULT")
    print("Official counts:", official_counts)
    print("Relaxed counts:", relaxed_counts)
    print(
        "Official cross-split values:",
        report["official_total_cross_split_values"],
    )
    print(
        "Relaxed cross-split values:",
        report["relaxed_total_cross_split_values"],
    )
    print(contamination.to_string(index=False))


if __name__ == "__main__":
    main()
