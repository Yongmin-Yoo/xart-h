# PatentMatch source-row audit

This directory provides the source-row manifest referenced by the PatScout
paper's dataset-construction and provenance sections.

## Public files

- `patentmatch_source_row_manifest.csv`: one record for each of the 25,340
  source PatentMatch rows.
- `source_row_manifest_report.json`: hashes, cleaning rules, source counts,
  exclusion counts, and final partition counts.

After pre-partition cleaning, 25,276 rows remain: 24,176 released rows and 1,100 dependency-quarantine rows.

## Row accounting

| Status | Rows |
|---|---:|
| Included in released train/validation/test partitions | 24,176 |
| Included in dependency quarantine | 1,100 |
| Excluded before partitioning | 64 |
| Total source rows | 25,340 |

The 64 pre-partition exclusions consist of:

| Exclusion reason | Rows |
|---|---:|
| `conflicting_label_pair` | 22 |
| `duplicate_pair_after_first` | 42 |
| Total | 64 |

The original duplicate audit flags 53 rows after the first occurrence, and
the conflicting-label audit covers 22 rows. Eleven rows satisfy both flags.
The manifest therefore applies conflict precedence: all rows in a
conflicting-label pair group are classified as `conflicting_label_pair`;
only non-conflicting later duplicates are classified as
`duplicate_pair_after_first`.

## Pair identity

Claim and passage text are normalized by lowercasing, stripping leading and
trailing whitespace, and collapsing consecutive whitespace. The manifest
does not redistribute full claim or passage text. Instead it records SHA-1
hashes of normalized claim text, normalized passage text, and their combined
pair identity.

For non-conflicting duplicate pairs, source order is deterministic:
the original PatentMatch train file precedes the original test file, and
row order within each source file is preserved. The first occurrence is
retained.

## Final partition counts

| Partition | Rows |
|---|---:|
| Train | 16,912 |
| Validation | 3,626 |
| Test | 3,638 |
| Dependency quarantine | 1,100 |

The pre-partition exclusion decision and the later dependency-aware split
assignment are separate fields in the manifest.
