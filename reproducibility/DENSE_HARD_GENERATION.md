# Dense-Hard candidate generation

This document describes the validated reconstruction of the Dense-Hard
candidate-generation pipeline from the released public inputs, Random-U
eligibility rules, generation report, and canonical outputs.

The generator performs the complete workflow:

1. constructs the candidate-passage universe from the public test split;
2. resolves patent metadata and candidate document aliases;
3. excludes existing citations, references, duplicate passages, the
   selected Hard-U document and passage, and candidates from the same
   observed priority component;
4. applies CPC overlap and temporal eligibility;
5. computes PatentSBERTa CLS embeddings;
6. applies L2 normalization and cosine similarity;
7. selects the highest-cosine eligible candidate for each query; and
8. writes the generated Dense-Hard Parquet and CSV files.

## Public inputs

- Dataset: `yongminyoo91/xart-h`, revision `v1.0.1`
- Metadata: `data/random_u/patent_metadata.csv`
- Matched target queries:
  `results/random_u/generated/test_random_u.parquet`
- Canonical audit target:
  `reproducibility/artifacts/table5/common/test_dense_hard_u.parquet`

## Eligibility and date rule

A candidate must satisfy the released Random-U eligibility constraints.
The temporal criterion is strictly:

    candidate_publication_date < target_date

Target-date resolution follows `_date`, `_parsed_date`, and `date`, in
that order.

## Dense retriever

- Retriever: `AI-Growth-Lab/PatentSBERTa`
- Revision: `3ff1d553c861d8f5bfd902333d97fc95eb6b4c8f`
- Pooling: CLS token
- Embedding normalization: L2
- Similarity: cosine
- Maximum token length: 384
- GPU inference precision: BF16 with float32 normalized output

## Pool audit

The released test split reconstructs:

- 1,642 target queries
- 2,079 unique candidate passages
- 208,123 eligible query-candidate pairs
- minimum pool size 1
- median pool size 107
- maximum pool size 393

## Run the eligibility audit

    python reproducibility/scripts/generate_dense_hard_candidates.py       --metadata data/random_u/patent_metadata.csv       --random-u results/random_u/generated/test_random_u.parquet       --output /tmp/patscout-dense-hard-pool-audit       --pool-only

## Run full GPU generation

    python reproducibility/scripts/generate_dense_hard_candidates.py       --metadata data/random_u/patent_metadata.csv       --random-u results/random_u/generated/test_random_u.parquet       --canonical reproducibility/artifacts/table5/common/test_dense_hard_u.parquet       --output /tmp/patscout-dense-hard       --batch-size 256

The canonical file is used only for post-generation auditing. It is not
used to choose or overwrite generated candidates. Small ranking changes
among nearly tied candidates can occur across CUDA and BF16 execution
environments; these are reported in the generated near-tie audit.

## GPU execution validation

The reconstructed pipeline completed full GPU execution on
**NVIDIA L4** with compute capability **8.9** and BF16
support.

Validated execution results:

- target queries: 1,642
- candidate passages: 2,079
- eligible query-candidate pairs: 208,123
- generated Dense-Hard rows: 1,642
- exact passage selections: 1,618 / 1,642
- exact document selections: 1,624 / 1,642
- canonical candidates missing from the universe: 0
- canonical candidates failing eligibility: 0
- maximum canonical rank: 3
- maximum cosine gap: 0.001468122
- near-tie validation: passed

The canonical artifact was used only for post-generation comparison.
Generated selections were determined independently by PatentSBERTa
CLS embeddings, L2 normalization, and cosine top-1 retrieval.
