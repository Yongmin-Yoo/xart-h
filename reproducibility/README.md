# PatScout Reproducibility Package

This directory contains the controlled-candidate and Dense-Hard
reproduction materials used for Tables 4 and 5.

## Contents

- `scripts/reproduce_table4.py`
  - Hard-only versus Mixed-U controlled evaluation
  - MiniLM, DeBERTa, and Qwen
  - Three-seed probability ensembles
  - Query-level bootstrap and Holm correction

- `scripts/reproduce_table5.py`
  - Dense-Hard evaluation
  - DeBERTa and Qwen
  - Full, passage-different, and document-different subsets
  - Query-level bootstrap and global Holm correction

- `artifacts/table4/`
  - Ordered prediction files
  - Frozen configurations and statistical inputs
  - Qwen LoRA adapters where available

- `artifacts/table5/`
  - Dense-Hard inputs
  - Overlap alignment
  - DeBERTa and Qwen predictions

- `reproduced/`
  - Machine-readable reproduced point estimates
  - Bootstrap intervals
  - Paired comparisons
  - Reproduction configurations

## Table 4

```bash
python reproducibility/scripts/reproduce_table4.py \
  --artifacts reproducibility/artifacts \
  --output reproducibility/reproduced/table4 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed 42

  ```

## Table 5

```bash
python reproducibility/scripts/reproduce_table5.py   --artifacts reproducibility/artifacts   --output reproducibility/reproduced/table5   --bootstrap-repetitions 2000   --bootstrap-seed 42
```

## Statistical protocol

- Model seeds: 13, 42, and 77
- Ensemble: arithmetic mean of cited probabilities
- Bootstrap unit: query
- Bootstrap repetitions: 2,000
- Confidence level: 95%
- Multiple-comparison correction: Holm

## Notes

The repository directory name and dataset identifier are retained as
technical identifiers. The benchmark name is PatScout.

## Validated controlled execution bridge

The Table 4 controlled execution bridge has been successfully validated
on an NVIDIA L4 GPU. The validation covered MiniLM native one-logit
loading and sigmoid inference, Qwen balanced claim-passage truncation
and public LoRA inference, and direct integration with the statistical
reproduction entry point.

The reported Table 4 statistics are reproducible from the canonical
released predictions. Full configuration and execution instructions are
provided in [`CONTROLLED_TRAINING.md`](CONTROLLED_TRAINING.md).

## Dense-Hard candidate generation

The complete Dense-Hard candidate-generation workflow is provided in
[`scripts/generate_dense_hard_candidates.py`](scripts/generate_dense_hard_candidates.py).
It reconstructs the released eligibility pool, applies the documented
date, citation, CPC, component, document, and passage exclusions,
computes PatentSBERTa CLS embeddings, performs L2-normalized cosine
top-1 selection, and saves the generated Dense-Hard files.

Input paths, environment requirements, execution commands, and audit
criteria are documented in
[`DENSE_HARD_GENERATION.md`](DENSE_HARD_GENERATION.md).

## Validated Dense-Hard generation

The reconstructed Dense-Hard generation pipeline was successfully
validated on an NVIDIA L4 GPU. Candidate-pool reconstruction,
PatentSBERTa CLS embedding, L2-normalized cosine top-1 selection, output
serialization, and canonical near-tie auditing completed successfully.
The run generated 1,642 Dense-Hard rows from 208,123 eligible pairs.
