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
