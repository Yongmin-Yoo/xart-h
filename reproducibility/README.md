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

## Table 5 Qwen environment and public artifact paths

The repository-level `requirements.txt` is not the canonical Qwen
environment because it constrains Transformers to versions below 5.
Table 5 Qwen inference must use the separate environment specification:

`environments/table5_qwen_cu128.txt`

Canonical versions:

- PyTorch 2.11.0+cu128
- CUDA 12.8
- Transformers 5.18.0
- PEFT 0.21.1
- NVIDIA L4 with BF16 and SDPA

Create a dedicated environment rather than installing the common
requirements into it:

    python -m venv .venv-table5-qwen
    source .venv-table5-qwen/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r reproducibility/environments/table5_qwen_cu128.txt

The historical inference implementation expects artifacts under
`experiments/...`, whereas the released files are stored under
`reproducibility/artifacts/...`. The public launcher validates the
released files and constructs a temporary compatibility tree:

    python reproducibility/scripts/run_table5_qwen_public.py       --check-only

Run all six canonical adapter evaluations with:

    python reproducibility/scripts/run_table5_qwen_public.py       --output reproducibility/generated/table5_qwen_recomputed

Public path mapping:

- historical Qwen adapters:
  `experiments/model_generalization_v1/qwen3_reranker_0.6b_control/`
- released adapters:
  `reproducibility/artifacts/table4/qwen_adapters/`
- historical Dense-Hard input:
  `experiments/dense_hard_v1/test_dense_hard_u.parquet`
- released Dense-Hard input:
  `reproducibility/artifacts/table5/common/test_dense_hard_u.parquet`
- historical frozen Qwen evaluation:
  `experiments/dense_hard_v1/qwen_frozen_evaluation/`
- released Qwen artifacts:
  `reproducibility/artifacts/table5/qwen/`

The launcher does not copy or modify adapters and does not use the
canonical saved predictions to generate new probabilities.

<!-- TABLE5_QWEN_PUBLIC_EXECUTION_START -->
## Table 5 Qwen public execution environment

The repository-level `requirements.txt` is not the canonical Qwen
environment. Table 5 Qwen inference uses the dedicated environment file:

    reproducibility/environments/table5_qwen_cu128.txt

The validated runtime is PyTorch 2.11.0+cu128, CUDA 12.8,
Transformers 5.18.0, and PEFT 0.21.1 on an NVIDIA L4 GPU with BF16
and SDPA. The public launcher is:

    python reproducibility/scripts/run_table5_qwen_public.py --check-only

For full inference:

    python reproducibility/scripts/run_table5_qwen_public.py       --output /path/to/qwen_table5_output

The launcher maps the currently released artifacts to the historical
layout expected by `experiments/run_qwen_dense_hard_recomputed.py`.
The bridge connects:

- `reproducibility/artifacts/table4/qwen_adapters/` to the historical
  adapter directory;
- `reproducibility/artifacts/table5/common/test_dense_hard_u.parquet`
  to the historical Dense-Hard input path; and
- `reproducibility/artifacts/table5/qwen/` to the historical prediction
  template directory.

Six public adapters (`hard_only` and `mixed_u`, seeds 13, 42, and 77)
were executed on all 1,642 Dense-Hard rows. The resulting 9,852
predictions reproduced the released predictions exactly: all six MAE
values were 0.0, all maximum absolute differences were 0.0, all
correlations were 1.0, and no probability was missing.
<!-- TABLE5_QWEN_PUBLIC_EXECUTION_END -->
