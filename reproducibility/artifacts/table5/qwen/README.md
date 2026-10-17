# Canonical Qwen Dense-Hard recomputation

This directory contains the canonical recomputation of the Qwen3-Reranker-0.6B Dense-Hard evaluation used for Table 5.

## Canonical configuration

- Model: `Qwen/Qwen3-Reranker-0.6B`
- Model revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`
- Dataset revision: `4a3ad685a395252c66932c49e98a00728089b614`
- Conditions: Hard-only and Mixed-U
- Seeds: 13, 42, 77
- Queries: 1,642
- Inference: BF16, SDPA, maximum length 384, batch size 36
- Bootstrap: 2,000 paired query-level repetitions, seed 42
- Multiple-testing correction: Holm over the six Qwen comparisons and the full twelve-comparison Table 5 family

## Patent-specific instruction

> Given a patent claim, determine whether the prior-art passage is examiner-cited evidence for that claim

## Results

| Subset | Metric | Hard-only | Mixed-U | Mixed-U − Hard-only | 95% CI | Global Holm p |
|---|---:|---:|---:|---:|---:|---:|
| document_disjoint_1389 | pairwise_cited_over_dense_hard | 0.625150 | 0.722558 | +0.097408 | [0.073254, 0.121527] | 0.011994 |
| document_disjoint_1389 | roc_auc | 0.624932 | 0.727186 | +0.102254 | [0.084561, 0.121276] | 0.011994 |
| passage_disjoint_1555 | pairwise_cited_over_dense_hard | 0.618542 | 0.730150 | +0.111608 | [0.088842, 0.132573] | 0.011994 |
| passage_disjoint_1555 | roc_auc | 0.618190 | 0.735095 | +0.116905 | [0.100684, 0.133712] | 0.011994 |
| primary_all_1642 | pairwise_cited_over_dense_hard | 0.607998 | 0.732268 | +0.124269 | [0.101945, 0.145251] | 0.011994 |
| primary_all_1642 | roc_auc | 0.608301 | 0.736472 | +0.128171 | [0.111334, 0.144451] | 0.011994 |

All six Mixed-U improvements have confidence intervals above zero and remain significant after the global twelve-comparison Holm correction.

## Important provenance note

These canonical predictions replace the earlier provisional Qwen Dense-Hard predictions that were produced with an inconsistent inference pipeline. The earlier predictions must not be used for the paper's final Table 5.

See `artifact_manifest.json` for SHA256 hashes and full artifact provenance.


## Statistical reproduction

The canonical statistical evaluation is implemented in:

```text
evaluation/qwen_dense_hard_statistics_recomputed.py
```

Set `XARTH_ARTIFACT_ROOT` to the restored experiment-artifact root
containing the `experiments/` directory:

```bash
XARTH_ARTIFACT_ROOT=/path/to/XART-H-artifacts \
python evaluation/qwen_dense_hard_statistics_recomputed.py
```

The script does not train or load a language model. It recomputes
three-seed ensembles, query-level bootstrap confidence intervals,
paired tests, disjoint-sensitivity analyses, and Holm corrections
from saved predictions.


## Canonical inference reproduction

The canonical six-adapter Dense-Hard inference is implemented in:

```text
experiments/run_qwen_dense_hard_recomputed.py
```

It evaluates Hard-only and Mixed-U adapters for seeds 13, 42,
and 77 without retraining the adapters.

```bash
XARTH_ARTIFACT_ROOT=/path/to/XART-H-artifacts \
python experiments/run_qwen_dense_hard_recomputed.py
```

The canonical runtime is torch 2.11.0+cu128, CUDA 12.8,
transformers 5.18.0, peft 0.21.1, BF16, SDPA, maximum
length 384, and evaluation batch size 36.

## Public environment and path bridge

The root-level `requirements.txt` is not the canonical Qwen environment.
Use the dedicated CUDA 12.8 environment file:

    python -m pip install       -r reproducibility/environments/table5_qwen_cu128.txt

Validate the released artifact layout without model inference:

    python reproducibility/scripts/run_table5_qwen_public.py       --check-only

Run the six existing LoRA adapters against the released Dense-Hard input:

    python reproducibility/scripts/run_table5_qwen_public.py       --output reproducibility/generated/table5_qwen_recomputed

The launcher maps the historical `experiments/...` layout expected by
`experiments/run_qwen_dense_hard_recomputed.py` to the released
`reproducibility/artifacts/...` paths through a temporary compatibility
tree. It does not alter the canonical inference implementation,
adapters, or Dense-Hard input.

<!-- EXACT_PUBLIC_REPRODUCTION_START -->
## Exact public inference reproduction

Install the dedicated environment rather than the repository-level
`requirements.txt`:

    python -m pip install       -r reproducibility/environments/table5_qwen_cu128.txt

The canonical runtime is PyTorch 2.11.0+cu128 with CUDA 12.8,
Transformers 5.18.0, PEFT 0.21.1, NVIDIA L4, BF16, SDPA,
maximum sequence length 384, and evaluation batch size 36.

Validate the released artifact mapping without inference:

    python reproducibility/scripts/run_table5_qwen_public.py --check-only

Run all six public LoRA adapters:

    python reproducibility/scripts/run_table5_qwen_public.py       --output /path/to/qwen_table5_output

The launcher creates a temporary compatibility tree for the historical
`experiments/...` paths and removes it after execution. It does not
duplicate or modify the released adapters.

GPU execution validation covered `hard_only` and `mixed_u` for seeds
13, 42, and 77. Each run contained 1,642 predictions, for 9,852 total
predictions. Every run matched its released prediction file exactly:
MAE 0.0, maximum absolute probability difference 0.0, correlation 1.0,
and zero missing probabilities.
<!-- EXACT_PUBLIC_REPRODUCTION_END -->
