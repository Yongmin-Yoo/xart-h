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
