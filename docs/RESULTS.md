# Current Benchmark Results

These results use XART-H v1.0.1. Supervised values are
means over seeds 13, 42, and 77 unless otherwise stated.

## Test Results

| Model | Cited/U AUC | X/A AUC | 3-way Accuracy | Macro-F1 |
|:--|--:|--:|--:|--:|
| Random scoring | 0.499 | 0.507 | N/A | N/A |
| TF-IDF | 0.347 | 0.516 | N/A | N/A |
| BM25 | 0.329 | 0.513 | N/A | N/A |
| MiniLM bi-encoder | 0.528 | 0.524 | N/A | N/A |
| PatentSBERTa | 0.495 | 0.511 | N/A | N/A |
| MiniLM cross-encoder | 0.426 | 0.524 | N/A | N/A |
| Qwen3-Reranker-0.6B | 0.633 | 0.516 | N/A | N/A |
| Qwen3-8B direct | 0.531 | 0.489 | 0.336 | 0.250 |
| MiniLM supervised | 0.493 | 0.501 | 0.325 | 0.313 |
| DeBERTa-v3-base | 0.746 | 0.507 | 0.453 | 0.450 |

## Direct and Hierarchical Prediction

| Model | Method | Accuracy | Macro-F1 | P-C/U | P-X/A |
|:--|:--|--:|--:|--:|--:|
| MiniLM | Direct | 0.325 | 0.313 | 0.502 | 0.492 |
| MiniLM | Hierarchical | 0.325 | 0.290 | 0.499 | 0.499 |
| DeBERTa | Direct | 0.453 | 0.450 | 0.747 | 0.517 |
| DeBERTa | Hierarchical | 0.446 | 0.432 | 0.750 | 0.500 |
| Qwen3-8B | Direct | 0.336 | 0.250 | 0.535 | 0.488 |
| Qwen3-8B | Hierarchical | 0.347 | 0.291 | 0.571 | 0.495 |

## Statistical Summary

For Qwen3-8B, hierarchical prompting improves macro-F1
by 0.041 and cited-versus-U ROC-AUC by 0.046. Both
differences remain significant after family-wise Holm
correction with adjusted p = 0.013.

For DeBERTa, direct prediction improves macro-F1 over
hierarchical prediction by 0.019, with a query-level
95% confidence interval of [0.006, 0.032] and
Holm-adjusted p = 0.036.

Detailed confidence intervals and corrected tests are
provided in the result files. Subgroup analyses remain pending.


## Metric Correction

Supervised ranking metrics use task-specific scores:
`p(A)+p(X)` for cited-versus-U ranking,
`p(X)/(p(A)+p(X))` for X-versus-A ranking, and
`2p(X)+p(A)` for graded nDCG. Classification and binary
ROC-AUC results are unchanged.

## Random U versus Hard U

The ablation uses 1,642 test queries for which both Hard U and matched Random U candidates are available. Random U candidates satisfy the same temporal, citation, priority, application, component, CPC, and duplication constraints as Hard U but are sampled without BM25-based selection. Random U generation achieved 97.9% coverage on the eligible test queries, with no detected eligibility or cross-split violations.

| Model | Random similarity | Hard similarity | Random C/U AUC | Hard C/U AUC | Random pairwise | Hard pairwise |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.019 | 0.086 | 0.775 | 0.342 | 0.787 | 0.331 |
| BM25 | 6.443 | 23.873 | 0.748 | 0.325 | 0.774 | 0.290 |
| MiniLM | 0.208 | 0.347 | 0.806 | 0.524 | 0.813 | 0.520 |
| PatentSBERTa | 0.368 | 0.475 | 0.775 | 0.490 | 0.797 | 0.494 |

Hard U candidates are more similar to the claims under all four scoring models. Cited-versus-U ROC-AUC and pairwise accuracy decrease substantially when Random U is replaced with Hard U. All paired query-level differences remain significant after Holm correction, with adjusted p = 0.003. The lexical AUC values below 0.5 indicate that BM25-selected Hard U passages can be more lexically similar to the claims than examiner-cited passages.

The analysis uses 2,000 paired query-level bootstrap samples with seed 42. Confidence intervals are reported in `results/random_u/random_vs_hard_summary.csv` and `results/random_u/random_vs_hard_results.json`. Similarity scales are model specific. The frozen XART-H v1.0.1 dataset release is unchanged.

<!-- ROBUSTNESS_RESULTS_START -->
## Temporal Robustness

Temporal robustness is evaluated on the 1,678 U-eligible
test queries. The date of each query's unique U row is used
as the temporal anchor. Chronological bins contain 568 early,
534 middle, and 576 late queries, with median distances of
266, 336, and 427 days from the end of the training period.

| System | Early M-F1 | Middle M-F1 | Late M-F1 | Early C/U AUC | Middle C/U AUC | Late C/U AUC |
|:--|--:|--:|--:|--:|--:|--:|
| MiniLM direct | 0.308 | 0.303 | 0.313 | 0.504 | 0.476 | 0.510 |
| MiniLM hierarchical | 0.299 | 0.270 | 0.298 | 0.522 | 0.456 | 0.482 |
| DeBERTa direct | 0.500 | 0.441 | 0.428 | 0.810 | 0.718 | 0.749 |
| DeBERTa hierarchical | 0.471 | 0.427 | 0.415 | 0.813 | 0.721 | 0.742 |
| Qwen3-8B direct | 0.252 | 0.235 | 0.242 | 0.573 | 0.526 | 0.483 |
| Qwen3-8B hierarchical | 0.297 | 0.290 | 0.286 | 0.556 | 0.591 | 0.580 |

DeBERTa direct, DeBERTa hierarchical, and Qwen3-8B
direct exhibit significant early-to-late changes in
cited-versus-U ROC-AUC after Holm correction. These
results show a temporal association but do not establish
causal temporal drift. The analysis uses 2,000 paired
query-level bootstrap samples with seed 42.

Detailed results are available under `results/temporal/`.

## Dependency Relaxation

The official dependency-controlled split has zero
cross-split overlap for every audited relation. A relaxed
row-level temporal reconstruction reassigns 1,100
quarantined X/A rows and produces test contamination rates
of 5.92% for dependency components, 6.84% for
supercomponents, and 7.14% for priority components.

DeBERTa was trained on the official and relaxed X/A
partitions with seeds 13, 42, and 77. Both conditions were
evaluated on the same frozen official X/A test set.

| Metric | Official | Relaxed | Relaxed minus Official | Holm p |
|:--|--:|--:|--:|--:|
| Accuracy | 0.488 | 0.489 | 0.000 | 1.000 |
| Macro-F1 | 0.460 | 0.445 | -0.015 | 0.136 |
| ROC-AUC | 0.499 | 0.492 | -0.007 | 0.398 |
| Average precision | 0.501 | 0.490 | -0.011 | 0.136 |

No metric differs significantly after Holm correction.
The dependency controls eliminate measurable structural
overlap, but the introduced overlap does not produce
detectable X/A performance inflation in this experiment.
Both conditions remain near chance, so this result should
not be interpreted as general evidence of insensitivity to
dependency leakage.

Detailed audit and model-comparison results are available
under `results/dependency/`.
<!-- ROBUSTNESS_RESULTS_END -->

<!-- ROBUSTNESS_REPRODUCIBILITY_START -->
## Robustness Reproducibility

Temporal robustness can be reproduced with:

```bash
python experiments/run_temporal_robustness.py
```

The dependency partition audit can be reproduced with:

```bash
python experiments/run_dependency_reconstruction_audit.py \
  --output-dir results/dependency/reconstructed
```

The dependency model comparison can be reproduced with:

```bash
python experiments/run_dependency_robustness.py
```

The dependency scripts use the frozen XART-H v1.0.1 
release and the auxiliary quarantine rows in 
`data/dependency/dependency_quarantine_xa.csv`. The 
auxiliary file contains only the columns required to 
reconstruct the dependency-relaxed X/A partitions. Model 
checkpoints, prediction caches, and authentication tokens 
are not included.
<!-- ROBUSTNESS_REPRODUCIBILITY_END -->
