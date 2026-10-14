# Current Benchmark Results

These results use XART-H v1.0.1. Supervised values are means over seeds 13, 42, and 77. Random scoring uses NumPy `default_rng(42)`. The values in this document supersede all earlier provisional results.

## Main Test Results

### Binary discrimination

| Model | Cited/U ROC-AUC | X/A ROC-AUC |
|:--|--:|--:|
| Random scoring | 0.518 | 0.512 |
| TF-IDF | 0.348 | 0.516 |
| BM25 | 0.331 | 0.514 |
| MiniLM bi-encoder | 0.527 | 0.524 |
| PatentSBERTa | 0.495 | 0.511 |
| MiniLM cross-encoder | 0.427 | **0.525** |
| Qwen3-Reranker-0.6B | **0.546** | 0.513 |
| Qwen3-8B direct | 0.529 | 0.490 |
| MiniLM supervised | 0.478 ± 0.006 | 0.493 ± 0.004 |
| DeBERTa-v3-base supervised | 0.474 ± 0.010 | 0.508 ± 0.010 |

Qwen3-Reranker-0.6B has the highest cited-versus-U ROC-AUC in the main test comparison at 0.546. TF-IDF and BM25 produce values below 0.5 because BM25-selected Hard-U passages can be more lexically similar to the claim than cited evidence. X-versus-A discrimination remains close to chance for every evaluated system.

### Joint classification and ranking

| Model | Prediction | Accuracy | Macro-F1 | P-C/U | P-X/A |
|:--|:--|--:|--:|--:|--:|
| Qwen3-8B | Direct | **0.335** | 0.249 | **0.539** | 0.492 |
| Qwen3-8B | Hierarchical | 0.347 | 0.291 | 0.570 | 0.496 |
| MiniLM supervised | Direct | 0.313 ± 0.013 | 0.297 ± 0.011 | 0.460 | **0.506** |
| MiniLM supervised | Hierarchical | 0.319 ± 0.003 | 0.287 ± 0.001 | 0.475 | 0.495 |
| DeBERTa supervised | Direct | 0.329 ± 0.005 | **0.319 ± 0.011** | 0.483 | 0.502 |
| DeBERTa supervised | Hierarchical | 0.325 ± 0.004 | 0.315 ± 0.006 | 0.473 | 0.500 |

Qwen3-8B direct prediction has the highest direct three-way accuracy at 0.335. DeBERTa has the highest direct macro-F1 at 0.319. P-X/A remains close to 0.5, consistent with the binary X-versus-A results.

Machine-readable sources:

- `results/table3_recomputed/table3_baseline_results.json`
- `results/table3_recomputed/table3_qwen3_reranker_results.json`
- `results/table3_recomputed/qwen3_8b/qwen3_8b_bootstrap.json`
- `results/supervised_recomputed/summary.json`
- `results/supervised_recomputed/bootstrap/supervised_bootstrap_results.json`

## Candidate-Construction Sensitivity

The candidate-construction experiments use fixed queries and cited evidence while changing the U candidate construction. Random-U candidates satisfy the temporal, citation, priority, application, component, CPC, and duplication constraints used for Hard-U candidates, but they are sampled without BM25 selection.

Across the fixed candidate experiments, Mixed-U training improves discrimination on Random-U candidates while producing smaller reductions on the original Hard-U candidates. The direction and magnitude of the effect therefore depend on candidate construction. This result should not be summarized as a model-independent improvement on every U distribution.

The separate scoring-model audit confirms that Hard-U candidates are more similar to claims than Random-U candidates under TF-IDF, BM25, MiniLM, and PatentSBERTa. All paired Random-U versus Hard-U differences remain significant after Holm correction at adjusted p = 0.002999.

Machine-readable sources:

- `results/random_u/random_vs_hard_summary.csv`
- `results/random_u/random_vs_hard_results.json`
- `results/random_u/random_u_generation_report.json`

## Transfer to Dense-Hard Candidates

Dense-Hard uses one independently selected U candidate per query. The primary evaluation contains 1,642 queries. Passage-disjoint and document-disjoint sensitivity subsets contain 1,555 and 1,389 queries.

### Canonical Qwen3-Reranker results

| Subset | Metric | Hard-only | Mixed-U | Difference |
|:--|:--|--:|--:|--:|
| All 1,642 queries | ROC-AUC | 0.608 | 0.736 | +0.128 |
| All 1,642 queries | P-C/U | 0.608 | 0.732 | +0.124 |
| Passage-disjoint, 1,555 | ROC-AUC | 0.618 | 0.735 | +0.117 |
| Passage-disjoint, 1,555 | P-C/U | 0.619 | 0.730 | +0.112 |
| Document-disjoint, 1,389 | ROC-AUC | 0.625 | 0.727 | +0.102 |
| Document-disjoint, 1,389 | P-C/U | 0.625 | 0.723 | +0.097 |

The canonical Qwen3-Reranker inference reverses the earlier provisional conclusion. Mixed-U improves both ROC-AUC and P-C/U on Dense-Hard candidates. Every confidence interval for the Mixed-U minus Hard-only difference excludes zero. The global Holm-adjusted p-value across 12 Dense-Hard comparisons is 0.011994.

The canonical predictions use six fixed LoRA adapters, consisting of Hard-only and Mixed-U conditions with seeds 13, 42, and 77. The inference script uses raw claim and passage text, a patent-specific instruction, maximum length 384, and fixed Qwen answer-token IDs.

Machine-readable sources and reproduction scripts:

- `results/dense_hard_recomputed/qwen3_reranker_0.6b/statistical_analysis/qwen_dense_hard_paired_tests.csv`
- `results/dense_hard_recomputed/qwen3_reranker_0.6b/statistical_analysis/qwen_dense_hard_bootstrap_ci.csv`
- `results/dense_hard_recomputed/qwen3_reranker_0.6b/statistical_analysis/table5_global_12_holm_recomputed.csv`
- `experiments/run_qwen_dense_hard_recomputed.py`
- `evaluation/qwen_dense_hard_statistics_recomputed.py`

## Direct and Hierarchical Prediction

For Qwen3-8B, hierarchical prediction increases macro-F1 from 0.249 to 0.291, a difference of 0.042. It also increases cited-versus-U ROC-AUC from 0.529 to 0.576, a difference of 0.047. Both differences remain significant after Holm correction at adjusted p = 0.012994.

The hierarchical formulation does not produce the same pattern for every supervised model. Its effects should therefore be reported by model and metric rather than as a general advantage.

Machine-readable source:

- `results/table3_recomputed/qwen3_8b/qwen3_8b_bootstrap.json`

## Temporal Robustness

Temporal robustness is evaluated on 1,678 U-eligible test queries. The early, middle, and late bins contain 568, 534, and 576 queries.

| System | Early C/U AUC | Middle C/U AUC | Late C/U AUC | Late minus Early |
|:--|--:|--:|--:|--:|
| MiniLM direct | 0.468 | 0.480 | 0.463 | -0.005 |
| DeBERTa direct | 0.483 | 0.487 | 0.488 | +0.004 |
| Qwen3-8B direct | 0.573 | 0.522 | 0.481 | -0.092 |
| Qwen3-8B hierarchical | 0.556 | 0.588 | 0.579 | +0.023 |

Direct Qwen3-8B exhibits a significant early-to-late reduction in cited-versus-U ROC-AUC. The difference is -0.0916 with a 95% confidence interval of [-0.1280, -0.0532] and Holm-adjusted p = 0.005997. Its P-C/U also decreases from 0.589 to 0.492. The corresponding temporal differences for MiniLM, DeBERTa, and hierarchical Qwen3-8B are not significant after Holm correction.

Machine-readable sources:

- `results/temporal_recomputed/temporal_robustness_results.json`
- `results/temporal_recomputed/temporal_bin_results.csv`
- `results/temporal_recomputed/temporal_early_late_tests.csv`

## Dependency Relaxation

The official split has zero cross-split overlap under the audited dependency relations. A relaxed row-level reconstruction reassigns 1,100 quarantined X/A rows. The relaxed test partition contains 282 of 3,950 rows, or 7.14%, that overlap under the priority-component relation.

Both models below are evaluated on the same frozen official X/A test set.

| Metric | Official | Relaxed | Difference | Holm p |
|:--|--:|--:|--:|--:|
| Accuracy | 0.496 | 0.514 | +0.018 | 0.031984 |
| Macro-F1 | 0.474 | 0.491 | +0.018 | 0.031984 |
| ROC-AUC | 0.503 | 0.517 | +0.014 | 0.031984 |
| Average precision | 0.498 | 0.510 | +0.012 | 0.031984 |

Relaxing dependency control produces small but consistent improvements in this X/A experiment. All four differences remain significant after Holm correction. Absolute performance remains close to chance, so the result measures a modest partition effect rather than strong evidence-category discrimination.

Machine-readable sources:

- `results/dependency_recomputed/dependency_model_results.json`
- `results/dependency_recomputed/dependency_paired_tests.csv`
- `results/dependency_recomputed/dependency_test_predictions.parquet`

## Statistical Protocol

Confidence intervals and paired comparisons use 2,000 bootstrap replicates with seed 42. The resampling unit is the lowercased and whitespace-normalized claim query. Reported intervals are 95% percentile intervals. Two-sided bootstrap p-values use finite-sample correction. Holm correction is applied within each stated comparison family. The Dense-Hard global family contains 12 model, metric, and subset comparisons.

## Versioning

These results use dataset revision `4a3ad685a395252c66932c49e98a00728089b614`, corresponding to the frozen XART-H v1.0.1 release. Corrections to experimental outputs are recorded through versioned result directories and Git history.
