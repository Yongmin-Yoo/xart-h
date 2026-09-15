# Current Benchmark Results

These results use XART-H v1.0.1. Supervised values are
means over seeds 13, 42, and 77 unless otherwise stated.

## Test Results

| Model | Cited/U AUC | X/A AUC | 3-way Accuracy | Macro-F1 |
|:--|--:|--:|--:|--:|
| Random scoring | 0.499 | 0.507 | N/A | N/A |
| TF-IDF | 0.347 | 0.516 | N/A | N/A |
| BM25 | 0.323 | 0.513 | N/A | N/A |
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
provided in the result files. Random U, temporal, and
subgroup analyses remain pending.


## Metric Correction

Supervised ranking metrics use task-specific scores:
`p(A)+p(X)` for cited-versus-U ranking,
`p(X)/(p(A)+p(X))` for X-versus-A ranking, and
`2p(X)+p(A)` for graded nDCG. Classification and binary
ROC-AUC results are unchanged.
