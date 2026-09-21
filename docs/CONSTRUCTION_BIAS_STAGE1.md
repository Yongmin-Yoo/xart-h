# Construction-Bias Diagnostic

## Status

This directory contains a post-hoc diagnostic of surface-feature and
candidate-construction signals in the XART-H cited-versus-U task.

The frozen XART-H v1.0.1 dataset and existing official benchmark results
are not modified.

## Evaluation scopes

The full official test scope contains 5,316 rows and 1,808 normalized
queries.

The U-eligible test scope contains 5,056 rows and 1,678 normalized
queries.

The matched Hard-U versus Random-U scope contains 1,642 queries. Results
from these scopes must not be treated as if they were computed on
identical examples.

## Stage 1 results

On the U-eligible Hard-U test scope:

| Model | ROC-AUC | Pairwise accuracy |
|:--|--:|--:|
| BM25 | 0.330837 | 0.297795 |
| Validation-selected reversed BM25 | 0.669163 | 0.702205 |
| Length LR | 0.580391 | 0.569537 |
| Lexical LR | 0.699808 | 0.701311 |
| Combined LR | 0.700106 | 0.708164 |
| DeBERTa three-seed probability ensemble | 0.759593 | 0.759565 |

Relative to combined LR, the DeBERTa ensemble improves ROC-AUC by
0.059487, with a 95% paired query-bootstrap interval of
[0.046975, 0.072786]. Its pairwise-accuracy improvement is 0.051400,
with an interval of [0.032924, 0.069846].

The measured lexical and length baselines provide strong performance,
but DeBERTa significantly exceeds them. This difference does not
identify the nature of the signal used by DeBERTa.

## Matched Hard-U versus Random-U

| Model | Hard-U AUC | Random-U AUC |
|:--|--:|--:|
| BM25 | 0.325055 | 0.747611 |
| Validation-selected reversed BM25 | 0.674945 | 0.252389 |
| Length LR | 0.586843 | 0.474180 |
| Lexical LR | 0.707546 | 0.281202 |
| Combined LR | 0.707733 | 0.263080 |

The fixed surface baselines are highly sensitive to the candidate-selection
regime. DeBERTa has not yet been evaluated on Random U because the local
supervised artifacts contain logits but no model checkpoints.

## Post-hoc status

The reversed-BM25 analysis was motivated by the already observed test
result. Its direction was subsequently fixed using validation, but the
test split is not a previously unseen confirmatory evaluation.

## BM25 full-test value

The full-test BM25 cited-versus-U ROC-AUC was recomputed as
0.329350 using the documented implementation. The manuscript and
result summary report the rounded value 0.329. The corresponding
reversed score achieves an ROC-AUC of 0.670650.

The matched Hard-U result remains 0.325055 because it is computed on
a different 1,642-query evaluation subset. Its corresponding reversed
score is 0.674945. Values from the full and matched evaluation scopes
must not be treated as if they were computed on identical examples.

## Artifact policy

This branch includes aggregate configurations and results. It excludes
patent text, row-level scored files, model checkpoints, per-example logits,
authentication credentials, and private Google Drive data.

Generated: 2026-09-21T22:13:25.943359+00:00
