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

## Stage 2B: validation-defined lexical matching

Stage 2B evaluates whether DeBERTa retains an advantage after reducing
measured lexical and length differences within existing Hard-U queries.

For each U-eligible query, one cited candidate is selected as the nearest
candidate to the existing Hard U under five features: log-transformed BM25
score, token Jaccard similarity, claim-token coverage, log-transformed
passage length, and log-transformed passage-to-claim length ratio.

Features are standardized using training-split statistics. Matching uses
the maximum absolute standardized feature difference. No model prediction
or error information is used for candidate selection.

Matching calipers are fixed using validation nearest-pair distances:

| Caliper | Validation threshold |
|:--|--:|
| q25 | 0.751146 |
| q50, primary | 1.161164 |
| q75 | 1.637485 |

### Primary q50 result

The validation-defined q50 rule retains 870 of 1,678 test queries, or
51.8% of the U-eligible test population. Each retained query contributes
one cited candidate and one existing Hard U.

| Model | ROC-AUC | Pairwise accuracy |
|:--|--:|--:|
| BM25 | 0.441695 | 0.395402 |
| Validation-selected reversed BM25 | 0.558305 | 0.604598 |
| Length LR | 0.518228 | 0.513793 |
| Lexical LR | 0.563710 | 0.588506 |
| Combined LR | 0.565599 | 0.601149 |
| DeBERTa three-seed probability ensemble | 0.674500 | 0.695402 |

The DeBERTa ensemble exceeds combined LR by 0.108901 ROC-AUC, with a
95% paired query-bootstrap interval of [0.086389, 0.131857]. Its
pairwise-accuracy advantage is 0.094253, with an interval of
[0.055144, 0.133333].

### Matching sensitivity

| Test scope | Queries | Retention | Combined LR AUC | DeBERTa AUC |
|:--|--:|--:|--:|--:|
| q25 | 410 | 24.4% | 0.521868 | 0.644860 |
| q50, primary | 870 | 51.8% | 0.565599 | 0.674500 |
| q75 | 1,239 | 73.8% | 0.596959 | 0.693607 |

The direction of the DeBERTa advantage is consistent across all three
validation-defined calipers.

### Interpretation limits

Matching reduces but does not eliminate measured feature differences. On
the primary q50 subset, mean signed standardized differences range from
-0.134 to -0.029, while mean absolute standardized differences range from
0.368 to 0.492. Reversed BM25 also retains a pairwise accuracy of 0.605.

The matched subset must therefore be described as approximately balanced
under the measured features, not free of construction bias. The reduction
in model performance is not interpreted causally because matching changes
the evaluated query and cited-candidate population.

The results support the limited conclusion that the evaluated BM25,
overlap, and length baselines do not fully account for DeBERTa
discrimination. They do not establish that DeBERTa performs deep technical
reasoning or that unmeasured construction cues are absent.

## Artifact policy

This branch includes aggregate configurations and results. It excludes
patent text, row-level scored files, model checkpoints, per-example logits,
authentication credentials, and private Google Drive data.

Generated: 2026-09-21T22:13:25.943359+00:00
