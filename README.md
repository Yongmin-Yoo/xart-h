# XART-H

XART-H is a dependency-controlled temporal benchmark for hierarchical
patent prior-art retrieval and relevance assessment. It combines
examiner-assigned X and A citations with automatically constructed
hard U candidates.

## Dataset Download

The frozen dataset should be downloaded from Hugging Face:

**Hugging Face Dataset**

https://huggingface.co/datasets/yongminyoo91/xart-h

**Frozen release**

https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1

Experiments and external users should use the frozen Hugging Face release. Owner-side working copies are not public download locations.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "yongminyoo91/xart-h",
    revision="v1.0.1"
)
```

## Dataset Statistics

| Split | Queries | X | A | U | Pairs |
|:--|--:|--:|--:|--:|--:|
| Train | 8,374 | 8,462 | 8,450 | 8,275 | 25,187 |
| Validation | 1,809 | 1,813 | 1,813 | 1,656 | 5,282 |
| Test | 1,808 | 1,822 | 1,816 | 1,678 | 5,316 |
| **Total** | **11,991** | **12,097** | **12,079** | **11,609** | **35,785** |

## Tasks

1. Cited-versus-uncited retrieval: X/A vs. U
2. Fine-grained relevance assessment: X vs. A
3. Three-way classification: X/A/U
4. Query-level candidate ranking

## Model Inputs

Models must use only the following input columns:

- `text`: patent claim
- `text_b`: candidate prior-art passage

The prediction target is:

- `label`: A = 0, X = 1, U = 2

The following columns are provided only for auditing and reconstruction.
They must not be used as predictive features:

- `label_name`
- `_source_split`
- `_u_method`
- `_u_bm25_score`
- `_u_candidate_base`
- `_u_candidate_publication_date`
- `_u_target_component`
- `_u_candidate_component`

## Repository Structure

```text
configs/       Experiment configurations
scripts/       Construction and verification scripts
baselines/     Baseline implementations
evaluation/    Metrics and statistical tests
docs/          Benchmark and audit documentation
results/       Reproducible result summaries
```

## Data Integrity

XART-H uses chronological and dependency-controlled partitions. No
cross-partition overlap was detected under normalized claim and passage
text, patent application identifiers, cited-document identifiers,
observed priority links, and dependency components.

Complete patent-family disjointness is not claimed because priority
metadata is only partially observable.

## Provenance

The X/A records originate from PatentMatch:

> Julian Risch, Nicolas Alder, Christoph Hewel, and Ralf Krestel.
> PatentMatch: A Dataset for Matching Patent Claims & Prior Art.
> arXiv:2012.13919, 2020.

XART-H provides priority-link-augmented temporal partitions, hard U
candidates, hierarchical tasks, and leakage audits.

## License

XART-H is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Users should cite both PatentMatch and XART-H.

<!-- XARTH_RESULTS_START -->
## Current Results

All headline values use the frozen XART-H v1.0.1 release. Supervised
results are means over seeds 13, 42, and 77. Full tables, confidence
intervals, and machine-readable provenance are reported in
[`docs/RESULTS.md`](docs/RESULTS.md) and [`results/`](results/).

The corrected supervised Cited-versus-U results are **0.726 ± 0.007**
ROC-AUC for MiniLM and **0.745 ± 0.014** for DeBERTa. These replace the
earlier non-reproducible values of 0.478 and 0.474. Evaluation on the
1,642-query Table 4 matched subset produces 0.731 for MiniLM and 0.746
for DeBERTa, confirming consistency with the controlled-training
experiment.

For X-versus-A discrimination, MiniLM obtains 0.530 ± 0.001 ROC-AUC
and DeBERTa obtains 0.505 ± 0.009. Direct three-way macro-F1 is
0.409 ± 0.018 for MiniLM and 0.450 ± 0.007 for DeBERTa. Hierarchical
macro-F1 is 0.415 ± 0.006 and 0.430 ± 0.010, respectively.

Among non-fine-tuned systems, Qwen3-Reranker-0.6B obtains 0.546
Cited-versus-U ROC-AUC. Qwen3-8B obtains 0.335 direct three-way
accuracy and 0.291 hierarchical macro-F1.

The canonical Dense-Hard evaluation shows that Mixed-U training
improves Qwen3-Reranker ROC-AUC from 0.608 to 0.736 and pairwise
Cited-over-Dense-Hard accuracy from 0.608 to 0.732. All 12 Dense-Hard
comparisons remain significant after global Holm correction at
adjusted p = 0.011994.

The corrected temporal analysis retains the significant Qwen3-8B
direct decline from 0.573 in the early period to 0.481 in the late
period and additionally reports temporal results recomputed from the
corrected supervised logits.
<!-- XARTH_RESULTS_END -->
