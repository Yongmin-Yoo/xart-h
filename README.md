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

The Google Drive copy is retained only as an owner-side backup:

```text
/content/drive/MyDrive/PatentSearchBench/external/PatentMatch/releases/XART-H-v1.0
```

The Google Drive path is not a public download URL. Experiments and
external users should use the Hugging Face release.

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
