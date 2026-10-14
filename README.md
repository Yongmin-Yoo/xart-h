<div align="center">

# PatScout

### A Temporal and Dependency-Controlled Benchmark for Patent Prior-Art Search

[![Dataset](https://img.shields.io/badge/Dataset-PatScout-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/yongminyoo91/xart-h)
[![Release](https://img.shields.io/badge/Release-v1.0.1-2EA44F?style=for-the-badge)](https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1)
[![Results](https://img.shields.io/badge/Results-Reproducible-6F42C1?style=for-the-badge)](docs/RESULTS.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-555555?style=for-the-badge)](LICENSE)

<br>

**PatScout** is a controlled benchmark for patent prior-art search and
candidate assessment. It combines examiner-assigned **X** and **A** citations
with temporally eligible, retrieval-selected **U** candidates.

The benchmark supports binary discrimination, three-way evidence
classification, query-level ranking, temporal robustness analysis, and
dependency-control auditing.

<br>

[**Dataset**](https://huggingface.co/datasets/yongminyoo91/xart-h)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Frozen Release**](https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Results**](docs/RESULTS.md)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Artifacts**](results/)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**License**](LICENSE)

</div>

---

## Why PatScout?

Patent prior-art evaluation is vulnerable to temporal leakage, dependency
overlap, and unrealistically easy negative candidates. PatScout is designed
to make these effects visible, measurable, and reproducible.

<table>
<tr>
<td width="25%" align="center">
<strong>Temporal control</strong><br>
<sub>Chronological train, validation, and test partitions</sub>
</td>
<td width="25%" align="center">
<strong>Dependency control</strong><br>
<sub>Priority-link and component-aware leakage auditing</sub>
</td>
<td width="25%" align="center">
<strong>Hard candidates</strong><br>
<sub>Retrieval-selected and temporally eligible U passages</sub>
</td>
<td width="25%" align="center">
<strong>Versioned evaluation</strong><br>
<sub>Machine-readable outputs and bootstrap statistics</sub>
</td>
</tr>
</table>

> [!IMPORTANT]
> Use the frozen Hugging Face release **`v1.0.1`** for benchmark experiments.
> The GitHub repository contains evaluation code, audits, and versioned
> experimental outputs; the Hugging Face release is the canonical dataset
> download location.

---

## Quick start

### Install the dataset loader

~~~bash
pip install datasets
~~~

### Load the frozen release

~~~python
from datasets import load_dataset

dataset = load_dataset(
    "yongminyoo91/xart-h",
    revision="v1.0.1",
)

print(dataset)
print(dataset["train"][0])
~~~

### Labels

| Label | ID | Meaning |
|:--|:--:|:--|
| **A** | `0` | Relevant background prior art |
| **X** | `1` | Particularly relevant prior art |
| **U** | `2` | Uncited, temporally eligible hard candidate |

> [!NOTE]
> U denotes observed citation absence under the benchmark construction
> protocol. It should not be interpreted as adjudicated irrelevance.

---

## Benchmark at a glance

| Split | Queries | X | A | U | Pairs |
|:--|--:|--:|--:|--:|--:|
| Train | 8,374 | 8,462 | 8,450 | 8,275 | 25,187 |
| Validation | 1,809 | 1,813 | 1,813 | 1,656 | 5,282 |
| Test | 1,808 | 1,822 | 1,816 | 1,678 | 5,316 |
| **Total** | **11,991** | **12,097** | **12,079** | **11,609** | **35,785** |

### Evaluation tasks

1. **Cited vs. U**
   Distinguish cited X/A evidence from hard U candidates.

2. **X vs. A**
   Distinguish particularly relevant evidence from background evidence.

3. **X/A/U classification**
   Predict the three evidence classes jointly.

4. **Query-level ranking**
   Rank cited evidence above U and X evidence above A evidence.

---

<!-- PATSCOUT_RESULTS_START -->
## Main results

All headline results use the frozen **PatScout v1.0.1** release.
Supervised values are means over seeds **13, 42, and 77**.

<table>
<tr>
<td width="25%" align="center">
<strong>Best Cited/U AUC</strong><br>
<code>0.745 ± 0.014</code><br>
<sub>DeBERTa supervised</sub>
</td>
<td width="25%" align="center">
<strong>Best X/A AUC</strong><br>
<code>0.530 ± 0.001</code><br>
<sub>MiniLM supervised</sub>
</td>
<td width="25%" align="center">
<strong>Best Joint Macro-F1</strong><br>
<code>0.450 ± 0.007</code><br>
<sub>DeBERTa direct</sub>
</td>
<td width="25%" align="center">
<strong>Dense-Hard AUC</strong><br>
<code>0.608 → 0.736</code><br>
<sub>Qwen3-Reranker Mixed-U</sub>
</td>
</tr>
</table>

### Binary discrimination

| Model | Cited/U ROC-AUC | X/A ROC-AUC |
|:--|--:|--:|
| Random scoring | 0.518 | 0.512 |
| TF-IDF | 0.348 | 0.516 |
| BM25 | 0.331 | 0.514 |
| MiniLM bi-encoder | 0.527 | 0.524 |
| PatentSBERTa | 0.495 | 0.511 |
| MiniLM cross-encoder | 0.427 | 0.525 |
| Qwen3-Reranker-0.6B | 0.546 | 0.513 |
| Qwen3-8B direct | 0.529 | 0.490 |
| MiniLM supervised | **0.726 ± 0.007** | **0.530 ± 0.001** |
| DeBERTa-v3-base supervised | **0.745 ± 0.014** | 0.505 ± 0.009 |

### Joint classification

| Model | Prediction | Accuracy | Macro-F1 |
|:--|:--|--:|--:|
| Qwen3-8B | Direct | 0.335 | 0.249 |
| Qwen3-8B | Hierarchical | 0.347 | 0.291 |
| MiniLM supervised | Direct | 0.437 ± 0.018 | 0.409 ± 0.018 |
| MiniLM supervised | Hierarchical | 0.447 ± 0.006 | 0.415 ± 0.006 |
| DeBERTa supervised | Direct | **0.453 ± 0.007** | **0.450 ± 0.007** |
| DeBERTa supervised | Hierarchical | 0.443 ± 0.002 | 0.430 ± 0.010 |

For the complete result tables, seed-level outputs, confidence intervals, and
machine-readable sources, see [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Controlled analyses

### Candidate-construction sensitivity

Random-U candidates are substantially easier than retrieval-selected Hard-U
candidates. The released analysis evaluates TF-IDF, BM25, and supervised
systems on the same matched query set.

- Eligible test queries: **1,678**
- Matched Random-U queries: **1,642**
- Random-U generation coverage: **97.85%**
- Holm-adjusted significance value: **0.002999**

Artifacts:

- [`results/random_u/`](results/random_u/)
- [`results/random_u/generated/`](results/random_u/generated/)

### Dense-Hard evaluation

The canonical Qwen3-Reranker evaluation compares Hard-only and Mixed-U
training on dense retrieval candidates.

| Subset | Queries | ROC-AUC gain | Pairwise gain |
|:--|--:|--:|--:|
| All matched queries | 1,642 | +0.128 | +0.124 |
| Passage-disjoint | 1,555 | +0.117 | +0.112 |
| Document-disjoint | 1,389 | +0.102 | +0.097 |

The global 12-comparison Holm-adjusted value is **0.011994**.

Artifacts:

- [`results/dense_hard_recomputed/`](results/dense_hard_recomputed/)
- [`experiments/run_qwen_dense_hard_recomputed.py`](experiments/run_qwen_dense_hard_recomputed.py)
- [`evaluation/qwen_dense_hard_statistics_recomputed.py`](evaluation/qwen_dense_hard_statistics_recomputed.py)

### Prediction formulation

For Qwen3-8B, hierarchical prediction changes the three-way decision profile:

- Macro-F1: **0.249 → 0.291**
- Cited/U ROC-AUC: **0.529 → 0.576**
- Holm-adjusted value for the significant comparisons: **0.012994**

Artifacts:

- [`results/table3_recomputed/qwen3_8b/`](results/table3_recomputed/qwen3_8b/)
- [`experiments/run_qwen3_8b.py`](experiments/run_qwen3_8b.py)

### Temporal robustness

Several systems show lower Cited/U performance in the late test period.

Selected direct-model results:

| Model | Early | Late |
|:--|--:|--:|
| Qwen3-8B | 0.573 | 0.481 |
| MiniLM supervised | 0.779 | 0.697 |
| DeBERTa supervised | 0.809 | 0.743 |

The reported significant early-to-late comparisons have Holm-adjusted
values of approximately **0.006**.

Artifacts:

- [`results/temporal_recomputed/`](results/temporal_recomputed/)
- [`results/temporal_supervised_v2/`](results/temporal_supervised_v2/)

### Dependency relaxation

Relaxing dependency control produces small but consistent improvements in
X/A evaluation.

| Metric | Official | Relaxed | Change |
|:--|--:|--:|--:|
| Accuracy | 0.496 | 0.514 | +0.018 |
| Macro-F1 | 0.474 | 0.491 | +0.018 |
| ROC-AUC | 0.503 | 0.517 | +0.014 |
| Average precision | 0.498 | 0.510 | +0.012 |

All four comparisons have Holm-adjusted values of **0.031984**.

Artifacts:

- [`results/dependency_recomputed/`](results/dependency_recomputed/)
- [`experiments/run_dependency_reconstruction_audit.py`](experiments/run_dependency_reconstruction_audit.py)
- [`experiments/run_dependency_robustness.py`](experiments/run_dependency_robustness.py)

<!-- PATSCOUT_RESULTS_END -->

---

## Statistical protocol

Unless an artifact states otherwise:

- supervised models use seeds **13, 42, and 77**;
- uncertainty estimates use **2,000 bootstrap replicates**;
- bootstrap sampling uses seed **42**;
- reported intervals are **95% percentile intervals**;
- resampling is performed at the normalized-query level;
- paired comparisons use two-sided bootstrap tests;
- Holm correction is applied within each declared comparison family.

The Dense-Hard global family contains 12 model, metric, and subset
comparisons.

---

## Reproducing the benchmark

Clone the repository and check out the public experiment branch:

~~~bash
git clone --branch Yongmin-Yoo --single-branch \
  https://github.com/Yongmin-Yoo/xart-h.git

cd xart-h
~~~

Install the core dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Representative entry points:

~~~bash
# Table 3 non-supervised baselines
python experiments/run_table3_baselines.py

# Table 3 supervised systems
python experiments/train_table3_supervised_v2.py

# Qwen3-8B direct and hierarchical evaluation
python experiments/run_qwen3_8b.py

# Dependency reconstruction audit
python experiments/run_dependency_reconstruction_audit.py
~~~

> [!CAUTION]
> Some accelerator-dependent experiments were produced with environment
> versions recorded in their artifact manifests. Consult the relevant
> manifest before attempting an exact rerun, particularly for Qwen-based
> experiments.

---

## Repository structure

~~~text
.
├── configs/        Experiment configuration files
├── data/           Public audit and reconstruction inputs
├── docs/           Extended results and documentation
├── evaluation/     Metrics, bootstrap tests, and statistical utilities
├── experiments/    Training and evaluation entry points
├── results/        Versioned machine-readable outputs
├── scripts/        Dataset construction and integrity utilities
├── LICENSE         Creative Commons Attribution 4.0
├── README.md       Project overview
└── requirements.txt
~~~

Notable files:

- [`docs/RESULTS.md`](docs/RESULTS.md): canonical human-readable results
- [`results/RESULT_MANIFEST.json`](results/RESULT_MANIFEST.json): artifact manifest
- [`experiments/run_table3_baselines.py`](experiments/run_table3_baselines.py): baseline evaluation
- [`experiments/train_table3_supervised_v2.py`](experiments/train_table3_supervised_v2.py): supervised training
- [`evaluation/`](evaluation/): bootstrap and multiple-testing utilities

---

## Data integrity and limitations

PatScout uses chronological and dependency-controlled partitions. The release
includes audits for normalized-text overlap, document identifiers, observed
priority links, and reconstructed dependency components.

Important limitations:

- U means observed citation absence, not adjudicated irrelevance.
- Priority metadata are partially observable.
- Complete patent-family disjointness is not claimed.
- X/A discrimination remains difficult for the evaluated systems.
- Candidate construction materially affects Cited/U performance.
- Some large model checkpoints and training caches are not included in Git.

---

## Provenance

The original X/A records derive from:

> Julian Risch, Nicolas Alder, Christoph Hewel, and Ralf Krestel.
> **PatentMatch: A Dataset for Matching Patent Claims & Prior Art.**
> arXiv:2012.13919, 2020.

PatScout adds priority-link-augmented temporal partitions, retrieval-selected
U candidates, controlled evaluation tasks, and leakage audits.

---

## Paper

This repository hosts the dataset release, evaluation code, and experimental
artifacts for:

> **PatScout: A Temporal and Dependency-Controlled Benchmark for Patent
> Prior-Art Search**

Final citation metadata will be added after publication.

---

## License and attribution

PatScout is released under the
[Creative Commons Attribution 4.0 International License](LICENSE).

Users should cite both **PatentMatch** and **PatScout** when publishing
results based on this benchmark.

---

<div align="center">

**PatScout · Controlled evaluation for patent prior-art search**

[Hugging Face](https://huggingface.co/datasets/yongminyoo91/xart-h)
&nbsp;•&nbsp;
[Results](docs/RESULTS.md)
&nbsp;•&nbsp;
[Artifacts](results/)
&nbsp;•&nbsp;
[CC BY 4.0](LICENSE)

</div>
