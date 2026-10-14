<div align="center">

# XART-H

### A dependency-controlled temporal benchmark for patent prior-art retrieval

[![Dataset](https://img.shields.io/badge/🤗_Dataset-XART--H-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/yongminyoo91/xart-h)
[![Release](https://img.shields.io/badge/Release-v1.0.1-2EA44F?style=for-the-badge)](https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1)
[![Results](https://img.shields.io/badge/Results-Reproducible-6F42C1?style=for-the-badge)](docs/RESULTS.md)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-555555?style=for-the-badge)](LICENSE)

<br>

**XART-H** combines examiner-assigned **X** and **A** patent citations
with temporally eligible, retrieval-selected **U** candidates.

It supports controlled evaluation of patent retrieval, evidence assessment,
joint classification, and query-level ranking.

<br>

[**Dataset**](https://huggingface.co/datasets/yongminyoo91/xart-h)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Frozen v1.0.1**](https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Results**](docs/RESULTS.md)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**Artifacts**](results/)
&nbsp;&nbsp;•&nbsp;&nbsp;
[**License**](LICENSE)

</div>

---

## Why XART-H?

Patent prior-art evaluation is vulnerable to temporal leakage, dependency
overlap, and unrealistically easy negative candidates. XART-H is designed to
make these effects visible and measurable.

<table>
<tr>
<td width="25%" align="center">
<strong>⏳ Temporal control</strong><br>
<sub>Chronological train, validation, and test partitions</sub>
</td>
<td width="25%" align="center">
<strong>🔗 Dependency control</strong><br>
<sub>Priority-link and component-aware auditing</sub>
</td>
<td width="25%" align="center">
<strong>🎯 Hard candidates</strong><br>
<sub>Retrieval-selected, temporally eligible U passages</sub>
</td>
<td width="25%" align="center">
<strong>📊 Reproducible results</strong><br>
<sub>Versioned outputs and bootstrap statistics</sub>
</td>
</tr>
</table>

> [!IMPORTANT]
> Use the frozen Hugging Face release **`v1.0.1`** for experiments.
> This GitHub repository contains code, audits, and corrected experimental
> outputs; it is not the canonical dataset download location.

---

## Quick start

### Install

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
   Rank cited evidence above U and X above A.

---

<!-- XARTH_RESULTS_START -->
## Results

All headline results use the frozen **XART-H v1.0.1** release.
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

### Main test results

| Model | Cited/U ROC-AUC | X/A ROC-AUC | Joint Accuracy | Joint Macro-F1 |
|:--|--:|--:|--:|--:|
| Random | 0.518 | 0.512 | — | — |
| TF-IDF | 0.348 | 0.516 | — | — |
| BM25 | 0.331 | 0.514 | — | — |
| MiniLM bi-encoder | 0.527 | 0.524 | — | — |
| PatentSBERTa | 0.495 | 0.511 | — | — |
| MiniLM cross-encoder | 0.427 | 0.525 | — | — |
| Qwen3-Reranker-0.6B | 0.546 | 0.513 | — | — |
| Qwen3-8B direct | 0.529 | 0.490 | 0.335 | 0.249 |
| MiniLM supervised | **0.726 ± 0.007** | **0.530 ± 0.001** | 0.437 ± 0.018 | 0.409 ± 0.018 |
| DeBERTa supervised | **0.745 ± 0.014** | 0.505 ± 0.009 | **0.453 ± 0.007** | **0.450 ± 0.007** |

### Query-level ranking

| Model | P-C/U | P-X/A |
|:--|--:|--:|
| Qwen3-8B direct | 0.539 | 0.492 |
| MiniLM supervised | 0.711 | 0.509 |
| DeBERTa supervised | **0.747** | **0.510** |

### Dense-Hard transfer

Mixed-U training substantially improves Qwen3-Reranker transfer to
PatentSBERTa-selected Dense-Hard candidates.

| Evaluation | Hard-only | Mixed-U | Difference |
|:--|--:|--:|--:|
| ROC-AUC | 0.608 | **0.736** | **+0.128** |
| Pairwise Cited-over-Dense-Hard | 0.608 | **0.732** | **+0.124** |

The global Holm-adjusted value across all 12 Dense-Hard comparisons is
**`p = 0.011994`**.

### Robustness findings

- **Temporal:** Qwen3-8B direct Cited/U ROC-AUC decreases from
  **0.573** in the early period to **0.481** in the late period.
- Corrected supervised logits show significant early-to-late declines:
  MiniLM direct **0.779 → 0.697** and DeBERTa direct **0.809 → 0.743**.
- **Dependency relaxation:** DeBERTa X/A ROC-AUC changes from
  **0.503 → 0.517**, with Holm-adjusted **`p = 0.031984`**.
- **Prediction formulation:** Qwen3-8B hierarchical prediction improves
  Macro-F1 from **0.249 → 0.291**.

For confidence intervals, per-seed values, matched-subset diagnostics, and
machine-readable provenance, see:

- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`results/table3_supervised_v2/`](results/table3_supervised_v2/)
- [`results/temporal_supervised_v2/`](results/temporal_supervised_v2/)
- [`results/dense_hard_recomputed/`](results/dense_hard_recomputed/)
- [`results/dependency_recomputed/`](results/dependency_recomputed/)
<!-- XARTH_RESULTS_END -->

---

## Model input policy

Models may use only the following fields:

~~~text
text      Patent claim
text_b    Candidate prior-art passage
label     A=0, X=1, U=2
~~~

> [!WARNING]
> Audit and reconstruction fields must not be used as predictive features.

<details>
<summary><strong>Show audit-only fields</strong></summary>

<br>

- `label_name`
- `_source_split`
- `_u_method`
- `_u_bm25_score`
- `_u_candidate_base`
- `_u_candidate_publication_date`
- `_u_target_component`
- `_u_candidate_component`

</details>

---

## Repository map

~~~text
xart-h/
├── baselines/       Baseline implementations
├── configs/         Experiment configurations
├── data/            Public audit and reconstruction inputs
├── docs/            Detailed benchmark documentation
├── evaluation/      Metrics and statistical testing
├── experiments/     Training and robustness experiments
├── results/         Versioned machine-readable outputs
└── scripts/         Construction and verification utilities
~~~

| Resource | Description |
|:--|:--|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Canonical benchmark tables and interpretation |
| [`results/`](results/) | Machine-readable results and audits |
| [`experiments/`](experiments/) | Training and robustness pipelines |
| [`evaluation/`](evaluation/) | Metrics, bootstrap tests, and Holm correction |
| [`scripts/`](scripts/) | Construction and integrity utilities |

---

## Data integrity and limitations

XART-H uses chronological and dependency-controlled partitions. No
cross-partition overlap was detected under normalized claim and passage text,
patent application identifiers, cited-document identifiers, observed priority
links, and dependency components.

Important limitations:

- **U means observed citation absence**, not adjudicated irrelevance.
- Priority metadata are partially observable; complete patent-family
  disjointness is not claimed.
- X/A discrimination remains close to chance for the evaluated systems.
- Candidate construction materially affects Cited/U performance.

---

## Provenance

The original X/A records derive from:

> Julian Risch, Nicolas Alder, Christoph Hewel, and Ralf Krestel.
> **PatentMatch: A Dataset for Matching Patent Claims & Prior Art.**
> arXiv:2012.13919, 2020.

XART-H adds priority-link-augmented temporal partitions, hard U candidates,
hierarchical evaluation tasks, and leakage audits.

---

## License and attribution

XART-H is released under the
[Creative Commons Attribution 4.0 International License](LICENSE).

Users should cite both **PatentMatch** and **XART-H** when publishing results
based on this benchmark.

---

<div align="center">

**XART-H · Controlled evaluation for patent prior-art retrieval**

[Hugging Face](https://huggingface.co/datasets/yongminyoo91/xart-h)
&nbsp;•&nbsp;
[Results](docs/RESULTS.md)
&nbsp;•&nbsp;
[CC BY 4.0](LICENSE)

</div>
