"""Post-hoc construction-bias diagnostic for XART-H v1.0.1.

This analysis was motivated by the observed test-set direction of the
BM25 cited-versus-U score. It must be described as an exploratory,
post-hoc diagnostic rather than a preregistered confirmatory analysis.

The script expects local XART-H data, supervised NPZ logits, and
matched Random-U artifacts. Existing official results are not modified.
"""

from pathlib import Path

MYDRIVE = Path("/content/drive/MyDrive")
PROJECT_ROOT = MYDRIVE / "PatentSearchBench" / "XART-H"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
SUPERVISED_ROOT = EXPERIMENT_ROOT / "supervised_v1"

import json
import shutil
from datetime import datetime, timezone

REPO = GIT_REPO

PUBLIC_SCRIPT_DIR = REPO / "experiments"
PUBLIC_RESULT_DIR = (
    REPO / "results" / "construction_bias"
)
PUBLIC_DOC_PATH = (
    REPO
    / "docs"
    / "CONSTRUCTION_BIAS_STAGE1.md"
)

PUBLIC_SCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
PUBLIC_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
PUBLIC_DOC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------------
# Recover Stage 1 code from this notebook's execution history.
# ---------------------------------------------------------------

stage1_marker = (
    "XART-H CONSTRUCTION-BIAS DIAGNOSTIC: STAGE 1"
)

stage1_source = None

try:
    history = (
        get_ipython()
        .history_manager
        .input_hist_raw
    )

    for cell in reversed(history):
        if stage1_marker in cell:
            stage1_source = cell
            break
except Exception:
    stage1_source = None

if stage1_source is not None:
    script_path = (
        PUBLIC_SCRIPT_DIR
        / "run_construction_bias_stage1.py"
    )

    script_path.write_text(
        '''"""Post-hoc construction-bias diagnostic for XART-H v1.0.1.

This analysis was motivated by the observed test-set direction of the
BM25 cited-versus-U score. It must be described as an exploratory,
post-hoc diagnostic rather than a preregistered confirmatory analysis.

The script expects local XART-H data, supervised NPZ logits, and
matched Random-U artifacts. Existing official results are not modified.
"""

from pathlib import Path

MYDRIVE = Path("/content/drive/MyDrive")
PROJECT_ROOT = MYDRIVE / "PatentSearchBench" / "XART-H"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
SUPERVISED_ROOT = EXPERIMENT_ROOT / "supervised_v1"

'''
        + stage1_source,
        encoding="utf-8",
    )

    print("Saved script:", script_path)
else:
    print(
        "WARNING: Stage 1 code was not found in notebook history."
    )
    print(
        "Do not commit until the script has been added manually."
    )

# ---------------------------------------------------------------
# Copy aggregate results only.
# ---------------------------------------------------------------

aggregate_files = [
    "construction_bias_config.json",
    "deberta_seed_mean_vs_ensemble.csv",
    "deberta_seed_metrics.csv",
    "deberta_vs_surface_bootstrap.csv",
    "feature_distributions.csv",
    "fixed_baselines_hard_vs_random.csv",
    "hard_u_baseline_results.csv",
    "selected_lr_hyperparameters.csv",
]

for filename in aggregate_files:
    source = SOURCE_RESULT_DIR / filename

    if source.exists():
        destination = (
            PUBLIC_RESULT_DIR / filename
        )
        shutil.copy2(source, destination)
        print("Copied:", destination)
    else:
        print("Missing:", source)

# ---------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------

documentation = f"""# Construction-Bias Diagnostic

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

## Unresolved BM25 discrepancy

The manuscript Table 3 reports a full-test BM25 cited-versus-U ROC-AUC of
0.323. The Stage 1 full-test recomputation produced 0.329350.

On the 1,642-query matched Hard-U subset, the recomputed scores exactly
match the existing score cache and produce ROC-AUC 0.325055.

The full-test discrepancy is under investigation. Until resolved,
0.329350 must not be described as an exact reproduction of Table 3.

## Artifact policy

This branch includes aggregate configurations and results. It excludes
patent text, row-level scored files, model checkpoints, per-example logits,
authentication credentials, and private Google Drive data.

Generated: {datetime.now(timezone.utc).isoformat()}
"""

PUBLIC_DOC_PATH.write_text(
    documentation,
    encoding="utf-8",
)

# ---------------------------------------------------------------
# Ignore row-level or private artifacts.
# ---------------------------------------------------------------

gitignore_path = REPO / ".gitignore"

gitignore_text = (
    gitignore_path.read_text(encoding="utf-8")
    if gitignore_path.exists()
    else ""
)

new_ignore_lines = [
    "results/construction_bias/*.parquet",
    "results/construction_bias/*predictions*",
    "results/construction_bias/*logits*",
    "results/construction_bias/checkpoints/",
]

for line in new_ignore_lines:
    if line not in gitignore_text:
        gitignore_text += "\n" + line

gitignore_path.write_text(
    gitignore_text.strip() + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("FILES PREPARED")
print("=" * 80)

run_command(
    ["git", "status", "--short"],
    cwd=REPO,
)

print("\nDo not commit or push yet.")
print("Run the discrepancy-audit cell next.")