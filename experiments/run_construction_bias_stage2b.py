"""Validation-defined lexical-matching diagnostic for XART-H.

This post-hoc diagnostic selects, within each U-eligible query, the cited
candidate nearest to the existing Hard U under measured lexical and length
features. Feature standardization is fitted on training data, and matching
calipers are selected from validation nearest-pair distances.

No model predictions or error information are used for matching. The q50
validation-defined caliper is the primary analysis; q25 and q75 are
sensitivity analyses.

This script is a sequential companion to
run_construction_bias_stage1.py and expects the Stage 1 objects and scores
to be available in the execution environment.
"""

# =====================================================================
# COMMIT AND PUSH STAGE 2B LEXICAL-MATCHING RESULTS
# =====================================================================

import os
import stat
import shutil
import subprocess
from pathlib import Path
from google.colab import userdata

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

GIT_REPO = Path(
    "/content/xart-h-construction-bias-repo"
)

SOURCE_PROJECT = Path(
    "/content/drive/MyDrive/PatentSearchBench/XART-H"
)

SOURCE_STAGE2_DIR = (
    SOURCE_PROJECT
    / "experiments"
    / "construction_bias_v1"
    / "lexical_matching"
)

BRANCH = "analysis/construction-bias-v1"

PUBLIC_SCRIPT_PATH = (
    GIT_REPO
    / "experiments"
    / "run_construction_bias_stage2b.py"
)

PUBLIC_RESULT_DIR = (
    GIT_REPO
    / "results"
    / "construction_bias"
    / "lexical_matching"
)

PUBLIC_DOC_PATH = (
    GIT_REPO
    / "docs"
    / "CONSTRUCTION_BIAS_STAGE1.md"
)

assert (GIT_REPO / ".git").exists(), GIT_REPO
assert SOURCE_STAGE2_DIR.exists(), SOURCE_STAGE2_DIR
assert PUBLIC_DOC_PATH.exists(), PUBLIC_DOC_PATH

PUBLIC_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Git helper
# ---------------------------------------------------------------------

def git(*args, check=True, env=None):
    result = subprocess.run(
        ["git", *args],
        cwd=GIT_REPO,
        text=True,
        capture_output=True,
        env=env,
    )

    print("$ git", " ".join(args))

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Git command failed: {' '.join(args)}"
        )

    return result


# ---------------------------------------------------------------------
# 1. Verify branch and initial state
# ---------------------------------------------------------------------

current_branch = git(
    "branch",
    "--show-current",
).stdout.strip()

assert current_branch == BRANCH, (
    f"Expected {BRANCH}, found {current_branch}"
)

print("\nInitial Git status:")
git("status", "--short")


# ---------------------------------------------------------------------
# 2. Recover the executed Stage 2B cell
# ---------------------------------------------------------------------

stage2_marker = (
    "XART-H CONSTRUCTION-BIAS DIAGNOSTIC: STAGE 2B"
)

stage2_source = None

try:
    history = (
        get_ipython()
        .history_manager
        .input_hist_raw
    )

    for cell in reversed(history):
        if stage2_marker in cell:
            stage2_source = cell
            break
except Exception:
    stage2_source = None

assert stage2_source is not None, (
    "Stage 2B code was not found in this notebook's "
    "execution history. Do not commit without the script."
)

stage2_preamble = '''"""Validation-defined lexical-matching diagnostic for XART-H.

This post-hoc diagnostic selects, within each U-eligible query, the cited
candidate nearest to the existing Hard U under measured lexical and length
features. Feature standardization is fitted on training data, and matching
calipers are selected from validation nearest-pair distances.

No model predictions or error information are used for matching. The q50
validation-defined caliper is the primary analysis; q25 and q75 are
sensitivity analyses.

This script is a sequential companion to
run_construction_bias_stage1.py and expects the Stage 1 objects and scores
to be available in the execution environment.
"""

'''

PUBLIC_SCRIPT_PATH.write_text(
    stage2_preamble + stage2_source,
    encoding="utf-8",
)

print("Saved script:", PUBLIC_SCRIPT_PATH)


# ---------------------------------------------------------------------
# 3. Copy aggregate Stage 2B results only
# ---------------------------------------------------------------------

aggregate_files = [
    "matching_calipers.json",
    "lexical_matching_balance.csv",
    "lexical_matching_config.json",
    "lexical_matching_results.csv",
    "primary_q50_deberta_comparisons.csv",
    "primary_q50_model_intervals.csv",
]

for filename in aggregate_files:
    source = SOURCE_STAGE2_DIR / filename
    assert source.exists(), source

    destination = (
        PUBLIC_RESULT_DIR / filename
    )

    shutil.copy2(
        source,
        destination,
    )

    print("Copied:", destination)

# Explicitly verify that row-level assignments are not copied.
assert not (
    PUBLIC_RESULT_DIR
    / "lexical_matching_assignments.parquet"
).exists(), (
    "Row-level assignment file must not be committed."
)


# ---------------------------------------------------------------------
# 4. Append Stage 2B documentation
# ---------------------------------------------------------------------

documentation = PUBLIC_DOC_PATH.read_text(
    encoding="utf-8"
)

stage2_section = """## Stage 2B: validation-defined lexical matching

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

"""

if (
    "## Stage 2B: validation-defined lexical matching"
    not in documentation
):
    artifact_marker = "## Artifact policy"

    assert artifact_marker in documentation, (
        "Could not locate the documentation insertion point."
    )

    documentation = documentation.replace(
        artifact_marker,
        stage2_section + artifact_marker,
    )

    PUBLIC_DOC_PATH.write_text(
        documentation,
        encoding="utf-8",
    )

    print("Stage 2B documentation appended.")
else:
    print("Stage 2B documentation already present.")


# ---------------------------------------------------------------------
# 5. Update .gitignore for nested row-level artifacts
# ---------------------------------------------------------------------

gitignore_path = GIT_REPO / ".gitignore"

gitignore_text = (
    gitignore_path.read_text(
        encoding="utf-8"
    )
    if gitignore_path.exists()
    else ""
)

ignore_entries = [
    "results/construction_bias/**/*.parquet",
    "results/construction_bias/**/*predictions*",
    "results/construction_bias/**/*logits*",
    "results/construction_bias/**/checkpoints/",
]

for entry in ignore_entries:
    if entry not in gitignore_text:
        gitignore_text += "\n" + entry

gitignore_path.write_text(
    gitignore_text.strip() + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# 6. Syntax validation
# ---------------------------------------------------------------------

syntax_check = subprocess.run(
    [
        "python",
        "-m",
        "py_compile",
        str(PUBLIC_SCRIPT_PATH),
    ],
    cwd=GIT_REPO,
    text=True,
    capture_output=True,
)

if syntax_check.returncode != 0:
    print(syntax_check.stderr)
    raise RuntimeError(
        "Stage 2B script failed syntax validation."
    )

print("Python syntax check: PASSED")


# ---------------------------------------------------------------------
# 7. Review and commit
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES BEFORE STAGING")
print("=" * 80)

git("status", "--short")

git(
    "add",
    ".gitignore",
    "docs/CONSTRUCTION_BIAS_STAGE1.md",
    "experiments/run_construction_bias_stage2b.py",
    "results/construction_bias/lexical_matching",
)

# Confirm that no Parquet file is staged.
staged_files = git(
    "diff",
    "--cached",
    "--name-only",
).stdout.splitlines()

for filename in staged_files:
    assert not filename.endswith(
        ".parquet"
    ), (
        f"Row-level Parquet was accidentally staged: "
        f"{filename}"
    )

print("\n" + "=" * 80)
print("STAGED DIFF")
print("=" * 80)

git("diff", "--cached", "--stat")

git(
    "commit",
    "-m",
    "Add validation-defined lexical matching diagnostic",
)

print("\n" + "=" * 80)
print("LOCAL COMMIT COMPLETE")
print("=" * 80)

git("log", "-2", "--oneline")
git("status", "--short")


# ---------------------------------------------------------------------
# 8. Secure push with Colab Secret
# ---------------------------------------------------------------------

token = userdata.get("GITHUB_TOKEN")

assert token, (
    "GITHUB_TOKEN is unavailable. "
    "Check Colab Secrets and notebook access."
)

askpass_path = Path(
    "/tmp/github-askpass-stage2b.sh"
)

askpass_path.write_text(
    """#!/bin/sh
case "$1" in
  *Username*) echo "x-access-token" ;;
  *Password*) echo "$GITHUB_TOKEN" ;;
  *) echo "" ;;
esac
""",
    encoding="utf-8",
)

askpass_path.chmod(
    askpass_path.stat().st_mode
    | stat.S_IXUSR
)

environment = os.environ.copy()
environment["GITHUB_TOKEN"] = token
environment["GIT_ASKPASS"] = str(
    askpass_path
)
environment["GIT_TERMINAL_PROMPT"] = "0"

push_result = subprocess.run(
    [
        "git",
        "push",
        "origin",
        BRANCH,
    ],
    cwd=GIT_REPO,
    text=True,
    capture_output=True,
    env=environment,
)

askpass_path.unlink(
    missing_ok=True
)
environment.pop(
    "GITHUB_TOKEN",
    None,
)
token = None

if push_result.stdout.strip():
    print(push_result.stdout.strip())

if push_result.stderr.strip():
    print(push_result.stderr.strip())

if push_result.returncode != 0:
    raise RuntimeError(
        "Git push failed. The local commit is safe; "
        "check token write permission and retry the push."
    )

print("\n" + "=" * 80)
print("STAGE 2B COMMIT AND PUSH COMPLETE")
print("=" * 80)

git("log", "-2", "--oneline")
git("status", "--short")

print(
    "\nBranch URL:\n"
    "https://github.com/Yongmin-Yoo/xart-h/"
    "tree/analysis/construction-bias-v1"
)