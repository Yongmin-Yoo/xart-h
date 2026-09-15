"""Exported from the XART-H experiment notebook.
See configs/experiments.json for the frozen setup.
"""

# ============================================================
# Publish XART-H experiment code and results
# GitHub: Yongmin-Yoo/xart-h
# Hugging Face: yongminyoo91/xart-h
# ============================================================

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from google.colab import userdata
except ImportError:
    userdata = None

try:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
    )
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "huggingface_hub>=0.34.0",
    ])
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
    )

# ------------------------------------------------------------
# 1. Authentication and repository settings
# ------------------------------------------------------------
GITHUB_REPO = "Yongmin-Yoo/xart-h"
HF_REPO = "yongminyoo91/xart-h"
HF_REVISION = "v1.0.1"

if userdata is None:
    raise RuntimeError(
        "This cell expects Google Colab Secrets."
    )

GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
HF_TOKEN = userdata.get("HF_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "Colab Secrets에 GITHUB_TOKEN이 없습니다."
    )

if not HF_TOKEN:
    raise ValueError(
        "Colab Secrets에 HF_TOKEN이 없습니다."
    )

github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

repo_response = requests.get(
    f"https://api.github.com/repos/{GITHUB_REPO}",
    headers=github_headers,
    timeout=60,
)

repo_response.raise_for_status()
repo_info = repo_response.json()
DEFAULT_BRANCH = repo_info["default_branch"]

print("GitHub repository:", GITHUB_REPO)
print("Default branch:", DEFAULT_BRANCH)

hf_api = HfApi(token=HF_TOKEN)
hf_identity = hf_api.whoami()

print("Hugging Face account:", hf_identity["name"])

# ------------------------------------------------------------
# 2. Local paths
# ------------------------------------------------------------
ZERO_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/zero_shot_v1"
)

SUPERVISED_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/supervised_v1"
)

QWEN_ROOT = Path(
    "/content/drive/MyDrive/PatentSearchBench/"
    "XART-H/experiments/qwen3_8b_zero_shot_v1"
)

STAGING = Path("/content/xart_h_github_release")
STAGING.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_text(relative_path, content):
    path = STAGING / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )
    return path


def copy_small_result(source, destination):
    source = Path(source)

    if not source.exists():
        print("WARNING missing:", source)
        return None

    if source.stat().st_size > 10 * 1024 * 1024:
        print("SKIP large file:", source)
        return None

    target = STAGING / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())

    print(
        "Staged:",
        destination,
        f"({target.stat().st_size / 1024:.1f} KB)",
    )

    return target


def get_github_file(path):
    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_REPO}/contents/{path}"
    )

    response = requests.get(
        url,
        headers=github_headers,
        params={"ref": DEFAULT_BRANCH},
        timeout=60,
    )

    if response.status_code == 404:
        return None, None

    response.raise_for_status()
    information = response.json()

    content = base64.b64decode(
        information["content"]
    ).decode("utf-8")

    return content, information["sha"]


def upload_github_file(
    local_path,
    repository_path,
    commit_message,
):
    local_path = Path(local_path)

    existing_content, existing_sha = get_github_file(
        repository_path
    )

    new_bytes = local_path.read_bytes()

    if (
        existing_content is not None
        and existing_content.encode("utf-8") == new_bytes
    ):
        print("UNCHANGED:", repository_path)
        return None

    payload = {
        "message": commit_message,
        "content": base64.b64encode(
            new_bytes
        ).decode("ascii"),
        "branch": DEFAULT_BRANCH,
    }

    if existing_sha:
        payload["sha"] = existing_sha

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_REPO}/contents/{repository_path}"
    )

    response = requests.put(
        url,
        headers=github_headers,
        json=payload,
        timeout=120,
    )

    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"GitHub upload failed for {repository_path}\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()
    commit_sha = result["commit"]["sha"]

    print(
        "UPLOADED:",
        repository_path,
        "commit:",
        commit_sha[:12],
    )

    return commit_sha


def update_marked_section(
    original,
    start_marker,
    end_marker,
    new_section,
):
    replacement = (
        f"{start_marker}\n"
        f"{new_section.strip()}\n"
        f"{end_marker}"
    )

    pattern = re.compile(
        re.escape(start_marker)
        + r".*?"
        + re.escape(end_marker),
        flags=re.DOTALL,
    )

    if pattern.search(original):
        return pattern.sub(replacement, original)

    return original.rstrip() + "\n\n" + replacement + "\n"

# ------------------------------------------------------------
# 4. Extract executed experiment cells from notebook history
# ------------------------------------------------------------
def notebook_history():
    try:
        shell = get_ipython()
        history = shell.history_manager.input_hist_raw
        return [
            cell
            for cell in history
            if isinstance(cell, str) and cell.strip()
        ]
    except Exception:
        return []


def find_history_cell(required_phrases):
    history = notebook_history()

    for cell in reversed(history):
        if all(
            phrase in cell
            for phrase in required_phrases
        ):
            return cell

    return None


def clean_notebook_cell(cell):
    lines = []

    for line in cell.splitlines():
        stripped = line.lstrip()

        # Remove notebook-only shell and magic lines.
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(
                "# Notebook command omitted: "
                + stripped
            )
        else:
            lines.append(line)

    header = (
        '"""Exported from the XART-H experiment notebook.\n'
        "See configs/experiments.json for the frozen setup.\n"
        '"""\n\n'
    )

    return header + "\n".join(lines).strip() + "\n"


SCRIPT_SEARCHES = {
    "experiments/run_lexical_baselines.py": [
        "xart_h_lexical_pilot",
        "TfidfVectorizer",
    ],
    "experiments/run_zero_shot_models.py": [
        "zero_shot_results.json",
        "PatentSBERTa",
        "Qwen3-Reranker",
    ],
    "experiments/train_supervised.py": [
        "supervised_v1",
        "TrainingArguments",
        "direct_3way",
    ],
    "experiments/run_qwen3_8b.py": [
        "QWEN3-8B RESULT",
        'run_mode("direct")',
        'run_mode("stage1")',
    ],
    "evaluation/qwen3_bootstrap.py": [
        "QWEN3-8B BOOTSTRAP RESULT",
        "qwen3_8b_bootstrap.json",
    ],
    "evaluation/supervised_bootstrap.py": [
        "SUPERVISED BOOTSTRAP RESULT",
        "supervised_bootstrap_results.json",
    ],
    "evaluation/family_holm.py": [
        "FAMILY-WISE HOLM RESULTS",
        "family_holm_p_value",
    ],
}

extracted_scripts = []
missing_scripts = []

for repository_path, phrases in SCRIPT_SEARCHES.items():
    cell = find_history_cell(phrases)

    if cell is None:
        missing_scripts.append(repository_path)
        print(
            "WARNING: notebook cell not found for",
            repository_path,
        )
        continue

    local_path = write_text(
        repository_path,
        clean_notebook_cell(cell),
    )

    extracted_scripts.append(
        (local_path, repository_path)
    )

print("\nExtracted scripts:", len(extracted_scripts))

# ------------------------------------------------------------
# 5. Experiment configuration
# ------------------------------------------------------------
experiment_config = {
    "dataset": {
        "repo_id": HF_REPO,
        "revision": HF_REVISION,
        "input_columns": ["text", "text_b"],
        "label_column": "label",
        "label_mapping": {
            "0": "A",
            "1": "X",
            "2": "U",
        },
    },
    "hardware": {
        "gpu": "NVIDIA L4 22.5 GB",
    },
    "supervised": {
        "models": {
            "minilm": (
                "cross-encoder/"
                "ms-marco-MiniLM-L6-v2"
            ),
            "deberta": "microsoft/deberta-v3-base",
        },
        "tasks": [
            "cited_vs_u",
            "x_vs_a",
            "direct_3way",
            "hierarchical_3way",
        ],
        "seeds": [13, 42, 77],
        "epochs": 2,
        "maximum_length": 384,
        "learning_rates": {
            "minilm": 2e-5,
            "deberta": 1e-5,
        },
        "train_batch_sizes": {
            "minilm": 256,
            "deberta": 32,
        },
        "evaluation_batch_sizes": {
            "minilm": 512,
            "deberta": 64,
        },
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "split": "test",
        "quantization": "4-bit NF4 double quantization",
        "compute_dtype": "bfloat16",
        "maximum_length": 1280,
        "tokens_per_text_field": 500,
        "thinking": False,
        "scoring": "constrained next-token logits",
    },
    "statistics": {
        "bootstrap_unit": (
            "lowercased and whitespace-normalized "
            "claim text"
        ),
        "bootstrap_repetitions": 2000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
        "multiple_testing": (
            "Holm correction within each "
            "comparison family"
        ),
    },
}

config_path = write_text(
    "configs/experiments.json",
    json.dumps(
        experiment_config,
        ensure_ascii=False,
        indent=2,
    ),
)

# ------------------------------------------------------------
# 6. Updated requirements
# ------------------------------------------------------------
requirements = """
datasets>=3.0.0,<5
huggingface_hub>=0.34.0
numpy>=1.26.0
pandas>=2.2.0
pyarrow>=18.0.0
scikit-learn>=1.5.0
scipy>=1.13.0
torch>=2.5.0
transformers>=4.51.0,<5
sentence-transformers>=3.4.0,<6
accelerate>=1.2.0,<2
bitsandbytes>=0.45.0
tqdm>=4.66.0
"""

requirements_path = write_text(
    "requirements.txt",
    requirements,
)

# ------------------------------------------------------------
# 7. Stage result files
# ------------------------------------------------------------
RESULT_FILES = [
    (
        ZERO_ROOT / "zero_shot_results.json",
        "results/zero_shot/zero_shot_results.json",
    ),
    (
        SUPERVISED_ROOT / "summary.json",
        "results/supervised/summary.json",
    ),
    (
        SUPERVISED_ROOT / "all_seed_results.json",
        "results/supervised/all_seed_results.json",
    ),
    (
        SUPERVISED_ROOT
        / "bootstrap"
        / "supervised_bootstrap_results.json",
        "results/supervised/"
        "supervised_bootstrap_results.json",
    ),
    (
        SUPERVISED_ROOT
        / "bootstrap"
        / "supervised_bootstrap_ci.csv",
        "results/supervised/"
        "supervised_bootstrap_ci.csv",
    ),
    (
        SUPERVISED_ROOT
        / "bootstrap"
        / "supervised_paired_tests.csv",
        "results/supervised/"
        "supervised_paired_tests.csv",
    ),
    (
        SUPERVISED_ROOT
        / "bootstrap"
        / "supervised_bootstrap_family_holm.json",
        "results/supervised/"
        "supervised_bootstrap_family_holm.json",
    ),
    (
        SUPERVISED_ROOT
        / "bootstrap"
        / "supervised_paired_tests_family_holm.csv",
        "results/supervised/"
        "supervised_paired_tests_family_holm.csv",
    ),
    (
        QWEN_ROOT / "qwen3_8b_results.json",
        "results/qwen3_8b/qwen3_8b_results.json",
    ),
    (
        QWEN_ROOT / "qwen3_8b_bootstrap.json",
        "results/qwen3_8b/qwen3_8b_bootstrap.json",
    ),
    (
        QWEN_ROOT / "qwen3_8b_bootstrap_ci.csv",
        "results/qwen3_8b/qwen3_8b_bootstrap_ci.csv",
    ),
    (
        QWEN_ROOT / "qwen3_8b_paired_tests.csv",
        "results/qwen3_8b/qwen3_8b_paired_tests.csv",
    ),
]

staged_results = []

for source, destination in RESULT_FILES:
    local_path = copy_small_result(
        source,
        destination,
    )

    if local_path is not None:
        staged_results.append(
            (local_path, destination)
        )

# ------------------------------------------------------------
# 8. Result manifest
# ------------------------------------------------------------
manifest_files = {}

for local_path, repository_path in (
    extracted_scripts + staged_results
):
    manifest_files[repository_path] = {
        "sha256": sha256_file(local_path),
        "bytes": local_path.stat().st_size,
    }

manifest = {
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": HF_REPO,
    "dataset_revision": HF_REVISION,
    "status": "intermediate experimental snapshot",
    "files": manifest_files,
    "excluded_artifacts": [
        "model checkpoints",
        "per-example prediction parquet files",
        "logit NPZ files",
        "authentication tokens",
    ],
    "pending_analyses": [
        "Random U versus Hard U",
        "temporal robustness",
        "CPC subgroup analysis",
        "length and lexical-overlap analysis",
        "confusion matrices and error analysis",
    ],
}

manifest_path = write_text(
    "results/RESULT_MANIFEST.json",
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
)

# ------------------------------------------------------------
# 9. Result documentation
# ------------------------------------------------------------
results_document = r"""
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
| MiniLM | Direct | 0.325 | 0.313 | 0.501 | 0.481 |
| MiniLM | Hierarchical | 0.325 | 0.290 | 0.498 | 0.496 |
| DeBERTa | Direct | 0.453 | 0.450 | 0.743 | 0.485 |
| DeBERTa | Hierarchical | 0.446 | 0.432 | 0.738 | 0.492 |
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
"""

results_doc_path = write_text(
    "docs/RESULTS.md",
    results_document,
)

# ------------------------------------------------------------
# 10. Update GitHub README
# ------------------------------------------------------------
github_readme, _ = get_github_file("README.md")

if github_readme is None:
    github_readme = "# XART-H\n"

github_results_section = r"""
## Current Results

Current experimental results are available in
[`docs/RESULTS.md`](docs/RESULTS.md), with machine-readable
files under [`results/`](results/).

The strongest supervised cited-versus-U result is obtained by
DeBERTa-v3-base with a test ROC-AUC of 0.746. X-versus-A
performance remains close to chance across the evaluated model
families. Qwen3-8B benefits from hierarchical prompting, but
remains below the task-specific reranker and supervised DeBERTa.

These results use the frozen Hugging Face release
[`v1.0.1`](https://huggingface.co/datasets/yongminyoo91/xart-h/tree/v1.0.1).
"""

updated_github_readme = update_marked_section(
    github_readme,
    "<!-- XARTH_RESULTS_START -->",
    "<!-- XARTH_RESULTS_END -->",
    github_results_section,
)

github_readme_path = write_text(
    "README.md",
    updated_github_readme,
)

# ------------------------------------------------------------
# 11. Upload files to GitHub
# ------------------------------------------------------------
UPLOAD_FILES = []

UPLOAD_FILES.extend(extracted_scripts)
UPLOAD_FILES.extend(staged_results)

UPLOAD_FILES.extend([
    (
        config_path,
        "configs/experiments.json",
    ),
    (
        requirements_path,
        "requirements.txt",
    ),
    (
        manifest_path,
        "results/RESULT_MANIFEST.json",
    ),
    (
        results_doc_path,
        "docs/RESULTS.md",
    ),
    (
        github_readme_path,
        "README.md",
    ),
])

print("\nUploading to GitHub...")

github_commits = []

for local_path, repository_path in UPLOAD_FILES:
    commit_sha = upload_github_file(
        local_path=local_path,
        repository_path=repository_path,
        commit_message=(
            "Add reproducible experiment code and results"
        ),
    )

    if commit_sha:
        github_commits.append(commit_sha)

# ------------------------------------------------------------
# 12. Update Hugging Face dataset card
# Only main README is updated. v1.0.1 remains unchanged.
# ------------------------------------------------------------
print("\nUpdating Hugging Face dataset card...")

hf_readme_path = hf_hub_download(
    repo_id=HF_REPO,
    filename="README.md",
    repo_type="dataset",
    revision="main",
    token=HF_TOKEN,
)

hf_readme = Path(hf_readme_path).read_text(
    encoding="utf-8"
)

hf_results_section = r"""
## Benchmark Results

The following results use the frozen v1.0.1 release.

| Model | Cited/U AUC | X/A AUC | 3-way Accuracy | Macro-F1 |
|:--|--:|--:|--:|--:|
| TF-IDF | 0.347 | 0.516 | N/A | N/A |
| BM25 | 0.323 | 0.513 | N/A | N/A |
| Qwen3-Reranker-0.6B | 0.633 | 0.516 | N/A | N/A |
| Qwen3-8B direct | 0.531 | 0.489 | 0.336 | 0.250 |
| MiniLM supervised | 0.493 | 0.501 | 0.325 | 0.313 |
| DeBERTa-v3-base | 0.746 | 0.507 | 0.453 | 0.450 |

Supervised values are means over three seeds. Query-level
bootstrap intervals, corrected statistical tests, and
reproduction code are available in the
[XART-H GitHub repository](https://github.com/Yongmin-Yoo/xart-h).

Random U, temporal, and subgroup analyses are ongoing.
"""

updated_hf_readme = update_marked_section(
    hf_readme,
    "<!-- XARTH_BENCHMARK_RESULTS_START -->",
    "<!-- XARTH_BENCHMARK_RESULTS_END -->",
    hf_results_section,
)

local_hf_readme = STAGING / "HF_README.md"
local_hf_readme.write_text(
    updated_hf_readme,
    encoding="utf-8",
)

hf_commit = hf_api.upload_file(
    path_or_fileobj=str(local_hf_readme),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="dataset",
    revision="main",
    commit_message=(
        "Add preliminary XART-H benchmark results"
    ),
)

# ------------------------------------------------------------
# 13. Final report
# ------------------------------------------------------------
print("\n" + "=" * 72)
print("PUBLICATION UPDATE COMPLETE")
print("=" * 72)
print(
    "GitHub:",
    f"https://github.com/{GITHUB_REPO}"
)
print(
    "GitHub branch:",
    DEFAULT_BRANCH,
)
print(
    "GitHub files uploaded:",
    len(UPLOAD_FILES),
)
print(
    "GitHub commits created:",
    len(github_commits),
)
print(
    "Hugging Face:",
    f"https://huggingface.co/datasets/{HF_REPO}"
)
print(
    "Frozen dataset remains:",
    f"https://huggingface.co/datasets/"
    f"{HF_REPO}/tree/{HF_REVISION}"
)
print(
    "HF README commit:",
    getattr(hf_commit, "oid", str(hf_commit)),
)

if missing_scripts:
    print("\nScripts not found in current notebook history:")
    for path in missing_scripts:
        print(" -", path)

    print(
        "\nThese scripts were not uploaded. "
        "They can be exported from the notebooks "
        "where they were originally executed."
    )

print("\nLarge artifacts intentionally excluded:")
print(" - NPZ logits")
print(" - per-example prediction Parquet")
print(" - model checkpoints")
print("=" * 72)
