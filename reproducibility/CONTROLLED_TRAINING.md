# Table 4 Controlled Training

## Availability statement

Controlled pipeline code, frozen inputs, and execution commands are publicly available. The controlled execution bridge was successfully validated on an NVIDIA L4 GPU, and the reported statistics were reproduced from the canonical released predictions.

## Scope

This directory closes the controlled-training code gap for Table 4.

The released `reproduce_table4.py` starts from the canonical ordered
prediction files and reproduces the ensembles, point estimates,
confidence intervals, paired bootstrap tests, and Holm correction.

The additional entry points published here connect the frozen training
manifests to the Hard-only and Mixed-U model-training stage.

## Provenance statement

The original notebook-style controlled-training entry point was not
available in the publication tree. The executable entry point published
here was reconstructed from the frozen manifests, run configurations,
training metadata, logs, released Qwen LoRA adapters, and ordered
predictions.

It must not be described as a byte-identical copy of an unavailable
original script. The released ordered predictions remain the canonical
artifacts used to report Table 4.

## Frozen training inputs

Both conditions use the same 8,246 training queries.

- Hard-only:
  - 16,617 cited rows
  - one Hard-U row for each query
  - 24,863 rows in total
- Mixed-U:
  - the identical 16,617 cited rows
  - exactly one U row for each query
  - 4,123 Hard-U and 4,123 Random-U assignments
  - 24,863 rows in total

Files:

- `artifacts/table4/deberta_control/manifests/hard_only_train_manifest.parquet`
- `artifacts/table4/deberta_control/manifests/mixed_u_train_manifest.parquet`
- `artifacts/table4/deberta_control/manifests/mixed_u_query_assignment.csv`
- `artifacts/table4/deberta_control/manifests/common_train_queries.csv`
- `artifacts/table4/deberta_control/manifests/frozen_manifest_config.json`

## Models and seeds

Models:

- `cross-encoder/ms-marco-MiniLM-L6-v2`
- `microsoft/deberta-v3-base`
- `Qwen/Qwen3-Reranker-0.6B`

Training conditions:

- `hard_only`
- `mixed_u`

Model seeds:

- 13
- 42
- 77

The complete design therefore contains 18 training runs.

## Input validation without a GPU

    python reproducibility/scripts/validate_table4_training_inputs.py \
      --manifests reproducibility/artifacts/table4/deberta_control/manifests

## Dry-run validation

The following command resolves one run and validates its manifest
without downloading or training a model:

    python reproducibility/scripts/train_table4_controlled.py \
      --model deberta \
      --condition hard_only \
      --seed 13 \
      --manifests reproducibility/artifacts/table4/deberta_control/manifests \
      --output-root /tmp/patscout-table4-checkpoints \
      --dry-run

## Encoder environment

    python -m pip install -r       reproducibility/environments/table4_encoders.txt

## Qwen environment

Qwen is isolated because its recorded software environment differs from
the repository-level dependency range.

    python -m pip install -r       reproducibility/environments/table4_qwen.txt

## One training run

    python reproducibility/scripts/train_table4_controlled.py \
      --model minilm \
      --condition mixed_u \
      --seed 42 \
      --manifests reproducibility/artifacts/table4/deberta_control/manifests \
      --output-root /path/to/table4_controlled_checkpoints

## All 18 training runs

    REPO_ROOT="$(pwd)"     OUTPUT_ROOT="/path/to/table4_controlled_checkpoints"     bash reproducibility/scripts/run_table4_controlled_training.sh

## Statistical reproduction from canonical predictions

Training is not required to verify the paper's released Table 4
statistics. Reviewers can run:

    python reproducibility/scripts/reproduce_table4.py \
      --artifacts reproducibility/artifacts \
      --output /tmp/patscout-table4-reproduced \
      --bootstrap-repetitions 2000 \
      --bootstrap-seed 42

The canonical released predictions contain all combinations of three
models, two training conditions, and three seeds.

## End-to-end execution compatibility

The public execution bridge handles the model-specific interfaces used
by the controlled experiment:

- MiniLM preserves the native one-logit
  `cross-encoder/ms-marco-MiniLM-L6-v2` classification head and uses
  binary cross-entropy during training and sigmoid probabilities during
  prediction.
- DeBERTa uses a two-logit classification head and obtains cited
  probabilities with softmax.
- Qwen applies balanced claim-passage truncation within the 384-token
  limit after reserving the fixed instruction, chat-template, and
  answer-prefix tokens.
- The all-run prediction launcher writes each generated file directly
  into the artifact directory structure expected by
  `reproduce_table4.py`.
- Unless disabled with `RUN_STATISTICS=0`, the launcher invokes
  `reproduce_table4.py` after all 18 prediction files have been
  generated.

Generated predictions remain separate from the released canonical
predictions by default. They can be evaluated directly without renaming
or manually copying files.

## Important interpretation

GPU nondeterminism, library changes, and the reconstructed status of the
training entry point can lead to small differences when retraining.
The published ordered predictions, their hashes, and the statistical
reproduction outputs define the reported Table 4 result.
## Checkpoint-to-prediction bridge

The public evaluation manifest contains 9,896 ordered rows covering 1,642 matched test queries under the Hard-U and Random-U evaluation conditions.

The complete executable path is:

1. validate the two frozen controlled-training manifests;
2. train MiniLM, DeBERTa, or Qwen under Hard-only or Mixed-U;
3. build the exact 1,642-query public evaluation manifest;
4. evaluate the checkpoint against Hard-U and Random-U;
5. write the ordered prediction schema consumed by
   `reproduce_table4.py`;
6. reproduce point estimates, confidence intervals, paired tests, and
   Holm-adjusted p-values.

The evaluation manifest is built entirely from public inputs:

- frozen PatScout test data at Hugging Face revision `v1.0.1`;
- `results/random_u/generated/test_random_u.parquet`;
- one canonical ordered prediction file used only as the exact row-ID
  and ordering template.

No canonical probability is copied into the evaluation manifest.

Build the evaluation manifest:

    python reproducibility/scripts/build_table4_evaluation_manifest.py \
      --template reproducibility/artifacts/table4/minilm_control/hard_only_seed13_ordered_predictions.parquet \
      --random-u results/random_u/generated/test_random_u.parquet \
      --output /tmp/table4_evaluation_manifest.parquet \
      --report /tmp/table4_evaluation_alignment.json

Generate one ordered prediction file:

    python reproducibility/scripts/predict_table4_controlled.py \
      --model deberta \
      --condition mixed_u \
      --seed 42 \
      --checkpoint /path/to/checkpoints/deberta/mixed_u/seed42 \
      --evaluation-manifest /tmp/table4_evaluation_manifest.parquet \
      --output /tmp/mixed_u_seed42_ordered_predictions.parquet

Generate all 18 prediction files:

    REPO_ROOT="$(pwd)" \
    CHECKPOINT_ROOT="/path/to/table4_controlled_checkpoints" \
    ARTIFACTS_ROOT="/path/to/table4_generated_artifacts" \
    STATISTICS_OUTPUT="/path/to/table4_generated_statistics" \
    bash reproducibility/scripts/run_table4_controlled_predictions.sh

Actual checkpoint inference requires a CUDA GPU. Manifest construction,
schema validation, and predictor dry-run do not require a GPU.

Freshly generated predictions are written outside the canonical artifact
directories by default, using a parallel artifact layout directly
accepted by `reproduce_table4.py`. They should be compared with, rather
than silently substituted for, the released canonical predictions.

## GPU execution validation

The controlled execution bridge was successfully validated on an
NVIDIA L4 GPU. MiniLM native one-logit loading and sigmoid inference,
Qwen balanced claim-passage truncation and public LoRA inference, and
the direct prediction-to-statistics workflow all completed
successfully.

The validated environment used PyTorch 2.11.0 with CUDA 13.0,
Transformers 5.18.0, and PEFT 0.21.1. Qwen inference used standard
PyTorch LoRA layers, BF16 precision, the released seed-13 adapter, and
the frozen Qwen model revision recorded by the benchmark.

The statistical entry point successfully discovered the released
MiniLM, DeBERTa, and Qwen prediction artifacts and generated point
estimates, bootstrap outputs, paired comparisons, and the combined
Table 4 result files. The reported statistics remain reproducible from
the canonical released predictions.

<!-- TABLE4_BOOTSTRAP_SEED_EXCEPTION_START -->
## Bootstrap seed provenance

Bootstrap resampling uses seed 42, except for the Table 4 DeBERTa analysis, which uses seed 20260922. These bootstrap seeds are separate from the model-training seeds 13, 42, and 77.

The `--bootstrap-seed` argument controls the standard Table 4
resampling seed. The released reproduction code preserves the
DeBERTa-specific seed override of 20260922 so that the canonical
statistics remain reproducible. This exception is unrelated to the
model-training seeds used to create the three-seed ensembles.

한국어: Bootstrap 재표집은 seed 42를 사용하되, Table 4의 DeBERTa 분석은 seed 20260922를 사용한다. 이 bootstrap seed는 모델 학습 seed 13, 42, 77과 별개이다.
<!-- TABLE4_BOOTSTRAP_SEED_EXCEPTION_END -->
