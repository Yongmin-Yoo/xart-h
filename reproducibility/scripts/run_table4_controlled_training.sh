#!/usr/bin/env bash
set -euo pipefail

# PatScout Table 4 controlled training:
# 3 models × 2 conditions × 3 seeds = 18 runs.
#
# This script performs GPU training. Review resource requirements before
# execution. Set OUTPUT_ROOT to a directory outside the Git repository
# unless checkpoint publication is intended.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFESTS="${MANIFESTS:-${REPO_ROOT}/reproducibility/artifacts/table4/deberta_control/manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/table4_controlled_checkpoints}"

TRAIN="${REPO_ROOT}/reproducibility/scripts/train_table4_controlled.py"
VALIDATE="${REPO_ROOT}/reproducibility/scripts/validate_table4_training_inputs.py"

python "${VALIDATE}" --manifests "${MANIFESTS}"

for model in minilm deberta qwen; do
  for condition in hard_only mixed_u; do
    for seed in 13 42 77; do
      python "${TRAIN}" \
        --model "${model}" \
        --condition "${condition}" \
        --seed "${seed}" \
        --manifests "${MANIFESTS}" \
        --output-root "${OUTPUT_ROOT}"
    done
  done
done

echo "All 18 controlled-training runs completed."
