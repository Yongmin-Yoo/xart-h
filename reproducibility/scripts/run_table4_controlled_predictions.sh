#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/table4_controlled_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/table4_controlled_predictions}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${OUTPUT_ROOT}/evaluation}"

BUILDER="${REPO_ROOT}/reproducibility/scripts/build_table4_evaluation_manifest.py"
PREDICTOR="${REPO_ROOT}/reproducibility/scripts/predict_table4_controlled.py"

TEMPLATE="${REPO_ROOT}/reproducibility/artifacts/table4/minilm_control/hard_only_seed13_ordered_predictions.parquet"
RANDOM_U="${REPO_ROOT}/results/random_u/generated/test_random_u.parquet"

EVALUATION_MANIFEST="${EVALUATION_ROOT}/table4_evaluation_manifest.parquet"
ALIGNMENT_REPORT="${EVALUATION_ROOT}/table4_evaluation_alignment.json"

mkdir -p "${EVALUATION_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

python "${BUILDER}" \
  --template "${TEMPLATE}" \
  --random-u "${RANDOM_U}" \
  --output "${EVALUATION_MANIFEST}" \
  --report "${ALIGNMENT_REPORT}"

for model in minilm deberta qwen; do
  for condition in hard_only mixed_u; do
    for seed in 13 42 77; do
      checkpoint="${CHECKPOINT_ROOT}/${model}/${condition}/seed${seed}"
      output="${OUTPUT_ROOT}/${model}/${condition}_seed${seed}_ordered_predictions.parquet"

      python "${PREDICTOR}" \
        --model "${model}" \
        --condition "${condition}" \
        --seed "${seed}" \
        --checkpoint "${checkpoint}" \
        --evaluation-manifest "${EVALUATION_MANIFEST}" \
        --output "${output}"
    done
  done
done

echo "All 18 Table 4 ordered prediction files generated."
echo "Predictions: ${OUTPUT_ROOT}"
