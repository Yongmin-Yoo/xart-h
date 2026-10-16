#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/table4_controlled_checkpoints}"

# This root intentionally mirrors reproducibility/artifacts so that
# reproduce_table4.py can consume generated predictions directly.
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${REPO_ROOT}/table4_generated_artifacts}"

EVALUATION_ROOT="${EVALUATION_ROOT:-${ARTIFACTS_ROOT}/table4/evaluation}"
STATISTICS_OUTPUT="${STATISTICS_OUTPUT:-${REPO_ROOT}/table4_generated_statistics}"
RUN_STATISTICS="${RUN_STATISTICS:-1}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"

BUILDER="${REPO_ROOT}/reproducibility/scripts/build_table4_evaluation_manifest.py"
PREDICTOR="${REPO_ROOT}/reproducibility/scripts/predict_table4_controlled.py"
REPRODUCER="${REPO_ROOT}/reproducibility/scripts/reproduce_table4.py"

TEMPLATE="${REPO_ROOT}/reproducibility/artifacts/table4/minilm_control/hard_only_seed13_ordered_predictions.parquet"
RANDOM_U="${REPO_ROOT}/results/random_u/generated/test_random_u.parquet"

EVALUATION_MANIFEST="${EVALUATION_ROOT}/table4_evaluation_manifest.parquet"
ALIGNMENT_REPORT="${EVALUATION_ROOT}/table4_evaluation_alignment.json"

mkdir -p "${EVALUATION_ROOT}"
mkdir -p "${ARTIFACTS_ROOT}/table4/minilm_control"
mkdir -p "${ARTIFACTS_ROOT}/table4/deberta_control/ordered_evaluation/predictions"
mkdir -p "${ARTIFACTS_ROOT}/table4/qwen_control"

python "${BUILDER}" \
  --template "${TEMPLATE}" \
  --random-u "${RANDOM_U}" \
  --output "${EVALUATION_MANIFEST}" \
  --report "${ALIGNMENT_REPORT}"

for model in minilm deberta qwen; do
  for condition in hard_only mixed_u; do
    for seed in 13 42 77; do
      checkpoint="${CHECKPOINT_ROOT}/${model}/${condition}/seed${seed}"

      case "${model}" in
        minilm)
          output="${ARTIFACTS_ROOT}/table4/minilm_control/${condition}_seed${seed}_ordered_predictions.parquet"
          ;;
        deberta)
          output="${ARTIFACTS_ROOT}/table4/deberta_control/ordered_evaluation/predictions/${condition}_seed${seed}_ordered_predictions.parquet"
          ;;
        qwen)
          output="${ARTIFACTS_ROOT}/table4/qwen_control/qwen3_reranker_${condition}_seed${seed}_ordered_predictions.parquet"
          ;;
        *)
          echo "Unsupported model: ${model}" >&2
          exit 1
          ;;
      esac

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
echo "Artifacts root: ${ARTIFACTS_ROOT}"

if [[ "${RUN_STATISTICS}" == "1" ]]; then
  mkdir -p "${STATISTICS_OUTPUT}"

  python "${REPRODUCER}" \
    --artifacts "${ARTIFACTS_ROOT}" \
    --output "${STATISTICS_OUTPUT}" \
    --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}" \
    --bootstrap-seed "${BOOTSTRAP_SEED}"

  echo "Table 4 statistical reproduction complete."
  echo "Statistics: ${STATISTICS_OUTPUT}"
else
  echo "Statistical reproduction skipped because RUN_STATISTICS=${RUN_STATISTICS}."
fi
