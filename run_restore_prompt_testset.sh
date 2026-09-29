#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --job-name=sm_prompt_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=mig24gb
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/restore_prompt_testset/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/restore_prompt_testset/%j.err

set -euo pipefail

PROJECT_ROOT=/work/vita/alefevre/programs/SonicMaster
DATASET_ROOT=/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster
OUTPUT_ROOT=/scratch/alefevre/evaluation_ariel/restored_sonicmaster_prompt_testset
CHECKPOINT_PATH=${PROJECT_ROOT}/checkpoints/model.safetensors
CHECKPOINT_SHA256=d6af20753f79824321b62a797780ff04ff277590804793f56bd83a2f3b7a5ed4
CONFIG_PATH=${PROJECT_ROOT}/configs/tangoflux_config.yaml
DEGRADATION=${1:-}

if [[ -z "${DEGRADATION}" ]]; then
  echo "Usage: sbatch run_restore_prompt_testset.sh <degradation>" >&2
  exit 2
fi

MANIFEST=${DATASET_ROOT}/degraded/${DEGRADATION}_degraded/degradation_pairs.jsonl
OUTPUT_DIR=${OUTPUT_ROOT}/${DEGRADATION}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

module purge
module load gcc/13.2.0
module load python/3.11.7

cd "${PROJECT_ROOT}"
source venv_py311/bin/activate
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

actual_checkpoint_sha256=$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')
if [[ "${actual_checkpoint_sha256}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint hash mismatch: expected ${CHECKPOINT_SHA256}, got ${actual_checkpoint_sha256}" >&2
  exit 1
fi

expected_count=$(wc -l < "${MANIFEST}")
mkdir -p "${OUTPUT_DIR}"

echo "SonicMaster prompted test-set restoration"
echo "Degradation: ${DEGRADATION}"
echo "Manifest: ${MANIFEST}"
echo "Expected samples: ${expected_count}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Prompt field: prompt"
echo "Checkpoint SHA-256: ${CHECKPOINT_SHA256}"

python inference_ptload_batch.py \
  --config "${CONFIG_PATH}" \
  --model_ckpt "${CHECKPOINT_PATH}" \
  --infer_file "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_format wav \
  --use_jsonl_prompt \
  --batch_size 1 \
  --seed 0 \
  --resume \
  --strip_degradation_suffix

metadata_path=${OUTPUT_DIR}/restoration_metadata.jsonl
if [[ ! -f "${metadata_path}" ]]; then
  echo "Missing restoration metadata: ${metadata_path}" >&2
  exit 1
fi

metadata_count=$(wc -l < "${metadata_path}")
wav_count=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*_restored.wav' | wc -l)
if [[ "${metadata_count}" -ne "${expected_count}" ]]; then
  echo "Metadata count mismatch: expected ${expected_count}, found ${metadata_count}" >&2
  exit 1
fi
if [[ "${wav_count}" -ne "${expected_count}" ]]; then
  echo "WAV count mismatch: expected ${expected_count}, found ${wav_count}" >&2
  exit 1
fi

if ! jq -e --argjson count "${expected_count}" \
    -s 'map(.source_manifest_index) | sort == [range(0; $count)]' \
    "${metadata_path}" >/dev/null; then
  echo "Metadata source_manifest_index coverage is incomplete or duplicated" >&2
  exit 1
fi

echo "Restoration complete: ${metadata_count} metadata rows and ${wav_count} WAV files."
