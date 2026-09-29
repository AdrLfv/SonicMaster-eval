#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=sm_prompt_fig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=mig24gb
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/restore_figure_clips/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/restore_figure_clips/%j.err

set -euo pipefail

PROJECT_ROOT=/work/vita/alefevre/programs/SonicMaster
DATASET_ROOT=/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster
OUTPUT_ROOT=/scratch/alefevre/evaluation_ariel/restored_sonicmaster_prompt_figure_clips
CHECKPOINT_PATH=${PROJECT_ROOT}/checkpoints/model.safetensors
CHECKPOINT_SHA256=d6af20753f79824321b62a797780ff04ff277590804793f56bd83a2f3b7a5ed4
CONFIG_PATH=${PROJECT_ROOT}/configs/tangoflux_config.yaml

# Each item is degradation|zero-based manifest index|sample ID.
CLIP_SPECS=(
  "clarity|367|sample_00367_42341"
  "clarity|681|sample_00681_1541923"
  "mud|345|sample_00345_1268365"
  "mud|595|sample_00595_1341246"
  "clip|777|sample_00777_1055137"
  "clip|814|sample_00814_1781444"
)

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

mkdir -p "${OUTPUT_ROOT}"
if [[ -e "${OUTPUT_ROOT}/restoration_metadata.jsonl" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}/restoration_metadata.jsonl" >&2
  exit 1
fi

for spec in "${CLIP_SPECS[@]}"; do
  IFS='|' read -r degradation manifest_index sample_id <<< "${spec}"
  expected_output=${OUTPUT_ROOT}/${degradation}/${sample_id}_restored.wav
  if [[ -e "${expected_output}" ]]; then
    echo "Refusing to overwrite ${expected_output}" >&2
    exit 1
  fi
done

job_tmp=$(mktemp -d /tmp/sonicmaster_prompt_figure_clips.XXXXXX)
staging_dir=$(mktemp -d "${OUTPUT_ROOT}/.staging.XXXXXX")
cleanup() {
  rm -rf -- "${job_tmp}" "${staging_dir}"
}
trap cleanup EXIT

subset_manifest=${job_tmp}/figure_clips.jsonl
: > "${subset_manifest}"

for spec in "${CLIP_SPECS[@]}"; do
  IFS='|' read -r degradation manifest_index sample_id <<< "${spec}"
  source_manifest=${DATASET_ROOT}/degraded/${degradation}_degraded/degradation_pairs.jsonl
  degraded_path=${DATASET_ROOT}/degraded/${degradation}_degraded/${sample_id}_deg1.h5
  clean_path=${DATASET_ROOT}/clean/shard_0000.h5::/${sample_id}
  manifest_line=$((manifest_index + 1))
  entry=$(sed -n "${manifest_line}p" "${source_manifest}")

  if [[ -z "${entry}" ]]; then
    echo "Manifest line ${manifest_index} is missing from ${source_manifest}" >&2
    exit 1
  fi

  if ! jq -e \
      --arg expected_id "${sample_id}_${degradation}" \
      --arg expected_clean "${clean_path}" \
      --arg expected_degraded "${degraded_path}" \
      '.id == $expected_id
       and .clean_audio_path == $expected_clean
       and .degraded_audio_path == $expected_degraded
       and (.prompt | type == "string" and length > 0)' \
      <<< "${entry}" >/dev/null; then
    echo "Manifest validation failed for ${degradation} index ${manifest_index} (${sample_id})" >&2
    exit 1
  fi

  printf '%s\n' "${entry}" >> "${subset_manifest}"
done

echo "Validated six requested manifest entries."
echo "Using the per-sample 'prompt' field (not 'alt_prompt')."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Checkpoint SHA-256: ${CHECKPOINT_SHA256}"
echo "Output root: ${OUTPUT_ROOT}"

python inference_ptload_batch.py \
  --config "${CONFIG_PATH}" \
  --model_ckpt "${CHECKPOINT_PATH}" \
  --infer_file "${subset_manifest}" \
  --output_dir "${staging_dir}" \
  --output_format wav \
  --use_jsonl_prompt \
  --batch_size 1 \
  --seed 0

staging_metadata=${staging_dir}/restoration_metadata.jsonl
if [[ ! -f "${staging_metadata}" ]]; then
  echo "Inference completed without producing ${staging_metadata}" >&2
  exit 1
fi

final_metadata_tmp=${job_tmp}/restoration_metadata.jsonl
: > "${final_metadata_tmp}"

# Validate the complete staged result before publishing any output file.
for spec in "${CLIP_SPECS[@]}"; do
  IFS='|' read -r degradation manifest_index sample_id <<< "${spec}"
  degraded_path=${DATASET_ROOT}/degraded/${degradation}_degraded/${sample_id}_deg1.h5
  staged_audio=${staging_dir}/${sample_id}_deg1_restored.wav

  if [[ ! -f "${staged_audio}" ]]; then
    echo "Expected restored audio is missing: ${staged_audio}" >&2
    exit 1
  fi

  metadata_entry=$(jq -c --arg degraded "${degraded_path}" \
    'select(.degraded_audio_path == $degraded)' "${staging_metadata}")
  if [[ -z "${metadata_entry}" ]]; then
    echo "Expected metadata entry is missing for ${degraded_path}" >&2
    exit 1
  fi
done

for spec in "${CLIP_SPECS[@]}"; do
  IFS='|' read -r degradation manifest_index sample_id <<< "${spec}"
  degraded_path=${DATASET_ROOT}/degraded/${degradation}_degraded/${sample_id}_deg1.h5
  staged_audio=${staging_dir}/${sample_id}_deg1_restored.wav
  final_dir=${OUTPUT_ROOT}/${degradation}
  final_audio=${final_dir}/${sample_id}_restored.wav
  metadata_entry=$(jq -c --arg degraded "${degraded_path}" \
    'select(.degraded_audio_path == $degraded)' "${staging_metadata}")

  mkdir -p "${final_dir}"
  mv "${staged_audio}" "${final_audio}"

  jq -c \
    --arg restored_audio_path "${final_audio}" \
    --arg checkpoint_path "${CHECKPOINT_PATH}" \
    --arg checkpoint_sha256 "${CHECKPOINT_SHA256}" \
    --arg config_path "${CONFIG_PATH}" \
    '.restored_audio_path = $restored_audio_path
     | .prompt_used = .prompt
     | .inference_settings = {
         mode: "with_prompt",
         prompt_field: "prompt",
         model_checkpoint: $checkpoint_path,
         checkpoint_sha256: $checkpoint_sha256,
         config: $config_path,
         vae: "stabilityai/stable-audio-open-1.0/vae",
         num_inference_steps: 100,
         guidance_scale: 1,
         solver: "Euler",
         seed: 0,
         duration_sec: 30,
         sample_rate_hz: 44100,
         batch_size: 1,
         output_format: "wav",
         torch_dtype: "float32"
       }' <<< "${metadata_entry}" >> "${final_metadata_tmp}"
done

mv "${final_metadata_tmp}" "${OUTPUT_ROOT}/restoration_metadata.jsonl"

echo "Restoration complete."
find "${OUTPUT_ROOT}" -maxdepth 2 -type f -printf '%p\t%s bytes\n' | sort
