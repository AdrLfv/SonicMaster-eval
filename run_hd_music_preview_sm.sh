#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --job-name=sm_hd_preview
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=h100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/sm_hd_preview/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/sm_hd_preview/%j.err

# Create four-way SonicMaster listening bundles for the six clips selected in
# ARIEL/tmp/hd_music_sa3_attn_mlp_rk128_preview.  Each degradation directory
# contains degraded, clean, clean-decoded (Stable Audio Open VAE), and restored
# (SonicMaster) WAVs, plus the manifests used to produce them.

set -euo pipefail

PROJECT_ROOT=/work/vita/alefevre/programs/SonicMaster
PREVIEW_ROOT=/work/vita/alefevre/programs/ARIEL/tmp/hd_music_sa3_attn_mlp_rk128_preview
MANIFEST_DIR="${PREVIEW_ROOT}/manifests"
OUTPUT_ROOT="${PREVIEW_ROOT}/sonicmaster"
MODEL_CKPT="${PROJECT_ROOT}/checkpoints/model.safetensors"

DEGRADATIONS="airy big boom bright clarity clip comp dark mic mix mud punch real small stereo vocal volume warm xband"

cd "${PROJECT_ROOT}"
module purge
module load gcc/13.2.0
module load python/3.11.7
source venv_py311/bin/activate

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "Missing SonicMaster checkpoint: ${MODEL_CKPT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/clean_decoded"

# All selected degradation manifests share the same six clean audio entries.
# VAE reconstruction is therefore performed once and copied into each bundle.
python reconstruct_vae_baseline.py \
    --input_jsonl "${MANIFEST_DIR}/airy_pairs_first6.jsonl" \
    --output_dir "${OUTPUT_ROOT}/clean_decoded" \
    --audio_key clean_audio_path \
    --duration_sec 30 \
    --batch_size 6 \
    --output_format wav

for degradation in ${DEGRADATIONS}; do
    manifest="${MANIFEST_DIR}/${degradation}_pairs_first6.jsonl"
    bundle="${OUTPUT_ROOT}/${degradation}"
    restored_dir="${bundle}/restored_raw"

    [[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 1; }
    mkdir -p "${bundle}/degraded" "${bundle}/clean" "${bundle}/clean_decoded" "${bundle}/restored"
    cp "${manifest}" "${bundle}/manifest.jsonl"

    # Export direct from the HDF5 paths in the manifest, so all names and the
    # selected non-consecutive dataset indices are preserved exactly.
    python export_preview_audio.py --input-jsonl "${manifest}" --audio-key clean_audio_path --output-dir "${bundle}/clean"
    python export_preview_audio.py --input-jsonl "${manifest}" --audio-key degraded_audio_path --output-dir "${bundle}/degraded"

    while IFS= read -r clean_dataset; do
        decoded_source="${OUTPUT_ROOT}/clean_decoded/${clean_dataset}_airy_reconstructed.wav"
        [[ -f "${decoded_source}" ]] || { echo "Missing clean decoded WAV: ${decoded_source}" >&2; exit 1; }
        cp "${decoded_source}" "${bundle}/clean_decoded/${clean_dataset}.wav"
    done < <(jq -r '.clean_audio_dataset' "${manifest}")

    mkdir -p "${restored_dir}"
    python inference_ptload_batch.py \
        --config configs/tangoflux_config.yaml \
        --model_ckpt "${MODEL_CKPT}" \
        --infer_file "${manifest}" \
        --output_dir "${restored_dir}" \
        --output_format wav \
        --prompt ""

    while IFS=$'\t' read -r clean_dataset degraded_dataset; do
        restored_source="${restored_dir}/${degraded_dataset}_restored.wav"
        [[ -f "${restored_source}" ]] || { echo "Missing restored WAV: ${restored_source}" >&2; exit 1; }
        cp "${restored_source}" "${bundle}/restored/${clean_dataset}.wav"
    done < <(jq -r '[.clean_audio_dataset, .degraded_audio_dataset] | @tsv' "${manifest}")
done

echo "Completed SonicMaster preview bundles: ${OUTPUT_ROOT}"
