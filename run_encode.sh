#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=encode
#SBATCH --nodes=1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/encode/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/encode/%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=h100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

# Load modules
module purge
module load gcc/13.2.0
module load python/3.11.7

if [ -d "venv_py311" ] && [ -f "venv_py311/bin/activate" ]; then
  echo "Using existing virtual environment."
  source ./venv_py311/bin/activate
  pip install -r requirements_sonic.txt
else
  echo "Creating Python 3.11 virtual environment..."
  python -m venv venv_py311
  echo "Virtual environment created." 
  source ./venv_py311/bin/activate
  pip install -r requirements_sonic.txt
fi

# Get dataset type from command line argument
DATASET=$1
USE_SHARDS=$2
ENCODE_CLEAN=$3

if [ -z "$DATASET" ]; then
  echo "Error: No dataset type specified"
  echo "Usage: sbatch run_encode.sh <dataset_type> [--use_shards] [--encode_clean]"
  echo "  dataset_type can be: test, train, or a degradation name (e.g., airy, punch, etc.)"
  echo "  --use_shards: Save latents in HDF5 shards instead of individual .pt files"
  echo "  --encode_clean: Also encode clean audio (only for degradation types)"
  exit 1
fi

echo "Running VAE encoding for dataset: $DATASET"

# Build extra flags
EXTRA_FLAGS=""
if [ "$USE_SHARDS" = "--use_shards" ] || [ "$ENCODE_CLEAN" = "--use_shards" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --use_shards"
  echo "Using shards mode"
fi
if [ "$USE_SHARDS" = "--encode_clean" ] || [ "$ENCODE_CLEAN" = "--encode_clean" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --encode_clean"
  echo "Encoding clean audio as well"
fi

# Determine input path and audio key based on dataset type
if [ "$DATASET" = "test" ]; then
  IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/metadata.jsonl"
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_encoded"
  AUDIO_KEY="clean_audio_path"
elif [ "$DATASET" = "train" ]; then
  IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/train_sonicmaster/metadata.jsonl"
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/train_sonicmaster_encoded"
  AUDIO_KEY="clean_audio_path"
else
  # Assume it's a degradation type for training set
  IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/train_sonicmaster/degraded/train_${DATASET}_degraded/degradation_pairs.jsonl"
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/train_sonicmaster/degraded/train_${DATASET}_encoded"
  AUDIO_KEY="degraded_audio_path"
fi

echo "Input JSONL: $IN_JSONL"
echo "Output folder: $OUT_FOLDER"
echo "Audio key: $AUDIO_KEY"
echo "Extra flags: $EXTRA_FLAGS"

# Create output directory if it doesn't exist
mkdir -p "$OUT_FOLDER"

python preencode_latents_acce2.py \
  --input_jsonl "$IN_JSONL" \
  --output_dir "$OUT_FOLDER" \
  --duration_sec 30 \
  --batch_size 1 \
  $EXTRA_FLAGS
  