#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=encode
#SBATCH --nodes=1
#SBATCH --partition=standard
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

if [ -z "$DATASET" ]; then
  echo "Error: No dataset type specified"
  echo "Usage: sbatch run_encode.sh <dataset_type>"
  echo "  dataset_type can be: test, train, or a degradation name (e.g., airy, punch, etc.)"
  exit 1
fi

echo "Running VAE encoding for dataset: $DATASET"

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
  # Assume it's a degradation type
  IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DATASET}_degraded/degradation_pairs.jsonl"
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DATASET}_encoded"
  AUDIO_KEY="degraded_audio_path"
fi

echo "Input JSONL: $IN_JSONL"
echo "Output folder: $OUT_FOLDER"
echo "Audio key: $AUDIO_KEY"

# Create output directory if it doesn't exist
mkdir -p "$OUT_FOLDER"

python preencode_latents_acce2.py \
  --input_jsonl "$IN_JSONL" \
  --output_dir "$OUT_FOLDER" \
  --duration_sec 30 \
  --batch_size 16
