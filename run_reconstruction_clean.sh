#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=reconstruct_clean
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16  # CPUs per task
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/reconstruct_clean/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/reconstruct_clean/%j.err
#SBATCH --ntasks-per-node=1  # One task per GPU for proper DDP
#SBATCH --partition=h100
#SBATCH --cpus-per-task=16  # CPUs per task
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

echo "Running VAE reconstruction for clean audio"

# Build paths
IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/clean/metadata.jsonl"
OUT_FOLDER="/scratch/alefevre/evaluation_sonicmaster/clean_reconstructed"

if [ ! -f "$IN_JSONL" ]; then
  echo "Error: Could not find JSONL file: ${IN_JSONL}"
  exit 1
fi

echo "Input JSONL: $IN_JSONL"
echo "Output folder: $OUT_FOLDER"

# Create output directory if it doesn't exist
mkdir -p "$OUT_FOLDER"

python reconstruct_vae_baseline.py \
  --input_jsonl "$IN_JSONL" \
  --output_dir "$OUT_FOLDER" \
  --audio_key clean_audio_path \
  --duration_sec 30 \
  --batch_size 16 \
  --output_format wav

echo "Done: clean reconstruction complete"
