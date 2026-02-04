#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=restore
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/restore/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/restore/%j.err
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

# Get degradation type and optional prompt from command line arguments
DEGRADATION=$1
PROMPT=${2:-""}

if [ -z "$DEGRADATION" ]; then
  echo "Error: No degradation type specified"
  echo "Usage: sbatch run_restoration.sh <degradation_type> [prompt]"
  exit 1
fi

echo "Running restoration for degradation: $DEGRADATION"

# Build paths
IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_encoded/degradation_pairs.jsonl"

# Determine output folder based on whether prompt is provided
if [ -z "$PROMPT" ]; then
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored"
  echo "Running restoration WITHOUT prompt"
else
  OUT_FOLDER="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored_prompt"
  echo "Running restoration WITH prompt: $PROMPT"
fi

echo "Input JSONL: $IN_JSONL"
echo "Output folder: $OUT_FOLDER"

# Create output directory if it doesn't exist
mkdir -p "$OUT_FOLDER"

# Build the python command
CMD="python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file \"$IN_JSONL\" \
  --output_dir \"$OUT_FOLDER\" \
  --output_format hdf5"

# Add prompt if provided
if [ -n "$PROMPT" ]; then
  CMD="$CMD --prompt \"$PROMPT\""
fi

# Execute the command
eval $CMD
