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

# Get degradation type, optional prompt, and extra args from command line arguments
DEGRADATION=$1
PROMPT=${2:-""}
EXTRA_ARGS="${@:3}"

if [ -z "$DEGRADATION" ]; then
  echo "Error: No degradation type specified"
  echo "Usage: sbatch run_restoration.sh <degradation_type> [prompt] [extra_args...]"
  echo "  degradation_type: e.g., clip, airy, punch, etc."
  echo "  prompt: (optional) restoration prompt for the model. Use '' for no prompt."
  echo "  extra_args: (optional) forwarded to inference_ptload_batch.py (e.g., --num_examples 10)"
  exit 1
fi

echo "Running restoration for degradation: $DEGRADATION"

# Build paths — read from per-degradation subfolder
IN_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/degraded/${DEGRADATION}_degraded/degradation_pairs.jsonl"

if [ ! -f "$IN_JSONL" ]; then
  echo "Error: Could not find JSONL file: ${IN_JSONL}"
  exit 1
fi

# Determine output folder based on whether prompt is provided
USE_JSONL_PROMPT=0
if [ "$PROMPT" = "__USE_JSONL_PROMPT__" ]; then
  USE_JSONL_PROMPT=1
  PROMPT=""
  OUT_FOLDER="/scratch/alefevre/evaluation_sonicmaster/restored/${DEGRADATION}_sm_restored_prompt"
  echo "Running restoration WITH per-sample JSONL prompts"
elif [ -z "$PROMPT" ]; then
  OUT_FOLDER="/scratch/alefevre/evaluation_sonicmaster/restored/${DEGRADATION}_sm_restored"
  echo "Running restoration WITHOUT prompt"
else
  OUT_FOLDER="/scratch/alefevre/evaluation_sonicmaster/restored/${DEGRADATION}_sm_restored_prompt"
  echo "Running restoration WITH prompt: $PROMPT"
fi

echo "=========================================="
echo "Inference Configuration (SonicMaster):"
echo "=========================================="
echo "Degradation:   ${DEGRADATION}"
echo "JSONL:         ${IN_JSONL}"
echo "Output dir:    ${OUT_FOLDER}"
echo "Prompt:        ${PROMPT:-<none>}"
echo "Extra args:    ${EXTRA_ARGS:-<none>}"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p "$OUT_FOLDER"

# Build the python command
CMD="python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file \"$IN_JSONL\" \
  --output_dir \"$OUT_FOLDER\" \
  --output_format wav"

# Add prompt if provided
if [ "$USE_JSONL_PROMPT" -eq 1 ]; then
  CMD="$CMD --use_jsonl_prompt"
elif [ -n "$PROMPT" ]; then
  CMD="$CMD --prompt \"$PROMPT\""
fi

# Forward any additional CLI args
if [ -n "$EXTRA_ARGS" ]; then
  CMD="$CMD ${EXTRA_ARGS}"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD
