#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=compute_clap_embeddings
#SBATCH --nodes=1
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/compute_clap_embeddings/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/compute_clap_embeddings/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=h100
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

# Get degradation type and evaluation type from command line arguments
DEGRADATION=$1
EVAL_TYPE=${2:-degraded}

if [ -z "$DEGRADATION" ]; then
  echo "Error: No degradation type specified"
  echo "Usage: sbatch run_compute_clap_embeddings.sh <degradation_type> [eval_type]"
  echo "  eval_type: degraded (default), reconstructed, restored, restored_prompt"
  echo "  Example: sbatch run_compute_clap_embeddings.sh airy restored"
  exit 1
fi

echo "Computing CLAP embeddings for degradation: $DEGRADATION"
echo "Evaluation type: $EVAL_TYPE"

# Determine input JSONL and audio key based on eval_type
if [ "$EVAL_TYPE" = "reconstructed" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_reconstructed/reconstructed_degradation_pairs.jsonl"
  AUDIO_KEY="reconstructed_path"
elif [ "$EVAL_TYPE" = "restored" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored/evaluation_metadata.jsonl"
  AUDIO_KEY="restored_path"
elif [ "$EVAL_TYPE" = "restored_prompt" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored_prompt/evaluation_metadata.jsonl"
  AUDIO_KEY="restored_path"
else
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_degraded/degradation_pairs.jsonl"
  AUDIO_KEY="degraded_audio_path"
fi

echo "Input JSONL: $INPUT_JSONL"
echo "Audio key: $AUDIO_KEY"
echo ""

echo "========================================="
echo "Computing CLAP embeddings..."
echo "========================================="
python evaluation/extract_clap_gt.py --jsonref "$INPUT_JSONL" --audio_key "$AUDIO_KEY"

echo ""
echo "✅ CLAP embeddings computation complete!"
