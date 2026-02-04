#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate_aae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/evaluate_aae/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/evaluate_aae/%j.err

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

# Get degradation type and optional evaluation type from command line arguments
DEGRADATION=$1
EVAL_TYPE=${2:-degraded}

if [ -z "$DEGRADATION" ]; then
  echo "Error: No degradation type specified"
  echo "Usage: sbatch run_evaluate_aae.sh <degradation_type> [eval_type]"
  echo "  eval_type: degraded (default), reconstructed, restored, restored_prompt"
  exit 1
fi

echo "Running evaluation for degradation: $DEGRADATION"
echo "Evaluation type: $EVAL_TYPE"

# Determine input JSONL, output CSV, and audio key based on eval_type
if [ "$EVAL_TYPE" = "reconstructed" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_reconstructed/reconstructed_degradation_pairs.jsonl"
  OUTPUT_CSV="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_reconstructed/metrics_reconstructed_baseline.xlsx"
  AUDIO_KEY="reconstructed_path"
elif [ "$EVAL_TYPE" = "restored" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored/evaluation_metadata.jsonl"
  OUTPUT_CSV="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored/metrics_restored_baseline.xlsx"
  AUDIO_KEY="restored_path"
elif [ "$EVAL_TYPE" = "restored_prompt" ]; then
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored_prompt/evaluation_metadata.jsonl"
  OUTPUT_CSV="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_restored_prompt/metrics_restored_prompt_baseline.xlsx"
  AUDIO_KEY="restored_path"
else
  INPUT_JSONL="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_degraded/degradation_pairs.jsonl"
  OUTPUT_CSV="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_degraded/metrics_degraded_baseline.xlsx"
  AUDIO_KEY="degraded_audio_path"
fi

echo "Input JSONL: $INPUT_JSONL"
echo "Output CSV: $OUTPUT_CSV"

python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref "$INPUT_JSONL" \
  --audio_key "$AUDIO_KEY" \
  --output_csv "$OUTPUT_CSV"
