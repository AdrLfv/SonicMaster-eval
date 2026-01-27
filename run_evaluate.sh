#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/evaluate/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/evaluate/%j.err

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

# Get degradation type from command line argument
DEGRADATION=$1

if [ -z "$DEGRADATION" ]; then
  echo "Error: No degradation type specified"
  echo "Usage: sbatch run_evaluate.sh <degradation_type>"
  exit 1
fi

echo "Running evaluation for degradation: $DEGRADATION"

python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_degraded/degradation_pairs.jsonl \
  --audio_key degraded_audio_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_${DEGRADATION}_degraded/metrics_degraded_baseline.xlsx
