#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=degrade
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=standard
#SBATCH --output=/work/vita/alefevre/programs/SonicMaster/logs/degrade/%j.out
#SBATCH --error=/work/vita/alefevre/programs/SonicMaster/logs/degrade/%j.err


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
  echo "Usage: sbatch run_degrade_train.sh <degradation_type>"
  exit 1
fi

echo "Running degradation: $DEGRADATION"

python dataset_scripts/degrade_final_chunks.py \
  --in_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/clean/metadata.jsonl \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/degraded/${DEGRADATION}_degraded_cropped \
  --deg_spec $DEGRADATION \
  --output_format hdf5 \
  --crop_to_original
