#!/bin/bash
# ============================================================
# run_reconstruction_degraded_all.sh
#
# Submit SonicMaster VAE reconstruction jobs for all (or
# selected) degradations. Loops over each {deg}_degraded
# subfolder and submits run_reconstruction_degraded.sh for each.
#
# Usage:
#   bash run_reconstruction_degraded_all.sh [--degradations "airy clip ..."] [--dry_run]
#
# Examples:
#   bash run_reconstruction_degraded_all.sh
#   bash run_reconstruction_degraded_all.sh --degradations "clip airy"
#   bash run_reconstruction_degraded_all.sh --dry_run
# ============================================================

set -e

PROJECT_ROOT=/work/vita/alefevre/programs/SonicMaster
DEGRADED_BASE="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/degraded"

ALL_DEGRADATIONS="airy big boom bright clarity clip comp dark mic mix mud punch real small stereo vocal volume warm xband"
DEGRADATIONS="$ALL_DEGRADATIONS"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --degradations) DEGRADATIONS="$2"; shift 2 ;;
    --dry_run)      DRY_RUN=1;         shift 1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

submit() {
  if [ $DRY_RUN -eq 1 ]; then
    echo "[DRY RUN] $*"
  else
    echo "+ $*"
    eval "$*"
  fi
}

echo "════════════════════════════════════════════════════════"
echo " SonicMaster VAE Reconstruction — Degraded Audio"
echo " Degraded base: ${DEGRADED_BASE}"
echo "════════════════════════════════════════════════════════"

for DEG in $DEGRADATIONS; do
  DEG_LC=$(echo "$DEG" | tr '[:upper:]' '[:lower:]')

  JSONL_PATH="${DEGRADED_BASE}/${DEG_LC}_degraded/degradation_pairs.jsonl"
  if [ ! -f "$JSONL_PATH" ]; then
    echo "⚠️  Skipping ${DEG_LC}: JSONL not found at ${JSONL_PATH}"
    continue
  fi

  echo ""
  echo "── ${DEG_LC} ─────────────────────────────────────────"
  submit sbatch "${PROJECT_ROOT}/run_reconstruction_degraded.sh" "$DEG_LC"
done

echo ""
echo "✅ Done submitting SonicMaster degraded reconstruction jobs"
