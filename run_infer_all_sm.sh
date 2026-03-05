#!/bin/bash
# ============================================================
# run_infer_all_sm.sh
#
# Submit SonicMaster restoration inference jobs for all (or
# selected) degradations. Loops over each {deg}_degraded
# subfolder in the test degraded directory and submits a
# run_restoration.sh job for each.
#
# Usage:
#   bash run_infer_all_sm.sh [--degradations "airy clip ..."] [--prompt "..."] [--dry_run]
#
# Examples:
#   bash run_infer_all_sm.sh
#   bash run_infer_all_sm.sh --degradations "clip airy"
#   bash run_infer_all_sm.sh --prompt "Restore the audio quality."
#   bash run_infer_all_sm.sh --dry_run
# ============================================================

set -e

PROJECT_ROOT=/work/vita/alefevre/programs/SonicMaster
DEGRADED_BASE="/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/degraded"

ALL_DEGRADATIONS="airy big boom bright clarity clip comp dark mic mix mud punch real small stereo vocal volume warm xband"
DEGRADATIONS="$ALL_DEGRADATIONS"
PROMPT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --degradations) DEGRADATIONS="$2"; shift 2 ;;
    --prompt)       PROMPT="$2";       shift 2 ;;
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
echo " SonicMaster Restoration Inference"
echo " Degraded base: ${DEGRADED_BASE}"
echo " Prompt: ${PROMPT:-<none>}"
echo "════════════════════════════════════════════════════════"

for DEG in $DEGRADATIONS; do
  DEG_LC=$(echo "$DEG" | tr '[:upper:]' '[:lower:]')

  # Verify JSONL exists
  JSONL_PATH="${DEGRADED_BASE}/${DEG_LC}_degraded/degradation_pairs.jsonl"
  if [ ! -f "$JSONL_PATH" ]; then
    echo "⚠️  Skipping ${DEG_LC}: JSONL not found at ${JSONL_PATH}"
    continue
  fi

  echo ""
  echo "── ${DEG_LC} ─────────────────────────────────────────"

  # Run 1: without prompt
  echo "  [no prompt]"
  submit sbatch "${PROJECT_ROOT}/run_restoration.sh" "$DEG_LC" ""

  # Run 2: with per-sample JSONL prompt
  echo "  [with JSONL prompt]"
  submit sbatch "${PROJECT_ROOT}/run_restoration.sh" "$DEG_LC" "__USE_JSONL_PROMPT__"
done

echo ""
echo "✅ Done submitting SonicMaster inference jobs"
