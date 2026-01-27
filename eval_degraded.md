## Punch

### Evaluate degraded audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/degradation_pairs.jsonl \
  --audio_key degraded_audio_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/metrics_degraded_baseline.xlsx
```

```bash
sbatch run_degrade.sh airy
sbatch run_degrade.sh big
sbatch run_degrade.sh boom
sbatch run_degrade.sh bright
sbatch run_degrade.sh clarity
sbatch run_degrade.sh clip
sbatch run_degrade.sh comp
sbatch run_degrade.sh dark
sbatch run_degrade.sh mic
sbatch run_degrade.sh mix
sbatch run_degrade.sh mud
sbatch run_degrade.sh punch
sbatch run_degrade.sh real
sbatch run_degrade.sh small
sbatch run_degrade.sh stereo
sbatch run_degrade.sh vocal
sbatch run_degrade.sh volume
sbatch run_degrade.sh warm
sbatch run_degrade.sh xband
```

```bash
sbatch run_evaluate.sh airy
sbatch run_evaluate.sh big
sbatch run_evaluate.sh boom
sbatch run_evaluate.sh bright
sbatch run_evaluate.sh clarity
sbatch run_evaluate.sh clip
sbatch run_evaluate.sh comp
sbatch run_evaluate.sh dark
sbatch run_evaluate.sh mic
sbatch run_evaluate.sh mix
sbatch run_evaluate.sh mud
sbatch run_evaluate.sh punch
sbatch run_evaluate.sh real
sbatch run_evaluate.sh small
sbatch run_evaluate.sh stereo
sbatch run_evaluate.sh vocal
sbatch run_evaluate.sh volume
sbatch run_evaluate.sh warm
sbatch run_evaluate.sh xband
```