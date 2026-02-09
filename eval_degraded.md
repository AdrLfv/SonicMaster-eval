## Punch

### Evaluate degraded audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/degradation_pairs.jsonl \
  --audio_key degraded_audio_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/metrics_degraded_baseline.xlsx
```

```bash
sbatch run_degrade_test.sh airy
sbatch run_degrade_test.sh big
sbatch run_degrade_test.sh boom
sbatch run_degrade_test.sh bright
sbatch run_degrade_test.sh clarity
sbatch run_degrade_test.sh clip
sbatch run_degrade_test.sh comp
sbatch run_degrade_test.sh dark
sbatch run_degrade_test.sh mic
sbatch run_degrade_test.sh mix
sbatch run_degrade_test.sh mud
sbatch run_degrade_test.sh punch
sbatch run_degrade_test.sh real
sbatch run_degrade_test.sh small
sbatch run_degrade_test.sh stereo
sbatch run_degrade_test.sh vocal
sbatch run_degrade_test.sh volume
sbatch run_degrade_test.sh warm
sbatch run_degrade_test.sh xband
```

```bash
sbatch run_evaluate_aae.sh airy
sbatch run_evaluate_aae.sh big
sbatch run_evaluate_aae.sh boom
sbatch run_evaluate_aae.sh bright
sbatch run_evaluate_aae.sh clarity
sbatch run_evaluate_aae.sh clip
sbatch run_evaluate_aae.sh comp
sbatch run_evaluate_aae.sh dark
sbatch run_evaluate_aae.sh mic
sbatch run_evaluate_aae.sh mix
sbatch run_evaluate_aae.sh mud
sbatch run_evaluate_aae.sh punch
sbatch run_evaluate_aae.sh real
sbatch run_evaluate_aae.sh small
sbatch run_evaluate_aae.sh stereo
sbatch run_evaluate_aae.sh vocal
sbatch run_evaluate_aae.sh volume
sbatch run_evaluate_aae.sh warm
sbatch run_evaluate_aae.sh xband
```

```bash
sbatch run_degrade_train.sh airy
sbatch run_degrade_train.sh big
sbatch run_degrade_train.sh boom
sbatch run_degrade_train.sh bright
sbatch run_degrade_train.sh clarity
sbatch run_degrade_train.sh clip
sbatch run_degrade_train.sh comp
sbatch run_degrade_train.sh dark
sbatch run_degrade_train.sh mic
sbatch run_degrade_train.sh mix
sbatch run_degrade_train.sh mud
sbatch run_degrade_train.sh punch
sbatch run_degrade_train.sh real
sbatch run_degrade_train.sh small
sbatch run_degrade_train.sh stereo
sbatch run_degrade_train.sh vocal
sbatch run_degrade_train.sh volume
sbatch run_degrade_train.sh warm
sbatch run_degrade_train.sh xband
```