
## Template

### On restored with prompts

```bash
python evaluation/extract_clap_gt.py \
    --jsonref 
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref 
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref 
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref 
```

### On restored no prompts

```bash
python evaluation/extract_clap_gt.py \
    --jsonref 
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref 
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref
```

### On reconstructed

```bash
python evaluation/extract_clap_gt.py \
    --jsonref
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref
  --audio_key reconstructed_path
```

## Big

### On restored with prompts

```bash
python evaluation/extract_clap_gt.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored_prompt/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored_prompt/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored_prompt/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored_prompt/evaluation_metadata_rank0.jsonl
```

### On restored no prompts

```bash
python evaluation/extract_clap_gt.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_restored/evaluation_metadata_rank0.jsonl
```

### On reconstructed

```bash
python evaluation/extract_clap_gt.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path
```


# Sbatch

## Template

### Run all 4 metrics on restored audio (no prompt)
sbatch run_evaluate_advanced.sh <degradation> restored

### Run only CLAP and FAD on restored audio with prompt
sbatch run_evaluate_advanced.sh <degradation> restored_prompt "clap,fad,aesthetics,kl_ssim"

### Run all metrics on reconstructed audio
sbatch run_evaluate_advanced.sh <degradation> reconstructed

### Run only KL-SSIM on degraded audio
sbatch run_evaluate_advanced.sh <degradation> degraded kl_ssim

## Airy

### Run all 4 metrics on degraded audio

sbatch run_evaluate_advanced.sh airy degraded

### Run all 4 metrics on reconstructed audio

sbatch run_evaluate_advanced.sh airy reconstructed

### Run all 4 metrics on restored audio (no prompt)

sbatch run_evaluate_advanced.sh airy restored

### Run all 4 metrics on restored audio (prompt)

sbatch run_evaluate_advanced.sh airy restored_prompt

