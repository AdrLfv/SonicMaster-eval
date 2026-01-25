
## Punch

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
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt/inference_20260124_181648/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt/inference_20260124_181648/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt/inference_20260124_181648/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt/inference_20260124_181648/evaluation_metadata_rank0.jsonl
```

### On restored no prompts

```bash
python evaluation/extract_clap_gt.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big/inference_20260124_175426/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big/inference_20260124_175426/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big/inference_20260124_175426/evaluation_metadata_rank0.jsonl
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/restored_big/inference_20260124_175426/evaluation_metadata_rank0.jsonl
```

### On reconstructed

```bash
python evaluation/extract_clap_gt.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_fad_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_kl_ssim_mass.py \
    --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
    --audio_key reconstructed_path
```

```bash
python evaluation/extract_aesthetics_scores.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path
```
