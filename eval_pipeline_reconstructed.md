## Punch

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded_reconstructed \
  --batch_size 16 \
  --output_format flac
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Clip

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded_reconstructed \
  --batch_size 16 \
  --output_format hdf5
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Small

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded_reconstructed \
  --batch_size 16 \
  --output_format flac
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Big

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed \
  --batch_size 16 \
  --output_format flac
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Dark

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded_reconstructed \
  --batch_size 16 \
  --output_format flac
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Warm

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded_reconstructed \
  --batch_size 16 \
  --output_format flac
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded_reconstructed/metrics_reconstructed_baseline.xlsx
```

## Real

### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded_reconstructed \
  --batch_size 16 \
  --output_format hdf5
```

### Step 2: Evaluate reconstructed audio

```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded_reconstructed/reconstructed_degradation_pairs.jsonl \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded_reconstructed/metrics_reconstructed_baseline.xlsx