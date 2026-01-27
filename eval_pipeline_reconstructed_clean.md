### Step 1: Reconstruct audio

```bash
python reconstruct_vae_baseline.py \
  --input_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_reconstructed \
  --batch_size 16 \
  --output_format flac \
  --audio_key clean_path
```

### Step 2: Evaluate reconstructed audio

```bash
python calculate_metrics.py \
  --jsonl_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_reconstructed/reconstructed_test_sonicmaster.jsonl \
  --metrics lsd mel sisdr \
  --audio_key reconstructed_path \
  --output_csv /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_reconstructed/metrics_reconstructed_baseline.xlsx
```
