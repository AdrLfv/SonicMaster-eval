
## Punch effect

### Step 1: Degrade test folder with punch
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded \
  --deg_spec punch
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_punch_prompt \
  --prompt "Add more impact and dynamic punch to the sound."
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref 
```
## Clip effect

### Step 1: Degrade test folder with clipping
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded \
  --deg_spec clip
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_clip_prompt \
  --prompt "Reduce the clipping and reconstruct the lost audio, please."
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Small effect

### Step 1: Degrade test folder with small
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded \
  --deg_spec small
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_small_prompt \
  --prompt "Clean this off any echoes!"
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Big effect

### Step 1: Degrade test folder with big
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded \
  --deg_spec big
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt \
  --prompt "Can you remove the excess reverb in this audio, please?"
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Dark effect

### Step 1: Degrade test folder with dark
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded \
  --deg_spec dark
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_dark_prompt \
  --prompt "Make the tone fuller and less sharp."
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Warm effect

### Step 1: Degrade test folder with warm
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded \
  --deg_spec warm
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_warm_prompt \
  --prompt "Make the sound warmer and more inviting."
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Real effect

### Step 1: Degrade test folder with real
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded \
  --deg_spec real
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_real_prompt \
  --prompt "Please, reduce the strong echo in this song."
```

### Step 4: Evaluate results
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

