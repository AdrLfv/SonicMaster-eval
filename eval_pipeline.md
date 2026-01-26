
## Punch

### Step 1: Degrade test folder with punch
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded \
  --deg_spec punch \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_punch \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_punch_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_punch_prompt \
  --prompt "Reduce the punch and reconstruct the lost audio, please." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec clip \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_clip \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clip_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_clip_prompt \
  --prompt "Reduce the clipping and reconstruct the lost audio, please." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec small \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_small \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_small_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_small_prompt \
  --prompt "Clean this off any echoes!" \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec big \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_big \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_big_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_big_prompt \
  --prompt "Can you remove the excess reverb in this audio, please?" \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec dark \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_dark \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_dark_prompt \
  --prompt "Make the tone fuller and less sharp." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec warm \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_warm \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_warm_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_warm_prompt \
  --prompt "Make the sound warmer and more inviting." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
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
  --deg_spec real \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_real \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_real_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_real_prompt \
  --prompt "Please, reduce the strong echo in this song." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Xband effect

### Step 1: Degrade test folder with xband
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_xband_degraded \
  --deg_spec xband \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_xband_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_xband_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_xband_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_xband \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_xband_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_xband_prompt \
  --prompt "Restore the full-spectrum balance and undo the multiband EQ color." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Mic effect

### Step 1: Degrade test folder with mic
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mic_degraded \
  --deg_spec mic \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mic_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mic_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mic_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mic \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mic_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mic_prompt \
  --prompt "Remove the microphone coloration and make the tone neutral again." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Bright effect

### Step 1: Degrade test folder with bright
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_bright_degraded \
  --deg_spec bright \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_bright_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_bright_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_bright_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_bright \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_bright_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_bright_prompt \
  --prompt "Bring back the top-end sparkle without making it harsh." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Airy effect

### Step 1: Degrade test folder with airy
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_airy_degraded \
  --deg_spec airy \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_airy_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_airy_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_airy_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_airy \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_airy_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_airy_prompt \
  --prompt "Restore the airy high-end texture and remove the dull veil." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Boom effect

### Step 1: Degrade test folder with boom
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_boom_degraded \
  --deg_spec boom \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_boom_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_boom_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_boom_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_boom \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_boom_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_boom_prompt \
  --prompt "Tighten the low end and remove the excessive boominess." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Clarity effect

### Step 1: Degrade test folder with clarity
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clarity_degraded \
  --deg_spec clarity \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clarity_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clarity_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clarity_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_clarity \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_clarity_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_clarity_prompt \
  --prompt "Clear up the smeared transients and improve articulation." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Mud effect

### Step 1: Degrade test folder with mud
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mud_degraded \
  --deg_spec mud \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mud_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mud_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mud_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mud \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mud_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mud_prompt \
  --prompt "Remove the muddiness and open up the midrange." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Vocal effect

### Step 1: Degrade test folder with vocal
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_vocal_degraded \
  --deg_spec vocal \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_vocal_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_vocal_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_vocal_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_vocal \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_vocal_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_vocal_prompt \
  --prompt "Bring the vocals back to their natural level and tone." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Comp effect

### Step 1: Degrade test folder with comp
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_comp_degraded \
  --deg_spec comp \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_comp_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_comp_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_comp_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_comp \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_comp_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_comp_prompt \
  --prompt "Undo the heavy compression and recover the natural dynamics." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Mix effect

### Step 1: Degrade test folder with mix
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mix_degraded \
  --deg_spec mix \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mix_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mix_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mix_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mix \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_mix_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_mix_prompt \
  --prompt "Reduce the artificial reverb blend and keep only natural ambience." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Volume effect

### Step 1: Degrade test folder with volume
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_volume_degraded \
  --deg_spec volume \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_volume_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_volume_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_volume_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_volume \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_volume_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_volume_prompt \
  --prompt "Restore the proper loudness without clipping or pumping." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

## Stereo effect

### Step 1: Degrade test folder with stereo
```bash
python dataset_scripts/degrade_final_chunks.py \
  --in_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --out_folder /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_stereo_degraded \
  --deg_spec stereo \
  --output_format hdf5
```

### Step 2: Encode degraded audio to latents
```bash
python preencode_latents_acce2.py \
  --input_jsonl /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_stereo_degraded/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_stereo_latents \
  --duration_sec 30 \
  --batch_size 16
```

### Step 3: Restore audio using the model (no prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_stereo_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_stereo \
  --output_format hdf5
```

### Step 3: Restore audio using the model (with prompt)
```bash
python inference_ptload_batch.py \
  --config configs/tangoflux_config.yaml \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_stereo_latents/degradation_pairs.jsonl \
  --output_dir /work/vita/datasets/audio/sonicmaster/audios/restored_stereo_prompt \
  --prompt "Rebuild the stereo width and maintain mono compatibility." \
  --output_format hdf5
```

### Step 4: Evaluate results (no prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

### Step 4: Evaluate results (with prompt)
```bash
python evaluation/evaluate_control_multiple_degs_mass.py \
  --jsonref
```

