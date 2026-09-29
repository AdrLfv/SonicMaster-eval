# Multi-GPU Parallel Inference

## Quick Start

Run inference on multiple GPUs:

```bash
bash run_inference_multi_gpu.sh 4  # Use 4 GPUs
```

Or manually:

```bash
torchrun --nproc_per_node=4 inference_ptload_batch.py --model_ckpt checkpoints/model.safetensors
```

## Single GPU (Original)

```bash
python inference_ptload_batch.py --model_ckpt checkpoints/model.safetensors
```

## How It Works

- Dataset is automatically split across available GPUs
- Each GPU processes its subset independently
- Results are saved to separate files per rank
- Rank 0 combines all metadata files at the end
- Speedup is approximately linear with number of GPUs

## Notes

- The script automatically detects if running in multi-GPU mode via environment variables
- No code changes needed to switch between single and multi-GPU modes
- All output files are saved to the same directory
- The default inference batch size is 16. Use `--batch_size 1` on a 24 GB MIG
  when restoring 30-second clips; larger batches can run out of memory during
  AutoencoderOobleck decoding.

## Memory-constrained single-GPU inference

```bash
python inference_ptload_batch.py \
  --model_ckpt checkpoints/model.safetensors \
  --batch_size 1
```

For long runs, `--resume` appends to `restoration_metadata.jsonl` and skips a
manifest index only when both its indexed metadata row and non-empty output
file exist. `--strip_degradation_suffix` converts an input such as
`sample_00001_deg1.h5` to `sample_00001_restored.wav`.

```bash
python inference_ptload_batch.py \
  --model_ckpt checkpoints/model.safetensors \
  --infer_file /path/to/degradation_pairs.jsonl \
  --output_dir /path/to/output \
  --output_format wav \
  --use_jsonl_prompt \
  --batch_size 1 \
  --resume \
  --strip_degradation_suffix
```

Every emitted metadata row includes `source_manifest_index`, `prompt_used`,
and the inference settings. Metadata is flushed after every batch so a Slurm
timeout does not discard progress.
