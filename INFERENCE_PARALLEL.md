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
