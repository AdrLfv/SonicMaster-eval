# Advanced Metrics Evaluation Commands

This document contains commands for running advanced metrics (CLAP, FAD, KL-SSIM, Aesthetics) on different audio types.

## Usage

Each metric has its own specialized script for better resource allocation:

```bash
# CLAP evaluation (requires GPU, runs on h100)
sbatch run_compute_clap_embeddings.sh <degradation> [eval_type]

# FAD evaluation (requires GPU, runs on h100)
sbatch run_evaluate_fad.sh <degradation> [eval_type]

# KL-SSIM evaluation (CPU only, runs on a100)
sbatch run_evaluate_kl_ssim.sh <degradation> [eval_type]

# Aesthetics evaluation (CPU only, runs on a100)
sbatch run_evaluate_aesthetics.sh <degradation> [eval_type]
```

- **degradation**: Type of degradation (airy, big, boom, etc.)
- **eval_type**: degraded (default), reconstructed, restored, restored_prompt

## Examples

```bash
# Run CLAP on restored audio (no prompt)
sbatch run_compute_clap_embeddings.sh airy restored

# Run FAD on restored audio with prompt
sbatch run_evaluate_fad.sh airy restored_prompt

# Run KL-SSIM on reconstructed audio
sbatch run_evaluate_kl_ssim.sh airy reconstructed

# Run Aesthetics on degraded audio
sbatch run_evaluate_aesthetics.sh airy degraded
```

## Batch Commands for restored

### CLAP evaluation
```bash
sbatch run_compute_clap_embeddings.sh airy restored
sbatch run_compute_clap_embeddings.sh big restored
sbatch run_compute_clap_embeddings.sh boom restored
sbatch run_compute_clap_embeddings.sh bright restored
sbatch run_compute_clap_embeddings.sh clarity restored
sbatch run_compute_clap_embeddings.sh clip restored
sbatch run_compute_clap_embeddings.sh comp restored
sbatch run_compute_clap_embeddings.sh dark restored
sbatch run_compute_clap_embeddings.sh mic restored
sbatch run_compute_clap_embeddings.sh mix restored
sbatch run_compute_clap_embeddings.sh mud restored
sbatch run_compute_clap_embeddings.sh punch restored
sbatch run_compute_clap_embeddings.sh real restored
sbatch run_compute_clap_embeddings.sh small restored
sbatch run_compute_clap_embeddings.sh stereo restored
sbatch run_compute_clap_embeddings.sh vocal restored
sbatch run_compute_clap_embeddings.sh volume restored
sbatch run_compute_clap_embeddings.sh warm restored
sbatch run_compute_clap_embeddings.sh xband restored
```

### FAD evaluation
```bash
sbatch run_evaluate_fad.sh airy restored
sbatch run_evaluate_fad.sh big restored
sbatch run_evaluate_fad.sh boom restored
sbatch run_evaluate_fad.sh bright restored
sbatch run_evaluate_fad.sh clarity restored
sbatch run_evaluate_fad.sh clip restored
sbatch run_evaluate_fad.sh comp restored
sbatch run_evaluate_fad.sh dark restored
sbatch run_evaluate_fad.sh mic restored
sbatch run_evaluate_fad.sh mix restored
sbatch run_evaluate_fad.sh mud restored
sbatch run_evaluate_fad.sh punch restored
sbatch run_evaluate_fad.sh real restored
sbatch run_evaluate_fad.sh small restored
sbatch run_evaluate_fad.sh stereo restored
sbatch run_evaluate_fad.sh vocal restored
sbatch run_evaluate_fad.sh volume restored
sbatch run_evaluate_fad.sh warm restored
sbatch run_evaluate_fad.sh xband restored
```

### KL-SSIM evaluation
```bash
sbatch run_evaluate_kl_ssim.sh airy restored
sbatch run_evaluate_kl_ssim.sh big restored
sbatch run_evaluate_kl_ssim.sh boom restored
sbatch run_evaluate_kl_ssim.sh bright restored
sbatch run_evaluate_kl_ssim.sh clarity restored
sbatch run_evaluate_kl_ssim.sh clip restored
sbatch run_evaluate_kl_ssim.sh comp restored
sbatch run_evaluate_kl_ssim.sh dark restored
sbatch run_evaluate_kl_ssim.sh mic restored
sbatch run_evaluate_kl_ssim.sh mix restored
sbatch run_evaluate_kl_ssim.sh mud restored
sbatch run_evaluate_kl_ssim.sh punch restored
sbatch run_evaluate_kl_ssim.sh real restored
sbatch run_evaluate_kl_ssim.sh small restored
sbatch run_evaluate_kl_ssim.sh stereo restored
sbatch run_evaluate_kl_ssim.sh vocal restored
sbatch run_evaluate_kl_ssim.sh volume restored
sbatch run_evaluate_kl_ssim.sh warm restored
sbatch run_evaluate_kl_ssim.sh xband restored
```

### Aesthetics evaluation
```bash
sbatch run_evaluate_aesthetics.sh airy restored
sbatch run_evaluate_aesthetics.sh big restored
sbatch run_evaluate_aesthetics.sh boom restored
sbatch run_evaluate_aesthetics.sh bright restored
sbatch run_evaluate_aesthetics.sh clarity restored
sbatch run_evaluate_aesthetics.sh clip restored
sbatch run_evaluate_aesthetics.sh comp restored
sbatch run_evaluate_aesthetics.sh dark restored
sbatch run_evaluate_aesthetics.sh mic restored
sbatch run_evaluate_aesthetics.sh mix restored
sbatch run_evaluate_aesthetics.sh mud restored
sbatch run_evaluate_aesthetics.sh punch restored
sbatch run_evaluate_aesthetics.sh real restored
sbatch run_evaluate_aesthetics.sh small restored
sbatch run_evaluate_aesthetics.sh stereo restored
sbatch run_evaluate_aesthetics.sh vocal restored
sbatch run_evaluate_aesthetics.sh volume restored
sbatch run_evaluate_aesthetics.sh warm restored
sbatch run_evaluate_aesthetics.sh xband restored
```

## Batch Commands for restored_prompt

### CLAP evaluation
```bash
sbatch run_compute_clap_embeddings.sh airy restored_prompt
sbatch run_compute_clap_embeddings.sh big restored_prompt
sbatch run_compute_clap_embeddings.sh boom restored_prompt
sbatch run_compute_clap_embeddings.sh bright restored_prompt
sbatch run_compute_clap_embeddings.sh clarity restored_prompt
sbatch run_compute_clap_embeddings.sh clip restored_prompt
sbatch run_compute_clap_embeddings.sh comp restored_prompt
sbatch run_compute_clap_embeddings.sh dark restored_prompt
sbatch run_compute_clap_embeddings.sh mic restored_prompt
sbatch run_compute_clap_embeddings.sh mix restored_prompt
sbatch run_compute_clap_embeddings.sh mud restored_prompt
sbatch run_compute_clap_embeddings.sh punch restored_prompt
sbatch run_compute_clap_embeddings.sh real restored_prompt
sbatch run_compute_clap_embeddings.sh small restored_prompt
sbatch run_compute_clap_embeddings.sh stereo restored_prompt
sbatch run_compute_clap_embeddings.sh vocal restored_prompt
sbatch run_compute_clap_embeddings.sh volume restored_prompt
sbatch run_compute_clap_embeddings.sh warm restored_prompt
sbatch run_compute_clap_embeddings.sh xband restored_prompt
```

### FAD evaluation
```bash
sbatch run_evaluate_fad.sh airy restored_prompt
sbatch run_evaluate_fad.sh big restored_prompt
sbatch run_evaluate_fad.sh boom restored_prompt
sbatch run_evaluate_fad.sh bright restored_prompt
sbatch run_evaluate_fad.sh clarity restored_prompt
sbatch run_evaluate_fad.sh clip restored_prompt
sbatch run_evaluate_fad.sh comp restored_prompt
sbatch run_evaluate_fad.sh dark restored_prompt
sbatch run_evaluate_fad.sh mic restored_prompt
sbatch run_evaluate_fad.sh mix restored_prompt
sbatch run_evaluate_fad.sh mud restored_prompt
sbatch run_evaluate_fad.sh punch restored_prompt
sbatch run_evaluate_fad.sh real restored_prompt
sbatch run_evaluate_fad.sh small restored_prompt
sbatch run_evaluate_fad.sh stereo restored_prompt
sbatch run_evaluate_fad.sh vocal restored_prompt
sbatch run_evaluate_fad.sh volume restored_prompt
sbatch run_evaluate_fad.sh warm restored_prompt
sbatch run_evaluate_fad.sh xband restored_prompt
```

### KL-SSIM evaluation
```bash
sbatch run_evaluate_kl_ssim.sh airy restored_prompt
sbatch run_evaluate_kl_ssim.sh big restored_prompt
sbatch run_evaluate_kl_ssim.sh boom restored_prompt
sbatch run_evaluate_kl_ssim.sh bright restored_prompt
sbatch run_evaluate_kl_ssim.sh clarity restored_prompt
sbatch run_evaluate_kl_ssim.sh clip restored_prompt
sbatch run_evaluate_kl_ssim.sh comp restored_prompt
sbatch run_evaluate_kl_ssim.sh dark restored_prompt
sbatch run_evaluate_kl_ssim.sh mic restored_prompt
sbatch run_evaluate_kl_ssim.sh mix restored_prompt
sbatch run_evaluate_kl_ssim.sh mud restored_prompt
sbatch run_evaluate_kl_ssim.sh punch restored_prompt
sbatch run_evaluate_kl_ssim.sh real restored_prompt
sbatch run_evaluate_kl_ssim.sh small restored_prompt
sbatch run_evaluate_kl_ssim.sh stereo restored_prompt
sbatch run_evaluate_kl_ssim.sh vocal restored_prompt
sbatch run_evaluate_kl_ssim.sh volume restored_prompt
sbatch run_evaluate_kl_ssim.sh warm restored_prompt
sbatch run_evaluate_kl_ssim.sh xband restored_prompt
```

### Aesthetics evaluation
```bash
sbatch run_evaluate_aesthetics.sh airy restored_prompt
sbatch run_evaluate_aesthetics.sh big restored_prompt
sbatch run_evaluate_aesthetics.sh boom restored_prompt
sbatch run_evaluate_aesthetics.sh bright restored_prompt
sbatch run_evaluate_aesthetics.sh clarity restored_prompt
sbatch run_evaluate_aesthetics.sh clip restored_prompt
sbatch run_evaluate_aesthetics.sh comp restored_prompt
sbatch run_evaluate_aesthetics.sh dark restored_prompt
sbatch run_evaluate_aesthetics.sh mic restored_prompt
sbatch run_evaluate_aesthetics.sh mix restored_prompt
sbatch run_evaluate_aesthetics.sh mud restored_prompt
sbatch run_evaluate_aesthetics.sh punch restored_prompt
sbatch run_evaluate_aesthetics.sh real restored_prompt
sbatch run_evaluate_aesthetics.sh small restored_prompt
sbatch run_evaluate_aesthetics.sh stereo restored_prompt
sbatch run_evaluate_aesthetics.sh vocal restored_prompt
sbatch run_evaluate_aesthetics.sh volume restored_prompt
sbatch run_evaluate_aesthetics.sh warm restored_prompt
sbatch run_evaluate_aesthetics.sh xband restored_prompt
```

## Batch Commands for reconstructed

### CLAP evaluation
```bash
sbatch run_compute_clap_embeddings.sh airy reconstructed
sbatch run_compute_clap_embeddings.sh big reconstructed
sbatch run_compute_clap_embeddings.sh boom reconstructed
sbatch run_compute_clap_embeddings.sh bright reconstructed
sbatch run_compute_clap_embeddings.sh clarity reconstructed
sbatch run_compute_clap_embeddings.sh clip reconstructed
sbatch run_compute_clap_embeddings.sh comp reconstructed
sbatch run_compute_clap_embeddings.sh dark reconstructed
sbatch run_compute_clap_embeddings.sh mic reconstructed
sbatch run_compute_clap_embeddings.sh mix reconstructed
sbatch run_compute_clap_embeddings.sh mud reconstructed
sbatch run_compute_clap_embeddings.sh punch reconstructed
sbatch run_compute_clap_embeddings.sh real reconstructed
sbatch run_compute_clap_embeddings.sh small reconstructed
sbatch run_compute_clap_embeddings.sh stereo reconstructed
sbatch run_compute_clap_embeddings.sh vocal reconstructed
sbatch run_compute_clap_embeddings.sh volume reconstructed
sbatch run_compute_clap_embeddings.sh warm reconstructed
sbatch run_compute_clap_embeddings.sh xband reconstructed
```

### FAD evaluation
```bash
sbatch run_evaluate_fad.sh airy reconstructed
sbatch run_evaluate_fad.sh big reconstructed
sbatch run_evaluate_fad.sh boom reconstructed
sbatch run_evaluate_fad.sh bright reconstructed
sbatch run_evaluate_fad.sh clarity reconstructed
sbatch run_evaluate_fad.sh clip reconstructed
sbatch run_evaluate_fad.sh comp reconstructed
sbatch run_evaluate_fad.sh dark reconstructed
sbatch run_evaluate_fad.sh mic reconstructed
sbatch run_evaluate_fad.sh mix reconstructed
sbatch run_evaluate_fad.sh mud reconstructed
sbatch run_evaluate_fad.sh punch reconstructed
sbatch run_evaluate_fad.sh real reconstructed
sbatch run_evaluate_fad.sh small reconstructed
sbatch run_evaluate_fad.sh stereo reconstructed
sbatch run_evaluate_fad.sh vocal reconstructed
sbatch run_evaluate_fad.sh volume reconstructed
sbatch run_evaluate_fad.sh warm reconstructed
sbatch run_evaluate_fad.sh xband reconstructed
```

### KL-SSIM evaluation
```bash
sbatch run_evaluate_kl_ssim.sh airy reconstructed
sbatch run_evaluate_kl_ssim.sh big reconstructed
sbatch run_evaluate_kl_ssim.sh boom reconstructed
sbatch run_evaluate_kl_ssim.sh bright reconstructed
sbatch run_evaluate_kl_ssim.sh clarity reconstructed
sbatch run_evaluate_kl_ssim.sh clip reconstructed
sbatch run_evaluate_kl_ssim.sh comp reconstructed
sbatch run_evaluate_kl_ssim.sh dark reconstructed
sbatch run_evaluate_kl_ssim.sh mic reconstructed
sbatch run_evaluate_kl_ssim.sh mix reconstructed
sbatch run_evaluate_kl_ssim.sh mud reconstructed
sbatch run_evaluate_kl_ssim.sh punch reconstructed
sbatch run_evaluate_kl_ssim.sh real reconstructed
sbatch run_evaluate_kl_ssim.sh small reconstructed
sbatch run_evaluate_kl_ssim.sh stereo reconstructed
sbatch run_evaluate_kl_ssim.sh vocal reconstructed
sbatch run_evaluate_kl_ssim.sh volume reconstructed
sbatch run_evaluate_kl_ssim.sh warm reconstructed
sbatch run_evaluate_kl_ssim.sh xband reconstructed
```

### Aesthetics evaluation
```bash
sbatch run_evaluate_aesthetics.sh airy reconstructed
sbatch run_evaluate_aesthetics.sh big reconstructed
sbatch run_evaluate_aesthetics.sh boom reconstructed
sbatch run_evaluate_aesthetics.sh bright reconstructed
sbatch run_evaluate_aesthetics.sh clarity reconstructed
sbatch run_evaluate_aesthetics.sh clip reconstructed
sbatch run_evaluate_aesthetics.sh comp reconstructed
sbatch run_evaluate_aesthetics.sh dark reconstructed
sbatch run_evaluate_aesthetics.sh mic reconstructed
sbatch run_evaluate_aesthetics.sh mix reconstructed
sbatch run_evaluate_aesthetics.sh mud reconstructed
sbatch run_evaluate_aesthetics.sh punch reconstructed
sbatch run_evaluate_aesthetics.sh real reconstructed
sbatch run_evaluate_aesthetics.sh small reconstructed
sbatch run_evaluate_aesthetics.sh stereo reconstructed
sbatch run_evaluate_aesthetics.sh vocal reconstructed
sbatch run_evaluate_aesthetics.sh volume reconstructed
sbatch run_evaluate_aesthetics.sh warm reconstructed
sbatch run_evaluate_aesthetics.sh xband reconstructed
```
