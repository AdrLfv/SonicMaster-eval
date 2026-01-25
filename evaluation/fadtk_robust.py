#!/usr/bin/env python3
"""
Robust FAD computation using fadtk with error handling for problematic audio files.
This script wraps fadtk functionality but adds try-except blocks to skip corrupted files.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from fadtk import FrechetAudioDistance
from fadtk.model_loader import CLAPLaionModel, VGGishModel
from fadtk.fad import calc_embd_statistics, calc_frechet_distance
from tqdm import tqdm

def get_audio_files(directory, recursive=False):
    """Get all audio files from a directory, excluding non-audio files.
    
    Args:
        directory: Path to directory
        recursive: If True, search subdirectories recursively (default: False)
    """
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}
    audio_files = []
    
    # Get all files in directory
    glob_pattern = '**/*' if recursive else '*'
    for file_path in Path(directory).glob(glob_pattern):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    return sorted(audio_files)

def compute_embeddings_robust(fad, audio_files, desc="Processing"):
    """Compute embeddings with error handling for problematic files."""
    embeddings = []
    failed_files = []
    
    for audio_file in tqdm(audio_files, desc=desc):
        try:
            # Load audio and get embedding
            wav_data = fad.load_audio(str(audio_file))
            embd = fad.ml.get_embedding(wav_data)
            embeddings.append(embd)
        except Exception as e:
            print(f"\n⚠️  Skipping {audio_file.name}: {str(e)}")
            failed_files.append(str(audio_file))
            continue
    
    if not embeddings:
        raise ValueError("No valid audio files could be processed!")
    
    # Stack embeddings
    embeddings = np.vstack(embeddings)
    
    if failed_files:
        print(f"\n⚠️  Failed to process {len(failed_files)} files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"   - {Path(f).name}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")
    
    return embeddings, failed_files

def compute_fad_robust(baseline_dir, eval_dir, model='clap-laion-music', recursive=False):
    """Compute FAD with robust error handling.
    
    Args:
        baseline_dir: Path to baseline audio directory or "fma_pop"
        eval_dir: Path to evaluation audio directory
        model: Model to use (default: clap-laion-music)
        recursive: Search subdirectories recursively (default: False)
    """
    print(f"🎵 Loading FAD model: {model}")
    
    # Instantiate the appropriate model loader
    if model == 'clap-laion-music':
        ml = CLAPLaionModel(type='music')
    elif model == 'clap-laion-audio':
        ml = CLAPLaionModel(type='audio')
    elif model == 'vggish':
        ml = VGGishModel(use_pca=False, use_activation=False)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    ml.load_model()
    fad = FrechetAudioDistance(ml=ml, audio_load_worker=8, load_model=False)
    
    # Handle fma_pop baseline (special keyword for fadtk built-in dataset)
    if baseline_dir.lower() == 'fma_pop':
        print(f"\n📂 Baseline: fma_pop (using fadtk built-in FMA Pop statistics)")
        # Load pre-computed fma_pop statistics from fadtk
        mu_baseline, cov_baseline = fad.load_stats('fma_pop')
        baseline_embeds = (mu_baseline, cov_baseline)  # Store as tuple
        baseline_failed = []
        print(f"   Loaded pre-computed statistics")
    else:
        print(f"\n📂 Baseline directory: {baseline_dir}")
        baseline_files = get_audio_files(baseline_dir)
        print(f"   Found {len(baseline_files)} audio files")
        
        if not baseline_files:
            raise ValueError(f"No audio files found in baseline directory: {baseline_dir}")
        
        # Compute embeddings with error handling
        print("\n🚀 Computing baseline embeddings...")
        baseline_embeds, baseline_failed = compute_embeddings_robust(fad, baseline_files, "Baseline")
    
    print(f"\n📂 Evaluation directory: {eval_dir}")
    eval_files = get_audio_files(eval_dir)
    print(f"   Found {len(eval_files)} audio files")
    
    if not eval_files:
        raise ValueError(f"No audio files found in evaluation directory: {eval_dir}")
    
    # Compute evaluation embeddings with error handling
    print("\n🚀 Computing evaluation embeddings...")
    eval_embeds, eval_failed = compute_embeddings_robust(fad, eval_files, "Evaluation")
    
    # Compute FAD score
    print("\n📊 Computing FAD score...")
    
    # Handle baseline: either pre-computed stats (tuple) or embeddings (array)
    if isinstance(baseline_embeds, tuple):
        mu_baseline, cov_baseline = baseline_embeds
    else:
        mu_baseline, cov_baseline = calc_embd_statistics(baseline_embeds)
    
    # Compute eval statistics
    mu_eval, cov_eval = calc_embd_statistics(eval_embeds)
    
    # Calculate FAD
    fad_score = calc_frechet_distance(mu_baseline, cov_baseline, mu_eval, cov_eval)
    
    print(f"\n✅ FAD Score: {fad_score:.4f}")
    if isinstance(baseline_embeds, tuple):
        print(f"   Baseline: fma_pop (pre-computed statistics)")
    else:
        print(f"   Baseline: {len(baseline_embeds)} samples ({len(baseline_failed)} failed)")
    print(f"   Evaluation: {len(eval_embeds)} files processed ({len(eval_failed)} failed)")
    
    return fad_score, baseline_failed, eval_failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robust FAD computation with error handling')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline audio directory (or "fma_pop")')
    parser.add_argument('--eval', type=str, required=True, help='Evaluation audio directory')
    parser.add_argument('--model', type=str, default='clap-laion-music', 
                        help='Model to use (default: clap-laion-music)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search subdirectories recursively (default: False)')
    args = parser.parse_args()
    
    try:
        fad_score, baseline_failed, eval_failed = compute_fad_robust(
            args.baseline, args.eval, args.model, recursive=args.recursive
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
