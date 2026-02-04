"""FAD (Fréchet Audio Distance) Computation

This script computes FAD scores to measure the quality of audio restoration.

FAD is similar to FID (Fréchet Inception Distance) used in image generation.
It measures the distance between two distributions of audio embeddings:
    - Lower FAD = more similar distributions = better quality
    - FAD of 0 = identical distributions (perfect restoration)

FAD is computed using CLAP embeddings and measures:
    1. Mean difference between embedding distributions
    2. Covariance difference between distributions

Usage:
    python extract_fad_mass.py --jsonref path/to/degradation_pairs.jsonl --audio_key restored_path

Outputs:
    - CSV file with FAD scores comparing clean vs degraded and clean vs output
    - Lower FAD for output means better restoration quality
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import soundfile as sf
import tempfile
from laion_clap import CLAP_Module
from scipy.linalg import sqrtm

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_audio_as_numpy

def read_entries_from_jsonl(jsonl_path):
    """Read all entries from a JSONL file.
    
    Each line in the JSONL should contain:
        - clean_path: Path to original clean audio
        - degraded_path: Path to degraded audio
        - restored_path or reconstructed_path: Path to model output
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        List of dictionary entries
    """
    # Fallback to rank0 file if combined file doesn't exist
    if not os.path.exists(jsonl_path):
        rank0_path = jsonl_path.replace('evaluation_metadata.jsonl', 'evaluation_metadata_rank0.jsonl')
        if os.path.exists(rank0_path):
            print(f"Warning: {jsonl_path} not found, using {rank0_path}")
            jsonl_path = rank0_path
        else:
            raise FileNotFoundError(f"Neither {jsonl_path} nor {rank0_path} found")
    
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)
    return entries



def extract_embeddings_from_jsonl(model, entries, audio_key='restored_path'):
    """Extract CLAP embeddings for clean, degraded, and output audio files.
    
    These embeddings will be used to compute FAD scores. We need embeddings from
    all three types of audio to compare:
        - Clean vs Degraded (how much quality was lost)
        - Clean vs Output (how much quality was recovered)
    
    Args:
        model: Loaded CLAP model
        entries: List of entries from JSONL file
        audio_key: Key to use for output audio path (default: 'restored_path')
    
    Returns:
        Tuple of (clean_embeddings, degraded_embeddings, output_embeddings)
        Each is a numpy array of shape (num_samples, embedding_dim)
    """
    clean_embeddings = []
    degraded_embeddings = []
    output_embeddings = []
    
    for entry in tqdm(entries, desc="Extracting embeddings"):
        clean_path = entry.get("clean_path") or entry.get("clean_audio_path")
        degraded_path = entry.get("degraded_path") or entry.get("degraded_audio_path")
        
        if audio_key in entry:
            output_path = entry[audio_key]
        else:
            print(f"Warning: {audio_key} not found in entry, skipping.")
            continue
        
        # Handle HDF5 dataset paths (format: /path/file.h5::/dataset)
        clean_file = clean_path.split('::')[0] if '::' in clean_path else clean_path
        degraded_file = degraded_path.split('::')[0] if '::' in degraded_path else degraded_path
        output_file = output_path.split('::')[0] if '::' in output_path else output_path
        
        if not os.path.exists(clean_file):
            print(f"⚠️  Missing clean file: {clean_file}")
            continue
        if not os.path.exists(degraded_file):
            print(f"⚠️  Missing degraded file: {degraded_file}")
            continue
        if not os.path.exists(output_file):
            print(f"⚠️  Missing output file: {output_file}")
            continue
        
        try:
            # Load audio from HDF5 or regular files
            clean_audio, _ = load_audio_as_numpy(clean_path, target_sr=48000)
            degraded_audio, _ = load_audio_as_numpy(degraded_path, target_sr=48000)
            output_audio, _ = load_audio_as_numpy(output_path, target_sr=48000)
            
            # Create temporary WAV files for CLAP
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as clean_tmp, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as degraded_tmp, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_tmp:
                
                sf.write(clean_tmp.name, clean_audio, 48000)
                sf.write(degraded_tmp.name, degraded_audio, 48000)
                sf.write(output_tmp.name, output_audio, 48000)
                
                clean_emb = model.get_audio_embedding_from_filelist([clean_tmp.name], use_tensor=False)
                degraded_emb = model.get_audio_embedding_from_filelist([degraded_tmp.name], use_tensor=False)
                output_emb = model.get_audio_embedding_from_filelist([output_tmp.name], use_tensor=False)
                
                os.unlink(clean_tmp.name)
                os.unlink(degraded_tmp.name)
                os.unlink(output_tmp.name)
            
            if len(clean_emb) == 0 or len(degraded_emb) == 0 or len(output_emb) == 0:
                print(f"❌ Empty embedding for entry")
                continue
            
            clean_embeddings.append(clean_emb[0])
            degraded_embeddings.append(degraded_emb[0])
            output_embeddings.append(output_emb[0])
        except Exception as e:
            print(f"❌ Failed on entry: {e}")
            continue
    
    return np.stack(clean_embeddings), np.stack(degraded_embeddings), np.stack(output_embeddings)

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two Gaussian distributions.
    
    The Fréchet distance measures how different two multivariate Gaussian
    distributions are. It considers both:
        1. Distance between means (mu1 - mu2)
        2. Difference in covariances (sigma1 vs sigma2)
    
    Formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
    
    Returns:
        Fréchet distance (scalar). Lower is better (more similar distributions).
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_fad(clean_embs, degraded_embs):
    """Compute FAD (Fréchet Audio Distance) between two sets of embeddings.
    
    Steps:
        1. Compute mean and covariance of each embedding set
        2. Calculate Fréchet distance between the two distributions
    
    Lower FAD means the two sets of audio are more similar in their
    perceptual/semantic characteristics.
    
    Args:
        clean_embs: Embeddings from clean audio (shape: num_samples x embedding_dim)
        degraded_embs: Embeddings from degraded/output audio (shape: num_samples x embedding_dim)
    
    Returns:
        FAD score (scalar). Lower is better.
            - FAD = 0: Perfect match
            - FAD < 5: Excellent quality
            - FAD < 10: Good quality
            - FAD > 20: Poor quality
    """
    mu1 = np.mean(clean_embs, axis=0)
    mu2 = np.mean(degraded_embs, axis=0)
    sigma1 = np.cov(clean_embs, rowvar=False)
    sigma2 = np.cov(degraded_embs, rowvar=False)
    fad = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute FAD (Frechet Audio Distance) using CLAP embeddings')
    parser.add_argument('--jsonref', type=str, required=True,
                        help='Path to JSONL file with clean/degraded/restored paths')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to output CSV file for metrics (defaults to jsonref parent dir)')
    parser.add_argument('--audio_key', type=str, default='restored_path',
                        help='JSON key for output audio path (default: restored_path, can use reconstructed_path)')
    args = parser.parse_args()
    
    jsonref = args.jsonref
    
    if args.output_csv:
        output_csv = args.output_csv
    else:
        json_parent = os.path.dirname(os.path.abspath(jsonref))
        folder_name = os.path.basename(json_parent) or "fad_results"
        output_csv = os.path.join(json_parent, f"fad_metrics_{folder_name}.csv")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\n📂 Reading JSONL: {jsonref}")
    print(f"🔑 Audio key: {args.audio_key}")
    
    print("\n📑 Loading entries from JSONL...")
    entries = read_entries_from_jsonl(jsonref)
    print(f"Found {len(entries)} entries.")

    print("\n🎧 Loading CLAP model...")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()

    print(f"\n🚀 Extracting embeddings...")
    clean_embs, degraded_embs, output_embs = extract_embeddings_from_jsonl(
        model, entries, audio_key=args.audio_key)

    if len(clean_embs) == 0 or len(degraded_embs) == 0 or len(output_embs) == 0:
        print(f"❌ No valid embeddings found")
        exit(1)

    print(f"\n📊 Computing FAD for {len(clean_embs)} samples...")
    fad_clean_degraded = compute_fad(clean_embs, degraded_embs)
    fad_clean_output = compute_fad(clean_embs, output_embs)

    folder_name = os.path.basename(os.path.dirname(jsonref)) or "fad_results"
    results = [{
        "folder": folder_name,
        "num_samples": len(clean_embs),
        "fad_clean_degraded": fad_clean_degraded,
        "fad_clean_output": fad_clean_output,
        "fad_improvement": fad_clean_degraded - fad_clean_output
    }]

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("\n✅ FAD results saved:")
    print(df)
    print(f"\nSaved to: {output_csv}")
