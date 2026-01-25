"""KL Divergence and SSIM Computation for Audio Quality Assessment

This script computes two complementary metrics for audio restoration quality:

1. **KL Divergence (Kullback-Leibler)**: Measures spectral distribution differences
   - Compares mel-spectrogram frequency distributions
   - Lower KL = more similar spectral content
   - Good for detecting timbral/frequency balance changes

2. **SSIM (Structural Similarity Index)**: Measures spectrogram structural similarity
   - Compares mel-spectrogram patterns (like image comparison)
   - Higher SSIM (0 to 1) = more similar structure
   - Good for detecting temporal/structural preservation

Usage:
    python extract_kl_ssim_mass.py --jsonref path/to/degradation_pairs.jsonl --audio_key restored_path

Outputs:
    - CSV file with KL and SSIM metrics
    - Optional: Raw numpy arrays for detailed analysis
"""

import os
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def get_log_mel(path, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Extract log-mel spectrogram from audio file.
    
    A mel-spectrogram is a time-frequency representation of audio that mimics
    human perception of frequency (mel scale). It's like a visual representation
    of how the audio sounds over time.
    
    Args:
        path: Path to audio file
        sr: Sample rate (default: 44100 Hz)
        n_fft: FFT window size (default: 2048)
        hop_length: Number of samples between frames (default: 512)
        n_mels: Number of mel frequency bins (default: 128)
    
    Returns:
        Log-mel spectrogram in dB scale (shape: n_mels x time_frames)
    """
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def compute_kl(mel1, mel2):
    """Compute KL divergence between two mel-spectrograms.
    
    KL divergence measures how one probability distribution (mel2) differs from
    a reference distribution (mel1). We average over time to get frequency distributions.
    
    Steps:
        1. Average mel-spectrograms over time to get frequency profiles
        2. Normalize to probability distributions (sum to 1)
        3. Compute KL divergence: sum(p * log(p/q))
    
    Args:
        mel1: Reference mel-spectrogram (clean audio)
        mel2: Comparison mel-spectrogram (degraded/output audio)
    
    Returns:
        KL divergence (scalar). Lower is better.
            - KL = 0: Identical distributions
            - KL < 0.1: Very similar
            - KL > 1.0: Very different
    """
    p = np.mean(mel1, axis=1)
    q = np.mean(mel2, axis=1)
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
    return kl

def compute_ssim(mel1, mel2):
    """Compute SSIM (Structural Similarity Index) between two mel-spectrograms.
    
    SSIM treats mel-spectrograms as images and measures structural similarity.
    It considers luminance, contrast, and structure, making it good for detecting
    temporal patterns and overall structure preservation.
    
    Steps:
        1. Crop to same size (in case of length differences)
        2. Normalize to [0, 1] range
        3. Compute SSIM using sliding window comparison
    
    Args:
        mel1: Reference mel-spectrogram (clean audio)
        mel2: Comparison mel-spectrogram (degraded/output audio)
    
    Returns:
        SSIM score (0 to 1). Higher is better.
            - SSIM = 1: Identical structure
            - SSIM > 0.9: Very similar
            - SSIM < 0.5: Very different
    """
    min_shape = (min(mel1.shape[0], mel2.shape[0]), min(mel1.shape[1], mel2.shape[1]))
    mel1 = mel1[:min_shape[0], :min_shape[1]]
    mel2 = mel2[:min_shape[0], :min_shape[1]]
    mel1 = (mel1 - mel1.min()) / (mel1.max() - mel1.min() + 1e-8)
    mel2 = (mel2 - mel2.min()) / (mel2.max() - mel2.min() + 1e-8)
    return ssim(mel1, mel2, data_range=1.0)

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
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)
    return entries

def evaluate_from_jsonl(jsonl_path, audio_key='restored_path', out_dir=None):
    """Evaluate audio restoration using KL divergence and SSIM metrics.
    
    For each audio sample, computes:
        - KL and SSIM between clean and degraded (baseline quality loss)
        - KL and SSIM between clean and output (restored quality)
    
    Better restoration means:
        - Lower KL for output than degraded
        - Higher SSIM for output than degraded
    
    Args:
        jsonl_path: Path to JSONL file with audio paths
        audio_key: Key for output audio path (default: 'restored_path')
        out_dir: Optional directory to save raw numpy arrays
    
    Returns:
        Tuple of (avg_kl_degraded, avg_ssim_degraded, avg_kl_output, avg_ssim_output)
    """
    kl_scores_degraded = []
    ssim_scores_degraded = []
    kl_scores_output = []
    ssim_scores_output = []

    entries = read_entries_from_jsonl(jsonl_path)

    for entry in tqdm(entries, desc="Processing entries"):
        clean_path = entry["clean_path"]
        degraded_path = entry["degraded_path"]
        
        if audio_key in entry:
            output_path = entry[audio_key]
        else:
            print(f"Warning: {audio_key} not found in entry, skipping.")
            continue

        if not os.path.exists(clean_path):
            print(f"Skipping missing file: {clean_path}")
            continue
        if not os.path.exists(degraded_path):
            print(f"Skipping missing file: {degraded_path}")
            continue
        if not os.path.exists(output_path):
            print(f"Skipping missing file: {output_path}")
            continue

        try:
            mel_clean = get_log_mel(clean_path)
            mel_degraded = get_log_mel(degraded_path)
            mel_output = get_log_mel(output_path)

            kl_degraded = compute_kl(mel_clean, mel_degraded)
            ssim_degraded = compute_ssim(mel_clean, mel_degraded)
            
            kl_output = compute_kl(mel_clean, mel_output)
            ssim_output = compute_ssim(mel_clean, mel_output)

            kl_scores_degraded.append(kl_degraded)
            ssim_scores_degraded.append(ssim_degraded)
            kl_scores_output.append(kl_output)
            ssim_scores_output.append(ssim_output)

        except Exception as e:
            print(f"Error processing entry: {e}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        folder_name = os.path.basename(os.path.dirname(jsonl_path))
        np.save(os.path.join(out_dir, f"{folder_name}_kl_degraded.npy"), np.array(kl_scores_degraded))
        np.save(os.path.join(out_dir, f"{folder_name}_ssim_degraded.npy"), np.array(ssim_scores_degraded))
        np.save(os.path.join(out_dir, f"{folder_name}_kl_output.npy"), np.array(kl_scores_output))
        np.save(os.path.join(out_dir, f"{folder_name}_ssim_output.npy"), np.array(ssim_scores_output))

    return (np.mean(kl_scores_degraded), np.mean(ssim_scores_degraded),
            np.mean(kl_scores_output), np.mean(ssim_scores_output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute KL divergence and SSIM metrics')
    parser.add_argument('--jsonref', type=str, required=True,
                        help='Path to JSONL file with clean/degraded/restored paths')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to output CSV file for metrics (defaults to jsonref parent dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save raw numpy arrays (optional)')
    parser.add_argument('--audio_key', type=str, default='restored_path',
                        help='JSON key for output audio path (default: restored_path, can use reconstructed_path)')
    args = parser.parse_args()
    
    jsonref = args.jsonref
    
    if args.output_csv:
        output_csv = args.output_csv
    else:
        json_parent = os.path.dirname(os.path.abspath(jsonref))
        folder_name = os.path.basename(json_parent) or "kl_ssim_results"
        output_csv = os.path.join(json_parent, f"kl_ssim_metrics_{folder_name}.csv")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\n📂 Reading JSONL: {jsonref}")
    print(f"🔑 Audio key: {args.audio_key}")
    
    print(f"\n🚀 Processing entries...")
    kl_degraded, ssim_degraded, kl_output, ssim_output = evaluate_from_jsonl(
        jsonref, audio_key=args.audio_key, out_dir=args.output_dir)
    
    folder_name = os.path.basename(os.path.dirname(jsonref)) or "kl_ssim_results"
    summary = [{
        "folder": folder_name,
        "avg_kl_degraded": kl_degraded,
        "avg_ssim_degraded": ssim_degraded,
        "avg_kl_output": kl_output,
        "avg_ssim_output": ssim_output,
        "kl_improvement": kl_degraded - kl_output,
        "ssim_improvement": ssim_output - ssim_degraded
    }]

    df = pd.DataFrame(summary)
    df.to_csv(output_csv, index=False)

    print("\n✅ KL/SSIM results saved:")
    print(df)
    print(f"\nSaved to: {output_csv}")
