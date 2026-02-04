"""Audiobox Aesthetics Score Extraction

This script computes production quality metrics using Meta's Audiobox Aesthetics toolbox.
It provides four key metrics for audio mastering quality:

1. **CE (Clarity/Cleanness)**: How clear and artifact-free the audio is
2. **CU (Coherence/Unity)**: How well-balanced and cohesive the mix is
3. **PC (Punch/Compression)**: Dynamic range and impact
4. **PQ (Production Quality)**: Overall professional production quality

These metrics are trained on professionally mastered music and correlate with
human perception of production quality.

Usage:
    python extract_aesthetics_scores.py --jsonref path/to/degradation_pairs.jsonl --audio_key restored_path

Outputs:
    - CSV file with CE, CU, PC, PQ scores for clean, degraded, and output audio
"""

import os
import sys
import json
import argparse
import pandas as pd
import h5py
import soundfile as sf
import tempfile
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_audio_as_numpy

try:
    from audiobox_aesthetics.infer import initialize_predictor
except ImportError:
    print("⚠️  audiobox_aesthetics not installed. Install with: pip install audiobox-aesthetics")
    exit(1)

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

def evaluate_aesthetics_from_jsonl(jsonl_path, audio_key='restored_path', predictor=None):
    """Evaluate audio aesthetics using Audiobox Aesthetics predictor.
    
    Computes CE, CU, PC, and PQ scores for clean, degraded, and output audio.
    Higher scores indicate better production quality.
    
    Args:
        jsonl_path: Path to JSONL file with audio paths
        audio_key: Key for output audio path (default: 'restored_path')
        predictor: Audiobox Aesthetics predictor instance
    
    Returns:
        Dictionary with mean scores for clean, degraded, and output audio
    """
    if predictor is None:
        predictor = initialize_predictor()
    
    clean_scores = []
    degraded_scores = []
    output_scores = []
    
    entries = read_entries_from_jsonl(jsonl_path)
    
    for entry in tqdm(entries, desc="Processing audio files"):
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
            print(f"Skipping missing file: {clean_file}")
            continue
        if not os.path.exists(degraded_file):
            print(f"Skipping missing file: {degraded_file}")
            continue
        if not os.path.exists(output_file):
            print(f"Skipping missing file: {output_file}")
            continue
        
        try:
            # Load audio from HDF5 or regular files
            clean_audio, _ = load_audio_as_numpy(clean_path)
            degraded_audio, _ = load_audio_as_numpy(degraded_path)
            output_audio, _ = load_audio_as_numpy(output_path)
            
            # Create temporary WAV files for predictor
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as clean_tmp, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as degraded_tmp, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_tmp:
                
                sf.write(clean_tmp.name, clean_audio, 44100)
                sf.write(degraded_tmp.name, degraded_audio, 44100)
                sf.write(output_tmp.name, output_audio, 44100)
                
                clean_result = predictor.forward([{"path": clean_tmp.name}])[0]
                degraded_result = predictor.forward([{"path": degraded_tmp.name}])[0]
                output_result = predictor.forward([{"path": output_tmp.name}])[0]
                
                os.unlink(clean_tmp.name)
                os.unlink(degraded_tmp.name)
                os.unlink(output_tmp.name)
            
            clean_scores.append(clean_result)
            degraded_scores.append(degraded_result)
            output_scores.append(output_result)
        except Exception as e:
            print(f"⚠️ Error processing entry: {e}")
            continue
    
    return {
        "clean": pd.DataFrame(clean_scores).mean().to_dict() if clean_scores else {},
        "degraded": pd.DataFrame(degraded_scores).mean().to_dict() if degraded_scores else {},
        "output": pd.DataFrame(output_scores).mean().to_dict() if output_scores else {},
        "num_samples": len(clean_scores)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Audiobox Aesthetics scores (CE, CU, PC, PQ)')
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
        folder_name = os.path.basename(json_parent) or "aesthetics_results"
        output_csv = os.path.join(json_parent, f"aesthetics_metrics_{folder_name}.csv")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\n📂 Reading JSONL: {jsonref}")
    print(f"🔑 Audio key: {args.audio_key}")
    print("\n🎨 Loading Audiobox Aesthetics predictor...")
    predictor = initialize_predictor()
    
    print("\n🚀 Processing audio files...")
    results = evaluate_aesthetics_from_jsonl(jsonref, audio_key=args.audio_key, predictor=predictor)
    
    if results["num_samples"] == 0:
        print("❌ No valid samples processed")
        exit(1)
    
    folder_name = os.path.basename(os.path.dirname(jsonref)) or "aesthetics_results"
    
    # Create summary rows
    summary = []
    for audio_type in ["clean", "degraded", "output"]:
        if results[audio_type]:
            row = {"folder": folder_name, "audio_type": audio_type}
            row.update(results[audio_type])
            summary.append(row)
    
    df = pd.DataFrame(summary)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Aesthetics scores saved to: {output_csv}")
    print(f"\n📊 Results ({results['num_samples']} samples):")
    print(df.to_string(index=False))
    
    # Print improvement metrics
    if results["clean"] and results["degraded"] and results["output"]:
        print("\n📈 Quality Improvements:")
        for metric in ["CE", "CU", "PC", "PQ"]:
            if metric in results["clean"]:
                clean_val = results["clean"][metric]
                degraded_val = results["degraded"][metric]
                output_val = results["output"][metric]
                improvement = output_val - degraded_val
                recovery = (output_val - degraded_val) / (clean_val - degraded_val + 1e-10) * 100
                print(f"  {metric}: {degraded_val:.3f} → {output_val:.3f} (improvement: {improvement:+.3f}, recovery: {recovery:.1f}%)")
