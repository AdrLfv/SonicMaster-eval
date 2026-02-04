"""CLAP Embedding Extraction and Similarity Evaluation

This script extracts CLAP (Contrastive Language-Audio Pretraining) embeddings from audio files
and computes similarity metrics between clean, degraded, and restored audio.

CLAP is a neural network model that creates semantic embeddings for audio, similar to how
CLIP works for images. Higher cosine similarity between embeddings means the audio sounds
more perceptually similar.

Usage:
    python extract_clap_gt.py --jsonref path/to/degradation_pairs.jsonl --audio_key restored_path

Outputs:
    - NPZ file with clean, degraded, and output embeddings
    - CSV file with similarity metrics showing improvement from degraded to restored
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa
import h5py
import tempfile
from laion_clap import CLAP_Module

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_audio_as_numpy

def load_mono_audio(path, target_sr=44100):
    """Load audio file and convert to mono at target sample rate.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 44100 Hz)
    
    Returns:
        Mono audio array at target sample rate
    """
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

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
    
    CLAP embeddings are high-dimensional vectors (typically 512-d) that capture
    the semantic content of audio. Similar-sounding audio will have embeddings
    that are close together in this vector space.
    
    Args:
        model: Loaded CLAP model
        entries: List of entries from JSONL file
        audio_key: Key to use for output audio path (default: 'restored_path')
    
    Returns:
        Tuple of (clean_embeddings, degraded_embeddings, output_embeddings, file_ids)
        Each embedding array has shape (num_samples, embedding_dim)
    """
    clean_embeddings = []
    degraded_embeddings = []
    output_embeddings = []
    file_ids = []
    
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
            print(f"Warning: file {clean_file} not found, skipping.")
            continue
        if not os.path.exists(degraded_file):
            print(f"Warning: file {degraded_file} not found, skipping.")
            continue
        if not os.path.exists(output_file):
            print(f"Warning: file {output_file} not found, skipping.")
            continue
        
        try:
            # Load audio from HDF5 or regular files
            clean_audio, clean_sr = load_audio_as_numpy(clean_path, target_sr=48000)
            degraded_audio, degraded_sr = load_audio_as_numpy(degraded_path, target_sr=48000)
            output_audio, output_sr = load_audio_as_numpy(output_path, target_sr=48000)
            
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
                
                # Clean up temporary files
                os.unlink(clean_tmp.name)
                os.unlink(degraded_tmp.name)
                os.unlink(output_tmp.name)
            
            clean_embeddings.append(clean_emb[0])
            degraded_embeddings.append(degraded_emb[0])
            output_embeddings.append(output_emb[0])
            file_ids.append(os.path.splitext(os.path.basename(clean_path.split('::')[0]))[0])
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    
    return (np.stack(clean_embeddings), np.stack(degraded_embeddings), 
            np.stack(output_embeddings), file_ids)


def save_embeddings(output_path, clean_embs, degraded_embs, output_embs, filenames):
    """Save all embeddings to a compressed NPZ file.
    
    Args:
        output_path: Path to save NPZ file
        clean_embs: Clean audio embeddings
        degraded_embs: Degraded audio embeddings
        output_embs: Model output embeddings
        filenames: List of file identifiers
    """
    np.savez(output_path, 
             clean_embeddings=clean_embs,
             degraded_embeddings=degraded_embs,
             output_embeddings=output_embs,
             filenames=filenames)

def compute_clap_similarity(clean_embs, degraded_embs, output_embs):
    """Compute cosine similarity between clean and degraded/output embeddings.
    
    Cosine similarity ranges from -1 to 1, where:
        - 1 means identical semantic content
        - 0 means orthogonal (unrelated)
        - -1 means opposite
    
    Higher similarity to clean audio means better perceptual quality.
    If clean_output_sim > clean_degraded_sim, the model improved the audio.
    
    Args:
        clean_embs: Clean audio embeddings
        degraded_embs: Degraded audio embeddings
        output_embs: Model output embeddings
    
    Returns:
        Tuple of (clean_degraded_similarity, clean_output_similarity)
    """
    clean_degraded_sim = np.mean([np.dot(c, d) / (np.linalg.norm(c) * np.linalg.norm(d)) 
                                   for c, d in zip(clean_embs, degraded_embs)])
    clean_output_sim = np.mean([np.dot(c, o) / (np.linalg.norm(c) * np.linalg.norm(o)) 
                                 for c, o in zip(clean_embs, output_embs)])
    return clean_degraded_sim, clean_output_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract CLAP embeddings and compute similarity')
    parser.add_argument('--jsonref', type=str, required=True,
                        help='Path to JSONL file with clean/degraded/restored paths')
    parser.add_argument('--output_npz', type=str, default=None,
                        help='Path to output NPZ file for embeddings (defaults to jsonref parent dir)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to output CSV file for metrics (defaults to jsonref parent dir)')
    parser.add_argument('--audio_key', type=str, default='restored_path',
                        help='JSON key for output audio path (default: restored_path, can use reconstructed_path)')
    args = parser.parse_args()
    
    jsonref = args.jsonref
    
    if args.output_npz:
        output_npz = args.output_npz
    else:
        json_parent = os.path.dirname(os.path.abspath(jsonref))
        folder_name = os.path.basename(json_parent) or "clap_results"
        output_npz = os.path.join(json_parent, f"clap_embeddings_{folder_name}.npz")
    
    if args.output_csv:
        output_csv = args.output_csv
    else:
        json_parent = os.path.dirname(os.path.abspath(jsonref))
        folder_name = os.path.basename(json_parent) or "clap_results"
        output_csv = os.path.join(json_parent, f"clap_metrics_{folder_name}.csv")

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\n📂 Reading JSONL: {jsonref}")
    print(f"🔑 Audio key: {args.audio_key}")
    print("\n🎧 Loading CLAP model...")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()

    print("\n📑 Reading entries from JSONL...")
    entries = read_entries_from_jsonl(jsonref)
    print(f"Found {len(entries)} entries.")

    print("\n🔊 Extracting embeddings...")
    clean_embs, degraded_embs, output_embs, filenames = extract_embeddings_from_jsonl(
        model, entries, audio_key=args.audio_key)
    
    print(f"\n💾 Saving embeddings to {output_npz}")
    save_embeddings(output_npz, clean_embs, degraded_embs, output_embs, filenames)
    print(f"Saved {len(filenames)} embeddings.")
    
    print("\n📊 Computing CLAP similarity metrics...")
    clean_degraded_sim, clean_output_sim = compute_clap_similarity(clean_embs, degraded_embs, output_embs)
    
    folder_name = os.path.basename(os.path.dirname(jsonref)) or "clap_results"
    results = [{
        "folder": folder_name,
        "num_samples": len(filenames),
        "clean_degraded_similarity": clean_degraded_sim,
        "clean_output_similarity": clean_output_sim,
        "improvement": clean_output_sim - clean_degraded_sim
    }]
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ CLAP metrics saved to: {output_csv}")
    print(f"\n📈 Results:")
    print(f"  Clean-Degraded Similarity: {clean_degraded_sim:.4f}")
    print(f"  Clean-Output Similarity: {clean_output_sim:.4f}")
    print(f"  Improvement: {clean_output_sim - clean_degraded_sim:.4f}")
