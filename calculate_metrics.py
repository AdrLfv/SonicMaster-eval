import os
import re
import json
import sys
import soundfile as sf
import torch
import torchaudio
import argparse
import numpy as np
import warnings
from tqdm import tqdm
import subprocess
from scipy.linalg import sqrtm
from scipy import signal
import librosa
import pandas as pd

warnings.filterwarnings("ignore")

# Import metrics from centralized metrics module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    multi_mel_snr,
    log_spectral_distance,
    RobustMelDistance,
    SISDRMetric,
)


def get_log_mel_from_array(audio_array, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute log mel spectrogram from audio array (SonicMaster-compatible).
    
    Args:
        audio_array: Audio signal as numpy array
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
    
    Returns:
        Log mel spectrogram
    """
    mel = librosa.feature.melspectrogram(
        y=audio_array, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

try:
    sys.path.insert(0, '/work/vita/alefevre/programs/zimtohrli/cpp/zimt')
    import pyohrli
    ZIMTOHRLI_AVAILABLE = True
except ImportError:
    ZIMTOHRLI_AVAILABLE = False
    print("Warning: Zimtohrli Python wrapper not available. Zimtohrli metrics will be skipped.")
    print("If you consider using Zimtohrli metrics, please clone and set up their repo: https://github.com/google/zimtohrli")


def load_audio(file_path):
    try:
        wav, samplerate = sf.read(file_path)
        if wav.ndim > 1:
            wav = wav.T
        else:
            wav = wav[np.newaxis, :]
        return torch.from_numpy(wav).float(), samplerate
    except Exception:
        return None, None

def compute_zimtohrli_distance(clean_audio, test_audio, sr=48000, warn_resample=True):
    """Compute Zimtohrli distance and MOS using Python wrapper."""
    if not ZIMTOHRLI_AVAILABLE:
        return None, None
    
    try:
        import librosa
        
        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=0)
        if test_audio.ndim > 1:
            test_audio = test_audio.mean(axis=0)
        
        # Resample to 48kHz if needed (Zimtohrli requires 48kHz)
        if sr != 48000:
            if warn_resample:
                print(f"\nWarning: Resampling audio from {sr} Hz to 48000 Hz for Zimtohrli calculation.")
                print("This may introduce artifacts that could bias the metric.")
            clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=48000)
            test_audio = librosa.resample(test_audio, orig_sr=sr, target_sr=48000)
        
        min_len = min(len(clean_audio), len(test_audio))
        clean_audio = clean_audio[:min_len]
        test_audio = test_audio[:min_len]
        
        clean_audio = np.clip(clean_audio, -1.0, 1.0).astype(np.float32)
        test_audio = np.clip(test_audio, -1.0, 1.0).astype(np.float32)
        
        metric = pyohrli.Pyohrli()
        distance = metric.distance(clean_audio, test_audio)
        mos = pyohrli.mos_from_zimtohrli(distance)
        
        return distance, mos
    except Exception as e:
        print(f"Warning: Failed to compute Zimtohrli distance: {e}")
        return None, None

def run_zimtohrli_compare(compare_bin, clean_path, test_path):
    """Run the Zimtohrli compare CLI and return the MOS score as float."""
    cmd = [
        compare_bin,
        "-path_a", clean_path,
        "-path_b", test_path,
        "-zimtohrli", "true",
        "-visqol", "false",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Warning: Failed to run compare binary on {test_path}: {exc}")
        return None

    match = re.search(r"Zimtohrli=([0-9eE+\-\.]+)", result.stdout)
    if not match:
        print(f"Warning: Could not parse Zimtohrli output for {test_path}")
        return None
    try:
        return float(match.group(1))
    except ValueError:
        print(f"Warning: Invalid Zimtohrli numeric value for {test_path}: {match.group(1)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Calculate audio quality metrics for restoration/reconstruction results from JSONL file")
    parser.add_argument("jsonl_file", type=str, help="Path to JSONL file with 'clean_path' and either 'restored_path'")
    parser.add_argument("--metrics", type=str, nargs='+', 
                        default=['multi_mel_snr', 'zimtohrli', 'lsd'],
                        choices=['multi_mel_snr', 'zimtohrli', 'lsd', 'mel', 'robust_mel', 'sisdr', 'all'],
                        help="Metrics to compute. Options: multi_mel_snr, zimtohrli, lsd, mel, sisdr, all. Default: all metrics.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode for all computations (useful when GPU memory is full)")
    parser.add_argument("--zim_compare_bin", type=str, default=None,
                        help="Path to Zimtohrli compare binary. Default: $HOME/go/bin/compare or /work/vita/alefevre/programs/zimtohrli/build/cpp/zimt/compare")
    parser.add_argument("--compute_degraded", action="store_true",
                        help="Also compute metrics for degraded vs clean (baseline). Requires degraded_path in JSONL.")
    parser.add_argument("--audio_key", type=str, default="auto",
                        help="Key to use for audio path in JSONL. Options: 'auto' (tries restored_path, then reconstructed_path, then degraded_path), 'restored_path', 'reconstructed_path', 'degraded_path'. Default: auto")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save metrics as CSV/Excel file (.csv or .xlsx). If not provided, only JSONL output is saved.")
    args = parser.parse_args()
    
    # Force CPU mode if requested - do this FIRST to prevent any GPU allocation
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch
        torch.set_default_device('cpu')
        print("INFO: Forced CPU mode enabled. GPU will not be used.")
    
    # Handle 'all' metrics option
    if 'all' in args.metrics:
        args.metrics = ['multi_mel_snr', 'zimtohrli', 'lsd', 'mel', 'sisdr']
    
    # Support both 'robust_mel' and 'mel' as aliases
    args.metrics = ['mel' if m == 'robust_mel' else m for m in args.metrics]
    
    # Determine which metrics to compute
    compute_multi_mel_snr = 'multi_mel_snr' in args.metrics
    compute_zimtohrli = 'zimtohrli' in args.metrics
    compute_lsd = 'lsd' in args.metrics
    compute_mel = 'mel' in args.metrics
    compute_sisdr = 'sisdr' in args.metrics
    
    # Set default Zimtohrli compare binary path if not provided
    if compute_zimtohrli and args.zim_compare_bin is None:
        possible_paths = [
            os.path.expanduser("~/go/bin/compare"),
            "/work/vita/alefevre/programs/zimtohrli/build/cpp/zimt/compare",
            "/work/vita/alefevre/programs/zimtohrli/build/compare",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.zim_compare_bin = path
                print(f"Using Zimtohrli compare binary at: {path}")
                break
    
    all_target_paths = []
    all_output_paths = []
    restoration_records = []

    print("--- Loading restoration results from JSONL ---")
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                restoration_records.append(record)
            except Exception as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
    print(f"Loaded {len(restoration_records)} restoration records")
    
    metrics_to_compute = ', '.join(args.metrics)
    print(f"\n--- Computing metrics: {metrics_to_compute} ---")
    
    mel_snr_scores = []
    lsd_scores = []
    mel_scores = []
    sisdr_scores = []
    zim_restored_distances = []
    zim_restored_mos_scores = []
    zim_degraded_distances = []
    zim_degraded_mos_scores = []
    zim_resample_warned = False
    
    # Initialize metric modules if needed
    mel_metric = None
    sisdr_metric = None
    if compute_mel:
        mel_metric = RobustMelDistance(sample_rate=44100)
    if compute_sisdr:
        sisdr_metric = SISDRMetric()
    
    # Counters for debugging
    processed_count = 0
    skipped_no_paths = 0
    skipped_no_files = 0
    skipped_load_failed = 0
    
    for record in tqdm(restoration_records, desc="Computing metrics", ncols=100):
        try:
            # Check if record has required fields
            # Support both 'restored_path' (for restoration)
            if 'clean_path' not in record:
                skipped_no_paths += 1
                continue
            
            output_path = None
            use_degraded_as_output = False
            
            # Determine which audio path to use based on --audio_key flag
            if args.audio_key == 'auto':
                # Auto mode: try restored_path, then reconstructed_path, then degraded_path
                if 'restored_path' in record:
                    output_path = record['restored_path']
                elif 'reconstructed_path' in record:
                    output_path = record['reconstructed_path']
                elif 'degraded_path' in record:
                    output_path = record['degraded_path']
                    use_degraded_as_output = True
            else:
                # Use specified key
                if args.audio_key in record:
                    output_path = record[args.audio_key]
                    if args.audio_key == 'degraded_path':
                        use_degraded_as_output = True
                else:
                    if rank == 0 and skipped_no_paths == 0:
                        print(f"Warning: Key '{args.audio_key}' not found in records. Available keys: {list(record.keys())}")
            
            if output_path is None:
                skipped_no_paths += 1
                continue
            
            # Ensure metrics dict exists in record
            if 'metrics' not in record:
                record['metrics'] = {}
            
            target_path = record['clean_path']
            # For AAE computation, we need the original degraded_path (if different from output)
            degraded_path = record.get('degraded_path', None) if not use_degraded_as_output else None
            degradation_name = record.get('degradation_name', None)

            if not os.path.exists(target_path):
                skipped_no_files += 1
                continue
            if not os.path.exists(output_path):
                skipped_no_files += 1
                continue
            
            # Skip audio loading if only computing FAD/FD
            if not (compute_multi_mel_snr or compute_lsd or compute_mel or compute_sisdr or compute_zimtohrli):
                continue
            
            target_wav, target_sr = load_audio(target_path)
            output_wav, output_sr = load_audio(output_path)

            if target_wav is None or output_wav is None:
                skipped_load_failed += 1
                continue
            if target_sr != output_sr:
                skipped_load_failed += 1
                continue
            if target_wav.shape[0] != output_wav.shape[0]:
                skipped_load_failed += 1
                continue

            min_len = min(target_wav.shape[-1], output_wav.shape[-1])
            target_wav = target_wav[..., :min_len]
            output_wav = output_wav[..., :min_len]

            if target_wav.shape[-1] == 0:
                skipped_load_failed += 1
                continue
            
            # Successfully loaded audio
            processed_count += 1
            
            if compute_multi_mel_snr:
                mel_snrs = []
                for ch in range(target_wav.shape[0]):
                    mel_snr_val = multi_mel_snr(target_wav[ch], output_wav[ch], sr=target_sr)
                    mel_snrs.append(mel_snr_val)
                
                avg_mel_snr = sum(mel_snrs) / len(mel_snrs)
                record['metrics']['multi_mel_snr'] = round(avg_mel_snr, 4)
                mel_snr_scores.append(avg_mel_snr)
            
            if compute_lsd:
                lsd_vals = []
                target_np = target_wav.numpy()
                output_np = output_wav.numpy()
                for ch in range(target_wav.shape[0]):
                    lsd_val = log_spectral_distance(target_np[ch], output_np[ch], sr=target_sr)
                    lsd_vals.append(lsd_val)
                
                avg_lsd = sum(lsd_vals) / len(lsd_vals)
                record['metrics']['lsd'] = round(avg_lsd, 4)
                lsd_scores.append(avg_lsd)
            
            if compute_mel:
                with torch.no_grad():
                    mel_val = mel_metric(target_wav, output_wav)
                record['metrics']['mel'] = round(mel_val.item(), 4)
                mel_scores.append(mel_val.item())
            
            if compute_sisdr:
                with torch.no_grad():
                    sisdr_val = sisdr_metric(output_wav, target_wav)
                record['metrics']['sisdr'] = round(sisdr_val.item(), 4)
                sisdr_scores.append(sisdr_val.item())

            if compute_zimtohrli and ZIMTOHRLI_AVAILABLE:
                target_np = target_wav.numpy()
                output_np = output_wav.numpy()
                
                zim_distance, zim_mos = compute_zimtohrli_distance(target_np, output_np, sr=target_sr, warn_resample=not zim_resample_warned)
                if not zim_resample_warned and target_sr != 48000:
                    zim_resample_warned = True
                if zim_distance is not None:
                    record['metrics']['zimtohrli_distance'] = round(zim_distance, 6)
                    record['metrics']['zimtohrli_mos'] = round(zim_mos, 4)
                    zim_restored_distances.append(zim_distance)
                    zim_restored_mos_scores.append(zim_mos)
                
                if args.compute_degraded and 'degraded_path' in record:
                    degraded_path = record['degraded_path']
                    if degraded_path and os.path.exists(degraded_path):
                        degraded_wav, degraded_sr = load_audio(degraded_path)
                        if degraded_wav is not None:
                            degraded_np = degraded_wav.numpy()
                            zim_deg_distance, zim_deg_mos = compute_zimtohrli_distance(target_np, degraded_np, sr=degraded_sr, warn_resample=False)
                            if zim_deg_distance is not None:
                                record['metrics']['zimtohrli_degraded_distance'] = round(zim_deg_distance, 6)
                                record['metrics']['zimtohrli_degraded_mos'] = round(zim_deg_mos, 4)
                                zim_degraded_distances.append(zim_deg_distance)
                                zim_degraded_mos_scores.append(zim_deg_mos)

            if compute_zimtohrli and args.zim_compare_bin and os.path.exists(args.zim_compare_bin):
                zim_score = run_zimtohrli_compare(args.zim_compare_bin, target_path, output_path)
                if zim_score is not None:
                    record['metrics']['zimtohrli_cli_score'] = zim_score

                if args.compute_degraded and 'degraded_path' in record:
                    degraded_path = record['degraded_path']
                    if degraded_path and os.path.exists(degraded_path):
                        zim_deg = run_zimtohrli_compare(args.zim_compare_bin, target_path, degraded_path)
                        if zim_deg is not None:
                            record['metrics']['zimtohrli_cli_degraded_score'] = zim_deg

        except Exception as e:
            output_file = record.get('restored_path')
            print(f"\nError processing {output_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print processing summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY:")
    print(f"  Total records: {len(restoration_records)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Skipped (no paths): {skipped_no_paths}")
    print(f"  Skipped (files not found): {skipped_no_files}")
    print(f"  Skipped (load failed): {skipped_load_failed}")
    print(f"{'='*60}")
    
    # Calculate overall average Multi-Mel-SNR
    if mel_snr_scores:
        overall_mel_snr = sum(mel_snr_scores) / len(mel_snr_scores)
        print(f"\n{'='*60}")
        print(f"Overall Multi-Mel-SNR: {overall_mel_snr:.4f}")
        print(f"Computed over {len(mel_snr_scores)} audio pairs")
        print(f"{'='*60}")
    
    # Calculate overall average LSD (for BABE2 comparison)
    if lsd_scores:
        overall_lsd = sum(lsd_scores) / len(lsd_scores)
        print(f"\n{'='*60}")
        print(f"Overall Log-Spectral Distance (LSD): {overall_lsd:.4f}")
        print(f"(Lower is better)")
        print(f"Computed over {len(lsd_scores)} audio pairs")
        print(f"{'='*60}")
    
    # Calculate overall average Mel Distance
    if mel_scores:
        overall_mel = sum(mel_scores) / len(mel_scores)
        print(f"\n{'='*60}")
        print(f"Overall Mel Distance: {overall_mel:.4f}")
        print(f"(Lower is better)")
        print(f"Computed over {len(mel_scores)} audio pairs")
        print(f"{'='*60}")
    
    # Calculate overall average SI-SDR
    if sisdr_scores:
        overall_sisdr = sum(sisdr_scores) / len(sisdr_scores)
        print(f"\n{'='*60}")
        print(f"Overall SI-SDR: {overall_sisdr:.4f} dB")
        print(f"(Higher is better)")
        print(f"Computed over {len(sisdr_scores)} audio pairs")
        print(f"{'='*60}")

    # Display Zimtohrli metrics
    if compute_zimtohrli:
        if zim_restored_distances:
            avg_zim_distance = sum(zim_restored_distances) / len(zim_restored_distances)
            avg_zim_mos = sum(zim_restored_mos_scores) / len(zim_restored_mos_scores)
            print(f"\n{'='*60}")
            print(f"Overall Zimtohrli Metrics (Restored vs Clean):")
            print(f"  Average Distance: {avg_zim_distance:.6f}")
            print(f"  Average MOS: {avg_zim_mos:.4f}")
            print(f"  Computed over {len(zim_restored_distances)} audio pairs")
            
            if zim_degraded_distances:
                avg_zim_deg_distance = sum(zim_degraded_distances) / len(zim_degraded_distances)
                avg_zim_deg_mos = sum(zim_degraded_mos_scores) / len(zim_degraded_mos_scores)
                print(f"\nOverall Zimtohrli Metrics (Degraded vs Clean):")
                print(f"  Average Distance: {avg_zim_deg_distance:.6f}")
                print(f"  Average MOS: {avg_zim_deg_mos:.4f}")
                print(f"  Computed over {len(zim_degraded_distances)} audio pairs")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("No Zimtohrli metrics computed.")
            if not ZIMTOHRLI_AVAILABLE:
                print("Reason: Zimtohrli Python wrapper not available.")
            else:
                print("Reason: No valid audio pairs processed or computation failed.")
            print(f"{'='*60}")
    
    # Save updated JSONL with metrics
    output_jsonl = args.jsonl_file.replace('.jsonl', '_with_metrics.jsonl')
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            return super().default(obj)
    
    with open(output_jsonl, 'w') as f:
        for record in restoration_records:
            f.write(json.dumps(record, cls=NumpyEncoder) + '\n')
    print(f"\nUpdated JSONL saved to: {output_jsonl}")
    
    # Export to CSV/Excel if requested
    if args.output_csv:
        try:
            # Flatten metrics into DataFrame
            rows = []
            for record in restoration_records:
                row = {
                    'id': record.get('id', record.get('source_id', 'N/A')),
                    'clean_path': record.get('clean_path', ''),
                    'output_path': record.get(args.audio_key if args.audio_key != 'auto' else 'restored_path', 
                                             record.get('reconstructed_path', record.get('degraded_path', ''))),
                }
                # Add all metrics
                if 'metrics' in record:
                    row.update(record['metrics'])
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Save to CSV or Excel based on extension
            if args.output_csv.endswith('.xlsx'):
                df.to_excel(args.output_csv, index=False)
            else:
                df.to_csv(args.output_csv, index=False)
            
            print(f"Metrics exported to: {args.output_csv}")
        except Exception as e:
            print(f"Warning: Failed to export metrics to {args.output_csv}: {e}")

if __name__ == "__main__":
    main()
