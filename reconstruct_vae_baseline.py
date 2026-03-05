import os
import json
import torch
import torchaudio
import h5py
import argparse
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from accelerate import Accelerator
import soundfile as sf

VAE_SR = 44100
DURATION_SEC = 30


def _waveform_from_numpy(audio_data, duration_sec, target_sr):
    """Convert a raw numpy array to a stereo float32 tensor at target_sr."""
    if audio_data.ndim == 1:
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0).repeat(2, 1)
    elif audio_data.ndim == 2:
        if audio_data.shape[0] == 2:
            waveform = torch.from_numpy(audio_data).float()
        elif audio_data.shape[1] == 2:
            waveform = torch.from_numpy(audio_data.T).float()
        else:
            waveform = torch.from_numpy(audio_data).float() if audio_data.shape[0] < audio_data.shape[1] else torch.from_numpy(audio_data.T).float()
    else:
        raise ValueError(f"Unexpected audio shape: {audio_data.shape}")

    # Infer sample rate from sample count
    inferred_sr = round(waveform.shape[1] / duration_sec)
    if inferred_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=inferred_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    # Trim or pad to exact target length
    target_length = int(target_sr * duration_sec)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))

    return waveform


def read_audio_file(filename, duration_sec=DURATION_SEC, target_sr=VAE_SR):
    """Read audio from wav, flac, sharded hdf5 (shard.h5::/dataset), or plain .h5 format."""
    # Sharded HDF5: shard.h5::/dataset_name
    if '::' in filename:
        sep = "::/" if "::/" in filename else "::"
        file_path, dataset_name = filename.split(sep, 1)
        dataset_name = dataset_name.lstrip("/")
        with h5py.File(file_path, 'r') as f:
            node = f[dataset_name]
            audio_data = node['audio'][:] if (isinstance(node, h5py.Group) and 'audio' in node) else node[:]
        return _waveform_from_numpy(audio_data, duration_sec, target_sr)

    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.wav', '.flac', '.mp3']:
        info = torchaudio.info(filename)
        sr = info.sample_rate
        num_frames = int(sr * duration_sec)
        waveform, sr = torchaudio.load(filename, num_frames=num_frames)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        target_length = int(target_sr * duration_sec)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        return waveform

    elif ext in ['.h5', '.hdf5']:
        with h5py.File(filename, 'r') as f:
            if 'audio' in f:
                audio_data = f['audio'][:]
            else:
                keys = list(f.keys())
                node = f[keys[0]]
                audio_data = node['audio'][:] if (isinstance(node, h5py.Group) and 'audio' in node) else node[:]
        return _waveform_from_numpy(audio_data, duration_sec, target_sr)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct audio through VAE encode-decode (baseline)')
    parser.add_argument('--input_jsonl', type=str,
                        help='Input JSONL file with audio paths (key: degraded_path or clean_path)')
    parser.add_argument('--input_folder', type=str,
                        help='Input folder containing standalone audio files (no JSONL needed)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for reconstructed audio')
    parser.add_argument('--audio_key', type=str, default='clean_audio_path',
                        help='JSON key for audio path (default: clean_audio_path)')
    parser.add_argument('--duration_sec', type=int, default=30, 
                        help='Duration in seconds to process (default: 30, use -1 for full length)')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for encoding/decoding (default: 16)')
    parser.add_argument('--output_format', type=str, default='flac', choices=['flac', 'wav', 'hdf5'],
                        help='Output audio format (default: flac)')
    args = parser.parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device

    print(f"Rank {accelerator.process_index}/{accelerator.num_processes} - Starting VAE reconstruction")
    if not args.input_jsonl and not args.input_folder:
        parser.error('Either --input_jsonl or --input_folder must be provided.')
    if args.input_jsonl and args.input_folder:
        parser.error('Please specify only one of --input_jsonl or --input_folder.')

    if args.input_jsonl:
        print(f"Input JSONL: {args.input_jsonl}")
    if args.input_folder:
        print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Audio key: {args.audio_key}")
    print(f"Device: {device}")

    # Build processing list
    if args.input_jsonl:
        with open(args.input_jsonl, 'r') as f:
            jsonl_entries = [json.loads(line) for line in f]
    else:
        supported_exts = ('.wav', '.flac', '.mp3', '.h5', '.hdf5')
        jsonl_entries = []
        for fname in sorted(os.listdir(args.input_folder)):
            full_path = os.path.join(args.input_folder, fname)
            if os.path.isfile(full_path) and fname.lower().endswith(supported_exts):
                jsonl_entries.append({args.audio_key: full_path})
        if not jsonl_entries:
            raise ValueError(f"No supported audio files found in {args.input_folder}")

    # Load VAE and prepare for multi-GPU
    print(f"Rank {accelerator.process_index} - Loading VAE model...")
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae.eval()
    vae.requires_grad_(False)
    vae = accelerator.prepare(vae)
    print(f"Rank {accelerator.process_index} - VAE model loaded")

    # Partition jsonl entries by rank
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    print(f"Rank {rank} - Total entries found: {len(jsonl_entries)}")
    rank_entries = jsonl_entries[rank::world_size]
    print(f"Rank {rank} - Entries assigned to this rank: {len(rank_entries)}")

    batch_waveforms = []
    batch_entries = []
    completed_entries = []

    if len(rank_entries) == 0:
        print(f"Rank {rank} - No entries to process. Exiting.")
        accelerator.wait_for_everyone()
        return

    print(f"Rank {rank} - Starting to process {len(rank_entries)} entries...")
    os.makedirs(args.output_dir, exist_ok=True)

    for entry in tqdm(rank_entries, desc=f"Rank {rank} Reconstructing", disable=not accelerator.is_main_process):
        try:
            audio_path = entry[args.audio_key]
            waveform = read_audio_file(audio_path, args.duration_sec)
            batch_waveforms.append(waveform)
            batch_entries.append(entry)

            if len(batch_waveforms) == args.batch_size or entry == rank_entries[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    latents = vae.encode(batch_tensor).latent_dist.mode()
                    reconstructed = vae.decode(latents).sample.cpu()

                for ent, recon_audio in zip(batch_entries, reconstructed):
                    # Use 'id' field if present, otherwise derive from audio path basename
                    if 'id' in ent:
                        base_stem = str(ent['id'])
                    else:
                        basename = os.path.basename(ent[args.audio_key])
                        base_stem = os.path.splitext(basename)[0]

                    # Save audio
                    if args.output_format == 'hdf5':
                        outpath = os.path.join(args.output_dir, f"{base_stem}_reconstructed.h5")
                        with h5py.File(outpath, 'w') as hf:
                            hf.create_dataset('audio', data=recon_audio.numpy(), compression='gzip')
                    else:
                        outpath = os.path.join(args.output_dir, f"{base_stem}_reconstructed.{args.output_format}")
                        audio_data = recon_audio.clamp(-1.0, 1.0).numpy().T  # [T, 2]
                        if args.output_format == 'flac':
                            sf.write(outpath, audio_data, samplerate=44100, format='FLAC')
                        else:
                            sf.write(outpath, audio_data, samplerate=44100, format='WAV')

                    # Full metadata passthrough + reconstructed_audio_path
                    out_entry = dict(ent)
                    out_entry['reconstructed_audio_path'] = outpath
                    completed_entries.append(out_entry)

                print(f"Rank {rank} - Saved batch of {len(batch_entries)} reconstructed files")
                batch_waveforms.clear()
                batch_entries.clear()

        except Exception as e:
            print(f"Error processing {entry.get(args.audio_key, 'unknown')} on rank {rank}: {e}")

    accelerator.wait_for_everyone()

    if world_size == 1:
        # Single-GPU: write directly
        output_jsonl = os.path.join(args.output_dir, "reconstruction_metadata.jsonl")
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for entry in completed_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Rank {rank} - Saved {len(completed_entries)} entries to {output_jsonl}")
    else:
        # Multi-GPU: write per-rank files, then rank 0 combines
        rank_jsonl_path = os.path.join(args.output_dir, f"reconstruction_metadata_rank{rank}.jsonl")
        with open(rank_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in completed_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Rank {rank} - Wrote {len(completed_entries)} entries to {rank_jsonl_path}")

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            output_jsonl = os.path.join(args.output_dir, "reconstruction_metadata.jsonl")
            with open(output_jsonl, 'w', encoding='utf-8') as out_f:
                for r in range(world_size):
                    rank_file = os.path.join(args.output_dir, f"reconstruction_metadata_rank{r}.jsonl")
                    if os.path.exists(rank_file):
                        with open(rank_file, 'r') as in_f:
                            out_f.write(in_f.read())
                        os.remove(rank_file)
            print(f"Saved combined metadata to {output_jsonl}")

    print(f"Rank {rank} - VAE reconstruction complete!")


if __name__ == "__main__":
    main()
