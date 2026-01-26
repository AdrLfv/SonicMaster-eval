import os
import json
import torch
import torchaudio
import h5py
import argparse
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from accelerate import Accelerator
from utils import pad_wav
import soundfile as sf


def read_audio_file(filename, duration_sec, target_sr=44100):
    """Read audio from wav, flac, or hdf5 format"""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.wav', '.flac', '.mp3']:
        info = torchaudio.info(filename)
        sample_rate = info.sample_rate
        num_frames = int(sample_rate * duration_sec) if duration_sec > 0 else -1
        
        if num_frames > 0:
            waveform, sr = torchaudio.load(filename, num_frames=num_frames)
        else:
            waveform, sr = torchaudio.load(filename)
        
        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Convert mono to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        
        # Pad if duration specified
        if duration_sec > 0:
            target_length = int(target_sr * duration_sec)
            if waveform.shape[1] < target_length:
                padded_left = pad_wav(waveform[0], target_length)
                padded_right = pad_wav(waveform[1], target_length)
                waveform = torch.stack([padded_left, padded_right])
            else:
                waveform = waveform[:, :target_length]
        
        return waveform
    
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(filename, 'r') as f:
            waveform = torch.from_numpy(f['audio'][:])
            
            if waveform.dim() == 2 and waveform.shape[1] == 2 and waveform.shape[0] > 2:
                waveform = waveform.T
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).repeat(2, 1)
            elif waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            
            if duration_sec > 0:
                target_length = int(target_sr * duration_sec)
                if waveform.shape[1] < target_length:
                    padded_left = pad_wav(waveform[0], target_length)
                    padded_right = pad_wav(waveform[1], target_length)
                    waveform = torch.stack([padded_left, padded_right])
                else:
                    waveform = waveform[:, :target_length]
        
        return waveform
    
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
    parser.add_argument('--audio_key', type=str, default='clean_path',
                        help='JSON key for audio path (default: clean_path)')
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
    jsonl_entries = jsonl_entries[rank::world_size]
    print(f"Rank {rank} - Entries assigned to this rank: {len(jsonl_entries)}")
    local_entries = jsonl_entries

    batch_waveforms = []
    batch_entries = []

    if len(jsonl_entries) == 0:
        print(f"Rank {rank} - No entries to process. Exiting.")
        return

    print(f"Rank {rank} - Starting to process {len(jsonl_entries)} entries...")
    os.makedirs(args.output_dir, exist_ok=True)

    for entry in tqdm(jsonl_entries, desc=f"Rank {rank} Reconstructing", disable=not accelerator.is_main_process):
        try:
            audio_path = entry[args.audio_key]
            waveform = read_audio_file(audio_path, args.duration_sec)
            batch_waveforms.append(waveform)
            batch_entries.append(entry)

            if len(batch_waveforms) == args.batch_size or entry == jsonl_entries[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    # Encode
                    latents = vae.encode(batch_tensor).latent_dist.mode()
                    # Decode immediately (no model inference)
                    reconstructed = vae.decode(latents).sample.cpu()

                for ent, recon_audio in zip(batch_entries, reconstructed):
                    basename = os.path.basename(ent[args.audio_key])
                    base_no_ext = os.path.splitext(basename)[0]
                    
                    # Save audio
                    if args.output_format == 'hdf5':
                        outpath = os.path.join(args.output_dir, f"{base_no_ext}_reconstructed.h5")
                        with h5py.File(outpath, 'w') as f:
                            f.create_dataset('audio', data=recon_audio.numpy(), compression='gzip')
                    else:
                        outpath = os.path.join(args.output_dir, f"{base_no_ext}_reconstructed.{args.output_format}")
                        audio_data = recon_audio.numpy().T  # [T, 2]
                        if args.output_format == 'flac':
                            sf.write(outpath, audio_data, samplerate=44100, format='FLAC')
                        else:
                            sf.write(outpath, audio_data, samplerate=44100, format='WAV')
                    
                    ent['reconstructed_path'] = outpath

                print(f"Rank {rank} - Saved batch of {len(batch_entries)} reconstructed files")
                batch_waveforms.clear()
                batch_entries.clear()

        except Exception as e:
            print(f"Error processing {entry.get(args.audio_key, 'unknown')} on rank {rank}: {e}")

    accelerator.wait_for_everyone()

    # Gather entries from all processes
    if accelerator.is_main_process:
        flat_entries = local_entries
    else:
        flat_entries = []

    if accelerator.is_main_process:
        if args.input_jsonl:
            output_jsonl = os.path.join(
                args.output_dir,
                f"reconstructed_{os.path.basename(args.input_jsonl)}"
            )
        else:
            folder_name = os.path.basename(os.path.normpath(args.input_folder))
            output_jsonl = os.path.join(
                args.output_dir,
                f"reconstructed_{folder_name}.jsonl"
            )

        with open(output_jsonl, 'w') as f:
            for entry in flat_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"Saved updated JSONL to {output_jsonl}")
    
    print(f"Rank {rank} - VAE reconstruction complete!")


if __name__ == "__main__":
    main()
