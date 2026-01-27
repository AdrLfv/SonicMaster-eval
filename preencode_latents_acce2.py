import os
import glob
import json
import torch
import torchaudio
import argparse
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from accelerate import Accelerator
from utils import pad_wav


def read_wav_file(filename, duration_sec):
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate
    num_frames = int(sample_rate * duration_sec)

    waveform, sr = torchaudio.load(filename, num_frames=num_frames)

    # Resample
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
    waveform = resampler(waveform)

    # Convert mono to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Pad each channel
    target_length = int(44100 * duration_sec)
    padded_left = pad_wav(waveform[0], target_length)
    padded_right = pad_wav(waveform[1], target_length)

    return torch.stack([padded_left, padded_right])


def main():
    parser = argparse.ArgumentParser(description='Pre-encode audio latents using VAE')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Input JSONL file with degraded audio paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for encoded latents')
    parser.add_argument('--duration_sec', type=int, default=30, help='Duration in seconds to process (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for encoding (default: 16)')
    args = parser.parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device

    input_jsonl = args.input_jsonl
    output_dir = args.output_dir
    duration_sec = args.duration_sec
    batch_size = args.batch_size

    print(f"Rank {accelerator.process_index}/{accelerator.num_processes} - Starting encoding process")
    print(f"Input JSONL: {input_jsonl}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # Read jsonl entries
    with open(input_jsonl, 'r') as f:
        jsonl_entries = [json.loads(line) for line in f]

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
    jsonl_entries = jsonl_entries[rank::world_size]  # Distribute entries across processes
    print(f"Rank {rank} - Entries assigned to this rank: {len(jsonl_entries)}")

    batch_waveforms = []
    batch_entries = []

    if len(jsonl_entries) == 0:
        print(f"Rank {rank} - No entries to process. Exiting.")
        return

    print(f"Rank {rank} - Starting to process {len(jsonl_entries)} entries...")
    os.makedirs(output_dir, exist_ok=True)

    for entry in tqdm(jsonl_entries, desc=f"Rank {rank} Encoding", disable=not accelerator.is_main_process):
        try:
            degraded_path = entry.get('degraded_audio_path') or entry.get('degraded_path')
            waveform = read_wav_file(degraded_path, duration_sec)
            batch_waveforms.append(waveform)
            batch_entries.append(entry)

            if len(batch_waveforms) == batch_size or entry == jsonl_entries[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    latents = vae.encode(batch_tensor).latent_dist.mode()
                    latents = latents.transpose(1, 2)  # [B, T, C]

                for ent, latent in zip(batch_entries, latents.cpu()):
                    degraded_path = ent.get('degraded_audio_path') or ent.get('degraded_path')
                    basename = os.path.basename(degraded_path)
                    outpath = os.path.join(output_dir, basename.replace(os.path.splitext(basename)[1], ".pt"))
                    torch.save(latent, outpath)
                    ent['degraded_latent_path'] = outpath

                print(f"Rank {rank} - Saved batch of {len(batch_entries)} latents")
                batch_waveforms.clear()
                batch_entries.clear()

        except Exception as e:
            degraded_path = entry.get('degraded_audio_path') or entry.get('degraded_path', 'unknown')
            print(f"Error processing {degraded_path} on rank {rank}: {e}")

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Gather all entries from all ranks and save updated jsonl
        with open(input_jsonl, 'r') as f:
            all_entries = [json.loads(line) for line in f]
        
        output_jsonl = os.path.join(output_dir, os.path.basename(input_jsonl))
        with open(output_jsonl, 'w') as f:
            for entry in all_entries:
                degraded_path = entry.get('degraded_audio_path') or entry.get('degraded_path')
                basename = os.path.basename(degraded_path)
                latent_path = os.path.join(output_dir, basename.replace(os.path.splitext(basename)[1], ".pt"))
                entry['degraded_latent_path'] = latent_path
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved updated JSONL to {output_jsonl}")
    
    print(f"Rank {rank} - Encoding complete!")


if __name__ == "__main__":
    main()
