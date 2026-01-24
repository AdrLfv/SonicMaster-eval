import os
import glob
import json
import torch
import torchaudio
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
    accelerator = Accelerator()
    device = accelerator.device

    input_jsonl = "/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_specific_punch_degraded/test_sonicmaster_punch.jsonl"
    output_dir = "/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_specific_punch_latents"
    duration_sec = 30
    batch_size = 16

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
            degraded_path = entry['degraded_path']
            waveform = read_wav_file(degraded_path, duration_sec)
            batch_waveforms.append(waveform)
            batch_entries.append(entry)

            if len(batch_waveforms) == batch_size or entry == jsonl_entries[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    latents = vae.encode(batch_tensor).latent_dist.mode()
                    latents = latents.transpose(1, 2)  # [B, T, C]

                for ent, latent in zip(batch_entries, latents.cpu()):
                    basename = os.path.basename(ent['degraded_path'])
                    outpath = os.path.join(output_dir, basename.replace(os.path.splitext(basename)[1], ".pt"))
                    torch.save(latent, outpath)
                    ent['degraded_latent_path'] = outpath

                print(f"Rank {rank} - Saved batch of {len(batch_entries)} latents")
                batch_waveforms.clear()
                batch_entries.clear()

        except Exception as e:
            print(f"Error processing {entry.get('degraded_path', 'unknown')} on rank {rank}: {e}")

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Gather all entries from all ranks and save updated jsonl
        with open(input_jsonl, 'r') as f:
            all_entries = [json.loads(line) for line in f]
        
        output_jsonl = os.path.join(output_dir, os.path.basename(input_jsonl))
        with open(output_jsonl, 'w') as f:
            for entry in all_entries:
                basename = os.path.basename(entry['degraded_path'])
                latent_path = os.path.join(output_dir, basename.replace(os.path.splitext(basename)[1], ".pt"))
                entry['degraded_latent_path'] = latent_path
                f.write(json.dumps(entry) + '\n')
        
        print(f"Saved updated JSONL to {output_jsonl}")
    
    print(f"Rank {rank} - Encoding complete!")


if __name__ == "__main__":
    main()
