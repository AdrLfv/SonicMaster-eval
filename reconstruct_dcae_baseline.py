import os
import json
import logging
import torch
import torchaudio
import h5py
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import soundfile as sf

from music_dcae.music_dcae_pipeline import MusicDCAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_audio_file(filename, duration_sec, target_sr=44100):
    """Read audio from wav, flac, or hdf5 format.
    
    Uses the same loading pattern as MusicDCAE.load_audio() for audio files.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.wav', '.flac', '.mp3']:
        # Use torchaudio.load like MusicDCAE.load_audio() - loads full file
        audio, sr = torchaudio.load(filename)
        
        # Convert mono to stereo (same as MusicDCAE.load_audio)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # Return audio with its actual sample rate (like MusicDCAE.load_audio)
        return audio, sr
    
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(filename, 'r') as f:
            waveform = torch.from_numpy(f['audio'][:])
            # Try to get sample rate from hdf5, default to target_sr
            sr = int(f.attrs.get('sample_rate', target_sr)) if 'sample_rate' in f.attrs else target_sr
            
            if waveform.dim() == 2 and waveform.shape[1] == 2 and waveform.shape[0] > 2:
                waveform = waveform.T
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).repeat(2, 1)
            elif waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
        
        return waveform, sr
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct audio through DCAE encode-decode (ACE-Step baseline)')
    parser.add_argument('--input_jsonl', type=str,
                        help='Input JSONL file with audio paths (key: degraded_path or clean_path)')
    parser.add_argument('--input_folder', type=str,
                        help='Input folder containing standalone audio files (no JSONL needed)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for reconstructed audio')
    parser.add_argument('--audio_key', type=str, default='clean_path',
                        help='JSON key for audio path (default: clean_path)')
    parser.add_argument('--duration_sec', type=int, default=-1, 
                        help='Duration in seconds to process (default: -1 for full length)')
    parser.add_argument('--output_format', type=str, default='flac', choices=['flac', 'wav', 'hdf5'],
                        help='Output audio format (default: flac)')
    parser.add_argument('--dcae_checkpoint_path', type=str, required=True,
                        help='Path to music_dcae_f8c8 checkpoint directory')
    parser.add_argument('--vocoder_checkpoint_path', type=str, required=True,
                        help='Path to music_vocoder checkpoint directory')
    parser.add_argument('--source_sr', type=int, default=44100,
                        help='Source sample rate for input audio (default: 44100)')
    parser.add_argument('--output_sr', type=int, default=44100,
                        help='Output sample rate for reconstructed audio (default: 44100)')
    args = parser.parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f"Rank {accelerator.process_index}/{accelerator.num_processes} - Starting DCAE reconstruction")
    if not args.input_jsonl and not args.input_folder:
        parser.error('Either --input_jsonl or --input_folder must be provided.')
    if args.input_jsonl and args.input_folder:
        parser.error('Please specify only one of --input_jsonl or --input_folder.')

    if args.input_jsonl:
        logger.info(f"Input JSONL: {args.input_jsonl}")
    if args.input_folder:
        logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Audio key: {args.audio_key}")
    logger.info(f"DCAE checkpoint: {args.dcae_checkpoint_path}")
    logger.info(f"Vocoder checkpoint: {args.vocoder_checkpoint_path}")
    logger.info(f"Source SR: {args.source_sr}, Output SR: {args.output_sr}")
    logger.info(f"Device: {device}")

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

    # Load DCAE model (following ACE-Step pipeline pattern)
    logger.info(f"Rank {accelerator.process_index} - Loading DCAE model...")
    dcae = MusicDCAE(
        source_sample_rate=args.source_sr,
        dcae_checkpoint_path=args.dcae_checkpoint_path,
        vocoder_checkpoint_path=args.vocoder_checkpoint_path,
    )
    dcae.eval()
    dcae.requires_grad_(False)
    # Move to device with appropriate dtype (following ACE-Step pattern)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dcae = dcae.to(device).to(dtype)
    logger.info(f"Rank {accelerator.process_index} - DCAE model loaded on {device} with dtype {dtype}")

    # Partition jsonl entries by rank
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    logger.info(f"Rank {rank} - Total entries found: {len(jsonl_entries)}")
    jsonl_entries = jsonl_entries[rank::world_size]
    logger.info(f"Rank {rank} - Entries assigned to this rank: {len(jsonl_entries)}")
    local_entries = jsonl_entries

    if len(jsonl_entries) == 0:
        logger.info(f"Rank {rank} - No entries to process. Exiting.")
        return

    logger.info(f"Rank {rank} - Starting to process {len(jsonl_entries)} entries...")
    os.makedirs(args.output_dir, exist_ok=True)

    for entry in tqdm(jsonl_entries, desc=f"Rank {rank} Reconstructing", disable=not accelerator.is_main_process):
        try:
            audio_path = entry[args.audio_key]
            
            # Load audio following ACE-Step pattern exactly:
            # input_audio, sr = self.music_dcae.load_audio(input_audio_path)
            audio, sr = read_audio_file(audio_path, args.duration_sec, target_sr=args.source_sr)
            
            # Add batch dimension and move to device (following ACE-Step pattern)
            # input_audio = input_audio.unsqueeze(0)
            # input_audio = input_audio.to(device=self.device, dtype=self.dtype)
            audio = audio.unsqueeze(0).to(device=device, dtype=dtype)

            with torch.no_grad():
                # Encode using DCAE (following ACE-Step pattern exactly)
                # latents, _ = self.music_dcae.encode(input_audio, sr=sr)
                latents, _ = dcae.encode(audio, sr=sr)
                
                # Decode using DCAE
                # Don't pass audio_lengths to avoid incorrect trimming
                output_sr, pred_wavs = dcae.decode(
                    latents=latents,
                    sr=args.output_sr
                )

            # pred_wavs is a list with one element (batch size 1)
            recon_audio = pred_wavs[0]
            basename = os.path.basename(audio_path)
            base_no_ext = os.path.splitext(basename)[0]
            
            # recon_audio is [2, T] tensor, convert to numpy
            recon_audio_np = recon_audio.float().cpu().numpy()
            
            # Save audio
            if args.output_format == 'hdf5':
                outpath = os.path.join(args.output_dir, f"{base_no_ext}_reconstructed.h5")
                with h5py.File(outpath, 'w') as f:
                    f.create_dataset('audio', data=recon_audio_np, compression='gzip')
            else:
                outpath = os.path.join(args.output_dir, f"{base_no_ext}_reconstructed.{args.output_format}")
                audio_data = recon_audio_np.T  # [T, 2] for soundfile
                if args.output_format == 'flac':
                    sf.write(outpath, audio_data, samplerate=output_sr, format='FLAC')
                else:
                    sf.write(outpath, audio_data, samplerate=output_sr, format='WAV')
            
            entry['reconstructed_path'] = outpath

        except Exception as e:
            logger.error(f"Error processing {entry.get(args.audio_key, 'unknown')} on rank {rank}: {e}")
            import traceback
            traceback.print_exc()

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

        logger.info(f"Saved updated JSONL to {output_jsonl}")
    
    logger.info(f"Rank {rank} - DCAE reconstruction complete!")


if __name__ == "__main__":
    main()
