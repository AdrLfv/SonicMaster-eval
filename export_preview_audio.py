"""Export raw clean or degraded WAVs referenced by a preview JSONL manifest."""

import argparse
import json
import os

import soundfile as sf

from reconstruct_vae_baseline import read_audio_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--audio-key", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_jsonl, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle]

    for entry in entries:
        waveform = read_audio_file(entry[args.audio_key], duration_sec=30)
        stem = entry["clean_audio_dataset"]
        output_path = os.path.join(args.output_dir, f"{stem}.wav")
        sf.write(output_path, waveform.numpy().T, samplerate=44100, format="WAV")
        print(output_path)


if __name__ == "__main__":
    main()
