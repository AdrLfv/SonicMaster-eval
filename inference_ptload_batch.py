import argparse
import json
import math
import os
import re
import yaml
from datetime import datetime
import time
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import TangoFlux
from utils import Text2AudioDataset, read_wav_file

from diffusers import AutoencoderOobleck
from safetensors.torch import load_file
import soundfile as sf
import h5py


logger = get_logger(__name__)


def restored_filename(degraded_audio_path, output_format, strip_degradation_suffix=False):
    """Build a restored filename from the degraded input filename."""
    base_name = os.path.basename(degraded_audio_path)
    base_stem = os.path.splitext(base_name)[0]
    if strip_degradation_suffix:
        base_stem = re.sub(r"_deg\d+$", "", base_stem)
    extension = "h5" if output_format == "hdf5" else output_format
    return f"{base_stem}_restored.{extension}"


def load_completed_indices(metadata_path, output_dir, filenames, output_format,
                           strip_degradation_suffix):
    """Return manifest indices with both valid metadata and an existing output."""
    completed = set()
    if not os.path.isfile(metadata_path):
        return completed
    with open(metadata_path, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                source_index = int(entry["source_manifest_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"Ignoring invalid resume metadata line {line_number}: {exc}")
                continue
            if not 0 <= source_index < len(filenames):
                continue
            expected_path = os.path.join(
                output_dir,
                restored_filename(
                    filenames[source_index], output_format, strip_degradation_suffix
                ),
            )
            if (entry.get("restored_audio_path") == expected_path
                    and os.path.isfile(expected_path)
                    and os.path.getsize(expected_path) > 44):
                completed.add(source_index)
    return completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectified flow for text to audio generation task."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training and validation.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="prompt",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--alt_text_column",
        type=str,
        default="alt_prompt",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="clean_audio_path",
        help="The name of the column in the datasets containing the target audio paths.",
    )
    parser.add_argument(
        "--deg_audio_column",
        type=str,
        default="degraded_audio_path",
        help="The name of the column in the datasets containing the degraded audio paths.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tangoflux_config.yaml",
        help="Config file defining the model size as well as other hyper parameter.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Add prefix in text prompts.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )

    parser.add_argument(
        "--vae_batch_size", type=int, default=16, help="Batch size for VAE encoding."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size (default: 16). Reduce this for GPUs with limited memory.",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/outputs/seed27full10sec/epoch_40",
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--infer_file",
        type=str,
        default=None,
        help="Override infer_file from config (JSONL with degraded audio paths and latents).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir from config (directory for restored audio).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Restoration prompt to guide the model (e.g., 'Reduce the clipping and reconstruct the lost audio, please.'). Empty string for no prompt.",
    )
    parser.add_argument(
        "--use_jsonl_prompt",
        action="store_true",
        help="Use per-sample 'prompt' field from the input JSONL instead of --prompt.",
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        default="flac",
        choices=["flac", "wav", "hdf5"],
        help="Output audio format (default: flac)",
    )
    
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Append timestamp to output directory (creates inference_YYYYMMDD_HHMMSS subdirectory)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append metadata and skip indices whose metadata and output file already exist.",
    )

    parser.add_argument(
        "--strip_degradation_suffix",
        action="store_true",
        help="Strip a trailing _degN suffix from restored output filenames.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of Euler steps of the restoration flow (SonicMaster default elsewhere: 10).",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # accelerator_log_kwargs = {}
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        print(f"Rank {rank}/{world_size} using device {device}")
    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    config = load_config(args.config)

    per_device_batch_size = int(config["training"]["per_device_batch_size"])

    # Override config values with command line args if provided
    output_dir = args.output_dir if args.output_dir is not None else config["paths"]["output_dir"]
    jsonfile = args.infer_file if args.infer_file is not None else config["paths"]["infer_file"]
    
    if rank == 0:
        print(f"Using infer_file: {jsonfile}")
        print(f"Using output_dir: {output_dir}")

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     **accelerator_log_kwargs,
    # )


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    # if accelerator.is_main_process:
    #     if output_dir is None or output_dir == "":
    #         output_dir = "saved/" + str(int(time.time()))

    #         if not os.path.exists("saved"):
    #             os.makedirs("saved")

    #         os.makedirs(output_dir, exist_ok=True)

    #     elif output_dir is not None:
    #         os.makedirs(output_dir, exist_ok=True)

    #     os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
    #     with open("{}/summary.jsonl".format(output_dir), "a") as f:
    #         f.write(json.dumps(dict(vars(args))) + "\n\n")

    #     accelerator.project_configuration.automatic_checkpoint_naming = False

    #     wandb.init(
    #         project="Text to Audio Flow matching",
    #         settings=wandb.Settings(_disable_stats=True),
    #     )

    # accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}

    if jsonfile != "":
        data_files["infer"] = jsonfile

    from datasets import Dataset, DatasetDict
    with open(jsonfile, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    
    # Convert nested structures to strings for PyArrow compatibility
    for entry in jsonl_data:
        for key, value in entry.items():
            if isinstance(value, (list, dict)):
                entry[key] = json.dumps(value)
    
    infer_dataset = Dataset.from_dict({k: [d[k] for d in jsonl_data] for k in jsonl_data[0].keys()})
    raw_datasets = DatasetDict({"infer": infer_dataset})
    text_column, alt_text_column, audio_column, deg_audio_column = args.text_column, args.alt_text_column, args.audio_column, args.deg_audio_column

    model = TangoFlux(config=config["model"])
    # model.load_state_dict(torch.load(os.path.join(args.model_ckpt,"model_1.safetensors")))

    # Handle both directory path and full file path
    if os.path.isfile(args.model_ckpt):
        weights = load_file(args.model_ckpt)
    else:
        weights = load_file(os.path.join(args.model_ckpt,"model.safetensors"))
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae.to(device)
    vae.eval()

    ## Freeze vae
    # for param in vae.parameters():
    #     vae.requires_grad = False
    #     vae.eval()

    ## Freeze text encoder param
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        model.text_encoder.eval()

    prefix = args.prefix

    # with accelerator.main_process_first():
    #     infer_dataset = Text2AudioDataset(
    #         raw_datasets["infer"],
    #         prefix,
    #         text_column,
    #         audio_column,
    #         deg_audio_column,
    #         "duration",
    #         args.num_examples,
    #     )
    #     accelerator.print(
    #         "Num instances in train: {}, validation: {}, test: {}".format(
    #             train_dataset.get_num_instances(),
    #             eval_dataset.get_num_instances(),
    #             test_dataset.get_num_instances(),
    #         )
    #     )

    fs=44100
    filenames=[]
    input_metadata = []
    with open(jsonfile, "r", encoding="utf-8") as infile:
        for line in infile:
            a=json.loads(line)
            filenames.append(os.path.basename(a[args.deg_audio_column]))
            input_metadata.append(a)

    full_dataset = Text2AudioDataset(
        raw_datasets["infer"],
        prefix,
        text_column,
        alt_text_column,
        audio_column,
        deg_audio_column,
        "duration",
        args.num_examples,
        deg_latent_column="degraded_latent_path",
    )

    if args.use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_output_dir = os.path.join(output_dir, f"inference_{timestamp}")
    else:
        inference_output_dir = output_dir

    if rank == 0:
        os.makedirs(inference_output_dir, exist_ok=True)

    if args.resume and world_size > 1:
        raise ValueError("--resume currently supports single-process inference only")

    eval_jsonl_path = os.path.join(inference_output_dir, "restoration_metadata.jsonl")
    completed_indices = set()
    if args.resume:
        completed_indices = load_completed_indices(
            eval_jsonl_path,
            inference_output_dir,
            filenames,
            args.output_format,
            args.strip_degradation_suffix,
        )
        if completed_indices:
            print(f"Resume: skipping {len(completed_indices)} completed samples")

    if world_size > 1:
        dataset_size = len(full_dataset)
        indices = list(range(rank, dataset_size, world_size))
        indices = [index for index in indices if index not in completed_indices]
        infer_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Rank {rank}: Processing {len(infer_dataset)}/{dataset_size} samples")
    elif args.resume:
        indices = [
            index for index in range(len(full_dataset))
            if index not in completed_indices
        ]
        infer_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        infer_dataset = full_dataset

    if len(infer_dataset) == 0:
        print("All requested samples are already complete.")
        return

    infer_dataloader = DataLoader(
        infer_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=full_dataset.collate_fn,
    )



    total_batch_size = per_device_batch_size

    # Only show the progress bar once on each machine.
    tqdm(range(math.ceil(len(infer_dataloader) / total_batch_size)))


    model.eval()
    
    if world_size > 1:
        torch.distributed.barrier()
    
    if world_size > 1:
        eval_jsonl_path = os.path.join(inference_output_dir, f"restoration_metadata_rank{rank}.jsonl")
    metadata_mode = "a" if args.resume else "w"
    eval_jsonl_file = open(eval_jsonl_path, metadata_mode, encoding="utf-8")
    
    # Create overall progress bar
    total_batches = len(infer_dataloader)
    pbar = tqdm(total=total_batches, desc=f"Rank {rank} Restoring", disable=(rank != 0))
    
    for step, batch in enumerate(infer_dataloader):
        # inference_batch = next(iter(infer_dataloader))
        # with accelerator.accumulate(model) and torch.no_grad():
        with torch.no_grad():
            batch_start_time = time.time()  
            if len(batch) == 7:
                text, alt_text, audios, deg_audios, duration, valid_global_indices, deg_latent_paths = batch
            else:
                text, alt_text, audios, deg_audios, duration, valid_global_indices = batch
                deg_latent_paths = None

            deg_audio_list = []
            if deg_latent_paths and all(p and os.path.exists(p) for p in deg_latent_paths):
                for deg_latent_path in deg_latent_paths:
                    loaded_tensor=torch.load(deg_latent_path)
                    deg_audio_list.append(loaded_tensor)
            else:
                # Encode from raw degraded audio using VAE
                for deg_audio_path in deg_audios:
                    audio_waveform = read_wav_file(deg_audio_path, 30).float()
                    if audio_waveform.dim() == 1:
                        audio_waveform = audio_waveform.unsqueeze(0).repeat(2, 1)
                    audio_batch = audio_waveform.unsqueeze(0).to(device)
                    with torch.no_grad():
                        latent = vae.encode(audio_batch).latent_dist.sample()
                    deg_audio_list.append(latent.squeeze(0).transpose(0, 1).cpu())

            deg_audio_latent = torch.stack(deg_audio_list, dim=0)
            deg_audio_latent = deg_audio_latent.to(device)

            if not args.use_jsonl_prompt:
                text = [args.prompt]*len(text)

            inferred_result = model.inference_flow(
                deg_audio_latent,
                text,
                # audiocond_latents=audio_latent,
                audiocond_latents=None,
                num_inference_steps=args.num_inference_steps,
                timesteps=None,
                guidance_scale=1,
                duration=duration,
                seed=args.seed if args.seed is not None else 0,
                disable_progress=True,
                num_samples_per_prompt=1,
                callback_on_step_end=None,
                solver="Euler", #Euler or rk4
            )
            wave = vae.decode(inferred_result.transpose(2, 1)).sample.cpu()

            # Calculate time taken for this batch
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            for k in range(len(wave)):
                file_idx = int(valid_global_indices[k])
                output_name = restored_filename(
                    filenames[file_idx],
                    args.output_format,
                    args.strip_degradation_suffix,
                )
                restored_path = os.path.join(inference_output_dir, output_name)
                if args.output_format == 'hdf5':
                    with h5py.File(restored_path, 'w') as f:
                        f.create_dataset('audio', data=wave[k].numpy(), compression='gzip')
                else:
                    sf.write(restored_path, wave[k].numpy().T, samplerate=fs, format=args.output_format.upper())
                
                # Write metadata for evaluation - start with original metadata
                eval_entry = dict(input_metadata[file_idx])
                
                # Add/update inference-specific fields
                eval_entry.update({
                    "restored_audio_path": restored_path,
                    "source_manifest_index": file_idx,
                    "prompt_used": text[k],
                    "sample_rate": fs,
                    "duration_sec": 30,
                    "inference_time_seconds": batch_time / len(wave),
                    "timestamp": datetime.now().isoformat(),
                    "inference_settings": {
                        "mode": "with_prompt" if args.use_jsonl_prompt else "fixed_prompt",
                        "prompt_field": args.text_column if args.use_jsonl_prompt else None,
                        "model_checkpoint": os.path.abspath(args.model_ckpt),
                        "config": os.path.abspath(args.config),
                        "vae": "stabilityai/stable-audio-open-1.0/vae",
                        "num_inference_steps": args.num_inference_steps,
                        "guidance_scale": 1,
                        "solver": "Euler",
                        "seed": args.seed if args.seed is not None else 0,
                        "duration_sec": 30,
                        "sample_rate_hz": fs,
                        "batch_size": args.batch_size,
                        "output_format": args.output_format,
                    },
                })
                eval_jsonl_file.write(json.dumps(eval_entry) + "\n")
            eval_jsonl_file.flush()

            del inferred_result, wave, deg_audio_latent
            
            # Update progress bar after each batch
            pbar.update(1)
    
    pbar.close()
    eval_jsonl_file.close()
    print(f"\n✅ Rank {rank}: Evaluation metadata saved to: {eval_jsonl_path}")
    
    if world_size > 1:
        torch.distributed.barrier()
        if rank == 0:
            combined_jsonl = os.path.join(inference_output_dir, "restoration_metadata.jsonl")
            with open(combined_jsonl, "w") as outf:
                for r in range(world_size):
                    rank_file = os.path.join(inference_output_dir, f"restoration_metadata_rank{r}.jsonl")
                    if os.path.exists(rank_file):
                        with open(rank_file, "r") as inf:
                            outf.write(inf.read())
                        os.remove(rank_file)
            print(f"\n✅ Combined restoration metadata saved to: {combined_jsonl}")

    # if accelerator.is_main_process:


    # for i, out in enumerate(infer_outputs):
    #     torch.save(out.cpu(), os.path.join(inference_output_dir, f"sample_{i}.pt"))
    # for i, out in enumerate(wave_list):
    #     sf.write(os.path.join(inference_output_dir,f"sample_{i}.flac"), out.numpy().T, samplerate=fs, format='FLAC')


        # torch.save(out.cpu(), os.path.join(inference_output_dir, f"sample_{i}.wav"))
                # if accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1




if __name__ == "__main__":
    main()
