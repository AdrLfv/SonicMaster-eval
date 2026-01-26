import argparse
import json
import math
import os
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
        default="clean_path",
        help="The name of the column in the datasets containing the target audio paths.",
    )
    parser.add_argument(
        "--deg_audio_column",
        type=str,
        default="degraded_path",
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

    if config["paths"]["infer_file"] != "":
        data_files["infer"] = config["paths"]["infer_file"]

    from datasets import Dataset, DatasetDict
    with open(config["paths"]["infer_file"], 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    for entry in jsonl_data:
        entry['degradation_tracking'] = json.dumps(entry['degradation_tracking'])
        entry['hidden_clipping'] = json.dumps(entry['hidden_clipping'])
        entry['degradations'] = json.dumps(entry['degradations'])
        entry['degradations_specifics'] = json.dumps(entry['degradations_specifics'])
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
    
    if world_size > 1:
        dataset_size = len(full_dataset)
        indices = list(range(rank, dataset_size, world_size))
        infer_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Rank {rank}: Processing {len(infer_dataset)}/{dataset_size} samples")
    else:
        infer_dataset = full_dataset

    infer_dataloader = DataLoader(
        infer_dataset,
        shuffle=False,
        # batch_size=config["training"]["per_device_batch_size"],
        batch_size=16,
        collate_fn=infer_dataset.collate_fn,
    )



    total_batch_size = per_device_batch_size

    # Only show the progress bar once on each machine.
    tqdm(range(math.ceil(len(infer_dataloader) / total_batch_size)))


    infer_outputs=[]
    # wave_list=[]
    model.eval()
    global_idx=0

    if args.use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_output_dir = os.path.join(output_dir, f"inference_{timestamp}")
    else:
        inference_output_dir = output_dir
    
    if rank == 0:
        os.makedirs(inference_output_dir, exist_ok=True)
    
    if world_size > 1:
        torch.distributed.barrier()
    
    eval_jsonl_path = os.path.join(inference_output_dir, f"evaluation_metadata_rank{rank}.jsonl")
    eval_jsonl_file = open(eval_jsonl_path, "w", encoding="utf-8")
    
    for step, batch in enumerate(infer_dataloader):
        # inference_batch = next(iter(infer_dataloader))
        # with accelerator.accumulate(model) and torch.no_grad():
        with torch.no_grad():
            batch_start_time = time.time()  
            text, alt_text, audios, deg_audios, duration, valid_global_indices, deg_latent_paths = batch

            deg_audio_list = []
            for deg_latent_path in deg_latent_paths:
                loaded_tensor=torch.load(deg_latent_path)
                deg_audio_list.append(loaded_tensor)

            deg_audio_latent = torch.stack(deg_audio_list, dim=0)
            deg_audio_latent = deg_audio_latent.to(device)

            text = [args.prompt]*len(text)

            inferred_result = model.inference_flow(
                deg_audio_latent,
                text,
                # audiocond_latents=audio_latent,
                audiocond_latents=None,
                num_inference_steps=100,
                timesteps=None,
                guidance_scale=1,
                duration=duration,
                seed=0,
                disable_progress=False,
                num_samples_per_prompt=1,
                callback_on_step_end=None,
                solver="Euler", #Euler or rk4
            )
            infer_outputs.append(inferred_result)
            wave_list=[]

            wave = vae.decode(inferred_result.transpose(2, 1)).sample.cpu()
            wave_list.append(wave)

            # Calculate time taken for this batch
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            for k in range(len(wave_list[0])):
                file_idx = valid_global_indices[k]
                if args.output_format == 'hdf5':
                    restored_filename = filenames[file_idx].replace(".pt", ".h5")
                    restored_path = os.path.join(inference_output_dir, restored_filename)
                    with h5py.File(restored_path, 'w') as f:
                        f.create_dataset('audio', data=wave_list[0][k].numpy(), compression='gzip')
                else:
                    restored_filename = filenames[file_idx].replace(".pt", f".{args.output_format}")
                    restored_path = os.path.join(inference_output_dir, restored_filename)
                    sf.write(restored_path, wave_list[0][k].numpy().T, samplerate=fs, format=args.output_format.upper())
                
                # Write metadata for evaluation
                deg_spec = input_metadata[file_idx].get("degradation_spec", "")
                deg_group = input_metadata[file_idx].get("degradation_group", "")
                eval_entry = {
                    "clean_path": input_metadata[file_idx].get(args.audio_column, ""),
                    "degraded_path": input_metadata[file_idx].get(args.deg_audio_column, ""),
                    "restored_path": restored_path,
                    "degradation_name": f"{deg_group}_sonicmaster_{deg_spec}",
                    "degradation_spec": deg_spec,
                    "degradation_group": deg_group,
                    "sample_rate": fs,
                    "inference_time_seconds": batch_time / len(wave_list[0])  # Time per audio in batch
                }
                eval_jsonl_file.write(json.dumps(eval_entry) + "\n")
    
    eval_jsonl_file.close()
    print(f"\n✅ Rank {rank}: Evaluation metadata saved to: {eval_jsonl_path}")
    
    if world_size > 1:
        torch.distributed.barrier()
        if rank == 0:
            combined_jsonl = os.path.join(inference_output_dir, "evaluation_metadata.jsonl")
            with open(combined_jsonl, "w") as outf:
                for r in range(world_size):
                    rank_file = os.path.join(inference_output_dir, f"evaluation_metadata_rank{r}.jsonl")
                    if os.path.exists(rank_file):
                        with open(rank_file, "r") as inf:
                            outf.write(inf.read())
            print(f"\n✅ Combined evaluation metadata saved to: {combined_jsonl}")

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
