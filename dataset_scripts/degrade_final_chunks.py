

# load
# roll how many augs
# decide which augs
# do them in order / roll on order?
# create a json entry?
# add sentences...
# optional - save all forms of degradation and sentences separately OR just save the final degraded with all associated sentences... and mix it all through distribution

# normalize after finishing, with clipping - skip?
# cut signals to N (original length)



# import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
import librosa
import scipy.signal as signal
import soundfile as sf
import h5py

import json, os
from glob import glob
import random

from deg_functions import *
from prompt_functions import *
from time import time
import argparse

import pyroomacoustics as pra


def save_shard(out_folder, shard_idx, shard_data, outfile, fs, deg_spec_selected):
    import logging
    if not shard_data:
        return []
    
    shard_path = os.path.join(out_folder, f"shard_{shard_idx:04d}.h5")
    logging.info(f"Saving shard {shard_idx} with {len(shard_data)} samples to {shard_path}")
    
    metadata_entries = []
    with h5py.File(shard_path, 'w') as f:
        for idx, (audio, original_id, ver_index, degrad_tracking, hc, prompt_tgt, alt_prompt_tgt, inpath, orig_audio, original_entry) in enumerate(shard_data):
            dataset_key = f"sample_{idx:05d}_{original_id}_deg{ver_index+1}"
            grp = f.create_group(dataset_key)
            grp.create_dataset('audio', data=audio, compression='gzip', compression_opts=4)
            
            # Create the degraded audio path in HDF5 format
            degraded_audio_path = f"{shard_path}::/{dataset_key}"
            
            # Start with initial information from input JSONL if available
            if original_entry:
                degraded_entry = dict(original_entry)
                if "source_id" not in degraded_entry:
                    degraded_entry["source_id"] = original_id
                if "clean_audio_path" not in degraded_entry:
                    degraded_entry["clean_audio_path"] = inpath
                if "duration" not in degraded_entry:
                    degraded_entry["duration"] = len(orig_audio[0]) / fs
            else:
                degraded_entry = {
                    "source_id": original_id,
                    "clean_audio_path": inpath,
                    "duration": len(orig_audio[0]) / fs
                }
            
            # Update fields that are modified by degradation
            degraded_entry.update({
                "id": f"{original_id}_{deg_spec_selected}",
                "degraded_audio_path": degraded_audio_path,
                "degraded_audio_dataset": dataset_key,
                "degraded_audio_shard": shard_path,
                "prompt": prompt_tgt,
                "alt_prompt": alt_prompt_tgt,
                "degradation_tracking": degrad_tracking,
                "hidden_clipping": hc
            })
            
            # Store metadata as HDF5 attributes
            for key, value in degraded_entry.items():
                if value is None:
                    continue
                if isinstance(value, (list, dict)):
                    grp.attrs[key] = json.dumps(value)
                else:
                    grp.attrs[key] = value
            
            metadata_entries.append(degraded_entry)
    
    # Write metadata to JSONL if outfile is provided and open
    if outfile and not outfile.closed:
        for degraded_entry in metadata_entries:
            json.dump(degraded_entry, outfile)
            outfile.write("\n")
    
    return metadata_entries


def main(fileindex=None):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Apply specific degradation to audio files')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--in_folder', type=str, help='Input folder containing clean audio files')
    input_group.add_argument('--in_jsonl', type=str, help='Input JSONL file with audio paths and metadata')
    parser.add_argument('--out_folder', type=str, required=True, help='Output folder for degraded audio files')
    parser.add_argument('--deg_spec', type=str, required=True, 
                        help='Degradation specification (e.g., punch, clip, comp, bright, etc.)')
    parser.add_argument('--output_format', type=str, default='flac', choices=['flac', 'wav', 'hdf5'],
                        help='Output audio format (default: flac)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: None, process all)')
    parser.add_argument('--use_shards', action='store_true',
                        help='Save degraded audio in HDF5 shards instead of individual files')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='Number of samples per shard when using --use_shards (default: 1000)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output (validates and continues from last valid entry)')
    parser.add_argument('--save_degraded_wav_samples_only', action='store_true',
                        help='Only degrade the first N samples and save them as WAV files, then stop. No shards/JSONL.')
    parser.add_argument('--num_wav_samples', type=int, default=10,
                        help='Number of WAV samples to save when using --save_degraded_wav_samples_only (default: 10)')
    args = parser.parse_args()
    
    timestart=time()

    random.seed(28)
    np.random.seed(28)

    out_folder = args.out_folder
    deg_spec_selected = args.deg_spec
    out_json = os.path.join(out_folder, 'degradation_pairs.jsonl')
    
    # In wav-samples-only mode, force max_samples and skip sharding/resume
    if args.save_degraded_wav_samples_only:
        args.max_samples = args.num_wav_samples
        args.use_shards = False
        args.resume = False
        logging.info(f"WAV samples only mode: will degrade {args.num_wav_samples} samples and save as WAV")

    # Sharding variables
    current_shard_idx = 0
    current_shard_data = []
    shard_size = args.shard_size if args.use_shards else None
    skip_count = 0
    
    # Resume mechanism: check for existing output
    if args.resume and os.path.exists(out_json):
        logging.info(f"Found existing output file: {out_json}")
        
        # Read existing JSONL and validate
        with open(out_json, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if lines:
            # Check if last line is complete JSON
            try:
                json.loads(lines[-1])
                logging.info(f"Last line is valid JSON")
                valid_lines = lines
            except json.JSONDecodeError:
                logging.warning(f"Last line is incomplete, removing it")
                valid_lines = lines[:-1]
            
            if valid_lines:
                # Find the last shard referenced
                last_entry = json.loads(valid_lines[-1])
                if args.use_shards and 'degraded_audio_shard' in last_entry:
                    last_shard_path = last_entry['degraded_audio_shard']
                    import re
                    match = re.search(r'shard_(\d+)\.h5', last_shard_path)
                    if match:
                        last_shard_num = int(match.group(1))
                        
                        # Count how many samples are in the last shard from JSONL
                        samples_in_last_shard = sum(1 for line in valid_lines 
                                                   if f"shard_{last_shard_num:04d}.h5" in line)
                        
                        # If last shard is full, move to next shard
                        if samples_in_last_shard >= args.shard_size:
                            current_shard_idx = last_shard_num + 1
                            logging.info(f"Last shard is full, starting new shard {current_shard_idx}")
                        else:
                            current_shard_idx = last_shard_num + 1  # Will increment to continue filling
                            logging.info(f"Last shard has {samples_in_last_shard} samples, will continue in next batch")
                        
                        # Delete incomplete next shard if it exists
                        next_shard_path = os.path.join(out_folder, f"shard_{current_shard_idx:04d}.h5")
                        if os.path.exists(next_shard_path):
                            logging.warning(f"Deleting incomplete shard: {next_shard_path}")
                            os.remove(next_shard_path)
                        
                        logging.info(f"Resuming from shard index {current_shard_idx}")
                
                # Rewrite JSONL with only valid lines
                with open(out_json, 'w', encoding='utf-8') as f:
                    f.writelines(valid_lines)
                
                skip_count = len(valid_lines)
                logging.info(f"Resuming: skipping first {skip_count} samples already processed")
        else:
            logging.info(f"Empty JSONL file, starting fresh")
    
    # Determine input mode and load audio file list
    if args.in_folder:
        in_folder = args.in_folder
        audio_extensions = ['*.flac', '*.wav', '*.mp3']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob(os.path.join(in_folder, ext)))
        audio_files = sorted(audio_files)
        input_entries = None
        logging.info(f"Input mode: folder ({in_folder})")
    else:
        in_jsonl = args.in_jsonl
        with open(in_jsonl, 'r', encoding='utf-8') as f:
            input_entries = [json.loads(line) for line in f]
        audio_files = [entry['clean_audio_path'] for entry in input_entries]
        logging.info(f"Input mode: JSONL ({in_jsonl})")
        logging.info(f"Loaded {len(input_entries)} entries from JSONL")
    
    # Apply max_samples limit if specified
    if args.max_samples is not None and args.max_samples > 0:
        original_count = len(audio_files)
        audio_files = audio_files[:args.max_samples]
        if input_entries is not None:
            input_entries = input_entries[:args.max_samples]
        logging.info(f"Limiting processing to {len(audio_files)} samples (out of {original_count} total)")

    mic_ir_folder='configs/smallpoli/irs'
    rir_folder='configs/realrirs'
    fs=44100
    stereo_thr=0.08
    
    os.makedirs(out_folder, exist_ok=True)
    samples_folder = os.path.join(out_folder, 'samples')
    os.makedirs(samples_folder, exist_ok=True)
    
    # Collect only valid RIR files (skip directories) and recurse into subfolders
    rir_extensions = ('.wav', '.flac', '.aif', '.aiff', '.ogg')
    rirs = []
    for root, _, files in os.walk(rir_folder):
        for fname in files:
            if fname.lower().endswith(rir_extensions):
                rirs.append(os.path.join(root, fname))
    N_rirs = len(rirs)
    logging.info(f"Discovered {N_rirs} real RIR files under {rir_folder}")
    if N_rirs == 0:
        raise FileNotFoundError(f"No real RIR files found under {rir_folder}. Expected audio files with extensions {rir_extensions}")
    logging.info("Sample RIR paths: %s", rirs[:5])



    degradation_groups = {
        "EQ": {
            "prob": 0.4,
            "options": {
                "xband": 7, "mic": 5, "bright": 3, "dark": 3, "airy": 2,
                "boom": 2, "clarity": 3, "mud": 3, "warm": 3, "vocal": 4
            }
        },
        "Dynamics": {
            "prob": 0.125,
            "options": {
                "comp": 2.5, "punch": 1
            }
        },
        "Reverb": {
            "prob": 0.225,
            "options": {
                "small": 0.15, "big": 0.15, "mix": 0.3, "real": 0.4
            }
        },
        "Amplitude": {
            "prob": 0.125,
            "options": {
                "clip": 3, "volume": 1
            }
        },
        "Stereo": {
            "prob": 0.125,
            "options": {
                "stereo": 1
            }
        }
    }

    # Find deg_group from deg_spec
    deg_group_selected = None
    for group, info in degradation_groups.items():
        if deg_spec_selected in info["options"]:
            deg_group_selected = group
            break
    
    if deg_group_selected is None:
        raise ValueError(f"Degradation spec '{deg_spec_selected}' not found in degradation_groups")
    
    logging.info(f"Selected degradation: {deg_group_selected} - {deg_spec_selected}")

    def choose_degradation(stereo_ok=True,vocal_enable=True):
        # Return the selected degradation from command line args
        return deg_group_selected, deg_spec_selected


    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # Skip already processed samples when resuming
    if skip_count > 0:
        audio_files = audio_files[skip_count:]
        if input_entries is not None:
            input_entries = input_entries[skip_count:]
        logging.info(f"After skipping, {len(audio_files)} samples remaining to process")
    
    # Save first N samples as WAV for verification (all samples in wav-only mode)
    if args.save_degraded_wav_samples_only:
        num_samples = len(audio_files)
    else:
        num_samples = min(10, len(audio_files))
    sample_indices = set(range(num_samples))
    logging.info(f"Will save first {num_samples} samples as WAV in {samples_folder}")
    
    # In wav-samples-only mode, skip JSONL creation entirely
    if args.save_degraded_wav_samples_only:
        outfile = None
    else:
        file_mode = "a" if skip_count > 0 else "w"
        outfile = open(out_json, file_mode, encoding="utf-8")
    
    try:
        song_counter=0
        for idx, inpath in enumerate(audio_files):
            # Get original entry if using JSONL input
            original_entry = input_entries[idx] if input_entries else None
            
            original_id = os.path.basename(inpath).replace('.flac', '').replace('.wav', '').replace('.h5', '').replace('.hdf5', '')
            logging.info(f"Processing {song_counter+1}/{len(audio_files)}: {original_id}")

            vocal_enable=True
            reduce_prompt_reverb=False

            degradation_counts = [1]  # Only one degradation per file
            single_degrad_counter = []

            #load audio
            #get diff std to allow destereo
            try:
                if '::' in inpath:
                    h5_file_path, dataset_name = inpath.split('::')
                    dataset_name = dataset_name.lstrip('/')
                    with h5py.File(h5_file_path, 'r') as h5f:
                        orig_audio = h5f[dataset_name]['audio'][:]
                    sr = fs
                    if orig_audio.ndim == 1:
                        orig_audio = orig_audio.astype(np.float32)
                    else:
                        orig_audio = orig_audio.T.astype(np.float32)
                else:
                    orig_audio, sr = librosa.load(inpath,sr=fs,mono=False)
            except Exception as e:
                logging.error(f"Failed to load audio file {original_id}: {str(e)}")
                song_counter+=1
                continue
                
            if orig_audio.ndim==1: #expand mono to stereo
                orig_audio = np.stack((orig_audio,orig_audio))
            elif orig_audio.shape[0]==1: #expand mono (1, N) to stereo (2, N)
                orig_audio = np.vstack((orig_audio,orig_audio))


            diff_std = np.std(orig_audio[0,:]-orig_audio[1,:])
            if diff_std>stereo_thr:
                stereo_ok=True
            else:
                stereo_ok=False

            for ver_index, count in enumerate(degradation_counts):

                audio=orig_audio
                audio = audio-np.mean(audio,axis=1,keepdims=True) #DC offset
                
                # degradations = set()
                final_prompt=[]
                final_alt_prompt=[]

                degrad_tracking={"EQ": [], "Dynamics": [], "Reverb": [], "Amplitude": [], "Stereo": []}
                try:

                    degrad_groups = set()
                    degrad_specific = set()
                    while len(degrad_groups) < count:
                        # degradations.add(choose_degradation())
                        deg_group, deg_spec = choose_degradation(stereo_ok,vocal_enable)
                        if deg_group not in degrad_groups:
                            degrad_groups.add(deg_group)
                            degrad_specific.add(deg_spec)

                    degrad_groups = list(degrad_groups)
                    degrad_specific = list(degrad_specific)
                    #degrads...
                    
                    while len(degrad_groups)==1 and ((degrad_groups[0]=="Stereo" and "stereo" in single_degrad_counter) or (degrad_specific[0]=="punch" and "punch" in single_degrad_counter) or (degrad_specific[0]=="clarity" and "clarity" in single_degrad_counter) or (degrad_specific[0]=="clip" and "clip" in single_degrad_counter)):
                        deg_group, deg_spec = choose_degradation(stereo_ok,vocal_enable)
                        degrad_groups[0] = deg_group
                        degrad_specific[0] = deg_spec

                    #HIDDEN CLIPPING FOR REVERB AND COMPRESSOR
                    hidden_clipping=False
                    hc=[hidden_clipping,0]
                    if "Amplitude" not in degrad_groups and ("comp" in degrad_specific or "Reverb" in degrad_groups):
                        if random.random()<0.15:
                            hidden_clipping=True


                    if "Dynamics" in degrad_groups:
                        if "comp" in degrad_specific:

                            # threshold_db = random.randint(-40,-35)
                            threshold_db = random.randint(-45,-38)
                            ratio = random.randint(60,450)/10.0
                            manual_gain_db = random.randint(160,250)/10.0
                            attack_ms = random.randint(3,150)
                            release_ms = random.randint(80,250)

                            out_audio = compress_audio_file(
                            audio.T,
                            threshold_db=threshold_db,
                            ratio=ratio,
                            attack_ms=attack_ms,
                            release_ms=release_ms,
                            manual_gain_db=manual_gain_db,
                            fs=fs
                            )

                            audio=out_audio.T

                            prompt,alt_prompt = prompts_compression(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            #check for amplitude - roll for clipping...
                            degrad_tracking["Dynamics"]=["comp",[threshold_db, ratio, attack_ms, release_ms, manual_gain_db]]

                        
                        elif "punch" in degrad_specific:
                            attack_ms=3
                            release_ms=150
                            lookahead_ms=10



                            audio,threshold_db,reduction_db=reduce_punch_auto_stereo(audio, fs, attack_ms=attack_ms, release_ms=release_ms, lookahead_ms=lookahead_ms)


                            prompt,alt_prompt = prompts_punch(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["Dynamics"]=["punch",[threshold_db, reduction_db, attack_ms, release_ms, lookahead_ms]]

                    if "EQ" in degrad_groups:
                        if "xband" in degrad_specific:

                            n_bands = random.randint(8, 12)
                            center_freqs = np.geomspace(40, 16000, n_bands)
                            Q = center_freqs[-1] / (center_freqs[-1]-center_freqs[-2]) / 2
                            gains_db = np.random.randint(-6,7,n_bands)
                            audio = apply_peak_eq(audio, center_freqs, Q, gains_db, fs)

                            prompt,alt_prompt = prompts_xband(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["xband",[n_bands, gains_db.tolist()]]


                        elif "mic" in degrad_specific:
                            mic_number=np.random.randint(20) #choose the mic index
                            audio,phonename = microphone_function(audio,mic_number,mic_ir_folder,fs)

                            prompt,alt_prompt = prompts_mics(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["mic",[mic_number,phonename]]

                        elif "bright" in degrad_specific:
                            gain=random.randint(6,15)
                            audio=reduce_brightness(audio, fs, gain_db=gain)

                            prompt,alt_prompt = prompts_brightness(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["bright",[gain]]

                        elif "dark" in degrad_specific:
                            gain=random.randint(6,15)
                            audio=reduce_darkness(audio, fs, gain_db=gain)

                            prompt,alt_prompt = prompts_darkness(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["dark",[gain]]


                        elif "airy" in degrad_specific:
                            gain=random.randint(10,20)
                            audio=reduce_air(audio, fs, gain_db=gain)

                            prompt,alt_prompt = prompts_airiness(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["airy",[gain]]


                        elif "boom" in degrad_specific: 
                            gain=random.randint(10,20)
                            audio=reduce_boom(audio, fs, gain_db=gain)

                            prompt,alt_prompt = prompts_boominess(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["boom",[gain]]


                        elif "clarity" in degrad_specific:
                            order=random.randint(3,5)
                            audio=remove_clarity(audio, order, fs)

                            prompt,alt_prompt = prompts_clarity(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["clarity",[order]]


                        elif "mud" in degrad_specific:
                            gain=random.randint(6,15)
                            audio=increase_muddiness(audio, fs, gain)

                            prompt,alt_prompt = prompts_muddiness(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["mud",[gain]]

                        elif "warm" in degrad_specific:
                            gain=random.randint(6,20)
                            audio=reduce_warmth(audio, fs, gain_db=gain)

                            prompt,alt_prompt = prompts_warmth(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["warm",[gain]]

                        elif "vocal" in degrad_specific:
                            gain=random.randint(6,20)
                            audio=lower_vocals3(audio, fs, gain) #bug fixed

                            prompt,alt_prompt = prompts_vocals(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["EQ"]=["vocal",[gain]]



                    if "Reverb" in degrad_groups:
                        if "real" not in degrad_specific: #then run pyroomacoustics
                            if "small" in degrad_specific:
                                sim_mode='simple'
                                a=np.array((3,3,2.5))
                                b=np.array((4,6,1.5))
                                absorption = random.random()*0.25+0.05
                                revword="small"
                            elif "big" in degrad_specific:
                                sim_mode='simple'
                                a=np.array((7,8,4))
                                b=np.array((8,10,10))
                                absorption = random.random()*0.25+0.05
                                revword="big"
                            elif "mix" in degrad_specific:
                                sim_mode='mix'
                                a=np.array((4,4,2.5)) #room size
                                b=np.array((4,3,1))
                                revword="mix"

                                how_many_abs_walls=np.random.randint(2)+1
                                which_walls=random.sample(range(6),how_many_abs_walls)

                                absmats=[]
                                for i in range(how_many_abs_walls): #absorptive walls
                                    c=np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.30, 0.35])
                                    d=np.array([0.50, 0.50, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.90, 0.95])
                                    coeffs=(d-c)*np.random.random_sample(len(c))+c
                                    mat = pra.Material(
                                    energy_absorption={
                                        'coeffs': coeffs,
                                        'center_freqs': np.array([125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000])
                                    }
                                    )
                                    absmats.append(mat)

                                refmats=[]
                                for i in range(6-how_many_abs_walls): #reflective walls
                                    c=np.array([0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.13, 0.15, 0.15, 0.17])
                                    d=np.array([0.15, 0.15, 0.15, 0.20, 0.20, 0.22, 0.23, 0.25, 0.25, 0.27])
                                    coeffs=(d-c)*np.random.random_sample(len(c))+c
                                    mat = pra.Material(
                                    energy_absorption={
                                        'coeffs': coeffs,
                                        'center_freqs': np.array([125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000])
                                    }
                                    )
                                    refmats.append(mat)

                                allmats=[]
                                k=0
                                m=0
                                for i in range(6):
                                    if i in which_walls:
                                        allmats.append(absmats[k])
                                        k+=1
                                        continue
                                    else:
                                        allmats.append(refmats[m])
                                        m+=1
                                        continue
                                    allmats.append()
                                
                                absorption = {
                                    'east': allmats[0],
                                    'west': allmats[1],
                                    'north': allmats[2],
                                    'south': allmats[3],
                                    'ceiling': allmats[4],
                                    'floor': allmats[5]
                                }


                            #get room size, source and mic positions
                            room_size=b*np.random.random_sample(3)+a
                            source_position=np.random.random_sample(3)*(0.8*room_size)+0.1*room_size #not too near the walls
                            mic_position=np.random.random_sample(3)*(0.8*room_size)+0.1*room_size #not too near the walls
                            source_position[2]=random.random()*room_size[2]*0.7+0.3 # source not near the ceiling, not on the floor either, at least 30 cm far
                            mic_position[2]=random.random()*room_size[2]*0.5+0.3 # mic not that high, but not near the floor either

                            audio=room_function(audio,room_size,source_position,mic_position,absorption,fs,sim_mode)

                            if revword=="mix":
                                centerfreqs=[125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000]
                                wall_materials_info = []
                                for wall_name, material in absorption.items():
                                    wall_info = {
                                        'wall': wall_name,
                                        'center_freqs': centerfreqs,
                                        'absorption': material.absorption_coeffs.tolist()
                                    }
                                    wall_materials_info.append(wall_info)


                                degrad_tracking["Reverb"]=[revword,[list(room_size),list(source_position),list(mic_position),wall_materials_info]]
                            else:
                                degrad_tracking["Reverb"]=[revword,[list(room_size),list(source_position),list(mic_position),absorption]]

                            

                        elif "real" in degrad_specific:

                            rir_index=np.random.randint(N_rirs)
                            rir_name=os.path.basename(rirs[rir_index]).split('.')[0]
                            ir, fs = librosa.load(rirs[rir_index],sr=44100,mono=False)
                            N=len(audio[0,:])

                            start_index=min(np.argmax(ir,1))
                            ir=ir[:,start_index:]

                            if ir.shape[0]==2:

                                ch1=signal.fftconvolve(audio[0,:],ir[0,:],mode='full')
                                ch2=signal.fftconvolve(audio[1,:],ir[1,:],mode='full')

                                out_audio=np.array((ch1,ch2))
                                audio=out_audio[:,:N]
                                # print(str(rir_name),str(output_file),str(np.max(np.abs(out_audio))))

                            elif ir.shape[0]==4:
                                ch1=audio[0,:]
                                ch2=audio[1,:]
                                convolved1 = [signal.fftconvolve(ch1, ir[i,:], mode='full') for i in range(4)]
                                convolved2 = [signal.fftconvolve(ch2, ir[i,:], mode='full') for i in range(4)]
                                ch1rev = convolved1[0] + 0.5 * convolved1[1] + 0.2 * convolved1[3]
                                ch2rev = convolved2[0] + 0.5 * convolved2[1] + 0.2 * convolved2[3]
                                # ch1rev = 0.6 * convolved1[0] + 0.3 * convolved1[1] + 0.1 * convolved1[3]
                                # ch2rev = 0.6 * convolved2[0] + 0.3 * convolved2[1] + 0.1 * convolved2[3]
                                out_audio=np.array((ch1rev,ch2rev))
                                audio=out_audio[:,:N]

                                # print(str(rir_name),str(output_file),str(np.max(np.abs(out_audio))))

                            degrad_tracking["Reverb"]=["real",[rir_index,rir_name]]

                        if "Stereo" in degrad_groups or "airy" in degrad_specific:
                            reduce_prompt_reverb=True
                        prompt,alt_prompt = prompts_reverb(2,reduce_prompt_reverb)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        reduce_prompt_reverb=False


                        


                    if "Stereo" in degrad_groups:
                        audio=destereo_audio(audio)

                        prompt,alt_prompt = prompts_stereo(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["Stereo"]=["stereo",["combined channels"]]

                    if "Amplitude" in degrad_groups:
                        if "clip" in degrad_specific:
                            clip_opts=[2,3,5]
                            # clip_int=random.randint(3,10)
                            clip_int=random.choice(clip_opts)
                            audio=clip_audio_choice(audio,clip_int)

                            prompt,alt_prompt = prompts_clipping(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["Amplitude"]=["clip",[clip_int]]

                        if "volume" in degrad_specific:
                            # vol_mult=random.randint(1,5)/100 #0.01 to 0.05
                            # vol_mult=random.randint(1,10)/1000 #0.001 to 0.01
                            vol_opts=[0.001, 0.003, 0.01, 0.05]
                            vol_mult=random.choice(vol_opts)

                            audio=lower_volume(audio,vol_mult) 

                            prompt,alt_prompt = prompts_volume(2)
                            final_prompt.append(prompt)
                            final_alt_prompt.append(alt_prompt)
                            degrad_tracking["Amplitude"]=["volume",[vol_mult]]

                    audio = audio-np.mean(audio,axis=1,keepdims=True) #DC offset

                    amp=np.max(np.abs(audio))
                    if hidden_clipping:
                        if amp>5:
                            hid_clip_int=5
                        elif amp>3:
                            hid_clip_int=3
                        else:
                            hid_clip_int=2
                        audio=clip_audio_choice(audio,hid_clip_int)
                        hc=[hidden_clipping,hid_clip_int]



                    #NORMALIZE unless amplitude degrad...
                    elif "Amplitude" not in degrad_groups and not hidden_clipping:
                        if amp>1:
                            norm_factor = random.randint(80,100)/100
                            audio = normalize(audio)*norm_factor
                        # put between 0.8 and 1.0, model will normalize all inputs when loading

                    # Save WAV sample after degradation for verification
                    if idx in sample_indices:
                        sample_wav_name = os.path.join(samples_folder, f"{original_id}_deg{ver_index+1}.wav")
                        sf.write(sample_wav_name, audio.T, samplerate=fs, format='WAV')
                        logging.info(f"Saved degraded sample WAV: {sample_wav_name}")

                    # In wav-samples-only mode, skip all output writing
                    if args.save_degraded_wav_samples_only:
                        continue

                    
                    # Generate prompts before using them
                    if len(degrad_groups)==1:
                        single_degrad_counter.append(degrad_specific[0])

                    random.shuffle(final_prompt) # shuffle the sentence order
                    random.shuffle(final_alt_prompt) # shuffle the sentence order
                    prompt_tgt=""
                    alt_prompt_tgt=""
                    for sentence in final_prompt: # connect the sentences
                        prompt_tgt += sentence + " "
                    for sentence in final_alt_prompt: # connect the sentences
                        alt_prompt_tgt += sentence + " "

                    prompt_tgt=prompt_tgt[:-1] #to remove the final space character
                    alt_prompt_tgt=alt_prompt_tgt[:-1] #to remove the final space character

                    if args.use_shards:
                        # Store audio data for later shard writing
                        current_shard_data.append((audio, original_id, ver_index, degrad_tracking, hc, prompt_tgt, alt_prompt_tgt, inpath, orig_audio, original_entry))
                        audio_out_name = None  # Will be set when shard is written
                    elif args.output_format == 'hdf5':
                        audio_out_name=os.path.join(out_folder,f"{original_id}_deg{ver_index+1}"+".h5")
                        with h5py.File(audio_out_name, 'w') as f:
                            f.create_dataset('audio', data=audio, compression='gzip')
                    else:
                        audio_out_name=os.path.join(out_folder,f"{original_id}_deg{ver_index+1}"+f".{args.output_format}")
                        sf.write(audio_out_name, audio.T, samplerate=fs, format=args.output_format.upper())

                    # Only write to JSONL if not using shards (shards will write later)
                    if not args.use_shards:
                        # Start with initial information from input JSONL if available
                        if original_entry:
                            degraded_entry = dict(original_entry)
                            # Only set these if not already present
                            if "source_id" not in degraded_entry:
                                degraded_entry["source_id"] = original_id
                            if "clean_audio_path" not in degraded_entry:
                                degraded_entry["clean_audio_path"] = inpath
                            if "duration" not in degraded_entry:
                                degraded_entry["duration"] = len(orig_audio[0]) / fs
                        else:
                            degraded_entry = {
                                "source_id": original_id,
                                "clean_audio_path": inpath,
                                "duration": len(orig_audio[0]) / fs
                            }
                        
                        # Update fields that are modified by degradation
                        degraded_entry.update({
                            "id": f"{original_id}_{deg_spec_selected}",
                            "degraded_audio_path": audio_out_name,
                            "prompt": prompt_tgt,
                            "alt_prompt": alt_prompt_tgt,
                            "degradation_tracking": degrad_tracking,
                            "hidden_clipping": hc
                        })

                        json.dump(degraded_entry, outfile)
                        outfile.write("\n")
                    
                    # Check if we need to save a shard
                    if args.use_shards and len(current_shard_data) >= shard_size:
                        save_shard(out_folder, current_shard_idx, current_shard_data, outfile, fs, deg_spec_selected)
                        current_shard_idx += 1
                        current_shard_data = []


                except Exception as e:
                    print('ERROR'+str(e))

            song_counter+=1
    
        # Save any remaining shard data
        if args.use_shards and current_shard_data:
            save_shard(out_folder, current_shard_idx, current_shard_data, outfile, fs, deg_spec_selected)
    finally:
        if outfile is not None and not outfile.closed:
            outfile.close()

    elapsed = time()-timestart
    logging.info(f"Processing completed in {elapsed:.2f} seconds")
    logging.info(f"Degraded audio saved to: {out_folder}")
    if not args.save_degraded_wav_samples_only:
        logging.info(f"JSONL file saved to: {out_json}")
    if args.use_shards:
        logging.info(f"Saved {current_shard_idx + (1 if current_shard_data else 0)} shards")

if __name__ == "__main__":
    main()