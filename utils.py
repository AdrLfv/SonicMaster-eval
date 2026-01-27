import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import torchaudio
import random
import itertools
import numpy as np
import h5py
import os

# Punch prompts for random selection
PROMPTS = [
    "Give this song a punch!",
    "Make the transients sharper, please.",
    "Increase the punchiness of the song by emphasizing the transients.",
    "Make the audio more punchy and energetic.",
    "Bring back the snap and attack of transients.",
    "Add more impact and dynamic punch to the sound.",
    "Make drums and hits sound more aggressive and tight.",
    "Increase the percussive clarity and definition."
]


import numpy as np


def sample_linear_plus_uniform(
    batch_size,
    skew_toward: str = "start",  # "start" = u=0, "end" = u=1
    uniform_weight=0.5,
    device=None
):
    """
    Sample u ∈ [0,1] from a mixture of:
    - uniform distribution
    - linear distribution: p(u) ∝ (1 - u) or u

    skew_toward:
        "start" → p(u) ∝ (1 - u) → higher density near 0
        "end"   → p(u) ∝ u       → higher density near 1
    """

    # Uniform component
    u_uniform = torch.rand(batch_size, device=device)

    # Linear distribution via inverse CDF sampling
    u_base = torch.rand(batch_size, device=device)
    if skew_toward == "start":
        u_linear = 1 - torch.sqrt(1 - u_base)
    elif skew_toward == "end":
        u_linear = torch.sqrt(u_base)
    else:
        raise ValueError("skew_toward must be 'start' or 'end'")

    # Mixture of uniform and linear
    mix_mask = torch.rand(batch_size, device=device) < uniform_weight
    u = torch.where(mix_mask, u_uniform, u_linear)

    return u


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        padded_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, padded_wav])
        return waveform


def read_wav_file(filename, duration_sec):
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.h5', '.hdf5']:
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
        
        sr = 44100
    else:
        try:
            info = torchaudio.info(filename)
            sample_rate = info.sample_rate

            # Calculate the number of frames corresponding to the desired duration
            num_frames = int(sample_rate * duration_sec)

            waveform, sr = torchaudio.load(filename, num_frames=num_frames)  # Faster!!!
        except Exception as e:
            raise FileNotFoundError(f"Error reading audio file '{filename}': {str(e)}")

    if waveform.shape[0] == 2:  ## Stereo audio
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
            resampled_waveform = resampler(waveform)
        else:
            resampled_waveform = waveform
        # print(resampled_waveform.shape)
        padded_left = pad_wav(
            resampled_waveform[0], int(44100 * duration_sec)
        )  ## We pad left and right seperately
        padded_right = pad_wav(resampled_waveform[1], int(44100 * duration_sec))

        return torch.stack([padded_left, padded_right])
    else:
        if sr != 44100:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=44100
            )[0]
        else:
            waveform = waveform[0]
        waveform = pad_wav(waveform, int(44100 * duration_sec)).unsqueeze(0)

        return waveform


class DPOText2AudioDataset(Dataset):
    def __init__(
        self,
        dataset,
        prefix,
        text_column,
        audio_w_column,
        audio_l_column,
        duration,
        num_examples=-1,
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios_w = list(dataset[audio_w_column])
        self.audios_l = list(dataset[audio_l_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio_w, audio_l, duration, text in zip(
            self.indices, self.audios_w, self.audios_l, self.durations, inputs
        ):
            self.mapper[index] = [audio_w, audio_l, duration, text]

        if num_examples != -1:
            self.inputs, self.audios_w, self.audios_l, self.durations = (
                self.inputs[:num_examples],
                self.audios_w[:num_examples],
                self.audios_l[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4, s5 = (
            self.inputs[index],
            self.audios_w[index],
            self.audios_l[index],
            self.durations[index],
            self.indices[index],
        )
        return s1, s2, s3, s4, s5

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


class Text2AudioDataset(Dataset):
    def __init__(
        self, dataset, prefix, text_column, alt_text_column, audio_column, deg_audio_column, duration, num_examples=-1, deg_latent_column=None
    ):

        inputs = list(dataset[text_column])
        # Randomly select a punch prompt for each sample
        # inputs = [random.choice(PROMPTS) for _ in range(len(dataset[audio_column]))]
        self.inputs = [prefix + inp for inp in inputs]
        alt_inputs = list(dataset[alt_text_column])
        # Randomly select a different punch prompt for each sample
        # alt_inputs = [random.choice(PROMPTS) for _ in range(len(dataset[audio_column]))]
        self.alt_inputs = [prefix + inp for inp in alt_inputs]
        self.audios = list(dataset[audio_column])
        self.deg_audios = list(dataset[deg_audio_column])
        
        # Support for preencoded degraded latents
        if deg_latent_column and deg_latent_column in dataset.column_names:
            self.deg_latents = list(dataset[deg_latent_column])
        else:
            self.deg_latents = None
        
        # self.durations = list(dataset[duration])
        self.durations = [30]*len(self.inputs)
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, deg_audio, duration, text, alt_text in zip(
            self.indices, self.audios, self.deg_audios, self.durations, inputs, alt_inputs
        ):
            self.mapper[index] = [audio, deg_audio, text, alt_text, duration]

        if num_examples != -1:
            self.inputs, self.alt_inputs, self.audios, self.deg_audios, self.durations = (
                self.inputs[:num_examples],
                self.alt_inputs[:num_examples],
                self.audios[:num_examples],
                self.deg_audios[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]
            if self.deg_latents:
                self.deg_latents = self.deg_latents[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.deg_latents:
            s1, s2, s3, s4, s5, s6, s7 = (
                self.inputs[index],
                self.alt_inputs[index],
                self.audios[index],
                self.deg_audios[index],
                self.durations[index],
                self.indices[index],
                self.deg_latents[index],
            )
            return s1, s2, s3, s4, s5, s6, s7
        else:
            s1, s2, s3, s4, s5, s6 = (
                self.inputs[index],
                self.alt_inputs[index],
                self.audios[index],
                self.deg_audios[index],
                self.durations[index],
                self.indices[index],
            )
            return s1, s2, s3, s4, s5, s6

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
