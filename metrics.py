import torch
import torchaudio
import numpy as np
import librosa

from torch.nn import functional as F
from torch import nn
from typing import Tuple

### Metrics are loss-like functions that do not backpropagate gradients.

class PESQMetric(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.resampler = (
            torchaudio.transforms.Resample(sample_rate, 16000)
            if sample_rate != 16000 else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        from pypesq import pesq
        
        if self.resampler is not None:
            inputs = self.resampler(inputs)
            targets = self.resampler(targets)

        inputs_np = inputs.cpu().numpy().astype("float64")
        targets_np = targets.cpu().numpy().astype("float64")
        batch_size = targets.shape[0]

        # Compute average pesq across batch size.
        val_pesq = (1.0 / batch_size) * sum(
            pesq(targets_np[i].reshape(-1), inputs_np[i].reshape(-1), 16000)
            for i in range(batch_size))
        return val_pesq
    

class LogSpectralDistance(nn.Module):
    """
    Log-Spectral Distance metric as nn.Module for training/validation.
    For batch file processing, use the standalone log_spectral_distance() function.
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Pre-create window and register as buffer so it moves with the module
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window.to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=self.window.to(targets.device),
                                 return_complex=True)

        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)

        diff = 10 * (torch.log10(input_mag + 1e-8) - torch.log10(target_mag + 1e-8))
        lsd = torch.sqrt(torch.mean(diff ** 2, dim=(1, 2)))  # Mean over F and T
        return torch.mean(lsd)  # Average over batch

class LTASDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Pre-create window and register as buffer so it moves with the module
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window.to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=self.window.to(targets.device),
                                 return_complex=True)

        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)

        input_ltas = torch.mean(input_mag, dim=2)  # Mean over time
        target_ltas = torch.mean(target_mag, dim=2)

        ltas_dist = torch.mean(torch.abs(input_ltas - target_ltas) / (target_ltas + 1e-8), dim=1)
        return torch.mean(10 * torch.log10(ltas_dist + 1e-8))  # Average over batch

class SISDRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets - torch.mean(targets, dim=-1, keepdim=True)
        inputs = inputs - torch.mean(inputs, dim=-1, keepdim=True)

        alpha = torch.sum(inputs * targets, dim=-1, keepdim=True) / (
            torch.sum(targets ** 2, dim=-1, keepdim=True) + 1e-8)
        s_target = alpha * targets
        e_noise = inputs - s_target

        sisdr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8))
        return torch.mean(sisdr)

class SNRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        noise = inputs - targets
        signal_power = torch.sum(targets ** 2, dim=-1)
        noise_power = torch.sum(noise ** 2, dim=-1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return torch.mean(snr)

class STFTDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Pre-create window and register as buffer so it moves with the module
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window.to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=self.window.to(targets.device),
                                 return_complex=True)

        dist = torch.abs(input_stft - target_stft)
        return torch.mean(torch.sqrt(torch.sum(dist ** 2, dim=(1, 2))))  # L2 norm then mean


class RobustMelDistance(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0, # Use magnitude, not power, for cleaner log conversion
            center=True
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute Mel Spectrograms
        input_mel = self.mel_transform(inputs)
        target_mel = self.mel_transform(targets)

        # 2. Robust Log conversion
        # Use 1e-5 (-100dB) to ignore inaudible digital silence differences
        input_log_mel = torch.log10(torch.clamp(input_mel, min=1e-5))
        target_log_mel = torch.log10(torch.clamp(target_mel, min=1e-5))

        # 3. Normalize by size (MSE-like) instead of Sum
        # This makes the metric independent of audio length
        # dist squared
        diff = input_log_mel - target_log_mel
        mse_loss = (diff ** 2).mean() # Mean over Batch, Freq, Time
        
        # Return RMSE (Root Mean Squared Error) in Log Domain
        return torch.sqrt(mse_loss)


class MelDistance(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_mel = self.mel_transform(inputs)
        target_mel = self.mel_transform(targets)

        # Convert to log-mel
        input_log_mel = torch.log10(input_mel + 1e-8)
        target_log_mel = torch.log10(target_mel + 1e-8)

        # Compute L2 norm of log-mel differences
        dist = torch.abs(input_log_mel - target_log_mel)
        return torch.mean(torch.sqrt(torch.sum(dist ** 2, dim=(1, 2))))
    

### Standalone metric functions for batch processing from files

def multi_mel_snr(reference, prediction, sr=48000):
    """
    Compute Multi-Mel-SNR between reference and prediction.
    Used for audio quality evaluation.
    
    Args:
        reference: Reference audio (torch.Tensor or numpy array)
        prediction: Predicted audio (torch.Tensor or numpy array)
        sr: Sample rate
    
    Returns:
        Average SNR across three mel configurations
    """
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()
    
    # Scale-invariant normalization
    alpha = torch.dot(reference, prediction) / (torch.dot(prediction, prediction) + 1e-8)
    prediction = alpha * prediction
    
    # Three mel configurations
    configs = [
        (512, 256, 80),    # (n_fft, hop_length, n_mels)
        (1024, 512, 128),
        (2048, 1024, 192)
    ]
    
    snrs = []
    for n_fft, hop, n_mels in configs:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, 
            n_mels=n_mels, f_min=0, f_max=24000, power=2.0
        )
        M_ref = mel(reference)
        M_pred = mel(prediction)
        snr = 10 * torch.log10(M_ref.pow(2).sum() / ((M_ref - M_pred).pow(2).sum() + 1e-8))
        snrs.append(snr.item())
    
    return sum(snrs) / len(snrs)


def log_spectral_distance(reference, prediction, sr=44100, n_fft=2048, hop_length=512):
    """
    Compute Log-Spectral Distance (LSD) between reference and prediction.
    This is the metric used in BABE2 paper for objective evaluation.
    
    Standard LSD formula: average over frames of sqrt(mean over frequencies of squared log-magnitude differences)
    
    Args:
        reference: Reference audio (torch.Tensor or numpy array)
        prediction: Predicted audio (torch.Tensor or numpy array)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
    
    Returns:
        LSD value in dB (lower is better)
    """
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()
    
    # Ensure 1D
    if reference.ndim > 1:
        reference = reference.squeeze()
    if prediction.ndim > 1:
        prediction = prediction.squeeze()
    
    # Compute STFT
    window = torch.hann_window(n_fft)
    ref_stft = torch.stft(reference, n_fft=n_fft, hop_length=hop_length,
                          window=window, return_complex=True)
    pred_stft = torch.stft(prediction, n_fft=n_fft, hop_length=hop_length,
                           window=window, return_complex=True)
    
    # Compute magnitude spectra (freq_bins x time_frames)
    ref_mag = torch.abs(ref_stft)
    pred_mag = torch.abs(pred_stft)
    
    # Compute LSD per frame: sqrt(mean over frequencies of squared log differences)
    # Standard formula: LSD = (1/T) * sum_t sqrt((1/K) * sum_k (log10(X[k,t]) - log10(Y[k,t]))^2)
    log_ref = torch.log10(ref_mag + 1e-8)
    log_pred = torch.log10(pred_mag + 1e-8)
    
    # Squared difference per frequency bin
    squared_diff = (log_ref - log_pred) ** 2
    
    # Mean over frequency bins, then sqrt, then mean over time frames
    lsd_per_frame = torch.sqrt(torch.mean(squared_diff, dim=0))  # Mean over freq bins
    lsd = torch.mean(lsd_per_frame)  # Mean over time frames
    
    return lsd.item()


def get_clap_embeddings(file_paths, model, processor, device, batch_size=16, sr=44100):
    """
    Get CLAP embeddings for FAD-CLAP calculation.
    CLAP model requires 48kHz audio.
    
    Args:
        file_paths: List of audio file paths
        model: CLAP model
        processor: CLAP processor
        device: torch device
        batch_size: Batch size for processing
        sr: Sample rate of input files
    
    Returns:
        Numpy array of embeddings
    """
    import soundfile as sf
    from tqdm import tqdm
    
    model.to(device)
    all_embeddings = []
    
    target_sr = 48000  # CLAP requires 48kHz
    
    if sr != target_sr:
        print(f"Warning: Audio files are at {sr} Hz but CLAP model requires 48000 Hz.")
        print("Resampling to 48kHz for FAD-CLAP calculation.")
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating embeddings", ncols=100, leave=False):
        batch_paths = file_paths[i:i+batch_size]
        audio_batch = []
        for path in batch_paths:
            try:
                wav, file_sr = sf.read(path)
                
                # Resample to 48kHz if needed
                if file_sr != target_sr:
                    from scipy import signal
                    num_samples = int(len(wav) * target_sr / file_sr)
                    wav = signal.resample(wav, num_samples)
                
                # Convert to mono if stereo
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                
                audio_batch.append(wav)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
        
        if not audio_batch:
            continue
        
        try:
            inputs = processor(audios=audio_batch, sampling_rate=target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)
            
            all_embeddings.append(audio_features.cpu().numpy())
        except Exception as e:
            print(f"Warning: Failed to process batch: {e}")
            continue
    
    if not all_embeddings:
        return np.array([])
    
    return np.concatenate(all_embeddings, axis=0)


def get_panns_embeddings(file_paths, device='cuda', batch_size=16, sr=32000):
    """
    Get PANNS embeddings for Fréchet Distance calculation (as used in BABE2 paper).
    PANNS uses 32kHz audio.
    
    Args:
        file_paths: List of audio file paths
        device: torch device
        batch_size: Batch size for processing
        sr: Sample rate of input files
    
    Returns:
        Numpy array of embeddings
    """
    import soundfile as sf
    from tqdm import tqdm
    
    try:
        from panns_inference import AudioTagging
    except ImportError:
        print("Error: PANNS not available. Install with: pip install panns-inference")
        return np.array([])
    
    # Initialize PANNS model
    at = AudioTagging(checkpoint_path=None, device=device)
    at.model.eval()
    
    all_embeddings = []
    target_sr = 32000  # PANNS uses 32kHz
    
    if sr != target_sr:
        print(f"Warning: Audio files are at {sr} Hz but PANNS requires 32000 Hz.")
        print("Resampling to 32kHz for FD calculation.")
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="  Calculating PANNS embeddings", ncols=100, leave=False):
        batch_paths = file_paths[i:i+batch_size]
        
        for path in batch_paths:
            try:
                wav, file_sr = sf.read(path)
                
                # Convert to mono if stereo
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                
                # Resample to 32kHz if needed
                if file_sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=file_sr, target_sr=target_sr)
                
                # Get embedding from PANNS
                with torch.no_grad():
                    _, embedding = at.inference(wav[np.newaxis, :])
                
                all_embeddings.append(embedding)
                
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
                continue
    
    if not all_embeddings:
        return np.array([])
    
    return np.concatenate(all_embeddings, axis=0)


def calculate_frechet_distance(embeddings1, embeddings2):
    """
    Calculate Fréchet Distance between two sets of embeddings.
    Used for both FAD-CLAP and FD-PANNS.
    
    Args:
        embeddings1: First set of embeddings (numpy array)
        embeddings2: Second set of embeddings (numpy array)
    
    Returns:
        Fréchet distance value (lower is better)
    """
    from scipy.linalg import sqrtm
    
    if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
        return None

    mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
    sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        return None

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fd_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fd_score
