import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
import librosa
import math
import logging

logger = logging.getLogger(__name__)

from scipy.signal import butter, lfilter, cheby2, sosfilt
from scipy.interpolate import interp1d

import scipy.signal as signal
import soundfile as sf
from glob import glob
import os
try:
    import pyroomacoustics as pra
except ImportError:
    pra = None

from compressor import FeedForwardCompressor




def normalize(audio):
    return audio / np.max(np.abs(audio))

def clip_audio(audio):
    return np.clip(audio, -1.0, 1.0)

def clip_audio_choice(audio,amplitude):
    return np.clip(amplitude*normalize(audio), -1.0, 1.0)

def destereo_audio(audio):
    return np.stack((np.sum(audio,0),np.sum(audio,0)))

def lower_volume(audio,amplitude):
    return amplitude*normalize(audio)

def shelf_filter(audio, sr, freq, gain_db, shelf_type='low', Q=0.707):
    """
    Apply a low-shelf or high-shelf filter to the audio signal.
    
    Parameters:
        audio (np.ndarray): Input audio signal
        sr (int): Sample rate
        freq (float): Corner frequency in Hz
        gain_db (float): Gain in dB (positive to boost, negative to cut)
        shelf_type (str): 'low' for low-shelf, 'high' for high-shelf
        Q (float): Quality factor (default = 0.707)
    
    Returns:
        np.ndarray: Filtered audio signal
    """
    assert shelf_type in ['low', 'high'], "shelf_type must be 'low' or 'high'"

    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    
    if shelf_type == 'low':
        b0 =    A*((A+1)-(A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 =  2*A*((A-1)-(A+1)*cos_w0)
        b2 =    A*((A+1)-(A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a0 =       (A+1)+(A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*cos_w0)
        a2 =       (A+1)+(A-1)*cos_w0 - 2*np.sqrt(A)*alpha
    else:  # high shelf
        b0 =    A*((A+1)+(A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*cos_w0)
        b2 =    A*((A+1)+(A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a0 =       (A+1)-(A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        a1 =  2*((A-1)-(A+1)*cos_w0)
        a2 =       (A+1)-(A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sosfilt(sos, audio)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    # print('Applied a low-pass filter.')
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)
    
# def lower_vocals(audio, fs, atten):
#     print('Lowered vocals by cutting mid frequencies (1-4 kHz).')
#     # samples_float = samples.astype(np.float32) / 32768.0
#     sos = cheby2(4, atten, [1000, 4000], 'bandstop', fs=fs, output='sos') #choose input params
#     filtered = sosfilt(sos, audio) # add orig+filtered?
#     return filtered

# def lower_vocals2(audio, fs, atten):
#     print('Lowered vocals by cutting mid frequencies (1-4 kHz).')
#     # samples_float = samples.astype(np.float32) / 32768.0
#     sos = cheby2(2, atten, [500, 3500], 'bandstop', fs=fs, output='sos') #choose input params
#     filtered = sosfilt(sos, audio) # add orig+filtered?
#     return filtered

def lower_vocals3(audio, fs, atten):
    # print('Lowered vocals by cutting mid frequencies (1-4 kHz).')
    # samples_float = samples.astype(np.float32) / 32768.0
    sos = cheby2(2, atten, [350, 3500], 'bandstop', fs=fs, output='sos') #choose input params
    filtered = sosfilt(sos, audio) # add orig+filtered?
    return filtered

def increase_muddiness(audio, fs, gain):
    # print('Increased muddiness by boosting low-mid frequencies (200-500 Hz).')
    # samples_float = samples.astype(np.float32) / 32768.0
    sos = cheby2(2, gain, [200, 500], 'bandpass', fs=fs, output='sos') #choose input params
    filtered = sosfilt(sos, audio)
    # mixed = 0.8 * audio + 0.2 * filtered
    # mixed = 0.2 * audio + 0.8 * filtered
    return filtered

def remove_clarity(audio, order, fs):
    # print('Reduced clarity by applying a low-pass filter to dull high frequencies.')
    # samples_float = samples.astype(np.float32) / 32768.0
    cutoff_freq = 4000
    filtered = lowpass_filter(audio, cutoff_freq, fs, order)
    return filtered

def reduce_punch(audio, fs): #TODO REPLACE? just lowpass filter with 8k/10k cutoff?
    # print('Reduced punch by softening transients with a low-pass filter.')
    # samples_float = samples.astype(np.float32) / 32768.0
    cutoff_freq = 2000
    smoothed = lowpass_filter(audio, cutoff_freq, fs)
    threshold = 0.7 * np.max(np.abs(smoothed))
    compressed = np.where(np.abs(smoothed) > threshold,
                          smoothed * 0.8,
                          smoothed)
    return compressed

def reduce_punch_auto(
    audio, sample_rate, attack_ms=5, release_ms=50, lookahead_ms=3
):
    """
    Adaptive punch reduction: auto threshold and gain reduction based on transient stats.

    Parameters:
        audio (np.ndarray): Mono audio signal (-1 to 1)
        sample_rate (int): Sample rate in Hz
        attack_ms (float): Envelope attack time in ms
        release_ms (float): Envelope release time in ms
        lookahead_ms (float): Lookahead time in ms

    Returns:
        np.ndarray: Processed audio
    """

    # Convert to absolute value for envelope
    attack = np.exp(-1.0 / (sample_rate * (attack_ms / 1000)))
    release = np.exp(-1.0 / (sample_rate * (release_ms / 1000)))
    lookahead_samples = int(sample_rate * lookahead_ms / 1000)

    padded_audio = np.concatenate([np.zeros(lookahead_samples), audio])
    envelope = np.zeros_like(padded_audio)

    gain = 0.0
    for i, sample in enumerate(np.abs(padded_audio)):
        if sample > gain:
            gain = attack * (gain - sample) + sample
        else:
            gain = release * (gain - sample) + sample
        envelope[i] = gain

    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-6)
    median_db = np.median(envelope_db)
    percentile_99_db = np.percentile(envelope_db, 99)

    # Adaptive threshold & reduction
    threshold_db = 0.7 * percentile_99_db + 0.3 * median_db
    reduction_db = min(max((percentile_99_db - median_db) * 0.5, 8), 15)
    reduction_lin = 10 ** (-reduction_db / 20)

    # Apply gain reduction where envelope exceeds threshold
    transient_mask = envelope_db > threshold_db
    processed = np.copy(audio)
    for i in range(len(audio)):
        idx = i + lookahead_samples
        if idx < len(transient_mask) and transient_mask[idx]:
            processed[i] *= reduction_lin

    # return processed
    return processed, threshold_db, reduction_db


def reduce_punch_auto_stereo(
    audio, sample_rate, attack_ms=5, release_ms=50, lookahead_ms=3
):
    """
    Adaptive punch reduction for stereo: auto threshold and gain reduction based on transient stats.

    Parameters:
        audio (np.ndarray): Stereo audio signal (-1 to 1), shape (2, N)
        sample_rate (int): Sample rate in Hz
        attack_ms (float): Envelope attack time in ms
        release_ms (float): Envelope release time in ms
        lookahead_ms (float): Lookahead time in ms

    Returns:
        np.ndarray: Processed stereo audio
        float: threshold in dB
        float: gain reduction in dB
    """
    assert audio.ndim == 2 and audio.shape[0] == 2, "Audio must be stereo with shape (2, N)"
    
    # Convert times to coefficients
    attack = np.exp(-1.0 / (sample_rate * (attack_ms / 1000)))
    release = np.exp(-1.0 / (sample_rate * (release_ms / 1000)))
    lookahead_samples = int(sample_rate * lookahead_ms / 1000)

    # Mono mixdown for envelope analysis
    mono_audio = np.mean(audio, axis=0)
    padded_audio = np.concatenate([np.zeros(lookahead_samples), mono_audio])
    envelope = np.zeros_like(padded_audio)

    # Envelope follower
    gain = 0.0
    for i, sample in enumerate(np.abs(padded_audio)):
        if sample > gain:
            gain = attack * (gain - sample) + sample
        else:
            gain = release * (gain - sample) + sample
        envelope[i] = gain

    # Envelope to dB
    envelope_db = 20 * np.log10(envelope + 1e-6)
    median_db = np.median(envelope_db)
    percentile_99_db = np.percentile(envelope_db, 99)

    # Adaptive threshold & reduction
    threshold_db = 0.7 * percentile_99_db + 0.3 * median_db
    reduction_db = min(max((percentile_99_db - median_db) * 0.7, 8), 15)
    reduction_lin = 10 ** (-reduction_db / 20)

    # Apply gain reduction where envelope exceeds threshold
    transient_mask = envelope_db > threshold_db
    processed = np.copy(audio)
    for ch in range(2):
        for i in range(audio.shape[1]):
            idx = i + lookahead_samples
            if idx < len(transient_mask) and transient_mask[idx]:
                processed[ch, i] *= reduction_lin

    return processed, threshold_db, reduction_db



def reduce_brightness(audio, fs, gain_db=6): #-6 to -12?
    gain_db = -1*gain_db # ---
    freq=6000
    filtered=shelf_filter(audio, fs, freq, gain_db, shelf_type='high', Q=0.707)
    return filtered

def reduce_darkness(audio, fs, gain_db=6): #+6 to +12?
    freq=6000
    filtered=shelf_filter(audio, fs, freq, gain_db, shelf_type='high', Q=0.707)
    return filtered

def reduce_warmth(audio, fs, gain_db=6): #-6 to -12?
    gain_db = -1*gain_db # ---
    freq=400
    filtered=shelf_filter(audio, fs, freq, gain_db, shelf_type='low', Q=0.707)
    return filtered

def reduce_boom(audio, fs, gain_db=6): #-6 to -12?
    gain_db = -1*gain_db # ---
    freq=120
    filtered=shelf_filter(audio, fs, freq, gain_db, shelf_type='low', Q=0.707)
    return filtered

def reduce_air(audio, fs, gain_db=6): #-6 to -12?
    gain_db = -1*gain_db # ---
    freq=10000
    filtered=shelf_filter(audio, fs, freq, gain_db, shelf_type='high', Q=0.707)
    return filtered


def microphone_function(input_audio,mic_number,ir_folder,fs):
    # fs=44100

    files=glob(ir_folder+'/*')
    files = [f for f in files if os.path.isfile(f) and f.endswith('.npy')]
    
    if not files:
        raise FileNotFoundError(f"No .npy files found in {ir_folder}")
    
    mic_number = mic_number % len(files)
    filename=files[mic_number]
    phonename=os.path.basename(filename)[:-4] #remove .npy extension

    impulse_response=np.load(filename, allow_pickle=False)
    impulse_response /= np.max(np.abs(impulse_response)) #use or not use?

    #detect peak to start from - eliminating time shifts
    impulse_response=impulse_response[np.argmax(impulse_response):]

    ch1=signal.fftconvolve(input_audio[0,:],impulse_response,mode='full')
    ch2=signal.fftconvolve(input_audio[1,:],impulse_response,mode='full')
    out_audio=np.array((ch1,ch2))

    return out_audio,phonename


def room_function(input_audio,room_size,source_pos,mic_pos,absorption,fs,sim_mode='simple'):

    max_order = 10
    if sim_mode=='simple':
        room = pra.ShoeBox(room_size, absorption=absorption, max_order=max_order, fs=fs)
    elif sim_mode=='mix':
        room = pra.ShoeBox(room_size, materials=absorption, max_order=max_order, fs=fs)


    room.add_source(source_pos)
    room.add_microphone_array(np.array([mic_pos]).T)
    room.compute_rir()

    rir = room.rir[0][0]  # First mic, first source

    rir=rir[np.argmax(rir):]

    ch1=signal.fftconvolve(input_audio[0,:],rir,mode='full')
    ch2=signal.fftconvolve(input_audio[1,:],rir,mode='full')
    out_audio=np.array((ch1,ch2))


    return out_audio

def eightband_eq_bandpass(input_audio,coeffs,fs):
    # bands = [(100, 300), (300, 1000), (1000, 4000), (4000, 8000)]
    bands = [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 4000), (4000, 6000), (6000, 10000), (10000, 20000)] #(1000, 2000)?
    modified = input_audio.copy()
    i=0
    for low, high in bands:
        # gain = np.random.uniform(-6, 6)
        gain=coeffs[i] # input dB adjustements
        sos = cheby2(4, 20, [low, high], 'bandpass', fs=sample_rate, output='sos')
        filtered = sosfilt(sos, modified)
        filtered = filtered*np.power(10,(gain/20))
        modified += filtered
        i+=1
        #modified += (1+gain) * filtered # let's dB it

    return out_audio



def db_to_gain(db):
    return 10 ** (db / 20)

def peaking_eq(f0, Q, gain_db, fs):
    """
    Create a second-order peaking EQ filter as a second-order section (SOS).
    
    f0: Center frequency (Hz)
    Q: Quality factor
    gain_db: Gain in dB
    fs: Sampling frequency (Hz)
    """
    A = db_to_gain(gain_db)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Normalize coefficients
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    # Convert to SOS format (for stability)
    # Format: [b0, b1, b2, a0, a1, a2]
    sos = np.array([b[0], b[1], b[2], a[0], a[1], a[2]])
    return sos

def apply_peak_eq(signal, bands, Q, gains_db, fs):
    """
    Apply 8-band equalizer to the signal.
    
    signal: input audio (1D numpy array)
    fs: sample rate in Hz
    gains_db: list or array of 8 gain values (in dB) for each band
    """
    # 8 standard band center frequencies (Hz) - common 1/3 octave bands
    # bands = [60, 170, 310, 600, 1000, 3000, 6000, 12000]
    # Q = 1.0  # bandwidth quality factor, tweak for wider/narrower bands
    
    # Convert signal to float32 for filtering stability
    filtered = signal.astype(np.float32)

    for gain_db, f0 in zip(gains_db, bands):
        sos = peaking_eq(f0, Q, gain_db, fs)
        # sosfilt expects sos in shape (n_sections, 6), so reshape accordingly
        sos = sos.reshape(1, 6)
        filtered = sosfilt(sos, filtered)

    return filtered


def coeff_exponential(ms, sample_rate=44100.0):
    time_in_samples = ms * sample_rate / 1000.0
    return math.exp(-1.0 / time_in_samples)


def compress_audio_file(
    audio,
    threshold_db=-18.0,
    ratio=4.0,
    attack_ms=5.0,
    release_ms=50.0,
    manual_gain_db=None,
    fs=44100
):
    # audio, fs = sf.read(filepath, always_2d=True)

    if audio.shape[1] < 2:
        raise ValueError("Input must be stereo. Mono files need to be converted before processing.")

    compressor = FeedForwardCompressor()
    compressor.init(fs, 2)
    compressor.set_threshold(threshold_db)
    compressor.set_ratio(ratio)
    compressor.set_attack_time(attack_ms)
    compressor.set_release_time(release_ms)

    compressed = compressor.process(audio)
    
    if manual_gain_db is not None:
        gain_factor = 10.0 ** (manual_gain_db / 20.0)
        compressed *= gain_factor
        # compressed = np.clip(compressed, -1.0, 1.0)

    return compressed


# ============================================================================
# NEW DEGRADATION FUNCTIONS — targeting real-world degradations
# ============================================================================

def _generate_colored_noise(n_samples, color='white'):
    """Generate colored noise in frequency domain.
    white=flat, pink=1/f, brown=1/f^2.
    """
    white = np.fft.rfft(np.random.randn(n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1.0  # avoid division by zero

    if color == 'pink':
        white /= np.sqrt(freqs)
    elif color == 'brown':
        white /= freqs

    noise = np.fft.irfft(white, n=n_samples)
    noise /= (np.max(np.abs(noise)) + 1e-12)
    return noise


def add_noise(audio, sample_rate, snr_db, noise_color='white'):
    """Add colored noise at a given SNR (dB).
    audio: shape (2, N) stereo or (N,) mono.
    noise_color: 'white', 'pink', or 'brown'.
    """
    if audio.ndim == 1:
        n_samples = len(audio)
        sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-12
        noise_rms_target = sig_rms / (10 ** (snr_db / 20.0))
        noise = _generate_colored_noise(n_samples, noise_color)
        noise = noise * noise_rms_target
        return audio + noise
    else:
        out = np.copy(audio)
        for ch in range(audio.shape[0]):
            sig_rms = np.sqrt(np.mean(audio[ch] ** 2)) + 1e-12
            noise_rms_target = sig_rms / (10 ** (snr_db / 20.0))
            noise = _generate_colored_noise(audio.shape[1], noise_color)
            noise = noise * noise_rms_target
            out[ch] += noise
        return out


def add_crackle(audio, sample_rate, density=0.001, amplitude_range=(0.3, 1.0)):
    """Simulate vinyl/tape impulsive noise (clicks and pops).
    density: fraction of samples that are click centers.
    amplitude_range: (min, max) amplitude of clicks relative to signal peak.
    """
    peak = np.max(np.abs(audio)) + 1e-12
    # Short decaying envelope for each click (~0.5ms)
    click_len = max(int(0.0005 * sample_rate), 4)
    envelope = np.exp(-np.linspace(0, 5, click_len))

    if audio.ndim == 1:
        n_samples = len(audio)
    else:
        n_samples = audio.shape[1]

    n_clicks = max(1, int(n_samples * density))
    positions = np.random.randint(0, n_samples - click_len, size=n_clicks)
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=n_clicks)
    polarities = np.random.choice([-1, 1], size=n_clicks)

    crackle = np.zeros(n_samples)
    for pos, amp, pol in zip(positions, amplitudes, polarities):
        crackle[pos:pos + click_len] += pol * amp * peak * envelope

    if audio.ndim == 1:
        return audio + crackle
    else:
        return audio + crackle[np.newaxis, :]


def highpass_filter(audio, sample_rate, cutoff_hz, order=4):
    """Butterworth high-pass filter. Handles mono and stereo."""
    nyq = 0.5 * sample_rate
    normalized = cutoff_hz / nyq
    if normalized >= 1.0:
        normalized = 0.99
    sos = butter(order, normalized, btype='high', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    else:
        out = np.copy(audio)
        for ch in range(audio.shape[0]):
            out[ch] = sosfilt(sos, audio[ch])
        return out


def telephone_filter(audio, sample_rate, low_hz=300, high_hz=3400, order=4):
    """Bandpass filter simulating telephone/radio bandwidth."""
    nyq = 0.5 * sample_rate
    low = low_hz / nyq
    high = high_hz / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    sos = butter(order, [low, high], btype='band', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    else:
        out = np.copy(audio)
        for ch in range(audio.shape[0]):
            out[ch] = sosfilt(sos, audio[ch])
        return out


def simulate_low_samplerate(audio, sample_rate, target_sr):
    """Downsample to target_sr then upsample back to original sample_rate.
    Simulates aliasing/anti-aliasing artifacts of low-quality ADC/DAC.
    """
    if target_sr >= sample_rate:
        return audio
    if audio.ndim == 1:
        down = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr, res_type='soxr_hq')
        up = librosa.resample(down, orig_sr=target_sr, target_sr=sample_rate, res_type='soxr_hq')
        # Match length
        if len(up) > len(audio):
            up = up[:len(audio)]
        elif len(up) < len(audio):
            up = np.pad(up, (0, len(audio) - len(up)))
        return up
    else:
        out = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            down = librosa.resample(audio[ch], orig_sr=sample_rate, target_sr=target_sr, res_type='soxr_hq')
            up = librosa.resample(down, orig_sr=target_sr, target_sr=sample_rate, res_type='soxr_hq')
            if len(up) > audio.shape[1]:
                up = up[:audio.shape[1]]
            elif len(up) < audio.shape[1]:
                up = np.pad(up, (0, audio.shape[1] - len(up)))
            out[ch] = up
        return out


def soft_clip(audio, drive=2.0, character='tanh'):
    """Soft saturation/clipping using a nonlinear transfer function.
    drive: pre-gain before nonlinearity (1.0=unity, higher=more saturation).
    character: 'tanh' (smooth), 'atan' (slightly harder), 'cubic' (asymmetric harmonics).
    Normalizes output to preserve original peak level.
    """
    peak_before = np.max(np.abs(audio)) + 1e-12
    driven = audio * drive

    if character == 'tanh':
        out = np.tanh(driven)
    elif character == 'atan':
        out = (2.0 / np.pi) * np.arctan(driven)
    elif character == 'cubic':
        out = driven - (driven ** 3) / 3.0
        out = np.clip(out, -1.0, 1.0)
    else:
        out = np.tanh(driven)

    peak_after = np.max(np.abs(out)) + 1e-12
    out = out * (peak_before / peak_after)
    return out


def add_hum(audio, sample_rate, frequency=50.0, harmonics=3, amplitude_db=-40):
    """Add electrical hum (50Hz or 60Hz) with harmonics.
    Each harmonic is 6dB quieter than the previous.
    """
    if audio.ndim == 1:
        n_samples = len(audio)
    else:
        n_samples = audio.shape[1]

    t = np.arange(n_samples) / sample_rate
    hum = np.zeros(n_samples)
    for h in range(1, harmonics + 1):
        amp = 10 ** ((amplitude_db - 6 * (h - 1)) / 20.0)
        phase = np.random.uniform(0, 2 * np.pi)
        hum += amp * np.sin(2 * np.pi * frequency * h * t + phase)

    if audio.ndim == 1:
        return audio + hum
    else:
        return audio + hum[np.newaxis, :]


def mp3_artifact_simulation(audio, sample_rate, bitrate_kbps=64):
    """Simulate MP3 compression artifacts (perceptual approximation).
    Applies bandwidth limitation matching the bitrate cutoff, adds subtle
    pre-echo simulation and shaped quantization noise.
    """
    # Approximate frequency cutoff for common bitrates
    cutoff_map = {32: 8000, 48: 10000, 64: 12000, 96: 15000, 128: 16000, 192: 18000, 256: 20000}
    closest_br = min(cutoff_map.keys(), key=lambda k: abs(k - bitrate_kbps))
    cutoff = cutoff_map[closest_br]

    # 1. Low-pass filter at cutoff
    nyq = 0.5 * sample_rate
    if cutoff < nyq:
        norm_cutoff = cutoff / nyq
        sos = butter(5, norm_cutoff, btype='low', output='sos')
        if audio.ndim == 1:
            filtered = sosfilt(sos, audio)
        else:
            filtered = np.copy(audio)
            for ch in range(audio.shape[0]):
                filtered[ch] = sosfilt(sos, audio[ch])
    else:
        filtered = np.copy(audio)

    # 2. Add subtle shaped quantization noise (scaled by bitrate)
    noise_level = 0.001 * (128.0 / max(bitrate_kbps, 32))
    if filtered.ndim == 1:
        qnoise = noise_level * np.random.randn(len(filtered))
        # Shape noise to be more prominent in high frequencies
        qnoise = np.diff(np.concatenate([[0], qnoise]))
        filtered = filtered + qnoise
    else:
        for ch in range(filtered.shape[0]):
            qnoise = noise_level * np.random.randn(filtered.shape[1])
            qnoise = np.diff(np.concatenate([[0], qnoise]))
            filtered[ch] += qnoise

    return filtered


def pitch_instability(audio, sample_rate, rate_hz=2.0, depth_cents=10.0):
    """Simulate tape wow & flutter: slow pitch modulation via sinusoidal LFO.
    rate_hz: LFO frequency in Hz.
    depth_cents: pitch deviation in cents.
    Same modulation on both channels for realism.
    """
    if audio.ndim == 1:
        n_samples = len(audio)
    else:
        n_samples = audio.shape[1]

    t = np.arange(n_samples) / sample_rate
    # LFO: modulate the read position
    # depth_cents → fractional speed deviation: 2^(cents/1200) - 1
    max_deviation = 2.0 ** (depth_cents / 1200.0) - 1.0
    lfo = max_deviation * np.sin(2 * np.pi * rate_hz * t)
    # Cumulative phase modulation → warped read indices
    warped = np.arange(n_samples) + np.cumsum(lfo)
    warped = np.clip(warped, 0, n_samples - 1)

    if audio.ndim == 1:
        interpolator = interp1d(np.arange(n_samples), audio, kind='linear',
                                fill_value='extrapolate')
        return interpolator(warped).astype(audio.dtype)
    else:
        out = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            interpolator = interp1d(np.arange(n_samples), audio[ch], kind='linear',
                                    fill_value='extrapolate')
            out[ch] = interpolator(warped)
        return out


def dc_offset(audio, offset=0.05):
    """Add a fixed DC offset to the signal.
    offset: value in [-1, 1] added to all samples.
    """
    return audio + offset