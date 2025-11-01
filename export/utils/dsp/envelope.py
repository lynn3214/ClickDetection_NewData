"""
Envelope extraction and TKEO utilities.
"""

import numpy as np
from scipy.signal import hilbert
from typing import Tuple


def compute_hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """
    Compute Hilbert envelope of signal.
    
    Args:
        audio: Input audio signal
        
    Returns:
        Envelope signal
    """
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)
    return envelope


def measure_envelope_width(envelope: np.ndarray,
                          peak_idx: int,
                          sample_rate: int,
                          db_threshold: float = -10) -> float:
    """
    Measure envelope width at specified dB level below peak.
    
    Args:
        envelope: Envelope signal
        peak_idx: Index of peak
        sample_rate: Sample rate (Hz)
        db_threshold: Threshold in dB below peak
        
    Returns:
        Width in milliseconds
    """
    if peak_idx < 0 or peak_idx >= len(envelope):
        return 0.0
        
    peak_value = envelope[peak_idx]
    if peak_value <= 0:
        return 0.0
        
    # Convert dB threshold to linear
    threshold = peak_value * 10**(db_threshold/20)
    
    # Find left crossing
    left_idx = peak_idx
    while left_idx > 0 and envelope[left_idx] > threshold:
        left_idx -= 1
        
    # Find right crossing
    right_idx = peak_idx
    while right_idx < len(envelope) - 1 and envelope[right_idx] > threshold:
        right_idx += 1
        
    # Calculate width in milliseconds
    width_samples = right_idx - left_idx
    width_ms = (width_samples / sample_rate) * 1000
    
    return width_ms


def compute_teager_kaiser(audio: np.ndarray) -> np.ndarray:
    """
    Compute Teager-Kaiser Energy Operator.
    TKEO[n] = x[n]^2 - x[n-1] * x[n+1]
    
    Args:
        audio: Input audio signal
        
    Returns:
        TKEO signal (length = len(audio) - 2)
    """
    if len(audio) < 3:
        return np.array([])
        
    tkeo = audio[1:-1]**2 - audio[:-2] * audio[2:]
    return tkeo


def smooth_tkeo(tkeo: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth TKEO signal using moving average.
    
    Args:
        tkeo: TKEO signal
        window_size: Smoothing window size
        
    Returns:
        Smoothed TKEO
    """
    if len(tkeo) < window_size:
        return tkeo
        
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(tkeo, kernel, mode='same')
    return smoothed


def compute_peak_factor(audio: np.ndarray) -> float:
    """
    Compute peak factor (crest factor) in dB.
    Peak factor = 20 * log10(peak / rms)
    
    Args:
        audio: Input audio signal
        
    Returns:
        Peak factor in dB
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    if rms > 0:
        return 20 * np.log10(peak / rms)
    else:
        return 0.0


def compute_energy_ratio(audio: np.ndarray,
                        sample_rate: int,
                        short_ms: float = 1.0,
                        long_ms: float = 5.0) -> float:
    """
    Compute ratio of short-term to long-term energy.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate (Hz)
        short_ms: Short window duration (ms)
        long_ms: Long window duration (ms)
        
    Returns:
        Energy ratio
    """
    short_samples = int(short_ms * sample_rate / 1000)
    long_samples = int(long_ms * sample_rate / 1000)
    
    if len(audio) < long_samples:
        return 0.0
        
    # Find peak
    peak_idx = np.argmax(np.abs(audio))
    
    # Short window centered on peak
    short_start = max(0, peak_idx - short_samples // 2)
    short_end = min(len(audio), short_start + short_samples)
    short_energy = np.sum(audio[short_start:short_end]**2)
    
    # Long window centered on peak
    long_start = max(0, peak_idx - long_samples // 2)
    long_end = min(len(audio), long_start + long_samples)
    long_energy = np.sum(audio[long_start:long_end]**2)
    
    if long_energy > 0:
        return short_energy / long_energy
    else:
        return 0.0


