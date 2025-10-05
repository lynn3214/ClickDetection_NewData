"""
Data augmentation pipeline for training samples.
Includes SNR mixing, time shifting, amplitude scaling, and optional EQ/reverb.
"""

import numpy as np
from typing import Optional, List
import random


class AugmentationPipeline:
    """Applies data augmentation to training samples."""
    
    def __init__(self,
                 sample_rate: int = 44100,
                 snr_range: tuple = (-5, 5),
                 time_shift_ms: float = 10.0,
                 amplitude_range: tuple = (0.8, 1.25),
                 apply_prob: float = 0.8):
        """
        Initialize augmentation pipeline.
        
        Args:
            sample_rate: Sample rate (Hz)
            snr_range: SNR range for mixing (dB)
            time_shift_ms: Maximum time shift (ms)
            amplitude_range: Amplitude scaling range
            apply_prob: Probability of applying augmentation
        """
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.time_shift_samples = int(time_shift_ms * sample_rate / 1000)
        self.amplitude_range = amplitude_range
        self.apply_prob = apply_prob
        
    def augment(self,
               signal: np.ndarray,
               noise: Optional[np.ndarray] = None,
               apply_snr_mix: bool = True,
               apply_time_shift: bool = True,
               apply_amplitude: bool = True) -> np.ndarray:
        """
        Apply augmentation pipeline.
        
        Args:
            signal: Input signal
            noise: Optional noise signal for SNR mixing
            apply_snr_mix: Whether to apply SNR mixing
            apply_time_shift: Whether to apply time shifting
            apply_amplitude: Whether to apply amplitude scaling
            
        Returns:
            Augmented signal
        """
        augmented = signal.copy()
        
        # SNR mixing
        if apply_snr_mix and noise is not None and random.random() < self.apply_prob:
            augmented = self.snr_mix(augmented, noise)
            
        # Time shifting
        if apply_time_shift and random.random() < self.apply_prob:
            augmented = self.time_shift(augmented)
            
        # Amplitude scaling
        if apply_amplitude and random.random() < self.apply_prob:
            augmented = self.amplitude_scale(augmented)
            
        return augmented
        
    def snr_mix(self,
               signal: np.ndarray,
               noise: np.ndarray,
               target_snr: Optional[float] = None) -> np.ndarray:
        """
        Mix signal with noise at specified SNR.
        
        Args:
            signal: Clean signal
            noise: Noise signal
            target_snr: Target SNR in dB (random if None)
            
        Returns:
            Mixed signal
        """
        if target_snr is None:
            target_snr = random.uniform(*self.snr_range)
            
        # Extract noise segment of same length
        if len(noise) > len(signal):
            start = random.randint(0, len(noise) - len(signal))
            noise_segment = noise[start:start + len(signal)]
        else:
            # Repeat noise if too short
            repeats = int(np.ceil(len(signal) / len(noise)))
            noise_segment = np.tile(noise, repeats)[:len(signal)]
            
        # Calculate signal and noise powers
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise_segment**2)
        
        # Calculate scaling factor for desired SNR
        if noise_power > 0:
            snr_linear = 10**(target_snr / 10)
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        else:
            noise_scale = 0
            
        # Mix
        mixed = signal + noise_scale * noise_segment
        
        return mixed
        
    def time_shift(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random time shift to signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Time-shifted signal
        """
        shift = random.randint(-self.time_shift_samples, self.time_shift_samples)
        
        if shift == 0:
            return signal
            
        shifted = np.zeros_like(signal)
        
        if shift > 0:
            # Shift right
            shifted[shift:] = signal[:-shift]
            # Fill left with reflection
            shifted[:shift] = signal[0]
        else:
            # Shift left
            shifted[:shift] = signal[-shift:]
            # Fill right with reflection
            shifted[shift:] = signal[-1]
            
        return shifted
        
    def amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random amplitude scaling.
        
        Args:
            signal: Input signal
            
        Returns:
            Scaled signal
        """
        scale = random.uniform(*self.amplitude_range)
        return signal * scale
        
    def apply_simple_eq(self,
                       signal: np.ndarray,
                       low_gain: float = None,
                       high_gain: float = None) -> np.ndarray:
        """
        Apply simple 2-band EQ (optional enhancement).
        
        Args:
            signal: Input signal
            low_gain: Low frequency gain (dB), random if None
            high_gain: High frequency gain (dB), random if None
            
        Returns:
            EQ'd signal
        """
        if low_gain is None:
            low_gain = random.uniform(-3, 3)
        if high_gain is None:
            high_gain = random.uniform(-3, 3)
            
        # Simple frequency domain EQ
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.sample_rate)
        
        # Split at 5kHz
        low_mask = freqs < 5000
        high_mask = freqs >= 5000
        
        # Apply gains
        fft[low_mask] *= 10**(low_gain / 20)
        fft[high_mask] *= 10**(high_gain / 20)
        
        # Back to time domain
        eq_signal = np.fft.irfft(fft, n=len(signal))
        
        return eq_signal
        
    def apply_light_reverb(self,
                          signal: np.ndarray,
                          decay: float = 0.3,
                          delay_ms: float = 20.0) -> np.ndarray:
        """
        Apply light reverb effect (optional enhancement).
        
        Args:
            signal: Input signal
            decay: Decay factor (0-1)
            delay_ms: Delay time (ms)
            
        Returns:
            Reverberated signal
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        if delay_samples >= len(signal):
            return signal
            
        reverb = signal.copy()
        reverb[delay_samples:] += decay * signal[:-delay_samples]
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(reverb))
        if peak > 1.0:
            reverb = reverb / peak
            
        return reverb