"""
Feature extraction for click detection.
Short-time energy, high-frequency content, spectral features, TKEO.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Optional, Tuple


class FeatureExtractor:
    """Extracts features for click detection."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 window_ms: float = 1.0,
                 step_ms: float = 0.25,
                 bandpass_low: float = 2000,
                 bandpass_high: float = 20000):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sample rate in Hz
            window_ms: Window size in milliseconds
            step_ms: Step size in milliseconds
            bandpass_low: Low cutoff for bandpass (Hz)
            bandpass_high: High cutoff for bandpass (Hz)
        """
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000)
        self.step_samples = int(step_ms * sample_rate / 1000)
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        
    def _frame_signal(self, audio: np.ndarray) -> np.ndarray:
        """
        Frame signal into overlapping windows.
        
        Args:
            audio: Input audio signal
            
        Returns:
            2D array of frames (n_frames, window_samples)
        """
        if len(audio) < self.window_samples:
            return audio.reshape(1, -1)
            
        n_frames = (len(audio) - self.window_samples) // self.step_samples + 1
        frames = np.zeros((n_frames, self.window_samples))
        
        for i in range(n_frames):
            start = i * self.step_samples
            end = start + self.window_samples
            frames[i] = audio[start:end]
            
        return frames
        
    def compute_ste(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Energy.
        
        Args:
            audio: Input audio signal
            
        Returns:
            STE values
        """
        frames = self._frame_signal(audio)
        ste = np.mean(frames**2, axis=1)
        return ste
        
    def compute_tkeo(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Teager-Kaiser Energy Operator.
        
        Args:
            audio: Input audio signal
            
        Returns:
            TKEO values per frame
        """
        frames = self._frame_signal(audio)
        tkeo_frames = []
        
        for frame in frames:
            # TKEO[n] = x[n]^2 - x[n-1]*x[n+1]
            tkeo = frame[1:-1]**2 - frame[:-2] * frame[2:]
            tkeo_frames.append(np.mean(tkeo))
            
        return np.array(tkeo_frames)
        
    def compute_hfc(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute High-Frequency Content.
        
        Args:
            audio: Input audio signal
            
        Returns:
            HFC values
        """
        frames = self._frame_signal(audio)
        hfc = []
        
        for frame in frames:
            # Compute FFT
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            magnitude = np.abs(fft)
            
            # Weight by frequency squared
            freqs = np.arange(len(magnitude))
            hfc_value = np.sum(magnitude * freqs**2)
            hfc.append(hfc_value)
            
        return np.array(hfc)
        
    def compute_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral centroid.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Spectral centroid values in Hz
        """
        frames = self._frame_signal(audio)
        centroids = []
        
        for frame in frames:
            # Compute FFT
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            magnitude = np.abs(fft)
            
            # Compute frequency bins
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            # Centroid = sum(f * mag) / sum(mag)
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            centroids.append(centroid)
            
        return np.array(centroids)
        
    def compute_spectral_flatness(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral flatness (Wiener entropy).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Spectral flatness values
        """
        frames = self._frame_signal(audio)
        flatness = []
        
        for frame in frames:
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            magnitude = np.abs(fft) + 1e-10  # Avoid log(0)
            
            # Geometric mean / Arithmetic mean
            geo_mean = np.exp(np.mean(np.log(magnitude)))
            arith_mean = np.mean(magnitude)
            
            if arith_mean > 0:
                flat = geo_mean / arith_mean
            else:
                flat = 0
            flatness.append(flat)
            
        return np.array(flatness)
        
    def compute_high_low_freq_ratio(self, audio: np.ndarray,
                                    split_freq: float = 12000) -> np.ndarray:
        """
        Compute ratio of high-frequency to low-frequency energy.
        
        Args:
            audio: Input audio signal
            split_freq: Frequency splitting point (Hz)
            
        Returns:
            High/low frequency energy ratio
        """
        frames = self._frame_signal(audio)
        ratios = []
        
        for frame in frames:
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            magnitude = np.abs(fft)**2
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            # Split at specified frequency
            low_mask = freqs < split_freq
            high_mask = freqs >= split_freq
            
            low_energy = np.sum(magnitude[low_mask])
            high_energy = np.sum(magnitude[high_mask])
            
            if low_energy > 0:
                ratio = high_energy / low_energy
            else:
                ratio = 0
            ratios.append(ratio)
            
        return np.array(ratios)
        
    def compute_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral flux (change in spectrum).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Spectral flux values
        """
        frames = self._frame_signal(audio)
        flux = [0]  # First frame has no previous frame
        
        prev_magnitude = None
        for frame in frames:
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            magnitude = np.abs(fft)
            
            if prev_magnitude is not None:
                # Sum of squared differences
                flux_value = np.sum((magnitude - prev_magnitude)**2)
                flux.append(flux_value)
            
            prev_magnitude = magnitude
            
        return np.array(flux)
        
    def robust_normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Robust normalization using median and MAD.
        z_robust = (x - median) / MAD
        
        Args:
            values: Input values
            
        Returns:
            Normalized values
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad > 0:
            z_robust = (values - median) / (1.4826 * mad)
        else:
            z_robust = np.zeros_like(values)
            
        return z_robust
        
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of feature arrays
        """
        features = {
            'ste': self.compute_ste(audio),
            'tkeo': self.compute_tkeo(audio),
            'hfc': self.compute_hfc(audio),
            'spectral_centroid': self.compute_spectral_centroid(audio),
            'spectral_flatness': self.compute_spectral_flatness(audio),
            'high_low_ratio': self.compute_high_low_freq_ratio(audio),
            'spectral_flux': self.compute_spectral_flux(audio)
        }
        
        # Add robust normalized versions
        features['ste_z'] = self.robust_normalize(features['ste'])
        features['tkeo_z'] = self.robust_normalize(features['tkeo'])
        features['hfc_z'] = self.robust_normalize(features['hfc'])
        
        return features
        
    def get_frame_times(self, audio_length: int) -> np.ndarray:
        """
        Get time values for each frame.
        
        Args:
            audio_length: Length of audio signal in samples
            
        Returns:
            Array of frame times in seconds
        """
        n_frames = (audio_length - self.window_samples) // self.step_samples + 1
        times = np.arange(n_frames) * self.step_samples / self.sample_rate
        return times