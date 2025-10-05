"""
Event-level feature extraction and quality scoring.
"""

import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import csv

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from utils.dsp.envelope import compute_peak_factor, compute_energy_ratio


class EventStatsExtractor:
    """Extracts event-level statistics and quality scores."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize event stats extractor.
        
        Args:
            sample_rate: Sample rate (Hz)
        """
        self.sample_rate = sample_rate
        
    def extract_event_stats(self,
                           audio: np.ndarray,
                           candidate: ClickCandidate) -> Dict[str, Any]:
        """
        Extract comprehensive event statistics.
        
        Args:
            audio: Full audio signal
            candidate: Click candidate
            
        Returns:
            Dictionary of event statistics
        """
        peak_idx = candidate.peak_idx
        
        # Extract local window around peak (20ms)
        window_samples = int(0.020 * self.sample_rate)
        start = max(0, peak_idx - window_samples // 2)
        end = min(len(audio), start + window_samples)
        segment = audio[start:end]
        
        # Basic stats
        peak_amplitude = np.max(np.abs(segment))
        rms = np.sqrt(np.mean(segment**2))
        snr_estimate = 20 * np.log10(peak_amplitude / (rms + 1e-10))
        
        # Peak factor
        peak_factor = compute_peak_factor(segment)
        
        # Energy ratio (short/long)
        energy_ratio = compute_energy_ratio(
            segment, self.sample_rate, 
            short_ms=1.0, long_ms=5.0
        )
        
        # Zero crossing rate
        zcr = self._compute_zcr(segment)
        
        # Spectral features
        spectral_stats = self._compute_spectral_stats(segment)
        
        # Envelope width (already in candidate)
        envelope_width = candidate.envelope_width
        
        # Quality score
        quality_score = self._compute_quality_score(
            peak_factor, energy_ratio, envelope_width,
            candidate.spectral_centroid, candidate.high_low_ratio
        )
        
        stats = {
            'peak_idx': peak_idx,
            'peak_time': candidate.peak_time,
            'peak_amplitude': float(peak_amplitude),
            'rms': float(rms),
            'snr_estimate': float(snr_estimate),
            'peak_factor': float(peak_factor),
            'energy_ratio': float(energy_ratio),
            'zcr': float(zcr),
            'envelope_width': float(envelope_width),
            'spectral_centroid': float(candidate.spectral_centroid),
            'high_low_ratio': float(candidate.high_low_ratio),
            'tkeo_value': float(candidate.tkeo_value),
            'ste_value': float(candidate.ste_value),
            'hfc_value': float(candidate.hfc_value),
            'confidence_score': float(candidate.confidence_score),
            'quality_score': float(quality_score),
            **spectral_stats
        }
        
        return stats
        
    def _compute_zcr(self, signal: np.ndarray) -> float:
        """
        Compute zero crossing rate.
        
        Args:
            signal: Input signal
            
        Returns:
            Zero crossing rate (normalized)
        """
        signs = np.sign(signal)
        zero_crossings = np.sum(np.abs(np.diff(signs))) / 2
        return zero_crossings / len(signal)
        
    def _compute_spectral_stats(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral statistics.
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of spectral features
        """
        # FFT
        fft = np.fft.rfft(signal * np.hanning(len(signal)))
        magnitude = np.abs(fft)
        power = magnitude ** 2
        freqs = np.fft.rfftfreq(len(signal), 1/self.sample_rate)
        
        # Spectral centroid (already have, but recompute for consistency)
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
            
        # Spectral bandwidth
        if np.sum(magnitude) > 0:
            bandwidth = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
            )
        else:
            bandwidth = 0
            
        # Spectral rolloff (95%)
        cumsum = np.cumsum(power)
        rolloff_threshold = 0.95 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Spectral flatness
        geo_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arith_mean = np.mean(magnitude)
        flatness = geo_mean / (arith_mean + 1e-10)
        
        return {
            'spectral_bandwidth': float(bandwidth),
            'spectral_rolloff': float(rolloff),
            'spectral_flatness': float(flatness)
        }
        
    def _compute_quality_score(self,
                              peak_factor: float,
                              energy_ratio: float,
                              envelope_width: float,
                              spectral_centroid: float,
                              high_low_ratio: float) -> float:
        """
        Compute overall quality score for event.
        
        Args:
            peak_factor: Peak factor (dB)
            energy_ratio: Short/long energy ratio
            envelope_width: Envelope width (ms)
            spectral_centroid: Spectral centroid (Hz)
            high_low_ratio: High/low frequency ratio
            
        Returns:
            Quality score (0-1)
        """
        # Peak factor contribution (target: 10-30 dB)
        pf_score = np.clip((peak_factor - 10) / 20, 0, 1)
        
        # Energy ratio contribution (target: > 1.8)
        er_score = np.clip(energy_ratio / 3.0, 0, 1)
        
        # Envelope width contribution (target: 0.2-1.8 ms)
        ew_ideal = 0.8
        ew_score = 1.0 - np.clip(abs(envelope_width - ew_ideal) / 1.5, 0, 1)
        
        # Spectral centroid contribution (target: > 8500 Hz)
        sc_score = np.clip((spectral_centroid - 5000) / 10000, 0, 1)
        
        # High/low ratio contribution (target: > 1.2)
        hl_score = np.clip(high_low_ratio / 2.0, 0, 1)
        
        # Weighted combination
        quality = (
            0.25 * pf_score +
            0.25 * er_score +
            0.20 * ew_score +
            0.15 * sc_score +
            0.15 * hl_score
        )
        
        return quality
        

def save_event_stats_csv(stats_list: List[Dict[str, Any]],
                        output_path: Path) -> None:
    """
    Save event statistics to CSV.
    
    Args:
        stats_list: List of event statistics dictionaries
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not stats_list:
        return
        
    fieldnames = list(stats_list[0].keys())
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_list)