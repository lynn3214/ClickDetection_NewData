"""
Adaptive threshold detector for dolphin clicks.
Multi-feature consistency triggering with file-internal normalization.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from detection.rules.features import FeatureExtractor
from utils.dsp.envelope import compute_hilbert_envelope, measure_envelope_width
from utils.dsp.teager import TeagerKaiserOperator


@dataclass
class DetectionParams:
    """Parameters for adaptive detection."""
    tkeo_threshold: float = 6.0  # Robust z-score threshold
    ste_threshold: float = 6.0
    hfc_threshold: float = 4.0
    high_low_ratio_threshold: float = 1.2
    envelope_width_min: float = 0.2  # ms
    envelope_width_max: float = 1.8  # ms
    spectral_centroid_min: float = 8500  # Hz
    refractory_ms: float = 1.5  # Minimum time between clicks
    

@dataclass
class ClickCandidate:
    """Detected click candidate."""
    peak_idx: int
    peak_time: float
    tkeo_value: float
    ste_value: float
    hfc_value: float
    spectral_centroid: float
    high_low_ratio: float
    envelope_width: float
    confidence_score: float


class AdaptiveDetector:
    """Adaptive click detector using multi-feature consensus."""
    
    def __init__(self,
                 sample_rate: int = 44100,
                 params: DetectionParams = None):
        """
        Initialize adaptive detector.
        
        Args:
            sample_rate: Sample rate in Hz
            params: Detection parameters
        """
        self.sample_rate = sample_rate
        self.params = params or DetectionParams()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            window_ms=1.0,
            step_ms=0.25
        )
        
        # Initialize TKEO
        self.tkeo_operator = TeagerKaiserOperator(smooth_window=5)
        
    def detect_clicks(self, audio: np.ndarray) -> List[ClickCandidate]:
        """
        Detect click candidates in audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            List of click candidates
        """
        # Extract all features
        features = self.feature_extractor.extract_all_features(audio)
        
        # Compute envelope
        envelope = compute_hilbert_envelope(audio)
        
        # Find candidate peaks using TKEO
        tkeo_peaks = self._find_tkeo_peaks(features['tkeo_z'])
        
        # Evaluate each candidate
        candidates = []
        for peak_idx in tkeo_peaks:
            candidate = self._evaluate_candidate(
                peak_idx, audio, features, envelope
            )
            if candidate is not None:
                candidates.append(candidate)
                
        # Apply refractory period
        candidates = self._apply_refractory_period(candidates)
        
        return candidates
        
    def _find_tkeo_peaks(self, tkeo_z: np.ndarray) -> np.ndarray:
        """
        Find peaks in normalized TKEO that exceed threshold.
        
        Args:
            tkeo_z: Normalized TKEO values
            
        Returns:
            Array of peak frame indices
        """
        # Simple peak detection: above threshold and local maximum
        above_threshold = tkeo_z > self.params.tkeo_threshold
        peaks = []
        
        for i in range(1, len(tkeo_z) - 1):
            if (above_threshold[i] and 
                tkeo_z[i] > tkeo_z[i-1] and 
                tkeo_z[i] > tkeo_z[i+1]):
                peaks.append(i)
                
        return np.array(peaks)
        
    def _evaluate_candidate(self,
                           frame_idx: int,
                           audio: np.ndarray,
                           features: Dict[str, np.ndarray],
                           envelope: np.ndarray) -> ClickCandidate:
        """
        Evaluate a candidate peak using all features.
        
        Args:
            frame_idx: Frame index of candidate
            audio: Full audio signal
            features: Extracted features
            envelope: Hilbert envelope
            
        Returns:
            ClickCandidate if valid, None otherwise
        """
        # Convert frame index to sample index
        step_samples = self.feature_extractor.step_samples
        sample_idx = frame_idx * step_samples
        
        # Check bounds
        if sample_idx >= len(audio) or frame_idx >= len(features['tkeo_z']):
            return None
            
        # Get feature values
        tkeo_z = features['tkeo_z'][frame_idx]
        ste_z = features['ste_z'][frame_idx]
        hfc_z = features['hfc_z'][frame_idx]
        centroid = features['spectral_centroid'][frame_idx]
        hl_ratio = features['high_low_ratio'][frame_idx]
        
        # Check primary thresholds
        if tkeo_z < self.params.tkeo_threshold:
            return None
        if ste_z < self.params.ste_threshold:
            return None
            
        # Check secondary criteria (at least one must pass)
        secondary_pass = (
            hfc_z >= self.params.hfc_threshold or
            hl_ratio >= self.params.high_low_ratio_threshold
        )
        if not secondary_pass:
            return None
            
        # Measure envelope width
        env_width = measure_envelope_width(
            envelope, sample_idx, self.sample_rate, db_threshold=-10
        )
        
        # Check envelope width
        if not (self.params.envelope_width_min <= env_width <= self.params.envelope_width_max):
            return None
            
        # Optional: check spectral centroid
        if centroid < self.params.spectral_centroid_min:
            return None
            
        # Calculate confidence score
        confidence = self._calculate_confidence(
            tkeo_z, ste_z, hfc_z, hl_ratio, env_width
        )
        
        # Create candidate
        peak_time = sample_idx / self.sample_rate
        candidate = ClickCandidate(
            peak_idx=sample_idx,
            peak_time=peak_time,
            tkeo_value=tkeo_z,
            ste_value=ste_z,
            hfc_value=hfc_z,
            spectral_centroid=centroid,
            high_low_ratio=hl_ratio,
            envelope_width=env_width,
            confidence_score=confidence
        )
        
        return candidate
        
    def _calculate_confidence(self,
                             tkeo_z: float,
                             ste_z: float,
                             hfc_z: float,
                             hl_ratio: float,
                             env_width: float) -> float:
        """
        Calculate confidence score for candidate.
        
        Args:
            tkeo_z: TKEO z-score
            ste_z: STE z-score
            hfc_z: HFC z-score
            hl_ratio: High/low frequency ratio
            env_width: Envelope width in ms
            
        Returns:
            Confidence score (0-1)
        """
        # Normalize each feature contribution
        tkeo_contrib = min(tkeo_z / 10, 1.0)
        ste_contrib = min(ste_z / 10, 1.0)
        hfc_contrib = min(hfc_z / 8, 1.0)
        ratio_contrib = min(hl_ratio / 2, 1.0)
        
        # Envelope width: ideal around 0.5-1.0 ms
        width_contrib = 1.0 - abs(env_width - 0.8) / 1.0
        width_contrib = max(0, min(width_contrib, 1.0))
        
        # Weighted average
        confidence = (
            0.3 * tkeo_contrib +
            0.25 * ste_contrib +
            0.2 * hfc_contrib +
            0.15 * ratio_contrib +
            0.1 * width_contrib
        )
        
        return confidence
        
    def _apply_refractory_period(self,
                                candidates: List[ClickCandidate]) -> List[ClickCandidate]:
        """
        Apply refractory period: merge candidates too close together.
        
        Args:
            candidates: List of candidates
            
        Returns:
            Filtered list with refractory period applied
        """
        if len(candidates) <= 1:
            return candidates
            
        # Sort by time
        candidates = sorted(candidates, key=lambda c: c.peak_time)
        
        refractory_s = self.params.refractory_ms / 1000
        filtered = []
        
        i = 0
        while i < len(candidates):
            current = candidates[i]
            
            # Look ahead for candidates within refractory period
            j = i + 1
            group = [current]
            while j < len(candidates):
                if candidates[j].peak_time - current.peak_time < refractory_s:
                    group.append(candidates[j])
                    j += 1
                else:
                    break
                    
            # Keep the one with highest confidence
            best = max(group, key=lambda c: c.confidence_score)
            filtered.append(best)
            
            i = j
            
        return filtered
        
    def batch_detect(self,
                    audio: np.ndarray,
                    chunk_duration: float = 60.0,
                    overlap: float = 0.5) -> List[ClickCandidate]:
        """
        Detect clicks in long audio using chunks.
        
        Args:
            audio: Input audio signal
            chunk_duration: Chunk duration in seconds
            overlap: Overlap duration in seconds
            
        Returns:
            List of all detected candidates
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        all_candidates = []
        
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Detect in chunk
            candidates = self.detect_clicks(chunk)
            
            # Adjust times to global coordinates
            for candidate in candidates:
                candidate.peak_idx += start
                candidate.peak_time = candidate.peak_idx / self.sample_rate
                
            all_candidates.extend(candidates)
            
            if end >= len(audio):
                break
                
        # Remove duplicates from overlaps (keep higher confidence)
        all_candidates = self._remove_duplicates(all_candidates, overlap)
        
        return all_candidates
        
    def _remove_duplicates(self,
                          candidates: List[ClickCandidate],
                          overlap: float) -> List[ClickCandidate]:
        """
        Remove duplicate detections from overlapping chunks.
        
        Args:
            candidates: List of candidates
            overlap: Overlap duration in seconds
            
        Returns:
            Deduplicated list
        """
        if len(candidates) <= 1:
            return candidates
            
        candidates = sorted(candidates, key=lambda c: c.peak_time)
        filtered = []
        
        i = 0
        while i < len(candidates):
            current = candidates[i]
            
            # Find duplicates (within a small time window)
            duplicates = [current]
            j = i + 1
            while j < len(candidates):
                if abs(candidates[j].peak_time - current.peak_time) < 0.01:  # 10ms
                    duplicates.append(candidates[j])
                    j += 1
                else:
                    break
                    
            # Keep best
            best = max(duplicates, key=lambda c: c.confidence_score)
            filtered.append(best)
            
            i = j
            
        return filtered