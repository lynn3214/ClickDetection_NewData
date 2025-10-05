"""
Click segmenter for extracting and saving click segments.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
from dataclasses import asdict

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from utils.dsp.envelope import compute_peak_factor


class ClickSegmenter:
    """Extracts and saves click segments."""
    
    def __init__(self,
                 sample_rate: int = 44100,
                 pre_ms: float = 2.0,
                 post_ms: float = 10.0,
                 min_peak_factor_db: float = 10.0):
        """
        Initialize click segmenter.
        
        Args:
            sample_rate: Sample rate (Hz)
            pre_ms: Pre-click duration (ms)
            post_ms: Post-click duration (ms)
            min_peak_factor_db: Minimum peak factor threshold (dB)
        """
        self.sample_rate = sample_rate
        self.pre_samples = int(pre_ms * sample_rate / 1000)
        self.post_samples = int(post_ms * sample_rate / 1000)
        self.min_peak_factor_db = min_peak_factor_db
        
    def extract_segment(self,
                       audio: np.ndarray,
                       peak_idx: int) -> Tuple[np.ndarray, bool]:
        """
        Extract click segment around peak.
        
        Args:
            audio: Full audio signal
            peak_idx: Peak sample index
            
        Returns:
            Tuple of (segment, is_valid)
        """
        # Calculate boundaries
        start_idx = max(0, peak_idx - self.pre_samples)
        end_idx = min(len(audio), peak_idx + self.post_samples)
        
        # Extract segment
        segment = audio[start_idx:end_idx]
        
        # Check peak factor
        peak_factor = compute_peak_factor(segment)
        is_valid = peak_factor >= self.min_peak_factor_db
        
        # Normalize
        if is_valid:
            segment = self._normalize_segment(segment)
            
        return segment, is_valid
        
    def _normalize_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Normalize segment to [-1, 1] range.
        
        Args:
            segment: Input segment
            
        Returns:
            Normalized segment
        """
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak
        return segment
        
    def extract_and_save(self,
                        audio: np.ndarray,
                        candidates: List[ClickCandidate],
                        output_dir: Path,
                        file_prefix: str = "click") -> List[Path]:
        """
        Extract segments and save to files.
        
        Args:
            audio: Full audio signal
            candidates: List of click candidates
            output_dir: Output directory
            file_prefix: Prefix for output files
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, candidate in enumerate(candidates):
            segment, is_valid = self.extract_segment(audio, candidate.peak_idx)
            
            if not is_valid:
                continue
                
            # Generate filename
            timestamp_ms = int(candidate.peak_time * 1000)
            filename = f"{file_prefix}_{timestamp_ms:08d}_{i:04d}.wav"
            filepath = output_dir / filename
            
            # Save
            sf.write(str(filepath), segment, self.sample_rate)
            saved_paths.append(filepath)
            
        return saved_paths
        
    def pad_to_length(self,
                     segment: np.ndarray,
                     target_length: int,
                     mode: str = 'constant') -> np.ndarray:
        """
        Pad segment to target length.
        
        Args:
            segment: Input segment
            target_length: Target length in samples
            mode: Padding mode ('constant', 'reflect', 'edge')
            
        Returns:
            Padded segment
        """
        if len(segment) >= target_length:
            return segment[:target_length]
            
        pad_total = target_length - len(segment)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        if mode == 'constant':
            padded = np.pad(segment, (pad_left, pad_right), mode='constant')
        elif mode == 'reflect':
            padded = np.pad(segment, (pad_left, pad_right), mode='reflect')
        elif mode == 'edge':
            padded = np.pad(segment, (pad_left, pad_right), mode='edge')
        else:
            raise ValueError(f"Unknown padding mode: {mode}")
            
        return padded
        
    def extract_centered_window(self,
                               audio: np.ndarray,
                               peak_idx: int,
                               window_ms: float = 200) -> np.ndarray:
        """
        Extract window centered on peak (for CNN input).
        
        Args:
            audio: Full audio signal
            peak_idx: Peak sample index
            window_ms: Window duration (ms)
            
        Returns:
            Centered window (padded if necessary)
        """
        window_samples = int(window_ms * self.sample_rate / 1000)
        half_window = window_samples // 2
        
        start_idx = peak_idx - half_window
        end_idx = peak_idx + half_window
        
        # Handle boundaries
        if start_idx < 0:
            segment = audio[:end_idx]
            segment = self.pad_to_length(segment, window_samples, mode='reflect')
        elif end_idx > len(audio):
            segment = audio[start_idx:]
            segment = self.pad_to_length(segment, window_samples, mode='reflect')
        else:
            segment = audio[start_idx:end_idx]
            
        return segment


def save_candidates_csv(candidates: List[ClickCandidate],
                       output_path: Path) -> None:
    """
    Save candidates to CSV file.
    
    Args:
        candidates: List of click candidates
        output_path: Output CSV path
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not candidates:
        return
        
    fieldnames = list(asdict(candidates[0]).keys())
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for candidate in candidates:
            writer.writerow(asdict(candidate))