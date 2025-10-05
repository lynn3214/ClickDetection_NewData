"""
Dataset construction for 0.2s training samples.
Handles positive samples from detected clicks and negative samples from noise.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import soundfile as sf
import json
from tqdm import tqdm
import random

from detection.candidate_finder.dynamic_threshold import ClickCandidate


class DatasetBuilder:
    """Builds 0.2s training samples from detected clicks and noise."""
    
    def __init__(self,
                 sample_rate: int = 44100,
                 window_ms: float = 200.0,
                 random_offset_ms: float = 20.0):
        """
        Initialize dataset builder.
        
        Args:
            sample_rate: Sample rate (Hz)
            window_ms: Window duration for samples (ms)
            random_offset_ms: Random time offset range (ms)
        """
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000)
        self.offset_samples = int(random_offset_ms * sample_rate / 1000)
        
    def build_positive_samples(self,
                              audio: np.ndarray,
                              candidates: List[ClickCandidate],
                              file_id: str) -> List[Dict[str, Any]]:
        """
        Build positive samples centered on detected clicks.
        
        Args:
            audio: Full audio signal
            candidates: List of click candidates
            file_id: Source file identifier
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for i, candidate in enumerate(candidates):
            # Random offset to add variability
            offset = random.randint(-self.offset_samples, self.offset_samples)
            center_idx = candidate.peak_idx + offset
            
            # Extract window
            segment = self._extract_centered_window(audio, center_idx)
            
            if segment is not None:
                sample = {
                    'waveform': segment,
                    'label': 1,  # Positive class
                    'file_id': file_id,
                    'candidate_idx': i,
                    'peak_time': candidate.peak_time,
                    'confidence': candidate.confidence_score
                }
                samples.append(sample)
                
        return samples
        
    def build_negative_samples(self,
                              noise_audio: np.ndarray,
                              file_id: str,
                              n_samples: int) -> List[Dict[str, Any]]:
        """
        Build negative samples from noise.
        
        Args:
            noise_audio: Noise audio signal
            n_samples: Number of samples to generate
            file_id: Source file identifier
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Random sampling from noise
        max_start = len(noise_audio) - self.window_samples
        if max_start <= 0:
            return samples
            
        for i in range(n_samples):
            start_idx = random.randint(0, max_start)
            segment = noise_audio[start_idx:start_idx + self.window_samples]
            
            # Normalize
            segment = self._normalize_segment(segment)
            
            sample = {
                'waveform': segment,
                'label': 0,  # Negative class
                'file_id': file_id,
                'candidate_idx': -1,
                'peak_time': start_idx / self.sample_rate,
                'confidence': 0.0
            }
            samples.append(sample)
            
        return samples
        
    def build_negative_from_rejected(self,
                                    audio: np.ndarray,
                                    rejected_candidates: List[ClickCandidate],
                                    file_id: str) -> List[Dict[str, Any]]:
        """
        Build negative samples from rejected candidates.
        
        Args:
            audio: Full audio signal
            rejected_candidates: List of rejected candidates
            file_id: Source file identifier
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for i, candidate in enumerate(rejected_candidates):
            segment = self._extract_centered_window(audio, candidate.peak_idx)
            
            if segment is not None:
                sample = {
                    'waveform': segment,
                    'label': 0,  # Negative class
                    'file_id': file_id,
                    'candidate_idx': -1,
                    'peak_time': candidate.peak_time,
                    'confidence': candidate.confidence_score
                }
                samples.append(sample)
                
        return samples
        
    def _extract_centered_window(self,
                                audio: np.ndarray,
                                center_idx: int) -> Optional[np.ndarray]:
        """
        Extract window centered on index.
        
        Args:
            audio: Audio signal
            center_idx: Center sample index
            
        Returns:
            Extracted window or None if out of bounds
        """
        half_window = self.window_samples // 2
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        # Check bounds
        if start_idx < 0 or end_idx > len(audio):
            # Pad if necessary
            segment = self._extract_with_padding(audio, center_idx)
        else:
            segment = audio[start_idx:end_idx]
            
        # Ensure correct length
        if len(segment) != self.window_samples:
            return None
            
        # Normalize
        segment = self._normalize_segment(segment)
        
        return segment
        
    def _extract_with_padding(self,
                             audio: np.ndarray,
                             center_idx: int) -> np.ndarray:
        """Extract window with reflection padding if needed."""
        half_window = self.window_samples // 2
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        # Calculate padding needed
        pad_left = max(0, -start_idx)
        pad_right = max(0, end_idx - len(audio))
        
        # Adjust extraction indices
        extract_start = max(0, start_idx)
        extract_end = min(len(audio), end_idx)
        
        segment = audio[extract_start:extract_end]
        
        # Apply padding
        if pad_left > 0 or pad_right > 0:
            segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
            
        return segment
        
    def _normalize_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Normalize segment to zero mean and unit variance.
        
        Args:
            segment: Input segment
            
        Returns:
            Normalized segment
        """
        # Remove DC
        segment = segment - np.mean(segment)
        
        # MAD-based normalization for robustness
        mad = np.median(np.abs(segment - np.median(segment)))
        if mad > 1e-10:
            segment = segment / (1.4826 * mad)
        else:
            # Fallback to RMS
            rms = np.sqrt(np.mean(segment**2))
            if rms > 1e-10:
                segment = segment / rms
                
        return segment.astype(np.float32)
        
    def save_dataset(self,
                    samples: List[Dict[str, Any]],
                    output_dir: Path,
                    split: str = 'train') -> Path:
        """
        Save dataset to disk.
        
        Args:
            samples: List of sample dictionaries
            output_dir: Output directory
            split: Dataset split name ('train', 'val', 'test')
            
        Returns:
            Path to saved dataset directory
        """
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save waveforms as numpy array
        waveforms = np.array([s['waveform'] for s in samples])
        labels = np.array([s['label'] for s in samples])
        
        np.save(split_dir / 'waveforms.npy', waveforms)
        np.save(split_dir / 'labels.npy', labels)
        
        # Save metadata
        metadata = []
        for s in samples:
            meta = {k: v for k, v in s.items() if k != 'waveform'}
            metadata.append(meta)
            
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved {len(samples)} samples to {split_dir}")
        print(f"  Positive: {np.sum(labels == 1)}")
        print(f"  Negative: {np.sum(labels == 0)}")
        
        return split_dir
        
    def load_dataset(self, dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load dataset from disk.
        
        Args:
            dataset_dir: Dataset directory
            
        Returns:
            Tuple of (waveforms, labels, metadata)
        """
        dataset_dir = Path(dataset_dir)
        
        waveforms = np.load(dataset_dir / 'waveforms.npy')
        labels = np.load(dataset_dir / 'labels.npy')
        
        with open(dataset_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return waveforms, labels, metadata
        
    def balance_dataset(self,
                       samples: List[Dict[str, Any]],
                       balance_ratio: float = 1.0) -> List[Dict[str, Any]]:
        """
        Balance positive and negative samples.
        
        Args:
            samples: List of samples
            balance_ratio: Ratio of negative to positive samples
            
        Returns:
            Balanced sample list
        """
        positive = [s for s in samples if s['label'] == 1]
        negative = [s for s in samples if s['label'] == 0]
        
        n_positive = len(positive)
        n_negative_target = int(n_positive * balance_ratio)
        
        # Undersample or oversample negatives
        if len(negative) > n_negative_target:
            negative = random.sample(negative, n_negative_target)
        elif len(negative) < n_negative_target:
            # Oversample with replacement
            negative = random.choices(negative, k=n_negative_target)
            
        balanced = positive + negative
        random.shuffle(balanced)
        
        return balanced