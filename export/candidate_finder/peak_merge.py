"""
Peak merging and click segment extraction.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
from dataclasses import asdict

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from utils.dsp.envelope import compute_peak_factor


# ==== peak_merge.py ====

class PeakMerger:
    """Merges nearby peaks and applies non-maximum suppression."""
    
    def __init__(self, refractory_ms: float = 1.5, sample_rate: int = 44100):
        """
        Initialize peak merger.
        
        Args:
            refractory_ms: Minimum time between peaks (ms)
            sample_rate: Sample rate (Hz)
        """
        self.refractory_samples = int(refractory_ms * sample_rate / 1000)
        
    def merge_peaks(self, candidates: List[ClickCandidate]) -> List[ClickCandidate]:
        """
        Merge peaks within refractory period.
        
        Args:
            candidates: List of click candidates
            
        Returns:
            Merged list of candidates
        """
        if len(candidates) <= 1:
            return candidates
            
        # Sort by peak index
        candidates = sorted(candidates, key=lambda c: c.peak_idx)
        merged = []
        
        i = 0
        while i < len(candidates):
            current = candidates[i]
            
            # Find all candidates within refractory period
            group = [current]
            j = i + 1
            while j < len(candidates):
                if candidates[j].peak_idx - current.peak_idx < self.refractory_samples:
                    group.append(candidates[j])
                    j += 1
                else:
                    break
                    
            # Keep the one with highest confidence
            best = max(group, key=lambda c: c.confidence_score)
            merged.append(best)
            
            i = j
            
        return merged
        
    def suppress_overlapping(self,
                           candidates: List[ClickCandidate],
                           window_samples: int) -> List[ClickCandidate]:
        """
        Apply non-maximum suppression based on confidence.
        
        Args:
            candidates: List of candidates
            window_samples: Window size for overlap check
            
        Returns:
            Filtered candidates
        """
        if len(candidates) <= 1:
            return candidates
            
        # Sort by confidence (descending)
        candidates = sorted(candidates, key=lambda c: c.confidence_score, reverse=True)
        
        keep = []
        suppressed_indices = set()
        
        for i, candidate in enumerate(candidates):
            if i in suppressed_indices:
                continue
                
            keep.append(candidate)
            
            # Suppress overlapping lower-confidence candidates
            for j in range(i + 1, len(candidates)):
                if j in suppressed_indices:
                    continue
                    
                if abs(candidates[j].peak_idx - candidate.peak_idx) < window_samples:
                    suppressed_indices.add(j)
                    
        # Sort by time
        keep = sorted(keep, key=lambda c: c.peak_idx)
        return keep