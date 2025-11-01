"""
Teager-Kaiser Energy Operator (TKEO) implementation.
Used for transient detection in dolphin clicks.
"""

import numpy as np
from typing import Tuple

# Import helper functions from envelope module
from .envelope import compute_teager_kaiser, smooth_tkeo


class TeagerKaiserOperator:
    """Teager-Kaiser Energy Operator with robust normalization."""
    
    def __init__(self, smooth_window: int = 5):
        """
        Initialize TKEO.
        
        Args:
            smooth_window: Window size for smoothing
        """
        self.smooth_window = smooth_window
        
    def compute(self, audio: np.ndarray, smooth: bool = True) -> np.ndarray:
        """
        Compute TKEO with optional smoothing.
        
        Args:
            audio: Input signal
            smooth: Whether to apply smoothing
            
        Returns:
            TKEO signal
        """
        tkeo = compute_teager_kaiser(audio)
        
        if smooth and len(tkeo) > 0:
            tkeo = smooth_tkeo(tkeo, self.smooth_window)
            
        return tkeo
        
    def robust_normalize(self, tkeo: np.ndarray) -> np.ndarray:
        """
        Robust normalization using median and MAD.
        
        Args:
            tkeo: TKEO signal
            
        Returns:
            Normalized TKEO (z-scores)
        """
        median = np.median(tkeo)
        mad = np.median(np.abs(tkeo - median))
        
        if mad > 0:
            z_robust = (tkeo - median) / (1.4826 * mad)
        else:
            z_robust = np.zeros_like(tkeo)
            
        return z_robust
        
    def detect_peaks(self, 
                    tkeo: np.ndarray,
                    threshold: float = 6.0,
                    min_distance: int = 44) -> np.ndarray:
        """
        Detect peaks in TKEO signal above threshold.
        
        Args:
            tkeo: TKEO signal
            threshold: Detection threshold (in robust z-scores)
            min_distance: Minimum distance between peaks (samples)
            
        Returns:
            Array of peak indices
        """
        # Normalize
        tkeo_z = self.robust_normalize(tkeo)
        
        # Find points above threshold
        above_threshold = tkeo_z > threshold
        
        # Find peaks
        peaks = []
        i = 0
        while i < len(above_threshold):
            if above_threshold[i]:
                # Found start of potential peak
                peak_start = i
                peak_end = i
                
                # Find end of peak region
                while peak_end < len(above_threshold) and above_threshold[peak_end]:
                    peak_end += 1
                    
                # Find maximum in this region
                peak_idx = peak_start + np.argmax(tkeo_z[peak_start:peak_end])
                peaks.append(peak_idx)
                
                # Skip to end + min_distance
                i = peak_end + min_distance
            else:
                i += 1
                
        return np.array(peaks)