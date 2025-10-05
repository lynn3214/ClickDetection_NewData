"""
Digital signal processing filters.
Bandpass, highpass, and pre-emphasis filters.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, firwin, lfilter
from typing import Optional


class BandpassFilter:
    """Butterworth bandpass filter."""
    
    def __init__(self,
                 lowcut: float,
                 highcut: float,
                 sample_rate: int,
                 order: int = 4):
        """
        Initialize bandpass filter.
        
        Args:
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            sample_rate: Sample rate (Hz)
            order: Filter order
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.sample_rate = sample_rate
        self.order = order
        
        # Design filter
        nyq = sample_rate / 2
        low = lowcut / nyq
        high = highcut / nyq
        self.sos = butter(order, [low, high], btype='band', output='sos')
        
    def apply(self, audio: np.ndarray, zero_phase: bool = True) -> np.ndarray:
        """
        Apply bandpass filter.
        
        Args:
            audio: Input audio signal
            zero_phase: Use zero-phase filtering (filtfilt)
            
        Returns:
            Filtered signal
        """
        if zero_phase:
            return sosfiltfilt(self.sos, audio)
        else:
            return sosfilt(self.sos, audio)


class HighpassFilter:
    """Butterworth highpass filter."""
    
    def __init__(self,
                 cutoff: float,
                 sample_rate: int,
                 order: int = 4):
        """
        Initialize highpass filter.
        
        Args:
            cutoff: Cutoff frequency (Hz)
            sample_rate: Sample rate (Hz)
            order: Filter order
        """
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.order = order
        
        # Design filter
        nyq = sample_rate / 2
        normal_cutoff = cutoff / nyq
        self.sos = butter(order, normal_cutoff, btype='high', output='sos')
        
    def apply(self, audio: np.ndarray, zero_phase: bool = True) -> np.ndarray:
        """
        Apply highpass filter.
        
        Args:
            audio: Input audio signal
            zero_phase: Use zero-phase filtering
            
        Returns:
            Filtered signal
        """
        if zero_phase:
            return sosfiltfilt(self.sos, audio)
        else:
            return sosfilt(self.sos, audio)


class PreEmphasisFilter:
    """Pre-emphasis filter for enhancing high frequencies."""
    
    def __init__(self, coefficient: float = 0.97):
        """
        Initialize pre-emphasis filter.
        
        Args:
            coefficient: Pre-emphasis coefficient (0-1)
        """
        self.coefficient = coefficient
        
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter: y[n] = x[n] - coef * x[n-1]
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pre-emphasized signal
        """
        return lfilter([1, -self.coefficient], [1], audio)
        
    def inverse(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply de-emphasis (inverse filter).
        
        Args:
            audio: Pre-emphasized signal
            
        Returns:
            De-emphasized signal
        """
        return lfilter([1], [1, -self.coefficient], audio)


def apply_bandpass(audio: np.ndarray,
                   lowcut: float,
                   highcut: float,
                   sample_rate: int,
                   order: int = 4,
                   zero_phase: bool = True) -> np.ndarray:
    """
    Apply bandpass filter to audio.
    
    Args:
        audio: Input audio
        lowcut: Low cutoff (Hz)
        highcut: High cutoff (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order
        zero_phase: Use zero-phase filtering
        
    Returns:
        Filtered audio
    """
    filt = BandpassFilter(lowcut, highcut, sample_rate, order)
    return filt.apply(audio, zero_phase)


def apply_highpass(audio: np.ndarray,
                   cutoff: float,
                   sample_rate: int,
                   order: int = 4,
                   zero_phase: bool = True) -> np.ndarray:
    """
    Apply highpass filter to audio.
    
    Args:
        audio: Input audio
        cutoff: Cutoff frequency (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order
        zero_phase: Use zero-phase filtering
        
    Returns:
        Filtered audio
    """
    filt = HighpassFilter(cutoff, sample_rate, order)
    return filt.apply(audio, zero_phase)