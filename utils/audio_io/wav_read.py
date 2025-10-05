"""
WAV file reading utilities with chunking support.
Handles long audio files with overlapping blocks.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Generator
import soundfile as sf
import librosa


class WavReader:
    """Handles WAV file reading with chunking and resampling."""
    
    def __init__(self, target_sr: int = 44100):
        """
        Initialize WAV reader.
        
        Args:
            target_sr: Target sample rate for resampling
        """
        self.target_sr = target_sr
        
    def read(self, filepath: Path, 
             start: Optional[float] = None,
             duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Read WAV file with optional segment selection.
        
        Args:
            filepath: Path to WAV file
            start: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio array and sample rate
        """
        filepath = Path(filepath)
        
        # Read audio
        audio, sr = sf.read(str(filepath), start=start, stop=duration)
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
            
        return audio, sr
        
    def read_chunks(self, 
                   filepath: Path,
                   chunk_duration: float = 60.0,
                   overlap: float = 0.5) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Read audio file in overlapping chunks.
        
        Args:
            filepath: Path to WAV file
            chunk_duration: Chunk duration in seconds
            overlap: Overlap duration in seconds
            
        Yields:
            Tuple of (audio_chunk, sample_rate, start_time)
        """
        filepath = Path(filepath)
        
        # Get file duration
        info = sf.info(str(filepath))
        total_duration = info.duration
        
        # Generate chunks
        start = 0.0
        step = chunk_duration - overlap
        
        while start < total_duration:
            # Read chunk
            chunk_audio, sr = self.read(filepath, start=start, duration=chunk_duration)
            
            yield chunk_audio, sr, start
            
            start += step
            
            # Last chunk adjustment
            if start + chunk_duration > total_duration and start < total_duration:
                start = total_duration - chunk_duration
                if start < 0:
                    start = 0
                    
    def normalize(self, audio: np.ndarray, method: str = 'rms') -> np.ndarray:
        """
        Normalize audio signal.
        
        Args:
            audio: Audio signal
            method: Normalization method ('rms', 'peak', 'mad')
            
        Returns:
            Normalized audio
        """
        if method == 'rms':
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms
        elif method == 'peak':
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
        elif method == 'mad':
            median = np.median(audio)
            mad = np.median(np.abs(audio - median))
            if mad > 0:
                audio = (audio - median) / (1.4826 * mad)
                
        return audio


def load_audio(filepath: Path, 
               target_sr: int = 44100,
               normalize: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate
        normalize: Whether to normalize audio
        
    Returns:
        Audio array and sample rate
    """
    reader = WavReader(target_sr)
    audio, sr = reader.read(filepath)
    
    if normalize:
        audio = reader.normalize(audio)
        
    return audio, sr