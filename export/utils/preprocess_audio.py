"""
Example audio preprocessing script (optional reference).
Users can adapt this to their own data pipeline.
"""

import soundfile as sf
import librosa
import numpy as np
from scipy import signal
from pathlib import Path

def preprocess_audio(input_file, output_dir, 
                    target_sr=44100,
                    bandpass_low=2000,
                    bandpass_high=20000,
                    segment_length_ms=500):
    """
    Example preprocessing function.
    
    Args:
        input_file: Input audio file
        output_dir: Output directory
        target_sr: Target sample rate
        bandpass_low: Low cutoff frequency (Hz)
        bandpass_high: High cutoff frequency (Hz)
        segment_length_ms: Segment length (ms)
    """
    # Load audio
    audio, sr = sf.read(input_file)
    
    # Convert to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Bandpass filter
    sos = signal.butter(4, [bandpass_low, bandpass_high], 
                       btype='band', fs=target_sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # Segment into 500ms clips
    segment_samples = int(segment_length_ms * target_sr / 1000)
    n_segments = len(audio) // segment_samples
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]
        
        output_file = output_dir / f"{Path(input_file).stem}_seg_{i:04d}.wav"
        sf.write(output_file, segment, target_sr)
    
    print(f"Generated {n_segments} segments")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    
    preprocess_audio(args.input, args.output_dir)