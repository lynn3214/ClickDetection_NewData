#!/usr/bin/env python3
"""
Resampling and filtering utilities for dolphin click detection.
"""

import argparse
import logging
from pathlib import Path
from typing import Union, List
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt, sosfiltfilt
import soundfile as sf
import scipy.io as sio
import h5py
from tqdm import tqdm
from datetime import datetime

# Global error collection list
error_logs: List[str] = []

def resample_and_hpf(
    x: np.ndarray,
    sr_orig: int,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    hp_order: int = 4
) -> np.ndarray:
    """
    Resample signal and apply high-pass filter (legacy function).
    
    Args:
        x: Input signal array
        sr_orig: Original sampling rate
        sr_target: Target sampling rate (default 44100)
        hp_cutoff: High-pass cutoff frequency in Hz (default 1000)
        hp_order: Filter order (default 4)
        
    Returns:
        Filtered and resampled signal as float32
    """
    # Handle empty input safely
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)

    # Resample
    if sr_orig != sr_target:
        g = np.gcd(sr_orig, sr_target)
        up = sr_target // g
        down = sr_orig // g
        y = resample_poly(x, up, down)
    else:
        y = x.copy()
    
    # High-pass filter
    if hp_cutoff > 0 and hp_cutoff < sr_target // 2 and len(y) > 0:
        sos = butter(hp_order, hp_cutoff, 'highpass', fs=sr_target, output='sos')
        y = sosfilt(sos, y)
    
    return y.astype(np.float32)


def resample_and_filter(
    x: np.ndarray,
    sr_orig: int,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    bandpass_low: float = 2000,
    bandpass_high: float = 20000,
    hp_order: int = 4,
    bp_order: int = 4,
    use_bandpass: bool = True
) -> np.ndarray:
    """
    Resample signal and apply filtering (highpass + optional bandpass).
    
    Args:
        x: Input signal array
        sr_orig: Original sampling rate
        sr_target: Target sampling rate (default 44100)
        hp_cutoff: High-pass cutoff frequency in Hz (default 1000)
        bandpass_low: Bandpass low cutoff (default 2000)
        bandpass_high: Bandpass high cutoff (default 20000)
        hp_order: High-pass filter order (default 4)
        bp_order: Bandpass filter order (default 4)
        use_bandpass: Whether to apply bandpass after highpass (default True)
        
    Returns:
        Filtered and resampled signal as float32
    """
    # Handle empty input safely
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)

    # Resample first
    if sr_orig != sr_target:
        g = np.gcd(sr_orig, sr_target)
        up = sr_target // g
        down = sr_orig // g
        y = resample_poly(x, up, down)
    else:
        y = x.copy()
    
    if len(y) == 0:
        return y.astype(np.float32)
    
    # Step 1: High-pass filter at 1000 Hz (remove very low frequencies)
    if hp_cutoff > 0 and hp_cutoff < sr_target // 2:
        sos_hp = butter(hp_order, hp_cutoff, 'highpass', fs=sr_target, output='sos')
        y = sosfilt(sos_hp, y)
    
    # Step 2: Bandpass filter 2-20 kHz (focus on dolphin click range)
    if use_bandpass and bandpass_low > 0 and bandpass_high < sr_target // 2:
        # Ensure valid frequency range
        if bandpass_low >= bandpass_high:
            bandpass_low = 2000
            bandpass_high = min(20000, sr_target // 2 - 100)
        
        nyq = sr_target / 2
        low = bandpass_low / nyq
        high = bandpass_high / nyq
        
        # Check validity
        if 0 < low < 1 and 0 < high < 1 and low < high:
            sos_bp = butter(bp_order, [low, high], 'bandpass', output='sos')
            y = sosfiltfilt(sos_bp, y)  # Zero-phase filtering
    
    return y.astype(np.float32)


def _read_mat_v73_h5(file_path: Path,
                     dataset_key: str = 'newFiltDat',
                     fs_key: str = 'fs') -> tuple:
    """Read v7.3 MAT file using h5py, return (wave, sr)."""
    with h5py.File(file_path, 'r') as f:
        # Read data
        if dataset_key not in f:
            raise KeyError(f'Key "{dataset_key}" not found in {file_path}')
        dset = f[dataset_key]
        data = dset[()]
        
        # Read sampling rate
        if fs_key in f:
            sr = int(f[fs_key][()])
        elif fs_key in dset.attrs:
            sr = int(dset.attrs[fs_key])
        else:
            raise KeyError(f'Cannot find sampling rate "{fs_key}" in file')
    return np.asarray(data), sr


def load_audio_file(file_path: Path, mat_channel: int = 10) -> tuple:
    """
    Load audio from .wav, .mat, or .pkl file.
    Returns mono waveform (float32) and original sample-rate.
    """
    file_path = Path(file_path)
    
    # Special handling for noise directory
    if file_path.parent.name == 'noise':
        import pickle
        noise = pickle.load(open(file_path, 'rb')).astype(np.float32)
        return noise, 96000
    
    suffix = file_path.suffix.lower()

    # WAV files
    if suffix == '.wav':
        audio, sr = sf.read(file_path)
        audio = audio.mean(1) if audio.ndim == 2 else audio
        return audio.astype(np.float32), int(sr)

    # MAT files
    if suffix == '.mat':
        try:
            mat = sio.loadmat(file_path, squeeze_me=True)
            data = mat['newFiltDat']
            sr = int(mat['fs'])
        except NotImplementedError:
            data, sr = _read_mat_v73_h5(file_path)
        
        # Handle channel orientation
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T
        
        # Select channel
        ch_idx = min(mat_channel, data.shape[1] - 1)
        audio = data[:, ch_idx]
        return audio.astype(np.float32), sr

    # Pickle files
    if suffix == '.pkl':
        import pickle
        noise = pickle.load(open(file_path, 'rb')).astype(np.float32)
        return noise, 96000

    raise ValueError(f'Unsupported file format: {suffix}')


def _process_waveform_and_save(wave: np.ndarray, sr_orig: int,
                               out_path: Path, cfg):
    """Process single waveform and save to file."""
    out_path = out_path.with_suffix('.wav')
    
    # Apply filtering
    y = resample_and_filter(
        wave, sr_orig,
        sr_target=cfg['sr_target'],
        hp_cutoff=cfg['hp_cutoff'],
        bandpass_low=cfg['bandpass_low'],
        bandpass_high=cfg['bandpass_high'],
        hp_order=cfg['hp_order'],
        bp_order=cfg['bp_order'],
        use_bandpass=cfg['use_bandpass']
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, cfg['sr_target'])
    logging.info(f"Saved: {out_path}")


def process_file(
    input_path: Path,
    output_path: Path,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    bandpass_low: float = 2000,
    bandpass_high: float = 20000,
    hp_order: int = 4,
    bp_order: int = 4,
    use_bandpass: bool = True,
    mat_channel: int = 10
) -> None:
    """Process a single audio file with improved filtering."""
    try:
        cfg = {
            'sr_target': sr_target,
            'hp_cutoff': hp_cutoff,
            'bandpass_low': bandpass_low,
            'bandpass_high': bandpass_high,
            'hp_order': hp_order,
            'bp_order': bp_order,
            'use_bandpass': use_bandpass,
            'mat_channel': mat_channel
        }
        
        audio, sr_orig = load_audio_file(input_path, mat_channel)
        output_path = output_path.with_suffix('.wav')
        
        if audio.ndim == 2:  # Multiple channels
            for i, row in enumerate(audio):
                out_file = output_path.with_suffix('').as_posix() + f'_{i:05d}.wav'
                _process_waveform_and_save(row, sr_orig, Path(out_file), cfg)
            logging.info(f"Processed {len(audio)} segments from: {input_path}")
        else:
            _process_waveform_and_save(audio, sr_orig, output_path, cfg)
            logging.info(f"Processed: {input_path} -> {output_path}")
            
    except Exception as e:
        error_msg = f"Error processing {input_path}: {str(e)}"
        error_logs.append(error_msg)
        logging.error(error_msg)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Resample and filter audio files')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory or file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--sr_target', type=int, default=44100,
                        help='Target sampling rate (default: 44100)')
    parser.add_argument('--hp_cutoff', type=int, default=1000,
                        help='High-pass cutoff frequency (default: 1000)')
    parser.add_argument('--bandpass_low', type=float, default=2000,
                        help='Bandpass low cutoff (default: 2000)')
    parser.add_argument('--bandpass_high', type=float, default=20000,
                        help='Bandpass high cutoff (default: 20000)')
    parser.add_argument('--hp_order', type=int, default=4,
                        help='High-pass filter order (default: 4)')
    parser.add_argument('--bp_order', type=int, default=4,
                        help='Bandpass filter order (default: 4)')
    parser.add_argument('--use_bandpass', action='store_true',
                        help='Apply bandpass filter (2-20 kHz)')
    parser.add_argument('--mat_channel', type=int, default=10,
                        help='Channel to extract from .mat files (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Process single file
        process_file(
            input_path, 
            output_path / input_path.name,
            sr_target=args.sr_target,
            hp_cutoff=args.hp_cutoff,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            hp_order=args.hp_order,
            bp_order=args.bp_order,
            use_bandpass=args.use_bandpass,
            mat_channel=args.mat_channel
        )
    elif input_path.is_dir():
        # Process all files in directory (THIS WAS MISSING!)
        files = list(input_path.rglob('*'))
        files = [f for f in files if (
            f.is_file() and
            f.name != '.DS_Store' and
            (f.suffix.lower() in ('.wav', '.mat', '.pkl') or
             f.parent.name == 'noise')
        )]
        
        for file_path in tqdm(files, desc="Processing files"):
            rel_path = file_path.relative_to(input_path)
            out_path = output_path / rel_path
            process_file(
                file_path,
                out_path,
                sr_target=args.sr_target,
                hp_cutoff=args.hp_cutoff,
                bandpass_low=args.bandpass_low,
                bandpass_high=args.bandpass_high,
                hp_order=args.hp_order,
                bp_order=args.bp_order,
                use_bandpass=args.use_bandpass,
                mat_channel=args.mat_channel
            )
    else:
        logging.error(f"Input path does not exist: {input_path}")
        return
    
    # Save error log if any
    if error_logs:
        log_file = output_path / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(error_logs))
        
        print(f"\nFound {len(error_logs)} errors. Details saved to: {log_file}")
        print("\nError summary:")
        for error in error_logs:
            print(f"- {error}")


if __name__ == '__main__':
    main()