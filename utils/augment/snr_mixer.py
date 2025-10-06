"""
SNR-based mixing with 0.2s window generation.
修正版：直接生成0.2s训练样本
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple
import random
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of signal."""
    return np.sqrt(np.mean(audio**2))


def pad_click_to_window(
    click: np.ndarray,
    window_samples: int,
    random_offset: bool = True,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Pad short click to 0.2s window with click centered (with optional random offset).
    
    Args:
        click: Short click segment (8-12ms)
        window_samples: Target window length (e.g., 8820 for 0.2s)
        random_offset: Whether to add random offset
        sample_rate: Sample rate
        
    Returns:
        Padded window with click embedded
    """
    if len(click) >= window_samples:
        # Click is already long enough, just crop
        return click[:window_samples]
    
    # Create empty window
    window = np.zeros(window_samples, dtype=np.float32)
    
    # Calculate center position
    center_idx = window_samples // 2
    
    # Add random offset (±20ms = ±882 samples)
    if random_offset:
        max_offset = int(0.02 * sample_rate)  # 20ms
        offset = random.randint(-max_offset, max_offset)
        center_idx += offset
        center_idx = np.clip(center_idx, len(click)//2, window_samples - len(click)//2)
    
    # Place click in window
    start_idx = center_idx - len(click) // 2
    end_idx = start_idx + len(click)
    window[start_idx:end_idx] = click
    
    return window


def mix_with_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> np.ndarray:
    """
    Mix signal with noise at target SNR.
    Both signal and noise should be same length.
    """
    if len(noise) != len(signal):
        # Match lengths
        if len(noise) < len(signal):
            n_repeats = int(np.ceil(len(signal) / len(noise)))
            noise = np.tile(noise, n_repeats)[:len(signal)]
        else:
            start = random.randint(0, len(noise) - len(signal))
            noise = noise[start:start + len(signal)]
    
    # Calculate RMS
    signal_rms = calculate_rms(signal)
    noise_rms = calculate_rms(noise)
    
    if signal_rms < 1e-10 or noise_rms < 1e-10:
        return signal
    
    # Calculate noise scaling
    target_noise_rms = signal_rms / (10 ** (target_snr_db / 20))
    noise_scale = target_noise_rms / noise_rms
    
    # Mix
    mixed = signal + noise * noise_scale
    
    # Prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / (max_val * 1.01)
    
    return mixed


def augment_click_with_noise_window(
    click_dir: Path,
    noise_dir: Path,
    output_dir: Path,
    snr_range: Tuple[float, float] = (-5, 15),
    n_augmentations: int = 3,
    sample_rate: int = 44100,
    window_ms: float = 200.0,
    save_clean: bool = True
):
    """
    Augment clicks by embedding in 0.2s windows with noise.
    
    Args:
        click_dir: Directory with short click segments
        noise_dir: Directory with noise segments
        output_dir: Output directory
        snr_range: SNR range in dB
        n_augmentations: Number of augmentations per click
        sample_rate: Sample rate
        window_ms: Window duration in ms
        save_clean: Save clean versions (click in window without noise)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    window_samples = int(window_ms * sample_rate / 1000)
    
    # Find files
    click_files = sorted(Path(click_dir).rglob('*.wav'))
    noise_files = sorted(Path(noise_dir).rglob('*.wav'))
    
    if not click_files:
        logger.error(f"No click files in {click_dir}")
        return
    
    if not noise_files:
        logger.error(f"No noise files in {noise_dir}")
        return
    
    logger.info(f"Found {len(click_files)} click files")
    logger.info(f"Found {len(noise_files)} noise segments")
    
    # Load noise segments
    noise_segments = []
    logger.info("Loading noise segments...")
    for nf in tqdm(noise_files, desc="Loading noise"):
        try:
            audio, sr = sf.read(nf)
            if sr != sample_rate:
                logger.warning(f"{nf.name}: sample rate mismatch, skipping")
                continue
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            noise_segments.append(audio)
        except Exception as e:
            logger.error(f"Error loading {nf.name}: {e}")
            continue
    
    if not noise_segments:
        logger.error("No valid noise segments loaded")
        return
    
    logger.info(f"Loaded {len(noise_segments)} noise segments")
    
    # Process clicks
    total_augmented = 0
    
    for click_file in tqdm(click_files, desc="Augmenting clicks"):
        try:
            # Load short click
            click_audio, sr = sf.read(click_file)
            if sr != sample_rate:
                logger.warning(f"{click_file.name}: sample rate mismatch")
                continue
            
            if click_audio.ndim == 2:
                click_audio = click_audio.mean(axis=1)
            
            # Save clean version (click in 0.2s window, no noise)
            if save_clean:
                clean_window = pad_click_to_window(
                    click_audio, window_samples, 
                    random_offset=False, sample_rate=sample_rate
                )
                out_path = output_dir / f"{click_file.stem}_clean.wav"
                sf.write(out_path, clean_window, sample_rate)
            
            # Create augmented versions
            for aug_idx in range(n_augmentations):
                # Pad click to 0.2s window with random offset
                click_window = pad_click_to_window(
                    click_audio, window_samples,
                    random_offset=True, sample_rate=sample_rate
                )
                
                # Random SNR
                target_snr = random.uniform(*snr_range)
                
                # Random noise
                noise = random.choice(noise_segments)
                
                # Mix
                mixed = mix_with_snr(click_window, noise, target_snr)
                
                # Save
                out_name = f"{click_file.stem}_snr{int(target_snr):+03d}_aug{aug_idx:02d}.wav"
                out_path = output_dir / out_name
                sf.write(out_path, mixed, sample_rate)
                
                total_augmented += 1
                
        except Exception as e:
            logger.error(f"Error processing {click_file.name}: {e}")
            continue
    
    logger.info(f"Augmentation completed: {total_augmented} files created")
    logger.info(f"All outputs are {window_ms}ms = {window_samples} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Augment clicks with 0.2s window generation'
    )
    parser.add_argument('--click-dir', type=str, required=True)
    parser.add_argument('--noise-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--snr-min', type=float, default=-5)
    parser.add_argument('--snr-max', type=float, default=15)
    parser.add_argument('--n-aug', type=int, default=3)
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--window-ms', type=float, default=200.0)
    parser.add_argument('--no-clean', action='store_true')
    
    args = parser.parse_args()
    
    augment_click_with_noise_window(
        click_dir=Path(args.click_dir),
        noise_dir=Path(args.noise_dir),
        output_dir=Path(args.output_dir),
        snr_range=(args.snr_min, args.snr_max),
        n_augmentations=args.n_aug,
        sample_rate=args.sample_rate,
        window_ms=args.window_ms,
        save_clean=not args.no_clean
    )


if __name__ == '__main__':
    main()