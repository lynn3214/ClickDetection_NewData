"""
Segment long noise files into shorter clips for augmentation.
Place at: utils/preprocessing/noise_segmenter.py
"""

import argparse
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def segment_noise_file(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration: float = 1.0,
    overlap: float = 0.5
) -> list:
    """
    Segment long noise audio into shorter clips.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        segment_duration: Duration of each segment (seconds)
        overlap: Overlap between segments (seconds)
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sample_rate)
    step_samples = int((segment_duration - overlap) * sample_rate)
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, step_samples):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
    
    return segments


def process_noise_directory(
    input_dir: Path,
    output_dir: Path,
    segment_duration: float = 1.0,
    overlap: float = 0.5,
    sample_rate: int = 44100
):
    """
    Process all noise files in directory.
    
    Args:
        input_dir: Input directory with noise files
        output_dir: Output directory for segments
        segment_duration: Duration of each segment (seconds)
        overlap: Overlap between segments (seconds)
        sample_rate: Expected sample rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all wav files
    noise_files = list(Path(input_dir).rglob('*.wav'))
    
    if not noise_files:
        logger.warning(f"No wav files found in {input_dir}")
        return
    
    logger.info(f"Found {len(noise_files)} noise files")
    
    total_segments = 0
    
    for noise_file in tqdm(noise_files, desc="Segmenting noise files"):
        try:
            # Load audio
            audio, sr = sf.read(noise_file)
            
            if sr != sample_rate:
                logger.warning(f"{noise_file.name}: sample rate {sr} != {sample_rate}, skipping")
                continue
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Segment
            segments = segment_noise_file(
                audio, sample_rate, segment_duration, overlap
            )
            
            # Save segments
            for seg_idx, segment in enumerate(segments):
                out_name = f"{noise_file.stem}_seg{seg_idx:04d}.wav"
                out_path = output_dir / out_name
                sf.write(out_path, segment, sample_rate)
            
            total_segments += len(segments)
            logger.info(f"{noise_file.name}: {len(segments)} segments")
            
        except Exception as e:
            logger.error(f"Error processing {noise_file.name}: {e}")
            continue
    
    logger.info(f"Total segments created: {total_segments}")


def main():
    parser = argparse.ArgumentParser(
        description='Segment long noise files into shorter clips'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory (e.g., data/noise_resampled)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory (e.g., data/noise_segments)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Segment duration in seconds (default: 1.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap in seconds (default: 0.5)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Expected sample rate (default: 44100)')
    
    args = parser.parse_args()
    
    process_noise_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        segment_duration=args.duration,
        overlap=args.overlap,
        sample_rate=args.sample_rate
    )
    
    logger.info("Noise segmentation completed")


if __name__ == '__main__':
    main()