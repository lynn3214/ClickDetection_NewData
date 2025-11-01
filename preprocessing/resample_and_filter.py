#!/usr/bin/env python3
"""
Resampling and high-pass filtering utilities for dolphin click detection.
修改: 跳过.mat文件处理(由convert_mat_to_wav.py专门处理)
"""

import argparse
import logging
from pathlib import Path
from typing import Union, List
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
import soundfile as sf
from tqdm import tqdm
from datetime import datetime

# 全局错误收集列表
error_logs: List[str] = []

def resample_and_hpf(
    x: np.ndarray,
    sr_orig: int,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    hp_order: int = 4
) -> np.ndarray:
    """
    Resample signal and apply high-pass filter.
    
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


def load_audio_file(file_path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio from .wav or .pkl file.
    Returns mono waveform (float32) and original sample-rate.
    
    注意: .mat文件不在此处理，由convert_mat_to_wav.py专门处理
    """
    file_path = Path(file_path)
    
    # 如果文件在 noise 目录下，直接当作 pickle 文件处理
    if file_path.parent.name == 'noise':
        import pickle
        noise = pickle.load(open(file_path, 'rb')).astype(np.float32)
        return noise, 96000
    
    suffix = file_path.suffix.lower()

    # ---------- .wav ----------
    if suffix == '.wav':
        audio, sr = sf.read(file_path)
        audio = audio.mean(1) if audio.ndim == 2 else audio
        return audio.astype(np.float32), int(sr)

    # ---------- .pkl ----------
    if suffix == '.pkl':
        import pickle
        noise = pickle.load(open(file_path, 'rb')).astype(np.float32)
        return noise, 96000

    raise ValueError(f'Unsupported file format: {suffix}')

    
def _process_waveform_and_save(wave: np.ndarray, sr_orig: int,
                               out_path: Path, cfg):
    """Process single waveform and save to file."""
    # Ensure output file has .wav suffix so soundfile can infer format
    out_path = out_path.with_suffix('.wav')
    y = resample_and_hpf(wave, sr_orig,
                         cfg['sr_target'], cfg['hp_cutoff'], cfg['hp_order'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, cfg['sr_target'])
    logging.debug(f"Saved: {out_path}")


def process_file(
    input_path: Path,
    output_path: Path,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    hp_order: int = 4,
    skip_mat: bool = True  # 新增参数：是否跳过MAT文件
) -> bool:
    """
    Process a single audio file.
    
    Returns:
        True if processed, False if skipped
    """
    # 跳过MAT文件
    if skip_mat and input_path.suffix.lower() == '.mat':
        logging.info(f"Skipped .mat file (use convert_mat_to_wav.py): {input_path.name}")
        return False
    
    try:
        cfg = {
            'sr_target': sr_target,
            'hp_cutoff': hp_cutoff,
            'hp_order': hp_order
        }
        
        audio, sr_orig = load_audio_file(input_path)
        
        if audio.ndim == 2:  # pickle 情况，多行 -> 已在名称中添加索引并写为 .wav
            for i, row in enumerate(audio):
                out_file = output_path.with_suffix('').as_posix() + f'_{i:05d}.wav'
                _process_waveform_and_save(row, sr_orig, Path(out_file), cfg)
            logging.info(f"Processed {len(audio)} segments from: {input_path}")
        else:
            # Ensure single-file outputs become .wav
            _process_waveform_and_save(audio, sr_orig, output_path, cfg)
            logging.info(f"Processed: {input_path.name} -> {output_path.with_suffix('.wav').name}")
        
        return True
            
    except Exception as e:
        error_msg = f"Error processing {input_path}: {str(e)}"
        error_logs.append(error_msg)
        logging.error(error_msg)
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Resample and filter audio files (skips .mat files by default)'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory or file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--sr_target', type=int, default=44100,
                        help='Target sampling rate (default: 44100)')
    parser.add_argument('--hp_cutoff', type=int, default=1000,
                        help='High-pass cutoff frequency (default: 1000)')
    parser.add_argument('--hp_order', type=int, default=4,
                        help='Filter order (default: 4)')
    parser.add_argument('--process-mat', action='store_true',
                        help='Process .mat files (default: skip them)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    skip_mat = not args.process_mat
    
    if skip_mat:
        logging.info("跳过 .mat 文件处理 (使用 convert_mat_to_wav.py 单独处理)")
    
    if input_path.is_file():
        process_file(
            input_path, 
            output_path / input_path.name,
            args.sr_target,
            args.hp_cutoff,
            args.hp_order,
            skip_mat
        )
    elif input_path.is_dir():
        # 处理所有文件
        files = list(input_path.rglob('*'))
        # 文件筛选逻辑
        files = [f for f in files if (
            f.is_file() and
            f.name != '.DS_Store' and  # 排除 .DS_Store
            (f.suffix.lower() in ('.wav', '.mat', '.pkl') or  # 有后缀的文件
             f.parent.name == 'noise')                        # noise目录下的文件
        )]
        
        # 统计信息
        mat_files = [f for f in files if f.suffix.lower() == '.mat']
        other_files = [f for f in files if f.suffix.lower() != '.mat']
        
        if skip_mat and mat_files:
            logging.warning(f"\n发现 {len(mat_files)} 个 .mat 文件将被跳过")
            logging.warning("请使用以下命令处理MAT文件:")
            logging.warning(f"  python convert_mat_to_wav.py")
            logging.warning(f"  然后将转换后的WAV文件放到 {output_path}")
        
        logging.info(f"\n将处理 {len(other_files)} 个非MAT文件")
        
        processed_count = 0
        skipped_count = 0
        
        for file_path in tqdm(files, desc="Processing files"):
            rel_path = file_path.relative_to(input_path)
            out_path = output_path / rel_path
            
            success = process_file(
                file_path,
                out_path,
                args.sr_target,
                args.hp_cutoff,
                args.hp_order,
                skip_mat
            )
            
            if success:
                processed_count += 1
            else:
                skipped_count += 1
        
        # 处理完成后，保存错误日志
        if error_logs:
            log_file = output_path / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(error_logs))
            
            print(f"\n发现 {len(error_logs)} 个错误。详细信息已保存到: {log_file}")
            print("\n错误摘要:")
            for error in error_logs[:10]:  # 只显示前10个
                print(f"- {error}")
            if len(error_logs) > 10:
                print(f"... 以及其他 {len(error_logs) - 10} 个错误")
        
        # 处理摘要
        print("\n" + "=" * 70)
        print("处理完成")
        print("=" * 70)
        print(f"已处理: {processed_count} 个文件")
        print(f"已跳过: {skipped_count} 个文件 (包括 {len(mat_files)} 个 .mat 文件)")
        print(f"错误数: {len(error_logs)}")
        
        if skip_mat and mat_files:
            print("\n下一步: 处理MAT文件")
            print(f"  1. 运行: python convert_mat_to_wav.py")
            print(f"  2. 将转换后的WAV文件移动到: {output_path}")
        
    else:
        logging.error(f"Input path does not exist: {input_path}")
        return


if __name__ == '__main__':
    main()