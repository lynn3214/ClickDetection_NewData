#!/usr/bin/env python3
"""
收集分散的click片段到统一目录
用途：将 detection_results/audio/file1/*.wav, file2/*.wav 等
     收集到 augmented_clicks/*.wav
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import logging


def collect_clicks(input_dir: Path, output_dir: Path, verbose: bool = False):
    """
    收集所有click片段
    
    Args:
        input_dir: 输入目录（如 detection_results/audio）
        output_dir: 输出目录（如 augmented_clicks）
        verbose: 详细日志
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # 查找所有wav文件
    wav_files = list(input_dir.rglob('*.wav'))
    logging.info(f"找到 {len(wav_files)} 个click片段")
    
    if len(wav_files) == 0:
        logging.error(f"未找到wav文件: {input_dir}")
        return
    
    # 复制文件（重命名避免冲突）
    for i, wav_file in enumerate(tqdm(wav_files, desc="收集片段", disable=not verbose)):
        # 构造新文件名：父目录名_原文件名
        parent_name = wav_file.parent.name
        new_name = f"{parent_name}_{wav_file.name}"
        
        # 处理可能的重复
        dst_path = output_dir / new_name
        counter = 1
        while dst_path.exists():
            stem = wav_file.stem
            new_name = f"{parent_name}_{stem}_{counter:03d}.wav"
            dst_path = output_dir / new_name
            counter += 1
        
        shutil.copy2(wav_file, dst_path)
    
    logging.info(f"✅ 收集完成：{len(wav_files)} 个片段 → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='收集click片段到统一目录'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='输入目录（detection_results/audio）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录（augmented_clicks）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细日志')
    
    args = parser.parse_args()
    
    collect_clicks(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()