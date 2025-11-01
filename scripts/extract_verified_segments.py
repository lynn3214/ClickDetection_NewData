#!/usr/bin/env python3
"""
根据Audacity验证后的标签提取音频片段
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def parse_audacity_label(label_file: Path):
    """
    解析Audacity标签文件
    
    格式：start\tend\tlabel
    
    Returns:
        [(start, end, label), ...]
    """
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2] if len(parts) > 2 else 'segment'
                labels.append((start, end, label))
    
    return labels


def extract_segments(audio_dir: Path,
                    labels_dir: Path,
                    output_dir: Path,
                    sample_rate: int = 44100):
    """
    根据标签提取片段
    
    Args:
        audio_dir: 音频文件目录
        labels_dir: 标签文件目录（*_labels.txt）
        output_dir: 输出目录
        sample_rate: 采样率
    """
    audio_dir = Path(audio_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有标签文件
    label_files = list(labels_dir.glob('*_labels.txt'))
    
    if not label_files:
        print(f"❌ 未找到标签文件: {labels_dir}")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    total_segments = 0
    
    for label_file in tqdm(label_files, desc="提取片段"):
        # 找到对应的音频文件
        audio_stem = label_file.stem.replace('_labels', '')
        
        # 尝试多种扩展名
        audio_file = None
        for ext in ['.wav', '.flac', '.mp3']:
            candidate = audio_dir / f"{audio_stem}{ext}"
            if candidate.exists():
                audio_file = candidate
                break
        
        if audio_file is None:
            print(f"⚠️  未找到音频文件: {audio_stem}")
            continue
        
        # 读取音频
        try:
            audio, sr = sf.read(audio_file)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 重采样
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        
        except Exception as e:
            print(f"❌ 读取音频失败 {audio_file}: {e}")
            continue
        
        # 解析标签
        labels = parse_audacity_label(label_file)
        
        if not labels:
            print(f"⚠️  标签文件为空: {label_file.name}")
            continue
        
        # 提取每个片段
        for i, (start, end, label) in enumerate(labels):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # 边界检查
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                continue
            
            segment = audio[start_sample:end_sample]
            
            # 归一化
            peak = np.max(np.abs(segment))
            if peak > 0:
                segment = segment / peak * 0.95
            
            # 保存
            filename = f"{audio_stem}_seg{i:04d}_{int(start*1000):06d}ms.wav"
            sf.write(output_dir / filename, segment, sample_rate)
            
            total_segments += 1
    
    print(f"\n✅ 提取完成")
    print(f"总片段数: {total_segments}")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='根据验证标签提取音频片段'
    )
    parser.add_argument('--audio', type=str, required=True,
                       help='音频文件目录')
    parser.add_argument('--labels', type=str, required=True,
                       help='标签文件目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='采样率，默认44100')
    
    args = parser.parse_args()
    
    extract_segments(
        audio_dir=Path(args.audio),
        labels_dir=Path(args.labels),
        output_dir=Path(args.output),
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()