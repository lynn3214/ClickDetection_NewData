#!/usr/bin/env python3
"""追踪噪音样本生成的完整流程，找出异常值产生的环节"""
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
sys.path.append('.')

from training.dataset.segments import DatasetBuilder

def trace_single_file(noise_file: Path, sample_rate: int = 44100):
    """追踪单个噪音文件的处理流程"""
    print("=" * 70)
    print(f"追踪文件: {noise_file.name}")
    print("=" * 70)
    
    # 1. 读取原始文件
    audio, sr = sf.read(noise_file)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    print(f"\n步骤1: 原始文件")
    print(f"  采样率: {sr}")
    print(f"  长度: {len(audio)} 样本")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"  Peak: {np.max(np.abs(audio)):.6f}")
    print(f"  范围: [{audio.min():.6f}, {audio.max():.6f}]")
    
    # 2. 重采样（如果需要）
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        print(f"\n步骤2: 重采样到 {sample_rate}Hz")
        print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        print(f"  Peak: {np.max(np.abs(audio)):.6f}")
    
    # 3. main.py中的RMS归一化
    audio_after_rms = audio.copy()
    rms = np.sqrt(np.mean(audio_after_rms**2))
    if rms > 1e-8:
        target_rms = 0.1
        audio_after_rms = audio_after_rms * (target_rms / rms)
    
    peak = np.max(np.abs(audio_after_rms))
    if peak > 0.95:
        audio_after_rms = audio_after_rms / peak * 0.95
    
    print(f"\n步骤3: main.py的RMS归一化")
    print(f"  目标RMS: 0.1")
    print(f"  RMS: {np.sqrt(np.mean(audio_after_rms**2)):.6f}")
    print(f"  Peak: {np.max(np.abs(audio_after_rms)):.6f}")
    print(f"  范围: [{audio_after_rms.min():.6f}, {audio_after_rms.max():.6f}]")
    
    # 4. 调用 build_negative_samples
    builder = DatasetBuilder(sample_rate=sample_rate)
    file_id = noise_file.stem
    
    print(f"\n步骤4: build_negative_samples 提取片段")
    # 只生成1个样本来测试
    negative_samples = builder.build_negative_samples(
        audio_after_rms, file_id, n_samples=1
    )
    
    if negative_samples:
        sample = negative_samples[0]['waveform']
        print(f"  样本长度: {len(sample)}")
        print(f"  RMS: {np.sqrt(np.mean(sample**2)):.6f}")
        print(f"  Peak: {np.max(np.abs(sample)):.6f}")
        print(f"  范围: [{sample.min():.6f}, {sample.max():.6f}]")
        
        if np.max(np.abs(sample)) > 10:
            print(f"\n⚠️ 发现异常大值: {np.max(np.abs(sample)):.2f}")
            print("问题出在 build_negative_samples 或 _normalize_segment 中")
            
            # 手动模拟_normalize_segment
            print(f"\n详细追踪 _normalize_segment:")
            segment = sample.copy()
            
            # 去均值
            segment = segment - np.mean(segment)
            print(f"  去均值后 Peak: {np.max(np.abs(segment)):.6f}")
            
            # MAD归一化
            mad = np.median(np.abs(segment - np.median(segment)))
            print(f"  MAD: {mad:.8f}")
            
            if mad > 1e-10:
                segment_norm = segment / (1.4826 * mad)
                print(f"  MAD归一化后 Peak: {np.max(np.abs(segment_norm)):.6f}")
            else:
                print(f"  ⚠️ MAD过小，使用RMS归一化")
                rms = np.sqrt(np.mean(segment**2))
                if rms > 1e-10:
                    segment_norm = segment / rms
                    print(f"  RMS归一化后 Peak: {np.max(np.abs(segment_norm)):.6f}")
        else:
            print(f"  ✅ 归一化正常")
    else:
        print("  ❌ 未生成样本")
    
    return audio_after_rms, negative_samples


def find_problematic_file():
    """从训练集中找出异常样本，反向追踪源文件"""
    print("=" * 70)
    print("从训练集反向追踪异常样本")
    print("=" * 70)
    
    # 加载训练集
    train_dir = Path("data/training_dataset/train")
    waveforms = np.load(train_dir / 'waveforms.npy')
    labels = np.load(train_dir / 'labels.npy')
    
    import json
    with open(train_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # 找出最大幅度的负样本
    neg_mask = labels == 0
    neg_waveforms = waveforms[neg_mask]
    neg_metadata = [m for m, l in zip(metadata, labels) if l == 0]
    
    # 找最大值
    abs_max = np.max(np.abs(neg_waveforms), axis=1)
    worst_idx = np.argmax(abs_max)
    worst_sample = neg_waveforms[worst_idx]
    worst_meta = neg_metadata[worst_idx]
    
    print(f"\n最异常样本:")
    print(f"  file_id: {worst_meta['file_id']}")
    print(f"  Peak: {np.max(np.abs(worst_sample)):.4f}")
    print(f"  RMS: {np.sqrt(np.mean(worst_sample**2)):.4f}")
    
    # 尝试找到源文件
    noise_dir = Path("data/noise_train_segs")
    source_file = noise_dir / f"{worst_meta['file_id']}.wav"
    
    if source_file.exists():
        print(f"\n✅ 找到源文件: {source_file}")
        print("\n开始追踪处理流程...\n")
        trace_single_file(source_file)
    else:
        print(f"\n❌ 源文件不存在: {source_file}")
        print("尝试搜索相似文件名...")
        
        # 搜索
        file_id_prefix = worst_meta['file_id'].split('_seg')[0]
        matching_files = list(noise_dir.glob(f"{file_id_prefix}*.wav"))
        
        if matching_files:
            print(f"找到 {len(matching_files)} 个匹配文件:")
            for f in matching_files[:3]:
                print(f"  - {f.name}")
            
            print(f"\n追踪第一个匹配文件...")
            trace_single_file(matching_files[0])
        else:
            print("未找到匹配文件")


if __name__ == "__main__":
    find_problematic_file()