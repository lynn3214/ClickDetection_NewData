#!/usr/bin/env python3
"""诊断原始噪音文件的幅度问题"""
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def diagnose_noise_sources(noise_dir: Path):
    """检查原始噪音文件的幅度统计"""
    
    noise_files = list(noise_dir.rglob('*.wav'))
    
    print("=" * 70)
    print("原始噪音文件诊断")
    print("=" * 70)
    print(f"找到 {len(noise_files)} 个噪音文件\n")
    
    if not noise_files:
        print(f"❌ 未找到噪音文件: {noise_dir}")
        return
    
    stats = []
    
    for noise_file in tqdm(noise_files[:50], desc="检查噪音文件"):  # 只检查前50个
        try:
            audio, sr = sf.read(noise_file)
            
            # 转单声道
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 统计信息
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            mad = np.median(np.abs(audio - np.median(audio)))
            
            stats.append({
                'file': noise_file.name,
                'length': len(audio),
                'rms': rms,
                'peak': peak,
                'mad': mad,
                'mean': np.mean(audio),
                'std': np.std(audio)
            })
            
        except Exception as e:
            print(f"❌ 读取失败 {noise_file.name}: {e}")
            continue
    
    if not stats:
        print("❌ 无有效统计数据")
        return
    
    # 转为数组便于分析
    rms_values = np.array([s['rms'] for s in stats])
    peak_values = np.array([s['peak'] for s in stats])
    mad_values = np.array([s['mad'] for s in stats])
    
    print("\n【整体统计】")
    print(f"RMS统计:")
    print(f"  均值: {rms_values.mean():.6f}")
    print(f"  中位数: {np.median(rms_values):.6f}")
    print(f"  最小: {rms_values.min():.6f}")
    print(f"  最大: {rms_values.max():.6f}")
    print(f"  标准差: {rms_values.std():.6f}")
    
    print(f"\n峰值统计:")
    print(f"  均值: {peak_values.mean():.6f}")
    print(f"  中位数: {np.median(peak_values):.6f}")
    print(f"  最小: {peak_values.min():.6f}")
    print(f"  最大: {peak_values.max():.6f}")
    
    print(f"\nMAD统计:")
    print(f"  均值: {mad_values.mean():.6f}")
    print(f"  中位数: {np.median(mad_values):.6f}")
    print(f"  最小: {mad_values.min():.6f}")
    print(f"  最大: {mad_values.max():.6f}")
    
    # 找出异常文件（RMS > 1.0 或 peak > 2.0）
    print("\n【异常文件（RMS > 1.0 或 Peak > 2.0）】")
    abnormal = [s for s in stats if s['rms'] > 1.0 or s['peak'] > 2.0]
    
    if abnormal:
        print(f"发现 {len(abnormal)} 个异常文件:\n")
        for s in sorted(abnormal, key=lambda x: x['rms'], reverse=True)[:20]:
            print(f"  {s['file']}")
            print(f"    RMS={s['rms']:.4f}, Peak={s['peak']:.4f}, MAD={s['mad']:.6f}")
    else:
        print("✅ 未发现异常文件")
    
    # 模拟归一化流程
    print("\n【模拟归一化流程】")
    print("对最异常的文件进行归一化测试...\n")
    
    # 找最大RMS的文件
    worst = max(stats, key=lambda x: x['rms'])
    worst_file = noise_dir / worst['file']
    
    print(f"测试文件: {worst['file']}")
    print(f"  原始RMS: {worst['rms']:.4f}")
    print(f"  原始Peak: {worst['peak']:.4f}")
    
    # 重新读取
    audio, sr = sf.read(worst_file)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # 步骤1: RMS归一化到0.1 (main.py做的)
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-8:
        audio_rms = audio * (0.1 / rms)
    else:
        audio_rms = audio.copy()
    
    peak = np.max(np.abs(audio_rms))
    if peak > 0.95:
        audio_rms = audio_rms / peak * 0.95
    
    print(f"\n  经过RMS归一化(0.1):")
    print(f"    RMS: {np.sqrt(np.mean(audio_rms**2)):.6f}")
    print(f"    Peak: {np.max(np.abs(audio_rms)):.6f}")
    
    # 步骤2: MAD归一化 (_normalize_segment做的)
    audio_mad = audio_rms - np.mean(audio_rms)
    mad = np.median(np.abs(audio_mad - np.median(audio_mad)))
    
    if mad > 1e-10:
        audio_mad = audio_mad / (1.4826 * mad)
    else:
        rms = np.sqrt(np.mean(audio_mad**2))
        if rms > 1e-10:
            audio_mad = audio_mad / rms
    
    print(f"\n  再经过MAD归一化:")
    print(f"    RMS: {np.sqrt(np.mean(audio_mad**2)):.6f}")
    print(f"    Peak: {np.max(np.abs(audio_mad)):.6f}")
    print(f"    MAD: {mad:.6f}")
    
    # 检查是否可能产生大值
    if np.max(np.abs(audio_mad)) > 10:
        print(f"\n  ⚠️ MAD归一化后峰值过大: {np.max(np.abs(audio_mad)):.2f}")
        print(f"  原因: MAD值过小 ({mad:.6f})，导致除法产生大值")
    
    print("\n" + "=" * 70)
    print("诊断结论:")
    print("=" * 70)
    
    if len(abnormal) > 0:
        print(f"⚠️ 发现 {len(abnormal)} 个异常噪音文件")
        print("问题原因:")
        print("  1. 原始噪音文件幅度过大（未在预处理中归一化）")
        print("  2. segment_noise.py 切割时未进行归一化")
        print("  3. resample_and_filter.py 重采样时未进行归一化")
        print("\n建议:")
        print("  方案1: 在 segment_noise.py 中添加RMS归一化")
        print("  方案2: 在 main.py 中使用更鲁棒的归一化方法")
        print("  方案3: 重新预处理噪音文件，确保幅度合理")
    else:
        print("✅ 原始噪音文件幅度正常")
        print("问题可能在数据处理流程中产生")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        noise_dir = Path(sys.argv[1])
    else:
        noise_dir = Path("data/noise_train_segs")
    
    diagnose_noise_sources(noise_dir)