# diagnose_dataset.py
"""诊断训练集幅度异常的原因"""
import numpy as np
from pathlib import Path
import json

def diagnose_dataset(dataset_dir):
    train_dir = Path(dataset_dir) / 'train'
    waveforms = np.load(train_dir / 'waveforms.npy')
    labels = np.load(train_dir / 'labels.npy')
    
    with open(train_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("=" * 70)
    print("训练集异常诊断")
    print("=" * 70)
    
    # 1. 按标签分析
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_waves = waveforms[pos_mask]
    neg_waves = waveforms[neg_mask]
    
    print("\n【按标签分析】")
    print(f"正样本 (label=1, click trains):")
    print(f"  数量: {len(pos_waves)}")
    print(f"  幅度范围: [{pos_waves.min():.4f}, {pos_waves.max():.4f}]")
    print(f"  均值: {pos_waves.mean():.6f}, 标准差: {pos_waves.std():.4f}")
    print(f"  RMS均值: {np.sqrt(np.mean(pos_waves**2, axis=1)).mean():.4f}")
    print(f"  异常值(>2.0): {np.sum(np.abs(pos_waves) > 2.0)}")
    
    print(f"\n负样本 (label=0, noise):")
    print(f"  数量: {len(neg_waves)}")
    print(f"  幅度范围: [{neg_waves.min():.4f}, {neg_waves.max():.4f}]")
    print(f"  均值: {neg_waves.mean():.6f}, 标准差: {neg_waves.std():.4f}")
    print(f"  RMS均值: {np.sqrt(np.mean(neg_waves**2, axis=1)).mean():.4f}")
    print(f"  异常值(>2.0): {np.sum(np.abs(neg_waves) > 2.0)}")
    
    # 2. 找出最异常的样本
    print("\n【最异常样本分析】")
    abs_max = np.max(np.abs(waveforms), axis=1)
    top10_idx = np.argsort(abs_max)[-10:][::-1]
    
    print("前10个最大幅度样本:")
    for rank, idx in enumerate(top10_idx, 1):
        sample = waveforms[idx]
        meta = metadata[idx]
        print(f"  {rank}. 样本{idx}: 标签={meta['label']}, "
              f"幅度范围=[{sample.min():.2f}, {sample.max():.2f}], "
              f"RMS={np.sqrt(np.mean(sample**2)):.4f}")
        print(f"     file_id={meta.get('file_id', 'N/A')}")
    
    # 3. 检查是否有特定来源的异常
    print("\n【按file_id统计异常】")
    file_id_stats = {}
    for idx, meta in enumerate(metadata):
        file_id = meta.get('file_id', 'unknown')
        label = meta['label']
        max_amp = np.max(np.abs(waveforms[idx]))
        
        if file_id not in file_id_stats:
            file_id_stats[file_id] = {'count': 0, 'max_amp': 0, 'label': label}
        
        file_id_stats[file_id]['count'] += 1
        file_id_stats[file_id]['max_amp'] = max(file_id_stats[file_id]['max_amp'], max_amp)
    
    # 只显示幅度>2的file_id
    abnormal_files = {k: v for k, v in file_id_stats.items() if v['max_amp'] > 2.0}
    if abnormal_files:
        print(f"发现 {len(abnormal_files)} 个file_id有异常幅度(>2.0):")
        for file_id, stats in sorted(abnormal_files.items(), 
                                     key=lambda x: x[1]['max_amp'], 
                                     reverse=True)[:20]:
            print(f"  {file_id}: 样本数={stats['count']}, "
                  f"最大幅度={stats['max_amp']:.2f}, 标签={stats['label']}")
    else:
        print("未发现file_id级别的异常")
    
    # 4. RMS分布分析
    print("\n【RMS分布分析】")
    rms_all = np.sqrt(np.mean(waveforms**2, axis=1))
    rms_pos = np.sqrt(np.mean(pos_waves**2, axis=1))
    rms_neg = np.sqrt(np.mean(neg_waves**2, axis=1))
    
    print(f"全部样本RMS: 均值={rms_all.mean():.4f}, 中位数={np.median(rms_all):.4f}, "
          f"最大={rms_all.max():.4f}")
    print(f"正样本RMS: 均值={rms_pos.mean():.4f}, 中位数={np.median(rms_pos):.4f}, "
          f"最大={rms_pos.max():.4f}")
    print(f"负样本RMS: 均值={rms_neg.mean():.4f}, 中位数={np.median(rms_neg):.4f}, "
          f"最大={rms_neg.max():.4f}")
    
    # 5. 统计结论
    print("\n【诊断结论】")
    if np.sum(np.abs(pos_waves) > 2.0) > np.sum(np.abs(neg_waves) > 2.0) * 2:
        print("⚠️ 正样本(click train)的异常值明显多于负样本")
        print("   可能原因: click train生成或SNR混合过程未正确归一化")
    elif np.sum(np.abs(neg_waves) > 2.0) > np.sum(np.abs(pos_waves) > 2.0) * 2:
        print("⚠️ 负样本(noise)的异常值明显多于正样本")
        print("   可能原因: 噪音片段归一化失败")
    else:
        print("⚠️ 正负样本都有异常，可能是通用的归一化问题")
    
    print("=" * 70)

if __name__ == "__main__":
    diagnose_dataset("data/training_dataset")