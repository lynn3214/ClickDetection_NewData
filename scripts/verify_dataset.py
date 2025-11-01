# verify_dataset.py
import numpy as np
from pathlib import Path

def verify_dataset(dataset_dir):
    train_dir = Path(dataset_dir) / 'train'
    waveforms = np.load(train_dir / 'waveforms.npy')
    labels = np.load(train_dir / 'labels.npy')
    
    print("=" * 60)
    print("数据集验证报告")
    print("=" * 60)
    print(f"样本数量: {len(waveforms)}")
    print(f"样本形状: {waveforms.shape}")
    print(f"\n幅度统计:")
    print(f"  最小值: {waveforms.min():.4f}")
    print(f"  最大值: {waveforms.max():.4f}")
    print(f"  均值: {waveforms.mean():.6f}")
    print(f"  标准差: {waveforms.std():.4f}")
    print(f"  中位数: {np.median(waveforms):.6f}")
    
    # 检查异常值
    outliers = np.sum(np.abs(waveforms) > 2.0)
    print(f"\n异常值检测 (>2.0): {outliers} 个")
    
    # RMS统计
    rms_per_sample = np.sqrt(np.mean(waveforms**2, axis=1))
    print(f"\nRMS统计:")
    print(f"  RMS均值: {rms_per_sample.mean():.4f}")
    print(f"  RMS中位数: {np.median(rms_per_sample):.4f}")
    print(f"  RMS标准差: {rms_per_sample.std():.4f}")
    
    # 标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n标签分布:")
    for u, c in zip(unique, counts):
        print(f"  类别 {u}: {c} 个 ({c/len(labels)*100:.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset("data/training_dataset")