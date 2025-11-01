#!/usr/bin/env python3
"""诊断测试集与训练集的重叠情况"""

from pathlib import Path
import json

# 3. 检查负样本来源
noise_train = Path('data/noise_train_segs')
noise_test = Path('data/noise_test_segs')

train_noise_files = list(noise_train.glob('*.wav'))
test_noise_files = list(noise_test.glob('*.wav'))

train_noise_sources = set('_'.join(f.stem.split('_')[:-1]) for f in train_noise_files)
test_noise_sources = set('_'.join(f.stem.split('_')[:-1]) for f in test_noise_files)

print(f"\n训练集噪音来自 {len(train_noise_sources)} 个源文件")
print(f"测试集噪音来自 {len(test_noise_sources)} 个源文件")

overlap_noise = train_noise_sources & test_noise_sources
if overlap_noise:
    print(f"⚠️ 警告：噪音源有重叠！{len(overlap_noise)} 个文件")
    print(f"示例: {list(overlap_noise)[:5]}")
