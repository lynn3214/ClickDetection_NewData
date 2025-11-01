#!/usr/bin/env python3
"""
将检测结果导出为Audacity标签格式
格式：start_time\tend_time\tlabel
"""

import argparse
import pandas as pd
from pathlib import Path


def export_to_audacity(csv_path: Path, 
                      output_path: Path,
                      window_ms: float = 120.0):
    """
    转换CSV为Audacity标签格式
    
    Args:
        csv_path: 检测结果CSV路径
        output_path: 输出txt路径
        window_ms: 片段长度（毫秒）
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print(f"⚠️  CSV为空: {csv_path}")
        return
    
    print(f"读取 {len(df)} 个检测事件")
    
    # 按文件分组
    grouped = df.groupby('source_file')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 为每个文件生成标签
    for source_file, group in grouped:
        # 标签文件名：原文件名.txt
        file_stem = Path(source_file).stem
        label_file = output_path.parent / f"{file_stem}_labels.txt"
        
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                peak_time = row['peak_time']
                
                # 计算窗口边界
                half_window = window_ms / 2 / 1000  # 转秒
                start = max(0, peak_time - half_window)
                end = peak_time + half_window
                
                # Audacity格式：start\tend\tlabel
                f.write(f"{start:.6f}\t{end:.6f}\tcandidate\n")
        
        print(f"✅ {file_stem}: {len(group)} 个标签 → {label_file}")
    
    print(f"\n总结：生成 {len(grouped)} 个标签文件")
    print(f"位置：{output_path.parent}")
    print("\n使用方法：")
    print("1. 在Audacity中打开对应音频文件")
    print("2. 选择 File → Import → Labels")
    print("3. 选择对应的 *_labels.txt 文件")
    print("4. 逐个听检并删除误检标签")
    print("5. File → Export → Export Labels 保存验证后的标签")


def main():
    parser = argparse.ArgumentParser(
        description='导出Audacity标签格式'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='检测结果CSV文件')
    parser.add_argument('--output', type=str, required=True,
                       help='输出标签文件（目录）')
    parser.add_argument('--window-ms', type=float, default=120.0,
                       help='片段长度（毫秒），默认120')
    
    args = parser.parse_args()
    
    export_to_audacity(
        csv_path=Path(args.input),
        output_path=Path(args.output),
        window_ms=args.window_ms
    )


if __name__ == '__main__':
    main()