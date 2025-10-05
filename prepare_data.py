#!/usr/bin/env python3
"""
数据准备脚本 - 新策略
根据文件名区分训练数据和测试数据
不处理noise文件，保持原始状态供后续统一处理
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml


def load_config(config_path: Path) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_high_snr_file(file_path: Path) -> bool:
    """检查文件名是否包含highsnr标记"""
    return 'highsnr' in file_path.name.lower()


def scan_labeled_directories(raw_wav_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    扫描标注数据目录，分离高SNR和低SNR文件
    
    返回:
        (high_snr_files, low_snr_files) 元组
    """
    labeled_dirs = [
        '06Nov21_left',
        '06Sept15_left',
        '9 Oct 06 Splash and Penn'
    ]
    
    high_snr_files = []
    low_snr_files = []
    
    for dir_name in labeled_dirs:
        dir_path = raw_wav_dir / dir_name
        if not dir_path.exists():
            logging.warning(f"标注目录未找到: {dir_path}")
            continue
        
        wav_files = list(dir_path.rglob('*.wav'))
        logging.info(f"在 {dir_name} 中找到 {len(wav_files)} 个WAV文件")
        
        for wav_file in wav_files:
            if is_high_snr_file(wav_file):
                high_snr_files.append(wav_file)
            else:
                low_snr_files.append(wav_file)
    
    return high_snr_files, low_snr_files


def organize_training_sources(
    raw_dir: Path,
    training_sources_dir: Path,
    high_snr_files: List[Path],
    copy_files: bool = True
) -> None:
    """
    组织训练数据源
    包括: high-SNR标注文件 + 其他wav源 + mat文件
    """
    training_sources_dir.mkdir(parents=True, exist_ok=True)
    
    raw_wav_dir = raw_dir / 'wav'
    raw_mat_dir = raw_dir / 'mat'
    
    # 1. 组织high-SNR标注文件
    if high_snr_files:
        high_snr_dir = training_sources_dir / 'high_snr_labeled'
        high_snr_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(high_snr_files)} 个high-SNR文件...")
        for src_file in high_snr_files:
            rel_path = src_file.relative_to(raw_wav_dir)
            dst_file = high_snr_dir / rel_path.parts[0] / src_file.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"High-SNR文件已组织到: {high_snr_dir}")
    
    # 2. 组织其他训练数据源（排除测试集目录）
    training_source_dirs = [
        'Dataport Dolphins Underwater Sounds Database',
        'Dolphin Clicks',
        'OceanParkdolphinclicks',
        'Watkins dolphin'
    ]
    
    for source_name in training_source_dirs:
        source_dir = raw_wav_dir / source_name
        if source_dir.exists():
            wav_files = list(source_dir.rglob('*.wav'))
            if wav_files:
                logging.info(f"组织 {len(wav_files)} 个文件从 {source_name}...")
                
                dst_dir = training_sources_dir / source_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                for src_file in wav_files:
                    rel_path = src_file.relative_to(source_dir)
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if copy_files:
                        shutil.copy2(src_file, dst_file)
                    else:
                        if dst_file.exists():
                            dst_file.unlink()
                        dst_file.symlink_to(src_file.resolve())
    
    # 3. 组织mat文件
    if raw_mat_dir.exists():
        mat_files = list(raw_mat_dir.rglob('*.mat'))
        if mat_files:
            logging.info(f"组织 {len(mat_files)} 个MAT文件...")
            
            mat_dst_dir = training_sources_dir / 'mat_files'
            mat_dst_dir.mkdir(parents=True, exist_ok=True)
            
            for src_file in mat_files:
                rel_path = src_file.relative_to(raw_mat_dir)
                dst_file = mat_dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                if copy_files:
                    shutil.copy2(src_file, dst_file)
                else:
                    if dst_file.exists():
                        dst_file.unlink()
                    dst_file.symlink_to(src_file.resolve())
            
            logging.info(f"MAT文件已组织到: {mat_dst_dir}")


def organize_test_data(
    raw_wav_dir: Path,
    test_raw_dir: Path,
    low_snr_files: List[Path],
    copy_files: bool = True
) -> None:
    """
    组织测试数据
    包括: 低SNR标注文件 + Singapore waters文件
    """
    test_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 组织低SNR标注文件
    if low_snr_files:
        low_snr_dir = test_raw_dir / 'low_snr_labeled'
        low_snr_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(low_snr_files)} 个低SNR文件...")
        for src_file in low_snr_files:
            rel_path = src_file.relative_to(raw_wav_dir)
            dst_file = low_snr_dir / rel_path.parts[0] / src_file.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"低SNR文件已组织到: {low_snr_dir}")
    
    # 2. 组织Singapore waters文件
    singapore_src = raw_wav_dir / 'Singapore_waters_detected_clicks'
    if singapore_src.exists():
        singapore_files = list(singapore_src.rglob('*.wav'))
        logging.info(f"组织 {len(singapore_files)} 个Singapore waters文件...")
        
        singapore_dir = test_raw_dir / 'singapore_waters'
        singapore_dir.mkdir(exist_ok=True)
        
        for src_file in singapore_files:
            rel_path = src_file.relative_to(singapore_src)
            dst_file = singapore_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"Singapore文件已组织到: {singapore_dir}")
    else:
        logging.warning(f"Singapore waters目录未找到: {singapore_src}")


def main():
    """主数据准备流程"""
    parser = argparse.ArgumentParser(
        description='准备训练和测试数据（不处理noise）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='原始数据目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='输出基础目录'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='创建符号链接而非复制文件（节省空间）'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细日志'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='仅显示将要执行的操作，不实际执行'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logging.warning(f"配置文件未找到: {config_path}，使用默认值")
        config = {}
    
    # 设置路径
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    raw_wav_dir = raw_dir / 'wav'
    test_raw_dir = output_dir / 'test_raw'
    training_sources_dir = raw_dir / 'training_sources'
    
    if not raw_wav_dir.exists():
        logging.error(f"原始WAV目录未找到: {raw_wav_dir}")
        return 1
    
    logging.info("="*70)
    logging.info("数据准备 - 新策略")
    logging.info("="*70)
    logging.info(f"原始目录: {raw_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"模式: {'符号链接' if args.symlink else '复制'}")
    
    # 扫描标注数据
    logging.info("\n扫描标注数据...")
    high_snr_files, low_snr_files = scan_labeled_directories(raw_wav_dir)
    
    logging.info(f"\n标注数据摘要:")
    logging.info(f"  High-SNR文件（用于训练）: {len(high_snr_files)}")
    logging.info(f"  Low-SNR文件（用于测试）: {len(low_snr_files)}")
    
    if args.dry_run:
        logging.info("\n=== 试运行模式 ===")
        logging.info("\n将组织high-SNR文件到: training_sources/")
        for f in high_snr_files[:5]:
            logging.info(f"  {f.relative_to(raw_wav_dir)}")
        if len(high_snr_files) > 5:
            logging.info(f"  ... 以及其他 {len(high_snr_files)-5} 个文件")
        
        logging.info("\n将组织low-SNR文件到: test_raw/low_snr_labeled/")
        for f in low_snr_files[:5]:
            logging.info(f"  {f.relative_to(raw_wav_dir)}")
        if len(low_snr_files) > 5:
            logging.info(f"  ... 以及其他 {len(low_snr_files)-5} 个文件")
        
        return 0
    
    # 组织训练数据
    logging.info("\n组织训练数据...")
    organize_training_sources(
        raw_dir=raw_dir,
        training_sources_dir=training_sources_dir,
        high_snr_files=high_snr_files,
        copy_files=not args.symlink
    )
    
    # 组织测试数据
    logging.info("\n组织测试数据...")
    organize_test_data(
        raw_wav_dir=raw_wav_dir,
        test_raw_dir=test_raw_dir,
        low_snr_files=low_snr_files,
        copy_files=not args.symlink
    )
    
    # 总结
    logging.info("\n" + "="*70)
    logging.info("数据准备完成")
    logging.info("="*70)
    logging.info("\n下一步操作:")
    logging.info("1. 对training_sources/运行resample_and_filter.py")
    logging.info("   python utils/resample_and_filter.py --input data/raw/training_sources --output data/resampled_training --verbose")
    logging.info("2. 对test_raw/运行resample_and_filter.py（单独目录）")
    logging.info("   python utils/resample_and_filter.py --input data/test_raw --output data/test_resampled --verbose")
    logging.info("3. 对noise目录运行resample_and_filter.py")
    logging.info("   python utils/resample_and_filter.py --input data/raw/noise --output data/noise_resampled --verbose")
    logging.info("4. 对重采样的训练数据运行clip_extractor.py")
    logging.info("   python data_scripts/clip_extractor.py --src_root data/resampled_training --dst_root data/clicks_multi_confidence --verbose")
    logging.info("5. 使用重采样的测试数据进行inference.py测试")
    
    logging.info("\n创建的目录结构:")
    logging.info(f"  {training_sources_dir}/")
    logging.info(f"    ├── high_snr_labeled/ ({len(high_snr_files)} 文件)")
    logging.info(f"    ├── [其他训练源目录]")
    logging.info(f"    └── mat_files/")
    logging.info(f"  {test_raw_dir}/")
    logging.info(f"    ├── low_snr_labeled/ ({len(low_snr_files)} 文件)")
    logging.info(f"    └── singapore_waters/")
    logging.info(f"\n注意: noise文件保持在 data/raw/noise/ 原始位置")
    logging.info(f"      需要单独对其运行resample_and_filter.py")
    
    return 0


if __name__ == '__main__':
    exit(main())