"""
Main CLI entry point for dolphin click detection pipeline.
Changes:
1. Fix batch-detect logic for saving audio segments
2. Fix build-dataset input paths and SNR mixing logic
3. Add 'collect-clicks' command
4. Add debug output and SNR validation
"""

import argparse
from pathlib import Path
import sys
import shutil  
from tqdm import tqdm 
import random

from utils.config import load_config
from utils.logging.logger import ProjectLogger
from utils.audio_io.manifest import scan_audio_files
#from utils.preprocessing.resample_and_filter import preprocess_audio_file
from detection.candidate_finder.dynamic_threshold import AdaptiveDetector, DetectionParams
from detection.segmenter.cropper import ClickSegmenter
from detection.features_event.event_stats import EventStatsExtractor, save_event_stats_csv
from detection.train_builder.cluster import TrainBuilder, save_trains_csv
from detection.fusion.decision import FusionDecider, FusionConfig
from detection.export.writer import ExportWriter
from training.dataset.segments import DatasetBuilder
from training.augment.pipeline import AugmentationPipeline
from models.cnn1d.model import create_model
from models.cnn1d.inference import ClickDetectorInference
from training.train.loop import Trainer, create_dataloaders
from training.eval.report import EvaluationReporter

import numpy as np
import torch
import soundfile as sf
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Dolphin Click Detection Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan audio files')
    scan_parser.add_argument('--input-dir', type=str, required=True,
                            help='Input directory to scan')
    scan_parser.add_argument('--output', type=str, required=True,
                            help='Output manifest file')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect click candidates')
    detect_parser.add_argument('--input', type=str, required=True,
                              help='Input audio file')
    detect_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory')
    detect_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                              help='Detection config file')
    
    # ========== Added: collect-clicks command ==========
    collect_parser = subparsers.add_parser(
        'collect-clicks',
        help='Collect all click segments into a single directory'
    )
    collect_parser.add_argument('--input', type=str, required=True,
                               help='Input directory (detection_results/audio)')
    collect_parser.add_argument('--output', type=str, required=True,
                               help='Output directory for collected clicks')
    collect_parser.add_argument('--verbose', '-v', action='store_true')
    
    # Batch detect command
    batch_detect_parser = subparsers.add_parser(
        'batch-detect', 
        help='Batch detect clicks in directory'
    )
    batch_detect_parser.add_argument('--input-dir', type=str, required=True,
                                    help='Input directory containing wav files')
    batch_detect_parser.add_argument('--output-dir', type=str, required=True,
                                    help='Output directory for results')
    batch_detect_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                                    help='Detection config file')
    batch_detect_parser.add_argument('--save-audio', action='store_true',
                                    help='Save extracted click segments')
    batch_detect_parser.add_argument('--recursive', action='store_true',
                                    help='Search recursively for wav files')
    batch_detect_parser.add_argument('--segment-ms', type=float, default=120.0,
                                    help='Segment length in milliseconds (default: 120)')
    
    # Trains command
    trains_parser = subparsers.add_parser('trains', help='Build click trains')
    trains_parser.add_argument('--events-csv', type=str, required=True,
                              help='Events CSV file')
    trains_parser.add_argument('--output', type=str, required=True,
                              help='Output trains CSV')
    trains_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                              help='Detection config file')
    
    # ========== Modified: build-dataset update parameter descriptions ==========
    dataset_parser = subparsers.add_parser('build-dataset',
                                          help='Build training dataset with SNR mixing')
    dataset_parser.add_argument('--events-dir', type=str, required=True,
                               help='Directory containing click wav files (augmented_clicks)')
    dataset_parser.add_argument('--noise-dir', type=str, required=True,
                               help='Directory containing noise segments (noise_train_segs)')
    dataset_parser.add_argument('--output-dir', type=str, required=True,
                               help='Output dataset directory')
    dataset_parser.add_argument('--config', type=str, default='configs/training.yaml',
                               help='Training config file')
    dataset_parser.add_argument('--save-wav', action='store_true',
                               help='Save mixed samples as wav files for inspection')
    dataset_parser.add_argument('--verbose', '-v', action='store_true')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train CNN model')
    train_parser.add_argument('--dataset-dir', type=str, required=True,
                             help='Dataset directory')
    train_parser.add_argument('--output-dir', type=str, required=True,
                             help='Output directory for checkpoints')
    train_parser.add_argument('--config', type=str, default='configs/training.yaml',
                             help='Training config file')
    train_parser.add_argument('--verbose', '-v', action='store_true',
                         help='Verbose logging')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Model checkpoint path')
    eval_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Test dataset directory')
    eval_parser.add_argument('--output-dir', type=str, required=True,
                            help='Output directory for reports')

    # Eval-wav command (新增)
    eval_wav_parser = subparsers.add_parser(
        'eval-wav', 
        help='Evaluate model on wav files (file-level classification)'
    )
    eval_wav_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Model checkpoint path')
    eval_wav_parser.add_argument('--positive-dir', type=str, 
                                default='data/test_resampled',
                                help='Directory with files containing clicks')
    eval_wav_parser.add_argument('--negative-dir', type=str,
                                default='data/noise_resampled',
                                help='Directory with noise files')
    eval_wav_parser.add_argument('--output-dir', type=str, required=True,
                                help='Output directory for results')
    eval_wav_parser.add_argument('--config', type=str, 
                                default='configs/eval_wav.yaml',
                                help='Evaluation config file')
    eval_wav_parser.add_argument('--file-threshold', type=float,
                                help='Window-level threshold (overrides config)')
    eval_wav_parser.add_argument('--min-positive-ratio', type=float,
                                help='Min positive ratio (overrides config)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export final detections')
    export_parser.add_argument('--input', type=str, required=True,
                              help='Input audio file')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Model checkpoint')
    export_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory')
    export_parser.add_argument('--config', type=str, default='configs/inference.yaml',
                              help='Inference config file')
    
    return parser


def cmd_scan(args):
    """Execute scan command."""
    logger = ProjectLogger()
    logger.info(f"Scanning directory: {args.input_dir}")
    
    manifest = scan_audio_files(
        Path(args.input_dir),
        extensions=['.wav'],
        recursive=True
    )
    
    manifest.to_csv(args.output, index=False)
    logger.info(f"Manifest saved to {args.output}")
    logger.info(f"Found {len(manifest)} audio files")


def cmd_detect(args):
    """Execute detect command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info(f"Detecting clicks in: {args.input}")
    
    # Load audio
    audio, sr = sf.read(args.input)
    
    # Initialize detector
    params = DetectionParams(
        tkeo_threshold=config['thresholds']['tkeo_z'],
        ste_threshold=config['thresholds']['ste_z'],
        hfc_threshold=config['thresholds']['hfc_z'],
        high_low_ratio_threshold=config['thresholds']['high_low_ratio'],
        envelope_width_min=config['envelope']['width_min_ms'],
        envelope_width_max=config['envelope']['width_max_ms'],
        spectral_centroid_min=config['thresholds']['spectral_centroid_min'],
        refractory_ms=config['refractory_ms']
    )
    
    detector = AdaptiveDetector(sample_rate=sr, params=params)
    
    # Detect
    candidates = detector.batch_detect(
        audio,
        chunk_duration=config['batch']['chunk_duration_s'],
        overlap=config['batch']['overlap_s']
    )
    
    logger.info(f"Detected {len(candidates)} candidates")
    
    # Extract event statistics
    stats_extractor = EventStatsExtractor(sample_rate=sr)
    stats_list = []
    
    for candidate in candidates:
        stats = stats_extractor.extract_event_stats(audio, candidate)
        stats_list.append(stats)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    events_csv = output_dir / 'events.csv'
    save_event_stats_csv(stats_list, events_csv)
    
    logger.info(f"Events saved to {events_csv}")

# ========== Added: collect-clicks command implementation ==========
def cmd_collect_clicks(args):
    """收集所有 click 片段到单个目录."""
    logger = ProjectLogger()
    logger.info("开始收集 click 片段...")
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 wav 文件
    wav_files = list(input_dir.rglob('*.wav'))
    logger.info(f"找到 {len(wav_files)} 个 click 片段")
    
    if not wav_files:
        logger.error(f"未找到任何 WAV 文件: {input_dir}")
        return
    
    # 复制到输出目录(重命名以避免冲突)
    for i, wav_file in enumerate(tqdm(wav_files, desc="收集片段")):
        # 保留原始文件前缀(来自父目录)
        parent_name = wav_file.parent.name
        new_name = f"{parent_name}_{wav_file.name}"
        shutil.copy2(wav_file, output_dir / new_name)
    
    logger.info(f"✅ 收集完成,保存到 {output_dir}")

# ========== Modified: batch-detect save audio segments ==========
def cmd_batch_detect(args):
    """Execute batch-detect command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 简化参数获取
    segment_ms = args.segment_ms
    
    # 扫描 wav 文件
    logger.info(f"扫描目录: {input_dir}")
    if args.recursive:
        wav_files = list(input_dir.rglob('*.wav'))
    else:
        wav_files = list(input_dir.glob('*.wav'))
    
    logger.info(f"找到 {len(wav_files)} 个 wav 文件")
    
    if not wav_files:
        logger.error(f"未找到任何 WAV 文件: {input_dir}")
        return
    
    # 初始化检测器
    params = DetectionParams(
        tkeo_threshold=config['thresholds']['tkeo_z'],
        ste_threshold=config['thresholds']['ste_z'],
        hfc_threshold=config['thresholds']['hfc_z'],
        high_low_ratio_threshold=config['thresholds']['high_low_ratio'],
        envelope_width_min=config['envelope']['width_min_ms'],
        envelope_width_max=config['envelope']['width_max_ms'],
        spectral_centroid_min=config['thresholds']['spectral_centroid_min'],
        refractory_ms=config['refractory_ms'],
        enable_transient_filter=config['transient'].get('enable_filter', True),
        min_dolphin_likelihood=config['transient'].get('min_dolphin_likelihood', 0.3)
    )
    
    all_stats = []
    total_candidates = 0
    
    # 处理每个文件
    for wav_file in tqdm(wav_files, desc="检测 clicks"):
        try:
            audio, sr = sf.read(wav_file)
            file_id = wav_file.stem
            
            # 检测
            detector = AdaptiveDetector(sample_rate=sr, params=params)
            candidates = detector.batch_detect(
                audio,
                chunk_duration=config['batch']['chunk_duration_s'],
                overlap=config['batch']['overlap_s']
            )
            
            total_candidates += len(candidates)
            logger.info(f"{wav_file.name}: 检测到 {len(candidates)} 个候选")
            
            # 提取统计信息
            stats_extractor = EventStatsExtractor(sample_rate=sr)
            for candidate in candidates:
                stats = stats_extractor.extract_event_stats(audio, candidate)
                stats['file_id'] = file_id
                stats['source_file'] = str(wav_file)
                all_stats.append(stats)
            
            # 保存音频片段
            if args.save_audio and candidates:
                audio_dir = output_dir / 'audio' / file_id
                audio_dir.mkdir(parents=True, exist_ok=True)
                
                segment_samples = int(segment_ms * sr / 1000)
                
                for i, candidate in enumerate(candidates):
                    # 提取固定长度片段(以峰值为中心)
                    half_window = segment_samples // 2
                    start_idx = max(0, candidate.peak_idx - half_window)
                    end_idx = min(len(audio), candidate.peak_idx + half_window)
                    
                    segment = audio[start_idx:end_idx]
                    
                    # 如需填充
                    if len(segment) < segment_samples:
                        pad_left = (segment_samples - len(segment)) // 2
                        pad_right = segment_samples - len(segment) - pad_left
                        segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
                    
                    # 归一化(避免后续削波)
                    peak_val = np.max(np.abs(segment))
                    if peak_val > 0:
                        segment = segment / peak_val * 0.95  # 留 5% 余量
                    
                    # 保存
                    timestamp_ms = int(candidate.peak_time * 1000)
                    filename = f"click_{i:04d}_{timestamp_ms:08d}ms.wav"
                    sf.write(audio_dir / filename, segment, sr)
                
        except Exception as e:
            logger.error(f"处理 {wav_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总 CSV
    if all_stats:
        csv_path = output_dir / 'all_events.csv'
        save_event_stats_csv(all_stats, csv_path)
        logger.info(f"✅ 保存 {len(all_stats)} 个事件到 {csv_path}")
        logger.info(f"✅ 处理完成! 总共检测到 {total_candidates} 个 click 候选")
    else:
        logger.warning("⚠️  未检测到任何事件")


def cmd_trains(args):
    """Execute trains command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info(f"Building trains from: {args.events_csv}")
    
    # Load events (this is simplified - in practice you'd reconstruct ClickCandidate objects)
    events_df = pd.read_csv(args.events_csv)
    
    # For this example, we'll need to import the candidates from somewhere
    # This is a placeholder - actual implementation would need to deserialize candidates
    logger.warning("Train building from CSV requires candidate objects - not fully implemented")
    
    # Initialize train builder
    train_builder = TrainBuilder(
        min_ici_ms=config['train']['min_ici_ms'],
        max_ici_ms=config['train']['max_ici_ms'],
        min_train_clicks=config['train']['min_train_clicks']
    )
    
    # Build trains (placeholder)
    # trains = train_builder.build_trains(candidates)
    
    logger.info("Train building command - implementation depends on serialization format")


def cmd_build_dataset(args):
    """Execute build-dataset command（修正版 - 分离训练/验证集生成）."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("构建训练数据集（click train序列 + SNR混合）")
    logger.info("=" * 60)

    # ========== 清理上次输出 ==========
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        logger.info(f"清理上次输出: {output_dir}")
        train_dir = output_dir / 'train'
        if train_dir.exists():
            shutil.rmtree(train_dir)
        val_dir = output_dir / 'val'
        if val_dir.exists():
            shutil.rmtree(val_dir)
        debug_dir = output_dir / 'debug_wavs'
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 初始化配置 ==========
    dataset_config = config['dataset']
    sample_rate = config.get('sample_rate', 44100)
    window_ms = dataset_config.get('window_ms', 120.0)
    
    logger.info(f"样本率: {sample_rate} Hz")
    logger.info(f"统一样本长度: 500ms ({int(0.5 * sample_rate)} 样本)")
    
    builder = DatasetBuilder(
        sample_rate=sample_rate,
        window_ms=window_ms,
        random_offset_ms=dataset_config['random_offset_ms'],
        unified_length_ms=500.0
    )
    
    # 初始化增强器
    augmentation_config = config.get('augmentation', {})
    augmenter = AugmentationPipeline(
        sample_rate=sample_rate,
        snr_range=tuple(augmentation_config.get('snr_range', [-5, 15])),
        time_shift_ms=augmentation_config.get('time_shift_ms', 10.0),
        amplitude_range=tuple(augmentation_config.get('amplitude_range', [0.8, 1.25])),
        apply_prob=augmentation_config.get('apply_prob', 0.8)
    )
    
    logger.info(f"\n增强设置:")
    logger.info(f"  SNR范围: {augmenter.snr_range} dB")
    logger.info(f"  时间偏移: ±{augmentation_config.get('time_shift_ms', 10.0)} ms")
    logger.info(f"  应用概率: {augmenter.apply_prob}")
    
    events_dir = Path(args.events_dir)
    noise_dir = Path(args.noise_dir)
    
    # ========== 加载噪声池 ==========
    logger.info(f"\n加载噪声文件池...")
    noise_files = list(noise_dir.rglob('*.wav'))
    
    if not noise_files:
        logger.error(f"未找到噪声文件: {noise_dir}")
        return
    
    logger.info(f"找到 {len(noise_files)} 个噪声文件")
    
    # 加载噪声池（最多100个文件）
    max_noise_files = min(len(noise_files), 100)
    noise_pool = []
    selected_noise_files = random.sample(noise_files, max_noise_files)
    min_noise_length = int(0.5 * sample_rate)  # 500ms
    
    for noise_file in tqdm(selected_noise_files, desc="加载噪声池"):
        try:
            noise_audio, sr = sf.read(noise_file)
            
            # 重采样
            if sr != sample_rate:
                import librosa
                noise_audio = librosa.resample(noise_audio, orig_sr=sr, target_sr=sample_rate)
            
            # 转单声道
            if noise_audio.ndim == 2:
                noise_audio = noise_audio.mean(axis=1)
            
            # 🔧 确保噪声足够长（如果短于500ms，重复填充）
            if len(noise_audio) < min_noise_length:
                repeats = int(np.ceil(min_noise_length / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)
                logger.debug(f"噪声文件 {noise_file.name} 过短，已重复填充")
            
            # RMS归一化到固定水平
            rms = np.sqrt(np.mean(noise_audio**2))
            if rms > 1e-8:
                target_rms = 0.1
                noise_audio = noise_audio * (target_rms / rms)
            
            # 峰值裁剪
            peak = np.max(np.abs(noise_audio))
            if peak > 0.95:
                noise_audio = noise_audio / peak * 0.95
            
            noise_pool.append(noise_audio)
            
        except Exception as e:
            logger.error(f"加载噪声失败 {noise_file}: {e}")
            continue

    logger.info(f"成功加载 {len(noise_pool)} 个噪声片段（已RMS归一化到0.1）")
    
    if len(noise_pool) == 0:
        logger.error("噪声池为空！")
        return
    
    # ========== 准备Click素材文件 ==========
    logger.info(f"\n准备Click素材文件...")
    positive_files = list(events_dir.rglob('*.wav'))

    if not positive_files:
        logger.error(f"未找到click文件: {events_dir}")
        return

    logger.info(f"找到 {len(positive_files)} 个click片段用于组建train")

    # ========== 生成Click Train序列样本 ==========
    train_config = dataset_config.get('click_train', {})
    enable_train = train_config.get('enable', True)

    if not enable_train:
        logger.error("❌ 必须启用click train生成！")
        return

    logger.info(f"\n生成Click Train序列样本...")

    # 🔧 新增：读取训练集和验证集样本数
    n_train_samples = train_config.get('n_samples', 8000)
    n_val_samples = train_config.get('val_samples', 2000)
    train_length_ms = train_config.get('train_length_ms', 500.0)
    min_clicks = train_config.get('min_clicks', 2)
    max_clicks = train_config.get('max_clicks', 5)
    ici_range_ms = tuple(train_config.get('ici_range_ms', [10.0, 80.0]))

    logger.info(f"  训练集样本数: {n_train_samples}")
    logger.info(f"  验证集样本数: {n_val_samples}")
    logger.info(f"  Train长度: {train_length_ms}ms")
    logger.info(f"  Clicks数范围: {min_clicks}-{max_clicks}")
    logger.info(f"  ICI范围: {ici_range_ms} ms")

    # 🔧 分别生成训练集和验证集
    logger.info(f"\n生成训练集 click trains...")
    train_samples = builder.build_click_train_samples(
        click_files=positive_files,
        n_train_samples=n_train_samples,
        train_length_ms=train_length_ms,
        min_clicks=min_clicks,
        max_clicks=max_clicks,
        ici_range_ms=ici_range_ms,
        sample_rate=sample_rate,
        noise_pool=noise_pool,
        augmenter=augmenter
    )

    logger.info(f"\n生成验证集 click trains...")
    val_positive_samples = builder.build_click_train_samples(
        click_files=positive_files,
        n_train_samples=n_val_samples,
        train_length_ms=train_length_ms,
        min_clicks=min_clicks,
        max_clicks=max_clicks,
        ici_range_ms=ici_range_ms,
        sample_rate=sample_rate,
        noise_pool=noise_pool,
        augmenter=augmenter
    )

    logger.info(f"  训练集正样本: {len(train_samples)}")
    logger.info(f"  验证集正样本: {len(val_positive_samples)}")
    
    # ========== 处理负样本 ==========
    logger.info(f"\n处理负样本...")
    
    balance_ratio = dataset_config.get('balance_ratio', 1.0)
    
    # 训练集负样本
    n_negative_train = int(len(train_samples) * balance_ratio)
    n_negative_per_file_train = max(1, n_negative_train // len(noise_files))
    
    # 验证集负样本
    n_negative_val = int(len(val_positive_samples) * balance_ratio)
    n_negative_per_file_val = max(1, n_negative_val // len(noise_files))
    
    logger.info(f"训练集目标负样本数: {n_negative_train}")
    logger.info(f"验证集目标负样本数: {n_negative_val}")
    
    train_negative_samples = []
    val_negative_samples = []
    
    # 🔧 使用不同的随机种子确保训练集和验证集的负样本不重叠
    train_noise_files = noise_files[:int(len(noise_files) * 0.8)]  # 80%用于训练
    val_noise_files = noise_files[int(len(noise_files) * 0.8):]    # 20%用于验证
    
    logger.info(f"训练集使用 {len(train_noise_files)} 个噪声文件")
    logger.info(f"验证集使用 {len(val_noise_files)} 个噪声文件")
    
    # 生成训练集负样本（添加RMS归一化）
    for noise_file in tqdm(train_noise_files, desc="生成训练集负样本"):
        try:
            audio, sr = sf.read(noise_file)
            
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 🔧 关键修复：添加RMS归一化（与噪音池加载保持一致）
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-8:
                target_rms = 0.1
                audio = audio * (target_rms / rms)
            
            # 峰值裁剪
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            file_id = noise_file.stem
            negative_samples = builder.build_negative_samples(
                audio, file_id, n_negative_per_file_train
            )
            train_negative_samples.extend(negative_samples)
            
        except Exception as e:
            logger.error(f"处理 {noise_file} 时出错: {e}")
            continue
    
    # 生成验证集负样本
    for noise_file in tqdm(val_noise_files, desc="生成验证集负样本"):
        try:
            audio, sr = sf.read(noise_file)
            
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 🔧 关键修复：添加RMS归一化
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-8:
                target_rms = 0.1
                audio = audio * (target_rms / rms)
            
            # 峰值裁剪
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            file_id = noise_file.stem
            negative_samples = builder.build_negative_samples(
                audio, file_id, n_negative_per_file_val
            )
            val_negative_samples.extend(negative_samples)
            
        except Exception as e:
            logger.error(f"处理 {noise_file} 时出错: {e}")
            continue
    
    logger.info(f"训练集负样本: {len(train_negative_samples)}")
    logger.info(f"验证集负样本: {len(val_negative_samples)}")
    
    # ========== 合并和平衡 ==========
    logger.info(f"\n平衡数据集...")
    
    # 训练集
    all_train_samples = train_samples + train_negative_samples
    balanced_train = builder.balance_dataset(all_train_samples, balance_ratio=balance_ratio)
    
    # 验证集
    all_val_samples = val_positive_samples + val_negative_samples
    balanced_val = builder.balance_dataset(all_val_samples, balance_ratio=balance_ratio)
    
    logger.info(f"平衡后训练集: {len(balanced_train)}")
    logger.info(f"平衡后验证集: {len(balanced_val)}")
    
    # 验证样本形状
    unique_lengths = set(len(s['waveform']) for s in balanced_train + balanced_val)
    if len(unique_lengths) > 1:
        logger.error(f"⚠️ 样本长度不一致: {unique_lengths}")
        return
    else:
        logger.info(f"✅ 所有样本长度统一: {list(unique_lengths)[0]} 样本")
    
    # 统计
    n_train_pos = sum(1 for s in balanced_train if s['label'] == 1)
    n_train_neg = len(balanced_train) - n_train_pos
    n_val_pos = sum(1 for s in balanced_val if s['label'] == 1)
    n_val_neg = len(balanced_val) - n_val_pos
    
    logger.info(f"\n最终组成:")
    logger.info(f"  训练集 - 正样本: {n_train_pos}, 负样本: {n_train_neg}")
    logger.info(f"  验证集 - 正样本: {n_val_pos}, 负样本: {n_val_neg}")
    
    # ========== 保存 ==========
    logger.info(f"\n保存到 {output_dir}")
    
    builder.save_dataset(balanced_train, output_dir, split='train')
    builder.save_dataset(balanced_val, output_dir, split='val')
    
    # ========== 保存音频样本用于验证 ==========
    if args.save_wav:
        debug_dir = output_dir / 'debug_wavs'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集Click Train样本
        if train_samples:
            num_examples = min(10, len(train_samples))
            for i, sample in enumerate(random.sample(train_samples, num_examples)):
                sample_path = debug_dir / f'train_click_train_{i:02d}.wav'
                sf.write(str(sample_path), sample['waveform'], sample_rate)
            logger.info(f"已保存 {num_examples} 个训练集Click Train样本")
        
        # 保存验证集Click Train样本
        if val_positive_samples:
            num_examples = min(5, len(val_positive_samples))
            for i, sample in enumerate(random.sample(val_positive_samples, num_examples)):
                sample_path = debug_dir / f'val_click_train_{i:02d}.wav'
                sf.write(str(sample_path), sample['waveform'], sample_rate)
            logger.info(f"已保存 {num_examples} 个验证集Click Train样本")
        
        # 保存负样本示例
        if train_negative_samples:
            num_examples = min(10, len(train_negative_samples))
            for i, sample in enumerate(random.sample(train_negative_samples, num_examples)):
                sample_path = debug_dir / f'train_noise_{i:02d}.wav'
                sf.write(str(sample_path), sample['waveform'], sample_rate)
            logger.info(f"已保存 {num_examples} 个训练集噪声样本")
        
        logger.info(f"调试音频样本保存到: {debug_dir}")
    
    # ========== 总结 ==========
    logger.info("\n" + "=" * 60)
    logger.info("✅ 数据集构建完成")
    logger.info("=" * 60)
    logger.info(f"数据集保存到: {output_dir}")
    logger.info(f"训练集Click Train: {len(train_samples)}")
    logger.info(f"验证集Click Train: {len(val_positive_samples)}")
    logger.info(f"SNR混合: 所有train均叠加持续背景噪音")
    logger.info(f"训练/验证最终数量: {len(balanced_train)}/{len(balanced_val)}")
    if args.save_wav:
        logger.info(f"调试音频样本: {debug_dir}")
    logger.info("=" * 60)

def cmd_train(args):
    """Execute train command (修正版 - 添加数据验证)."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("开始模型训练")
    logger.info("=" * 60)
    
    # Load dataset
    dataset_dir = Path(args.dataset_dir)
    builder = DatasetBuilder()
    
    logger.info(f"加载数据集: {dataset_dir}")
    
    try:
        train_waveforms, train_labels, train_metadata = builder.load_dataset(dataset_dir / 'train')
        val_waveforms, val_labels, val_metadata = builder.load_dataset(dataset_dir / 'val')
    except FileNotFoundError as e:
        logger.error(f"数据集文件未找到: {e}")
        logger.error(f"请先运行: python main.py build-dataset ...")
        return
    
    logger.info(f"训练集样本: {len(train_waveforms)}")
    logger.info(f"验证集样本: {len(val_waveforms)}")
    
    # 🔧 新增：数据形状验证
    logger.info(f"\n数据验证:")
    logger.info(f"  训练集形状: {train_waveforms.shape}")
    logger.info(f"  验证集形状: {val_waveforms.shape}")
    
    expected_length = config['model']['input_length']
    
    if train_waveforms.shape[1] != expected_length:
        logger.error(f"❌ 数据集样本长度 ({train_waveforms.shape[1]}) 与模型输入长度 ({expected_length}) 不匹配!")
        logger.error(f"请检查:")
        logger.error(f"  1. configs/training.yaml 中 model.input_length 是否为 {train_waveforms.shape[1]}")
        logger.error(f"  2. 数据集是否正确构建")
        return
    
    if val_waveforms.shape[1] != expected_length:
        logger.error(f"❌ 验证集样本长度不匹配!")
        return
    
    logger.info(f"✅ 数据形状验证通过")
    
    # 🔧 新增：标签分布验证
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    
    logger.info(f"\n标签分布:")
    logger.info(f"  训练集 - 负样本: {counts_train[0]}, 正样本: {counts_train[1]}")
    logger.info(f"  验证集 - 负样本: {counts_val[0]}, 正样本: {counts_val[1]}")
    logger.info(f"  训练集正负比: 1:{counts_train[0]/counts_train[1]:.2f}")
    logger.info(f"  验证集正负比: 1:{counts_val[0]/counts_val[1]:.2f}")
    
    # 🔧 新增：数据范围检查（检测异常值）
    train_min, train_max = train_waveforms.min(), train_waveforms.max()
    train_mean, train_std = train_waveforms.mean(), train_waveforms.std()
    
    logger.info(f"\n数据统计:")
    logger.info(f"  训练集范围: [{train_min:.4f}, {train_max:.4f}]")
    logger.info(f"  训练集均值: {train_mean:.4f}, 标准差: {train_std:.4f}")
    
    if train_max > 10.0 or train_min < -10.0:
        logger.warning(f"⚠️ 检测到异常大的幅度值，可能存在数据问题")
    
    # Create data loaders
    logger.info(f"\n创建数据加载器...")
    batch_size = config['training']['batch_size']
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_waveforms).float(),
            torch.from_numpy(train_labels).long()
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 🔧 CPU训练时设为0
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_waveforms).float(),
            torch.from_numpy(val_labels).long()
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  训练批次数: {len(train_loader)}")
    logger.info(f"  验证批次数: {len(val_loader)}")
    
    # Create model
    logger.info(f"\n创建模型...")
    model = create_model(config['model'])
    
    # 🔧 打印模型信息
    from models.cnn1d.model import count_parameters
    n_params = count_parameters(model)
    logger.info(f"  模型类型: {'Lightweight' if config['model'].get('use_lightweight', True) else 'Full'}")
    logger.info(f"  模型参数: {n_params:,}")
    logger.info(f"  输入长度: {config['model']['input_length']} 样本")
    
    # Calculate class weights
    class_weights = torch.FloatTensor(len(counts_train) / counts_train)
    logger.info(f"\n类别权重: {class_weights.tolist()}")
    
    # Initialize trainer
    logger.info(f"\n初始化训练器...")
    trainer = Trainer(
        model=model,
        device=config['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        class_weights=class_weights
    )
    
    logger.info(f"  设备: {config['device']}")
    logger.info(f"  学习率: {config['training']['learning_rate']}")
    logger.info(f"  权重衰减: {config['training']['weight_decay']}")
    logger.info(f"  最大轮数: {config['training']['num_epochs']}")
    logger.info(f"  早停耐心: {config['training']['early_stopping_patience']}")
    
    # Train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n开始训练...")
    logger.info(f"检查点保存到: {output_dir}")
    logger.info("=" * 60)
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            checkpoint_dir=output_dir
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 训练完成")
        logger.info("=" * 60)
        logger.info(f"最佳验证损失: {min(history['val_loss']):.4f}")
        logger.info(f"最佳验证准确率: {max(history['val_acc']):.4f}")
        logger.info(f"训练历史已保存")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 训练被用户中断")
        logger.info("部分检查点已保存")
    except Exception as e:
        logger.error(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def cmd_eval(args):
    """Execute eval command."""
    logger = ProjectLogger()
    
    logger.info("Evaluating model")
    
    # Load model
    inference = ClickDetectorInference.from_checkpoint(
        args.checkpoint,
        device='cpu',
        batch_size=32
    )
    
    # Load test dataset
    dataset_dir = Path(args.dataset_dir)
    builder = DatasetBuilder()
    test_waveforms, test_labels, _ = builder.load_dataset(dataset_dir)
    
    logger.info(f"Test samples: {len(test_waveforms)}")
    
    # Predict
    y_proba = inference.predict_batch(test_waveforms)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Generate report
    reporter = EvaluationReporter(Path(args.output_dir))
    
    # Convert to 2D probabilities
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])
    
    generated_files = reporter.generate_report(
        y_true=test_labels,
        y_pred=y_pred,
        y_proba=y_proba_2d,
        metadata={
            'checkpoint': args.checkpoint,
            'dataset': args.dataset_dir
        }
    )
    
    logger.info(f"Evaluation report generated at {args.output_dir}")


def cmd_export(args):
    """Execute export command."""
    logger = ProjectLogger()
    detection_config = load_config('configs/detection.yaml')
    inference_config = load_config(args.config)
    
    logger.info(f"Processing file: {args.input}")
    
    # Load audio
    audio, sr = sf.read(args.input)
    file_id = Path(args.input).stem
    
    # Step 1: Rule-based detection
    params = DetectionParams(
        tkeo_threshold=detection_config['thresholds']['tkeo_z'],
        ste_threshold=detection_config['thresholds']['ste_z'],
        hfc_threshold=detection_config['thresholds']['hfc_z'],
        high_low_ratio_threshold=detection_config['thresholds']['high_low_ratio'],
        envelope_width_min=detection_config['envelope']['width_min_ms'],
        envelope_width_max=detection_config['envelope']['width_max_ms'],
        spectral_centroid_min=detection_config['thresholds']['spectral_centroid_min'],
        refractory_ms=detection_config['refractory_ms']
    )
    
    detector = AdaptiveDetector(sample_rate=sr, params=params)
    candidates = detector.batch_detect(audio)
    
    logger.info(f"Rule-based detection: {len(candidates)} candidates")
    
    # Step 2: Extract 0.2s windows for model inference
    builder = DatasetBuilder(sample_rate=sr)
    windows = []
    
    for candidate in candidates:
        window = builder._extract_centered_window(audio, candidate.peak_idx)
        if window is not None:
            windows.append(window)
        else:
            windows.append(np.zeros(builder.window_samples))
    
    windows = np.array(windows)
    
    # Step 3: Model inference
    inference = ClickDetectorInference.from_checkpoint(
        args.checkpoint,
        device='cpu',
        batch_size=inference_config['batch_size']
    )
    
    model_scores = inference.predict_batch(windows)
    logger.info(f"Model inference completed")
    
    # Step 4: Fusion decision
    fusion_cfg = FusionConfig(
        high_confidence_threshold=inference_config['fusion']['high_confidence_threshold'],
        medium_confidence_threshold=inference_config['fusion']['medium_confidence_threshold'],
        train_consistency_required=inference_config['fusion']['train_consistency_required'],
        min_train_clicks=inference_config['fusion']['min_train_clicks'],
        max_ici_cv=inference_config['fusion']['max_ici_cv'],
        doublet_min_ici_ms=inference_config['doublet']['min_ici_ms'],
        doublet_max_ici_ms=inference_config['doublet']['max_ici_ms'],
        doublet_min_confidence=inference_config['doublet']['min_confidence']
    )
    
    decider = FusionDecider(config=fusion_cfg)
    accepted_indices, decision_info = decider.apply_fusion(candidates, model_scores)
    
    accepted_candidates = [candidates[i] for i in accepted_indices]
    
    logger.info(f"Fusion decision: {len(accepted_candidates)} accepted")
    logger.info(decider.get_statistics(decision_info))
    
    # Step 5: Build trains
    train_builder = TrainBuilder(
        min_ici_ms=detection_config['train']['min_ici_ms'],
        max_ici_ms=detection_config['train']['max_ici_ms'],
        min_train_clicks=detection_config['train']['min_train_clicks']
    )
    
    trains = train_builder.build_trains(accepted_candidates)
    logger.info(f"Built {len(trains)} click trains")
    
    # Step 6: Export results
    output_dir = Path(args.output_dir)
    exporter = ExportWriter(output_dir, sample_rate=sr)
    
    if inference_config['export']['export_events']:
        event_files = exporter.export_events(
            accepted_candidates,
            audio,
            file_id,
            export_audio=inference_config['export']['export_audio']
        )
        logger.info(f"Events exported to {event_files['csv']}")
    
    if inference_config['export']['export_trains']:
        train_files = exporter.export_trains(
            trains,
            accepted_candidates,
            audio,
            file_id,
            export_audio=inference_config['export']['export_audio']
        )
        logger.info(f"Trains exported to {train_files['csv']}")
    
    if inference_config['export']['create_summary']:
        summary_stats = {
            'total_candidates': len(candidates),
            'accepted_clicks': len(accepted_candidates),
            'rejection_rate': 1 - len(accepted_candidates) / len(candidates) if candidates else 0,
            'num_trains': len(trains),
            **decision_info
        }
        
        report_path = exporter.create_summary_report(file_id, summary_stats)
        logger.info(f"Summary report: {report_path}")
    
    logger.info("Export completed successfully")

def cmd_eval_wav(args):
    """
    执行eval-wav命令 - 评估500ms音频片段
    
    工作流程:
    1. 加载模型
    2. 扫描正/负样本目录
    3. 批量推理
    4. 计算指标并生成报告
    """
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("开始WAV文件评估(500ms片段模式)")
    logger.info("=" * 60)
    
    # ========== 1. 加载模型 ==========
    logger.info(f"\n加载模型: {args.checkpoint}")
    try:
        inference = ClickDetectorInference.from_checkpoint(
            args.checkpoint,
            device=config['inference']['device'],
            batch_size=config['inference']['batch_size']
        )
        logger.info(f"✅ 模型加载成功")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return
    
    # ========== 2. 扫描测试集文件 ==========
    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n扫描测试集文件...")
    logger.info(f"  正样本目录: {positive_dir}")
    logger.info(f"  负样本目录: {negative_dir}")
    
    # 递归查找所有wav文件
    positive_files = list(positive_dir.rglob('*.wav'))
    negative_files = list(negative_dir.rglob('*.wav'))
    
    logger.info(f"  找到正样本: {len(positive_files)} 个")
    logger.info(f"  找到负样本: {len(negative_files)} 个")
    
    if len(positive_files) == 0 or len(negative_files) == 0:
        logger.error("❌ 测试集文件数量不足!")
        return
    
    # ========== 3. 加载并预处理音频 ==========
    logger.info(f"\n加载音频文件...")
    
    def load_audio_files(file_list, label, sample_rate=44100):
        """加载音频文件并统一长度"""
        waveforms = []
        labels = []
        file_paths = []
        
        for wav_file in tqdm(file_list, desc=f"加载{label}样本"):
            try:
                audio, sr = sf.read(wav_file)
                
                # 重采样
                if sr != sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                # 转单声道
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                
                # 统一长度到500ms (22050样本)
                target_length = int(0.5 * sample_rate)
                if len(audio) > target_length:
                    audio = audio[:target_length]
                elif len(audio) < target_length:
                    # 中心padding
                    pad_total = target_length - len(audio)
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    audio = np.pad(audio, (pad_left, pad_right), mode='constant')
                
                waveforms.append(audio)
                labels.append(label)
                file_paths.append(str(wav_file))
                
            except Exception as e:
                logger.error(f"加载失败 {wav_file}: {e}")
                continue
        
        return np.array(waveforms), np.array(labels), file_paths
    
    # 加载正样本(label=1)
    pos_waveforms, pos_labels, pos_paths = load_audio_files(
        positive_files, label=1
    )
    
    # 加载负样本(label=0)
    neg_waveforms, neg_labels, neg_paths = load_audio_files(
        negative_files, label=0
    )
    
    # 合并
    all_waveforms = np.vstack([pos_waveforms, neg_waveforms])
    all_labels = np.concatenate([pos_labels, neg_labels])
    all_paths = pos_paths + neg_paths
    
    logger.info(f"\n数据加载完成:")
    logger.info(f"  总样本数: {len(all_waveforms)}")
    logger.info(f"  正样本: {np.sum(all_labels == 1)}")
    logger.info(f"  负样本: {np.sum(all_labels == 0)}")
    logger.info(f"  样本形状: {all_waveforms.shape}")
    
    # ========== 4. 模型推理 ==========
    logger.info(f"\n开始模型推理...")
    
    try:
        y_proba = inference.predict_batch(all_waveforms)
        logger.info(f"✅ 推理完成")
    except Exception as e:
        logger.error(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取置信度阈值
    threshold = config['thresholds']['confidence_threshold']
    y_pred = (y_proba >= threshold).astype(int)
    
    logger.info(f"  使用阈值: {threshold}")
    logger.info(f"  预测为正样本: {np.sum(y_pred == 1)}")
    logger.info(f"  预测为负样本: {np.sum(y_pred == 0)}")
    
    # ========== 5. 计算评估指标 ==========
    logger.info(f"\n计算评估指标...")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 基础指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 打印结果
    logger.info(f"\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(f"准确率 (Accuracy):  {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall):    {recall:.4f}")
    logger.info(f"F1分数 (F1 Score):  {f1:.4f}")
    logger.info(f"ROC AUC:            {roc_auc:.4f}")
    logger.info(f"PR AUC:             {pr_auc:.4f}")
    logger.info(f"\n混淆矩阵:")
    logger.info(f"  TN: {tn:<6} FP: {fp}")
    logger.info(f"  FN: {fn:<6} TP: {tp}")
    logger.info("=" * 60)
    
    # ========== 6. 保存结果 ==========
    logger.info(f"\n保存评估结果到: {output_dir}")
    
    # 6.1 保存预测结果CSV
    if config['output']['save_predictions']:
        results_df = pd.DataFrame({
            'file_path': all_paths,
            'true_label': all_labels,
            'predicted_label': y_pred,
            'confidence': y_proba,
            'correct': (all_labels == y_pred)
        })
        results_csv = output_dir / 'predictions.csv'
        results_df.to_csv(results_csv, index=False)
        logger.info(f"  ✅ 预测结果: {results_csv}")
    
    # 6.2 保存误分类文件
    if config['output']['save_misclassified_files']:
        misclassified = results_df[~results_df['correct']]
        if len(misclassified) > 0:
            misc_csv = output_dir / 'misclassified.csv'
            misclassified.to_csv(misc_csv, index=False)
            logger.info(f"  ✅ 误分类文件: {misc_csv} ({len(misclassified)}个)")
    
    # 6.3 保存混淆矩阵图
    if config['output']['save_confusion_matrix']:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix (Acc: {accuracy:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ 混淆矩阵图: {cm_path}")
    
    # 6.4 保存ROC曲线
    if config['output']['save_roc_curve']:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        roc_path = output_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ ROC曲线: {roc_path}")
    
    # 6.5 保存PR曲线
    if config['output']['save_pr_curve']:
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, linewidth=2,
                label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        pr_path = output_dir / 'pr_curve.png'
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ PR曲线: {pr_path}")
    
    # 6.6 保存详细报告
    if config['output']['generate_detailed_report']:
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估报告 - 500ms片段模式\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"模型: {args.checkpoint}\n")
            f.write(f"正样本目录: {positive_dir}\n")
            f.write(f"负样本目录: {negative_dir}\n")
            f.write(f"置信度阈值: {threshold}\n\n")
            
            f.write("数据集统计:\n")
            f.write(f"  总样本数: {len(all_waveforms)}\n")
            f.write(f"  正样本: {np.sum(all_labels == 1)}\n")
            f.write(f"  负样本: {np.sum(all_labels == 0)}\n\n")
            
            f.write("评估指标:\n")
            f.write(f"  准确率:  {accuracy:.4f}\n")
            f.write(f"  精确率:  {precision:.4f}\n")
            f.write(f"  召回率:  {recall:.4f}\n")
            f.write(f"  F1分数:  {f1:.4f}\n")
            f.write(f"  ROC AUC: {roc_auc:.4f}\n")
            f.write(f"  PR AUC:  {pr_auc:.4f}\n\n")
            
            f.write("混淆矩阵:\n")
            f.write(f"  真负例(TN): {tn}\n")
            f.write(f"  假正例(FP): {fp}\n")
            f.write(f"  假负例(FN): {fn}\n")
            f.write(f"  真正例(TP): {tp}\n\n")
            
            f.write("sklearn分类报告:\n")
            f.write(classification_report(
                all_labels, y_pred,
                target_names=['Negative', 'Positive']
            ))
        
        logger.info(f"  ✅ 详细报告: {report_path}")
    
    logger.info(f"\n" + "=" * 60)
    logger.info("✅ 评估完成!")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route command
    commands = {
        'scan': cmd_scan,
        'detect': cmd_detect,
        'batch-detect': cmd_batch_detect,
        'collect-clicks': cmd_collect_clicks,  # Added
        'trains': cmd_trains,
        'build-dataset': cmd_build_dataset,    # Fixed
        'train': cmd_train,
        'eval': cmd_eval,
        'eval-wav': cmd_eval_wav,
        'export': cmd_export
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        logger = ProjectLogger()
        logger.error(f"Command failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()