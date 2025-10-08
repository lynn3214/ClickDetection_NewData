"""
评估脚本：直接对 wav 文件进行分类测试
支持文件级和片段级的评估指标
新增功能：负样本降采样以平衡测试集
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import random

from models.cnn1d.inference import ClickDetectorInference
from training.eval.metrics import ModelEvaluator
from training.eval.report import EvaluationReporter
from utils.logging.logger import ProjectLogger
from training.dataset.segments import DatasetBuilder


class WavFileEvaluator:
    """评估器：处理 wav 文件的分类任务"""
    
    def __init__(self, 
                 inference: ClickDetectorInference,
                 sample_rate: int = 44100,
                 window_ms: float = 200.0,
                 stride_ms: float = 50.0):
        """
        初始化评估器
        
        Args:
            inference: 模型推理对象
            sample_rate: 采样率
            window_ms: 窗口大小(毫秒)
            stride_ms: 滑动步长(毫秒)
        """
        self.inference = inference
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000)
        self.stride_samples = int(stride_ms * sample_rate / 1000)
        self.dataset_builder = DatasetBuilder(sample_rate=sample_rate)
        self.logger = ProjectLogger()
        
    def extract_windows(self, audio: np.ndarray) -> np.ndarray:
        """
        从音频中提取滑动窗口
        
        Args:
            audio: 音频信号
            
        Returns:
            窗口数组 [n_windows, window_samples]
        """
        if len(audio) < self.window_samples:
            # 音频太短,填充到窗口大小
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded.reshape(1, -1)
        
        n_windows = (len(audio) - self.window_samples) // self.stride_samples + 1
        windows = np.zeros((n_windows, self.window_samples), dtype=np.float32)
        
        for i in range(n_windows):
            start = i * self.stride_samples
            end = start + self.window_samples
            window = audio[start:end]
            
            # 归一化窗口
            window = self.dataset_builder._normalize_segment(window)
            windows[i] = window
            
        return windows
    
    def predict_file(self, 
                     audio_path: Path,
                     file_threshold: float = 0.5,
                     min_positive_ratio: float = 0.1) -> Dict:
        """
        预测单个文件
        
        Args:
            audio_path: 音频文件路径
            file_threshold: 片段判定为 click 的阈值
            min_positive_ratio: 文件被判定为包含 click 的最小正样本比例
            
        Returns:
            预测结果字典
        """
        try:
            # 读取音频
            audio, sr = sf.read(audio_path)
            
            # 转为单声道
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 重采样(如果需要)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # 提取窗口
            windows = self.extract_windows(audio)
            
            # 模型推理
            probs = self.inference.predict_batch(windows)  # [n_windows]
            
            # 片段级预测
            window_predictions = (probs >= file_threshold).astype(int)
            
            # 文件级决策
            positive_ratio = np.mean(window_predictions)
            file_prediction = 1 if positive_ratio >= min_positive_ratio else 0
            
            return {
                'file_path': str(audio_path),
                'file_name': audio_path.name,
                'n_windows': len(windows),
                'mean_prob': float(np.mean(probs)),
                'max_prob': float(np.max(probs)),
                'positive_ratio': float(positive_ratio),
                'file_prediction': file_prediction,
                'window_predictions': window_predictions,
                'window_probs': probs
            }
            
        except Exception as e:
            self.logger.error(f"处理文件 {audio_path} 时出错: {str(e)}")
            return None
    
    def evaluate_directory(self,
                          data_dir: Path,
                          label: int,
                          file_threshold: float = 0.5,
                          min_positive_ratio: float = 0.1,
                          max_files: int = None) -> List[Dict]:
        """
        评估整个目录
        
        Args:
            data_dir: 数据目录
            label: 该目录的真实标签(0或1)
            file_threshold: 片段阈值
            min_positive_ratio: 文件判定阈值
            max_files: 最大文件数量(用于降采样)
            
        Returns:
            所有文件的预测结果列表
        """
        data_dir = Path(data_dir)
        wav_files = sorted(data_dir.rglob('*.wav'))
        
        # 如果指定了最大文件数，进行随机采样
        if max_files is not None and len(wav_files) > max_files:
            self.logger.info(f"对 {'正样本' if label == 1 else '负样本'} 进行降采样: {len(wav_files)} -> {max_files}")
            random.shuffle(wav_files)
            wav_files = wav_files[:max_files]
            wav_files = sorted(wav_files)  # 重新排序以便复现
        
        self.logger.info(f"处理目录: {data_dir}")
        self.logger.info(f"将评估 {len(wav_files)} 个 wav 文件")
        
        results = []
        for wav_file in tqdm(wav_files, desc=f"评估 {'正样本' if label == 1 else '负样本'}"):
            result = self.predict_file(wav_file, file_threshold, min_positive_ratio)
            if result is not None:
                result['true_label'] = label
                results.append(result)
        
        return results


def sample_negative_files(positive_count: int, 
                          negative_files: List[Path],
                          strategy: str,
                          pos_neg_ratio: str = None,
                          max_negative: int = None,
                          random_seed: int = 42) -> List[Path]:
    """
    对负样本文件进行降采样
    
    Args:
        positive_count: 正样本数量
        negative_files: 负样本文件列表
        strategy: 采样策略 ('balance', 'ratio', 'max', 'none')
        pos_neg_ratio: 正负比例，格式如 "1:2" (仅当strategy='ratio'时使用)
        max_negative: 最大负样本数(仅当strategy='max'时使用)
        random_seed: 随机种子
        
    Returns:
        采样后的负样本文件列表
    """
    random.seed(random_seed)
    
    if strategy == 'none':
        return negative_files
    
    elif strategy == 'balance':
        # 自动平衡：负样本数 = 正样本数
        target_count = positive_count
        
    elif strategy == 'ratio':
        # 按指定比例
        if pos_neg_ratio is None:
            raise ValueError("使用 'ratio' 策略时必须指定 --pos-neg-ratio")
        
        try:
            pos_ratio, neg_ratio = map(int, pos_neg_ratio.split(':'))
            target_count = int(positive_count * neg_ratio / pos_ratio)
        except:
            raise ValueError(f"无效的比例格式: {pos_neg_ratio}，应为 '1:2' 这样的格式")
    
    elif strategy == 'max':
        # 指定最大负样本数
        if max_negative is None:
            raise ValueError("使用 'max' 策略时必须指定 --max-negative-samples")
        target_count = max_negative
    
    else:
        raise ValueError(f"未知的采样策略: {strategy}")
    
    # 执行采样
    if len(negative_files) <= target_count:
        return negative_files
    
    sampled_files = random.sample(negative_files, target_count)
    return sorted(sampled_files)  # 排序以便复现


def evaluate_wav_dataset(checkpoint_path: str,
                        positive_dir: str,
                        negative_dir: str,
                        output_dir: str,
                        file_threshold: float = 0.5,
                        min_positive_ratio: float = 0.1,
                        device: str = 'cpu',
                        # 新增参数
                        negative_sampling: str = 'none',
                        pos_neg_ratio: str = None,
                        max_negative_samples: int = None,
                        random_seed: int = 42):
    """
    主评估函数
    
    Args:
        checkpoint_path: 模型检查点路径
        positive_dir: 正样本目录(包含 clicks)
        negative_dir: 负样本目录(噪音)
        output_dir: 输出目录
        file_threshold: 窗口级阈值
        min_positive_ratio: 文件级判定比例
        device: 设备
        negative_sampling: 负样本采样策略 ('none', 'balance', 'ratio', 'max')
        pos_neg_ratio: 正负样本比例，如 "1:2"
        max_negative_samples: 最大负样本数量
        random_seed: 随机种子
    """
    logger = ProjectLogger()
    logger.info("=" * 60)
    logger.info("WAV 文件评估开始")
    logger.info("=" * 60)
    
    # 加载模型
    logger.info(f"加载模型: {checkpoint_path}")
    inference = ClickDetectorInference.from_checkpoint(
        checkpoint_path,
        device=device,
        batch_size=32
    )
    
    # 初始化评估器
    evaluator = WavFileEvaluator(inference)
    
    # 获取文件列表
    positive_files = sorted(Path(positive_dir).rglob('*.wav'))
    negative_files = sorted(Path(negative_dir).rglob('*.wav'))
    
    logger.info(f"\n原始数据集统计:")
    logger.info(f"  正样本文件数: {len(positive_files)}")
    logger.info(f"  负样本文件数: {len(negative_files)}")
    logger.info(f"  正负比例: 1:{len(negative_files)/len(positive_files):.2f}")
    
    # 对负样本进行降采样
    if negative_sampling != 'none':
        logger.info(f"\n应用负样本采样策略: {negative_sampling}")
        negative_files = sample_negative_files(
            positive_count=len(positive_files),
            negative_files=negative_files,
            strategy=negative_sampling,
            pos_neg_ratio=pos_neg_ratio,
            max_negative=max_negative_samples,
            random_seed=random_seed
        )
        logger.info(f"降采样后负样本数: {len(negative_files)}")
        logger.info(f"新的正负比例: 1:{len(negative_files)/len(positive_files):.2f}")
    
    # 评估正样本
    logger.info("\n处理正样本(包含 dolphin clicks)...")
    positive_results = evaluator.evaluate_directory(
        Path(positive_dir),
        label=1,
        file_threshold=file_threshold,
        min_positive_ratio=min_positive_ratio
    )
    
    # 评估负样本（使用采样后的文件列表）
    logger.info("\n处理负样本(纯噪音)...")
    negative_results = []
    for wav_file in tqdm(negative_files, desc="评估 负样本"):
        result = evaluator.predict_file(wav_file, file_threshold, min_positive_ratio)
        if result is not None:
            result['true_label'] = 0
            negative_results.append(result)
    
    # 合并结果
    all_results = positive_results + negative_results
    
    if not all_results:
        logger.error("没有成功处理任何文件!")
        return
    
    logger.info(f"\n总共处理 {len(all_results)} 个文件")
    logger.info(f"  正样本: {len(positive_results)}")
    logger.info(f"  负样本: {len(negative_results)}")
    logger.info(f"  实际正负比例: 1:{len(negative_results)/len(positive_results):.2f}")
    
    # 提取文件级标签和预测
    y_true = np.array([r['true_label'] for r in all_results])
    y_pred = np.array([r['file_prediction'] for r in all_results])
    y_proba = np.array([r['mean_prob'] for r in all_results])
    
    # 计算指标
    logger.info("\n计算评估指标...")
    evaluator_metrics = ModelEvaluator()
    
    # 转换为二分类概率格式
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])
    
    metrics = evaluator_metrics.compute_metrics(y_true, y_pred, y_proba_2d)
    
    # 输出关键指标
    logger.info("\n" + "=" * 60)
    logger.info("文件级评估结果")
    logger.info("=" * 60)
    logger.info(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall):    {metrics['recall']:.4f}")
    logger.info(f"F1 分数:            {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC:            {metrics.get('roc_auc', 0):.4f}")
    logger.info(f"\n混淆矩阵:")
    logger.info(f"  TN: {metrics['true_negatives']:<6} FP: {metrics['false_positives']}")
    logger.info(f"  FN: {metrics['false_negatives']:<6} TP: {metrics['true_positives']}")
    
    # 保存详细结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存文件级结果CSV
    results_df = pd.DataFrame([{
        'file_name': r['file_name'],
        'file_path': r['file_path'],
        'true_label': r['true_label'],
        'predicted_label': r['file_prediction'],
        'mean_prob': r['mean_prob'],
        'max_prob': r['max_prob'],
        'positive_ratio': r['positive_ratio'],
        'n_windows': r['n_windows']
    } for r in all_results])
    
    results_csv = output_dir / 'file_level_results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\n文件级结果保存至: {results_csv}")
    
    # 生成完整报告
    reporter = EvaluationReporter(output_dir)
    report_files = reporter.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba_2d,
        metadata={
            'checkpoint': checkpoint_path,
            'positive_dir': positive_dir,
            'negative_dir': negative_dir,
            'file_threshold': file_threshold,
            'min_positive_ratio': min_positive_ratio,
            'total_files': len(all_results),
            'positive_files': len(positive_results),
            'negative_files': len(negative_results),
            'negative_sampling': negative_sampling,
            'pos_neg_ratio': pos_neg_ratio,
            'max_negative_samples': max_negative_samples,
            'random_seed': random_seed
        },
        report_name='wav_file_evaluation'
    )
    
    logger.info(f"\n完整评估报告保存至: {output_dir / 'wav_file_evaluation'}")
    
    # 寻找最优阈值
    logger.info("\n寻找最优文件判定阈值...")
    best_threshold, best_f1 = evaluator_metrics.find_optimal_threshold(
        y_true, y_proba, metric='f1'
    )
    logger.info(f"最优阈值(基于F1): {best_threshold:.3f}, F1={best_f1:.4f}")
    
    logger.info("\n评估完成!")
    
    return metrics, all_results


def main():
    parser = argparse.ArgumentParser(
        description='评估模型对 wav 文件的分类性能（支持负样本降采样）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
采样策略说明:
  none     - 不进行采样，使用所有负样本（默认）
  balance  - 自动平衡，使负样本数 = 正样本数（1:1）
  ratio    - 按指定比例采样，需配合 --pos-neg-ratio 使用
  max      - 限制负样本最大数量，需配合 --max-negative-samples 使用

示例:
  # 自动平衡正负样本（1:1）
  python eval_wav_files.py --checkpoint model.pt --negative-sampling balance
  
  # 设置正负比例为 1:2
  python eval_wav_files.py --checkpoint model.pt --negative-sampling ratio --pos-neg-ratio 1:2
  
  # 限制负样本最多1000个
  python eval_wav_files.py --checkpoint model.pt --negative-sampling max --max-negative-samples 1000
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--positive-dir', type=str, 
                       default='data/test_resampled',
                       help='包含 clicks 的正样本目录')
    parser.add_argument('--negative-dir', type=str,
                       default='data/noise_resampled',
                       help='纯噪音负样本目录')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--file-threshold', type=float, default=0.5,
                       help='窗口级判定阈值(默认0.5)')
    parser.add_argument('--min-positive-ratio', type=float, default=0.1,
                       help='文件被判定为包含click的最小正窗口比例(默认0.1)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    
    # 负样本采样参数
    sampling_group = parser.add_argument_group('负样本采样选项')
    sampling_group.add_argument('--negative-sampling', type=str, 
                               default='none',
                               choices=['none', 'balance', 'ratio', 'max'],
                               help='负样本采样策略（默认: none）')
    sampling_group.add_argument('--pos-neg-ratio', type=str,
                               help='正负样本比例，格式: "1:2" （用于 ratio 策略）')
    sampling_group.add_argument('--max-negative-samples', type=int,
                               help='最大负样本数量（用于 max 策略）')
    sampling_group.add_argument('--random-seed', type=int, default=42,
                               help='随机种子，用于可复现的采样（默认: 42）')
    
    args = parser.parse_args()
    
    evaluate_wav_dataset(
        checkpoint_path=args.checkpoint,
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        output_dir=args.output_dir,
        file_threshold=args.file_threshold,
        min_positive_ratio=args.min_positive_ratio,
        device=args.device,
        negative_sampling=args.negative_sampling,
        pos_neg_ratio=args.pos_neg_ratio,
        max_negative_samples=args.max_negative_samples,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()