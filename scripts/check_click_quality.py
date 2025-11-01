#!/usr/bin/env python3
"""
Click片段质量检查工具
功能：
1. 自动检测明显的误检片段（低质量）
2. 生成质量报告
3. 可选：自动移除低质量片段
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import shutil


class ClickQualityChecker:
    """Click质量检查器"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def check_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """
        检查单个click片段的质量
        
        返回质量指标：
        - peak_factor: 峰值因子（dB）
        - peak_to_rms_ratio: 峰值/RMS比值
        - zero_crossing_rate: 过零率
        - energy_concentration: 能量集中度
        - quality_score: 综合质量评分（0-1）
        """
        # 1. 峰值因子
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            peak_factor_db = 20 * np.log10(peak / rms)
            peak_to_rms_ratio = peak / rms
        else:
            peak_factor_db = 0
            peak_to_rms_ratio = 0
        
        # 2. 过零率（click应该有较高的过零率）
        zcr = self._compute_zcr(audio)
        
        # 3. 能量集中度（click的能量应该集中在短时间内）
        energy_concentration = self._compute_energy_concentration(audio)
        
        # 4. 峰值位置（应该在中间附近）
        peak_position = self._compute_peak_position(audio)
        
        # 5. 综合质量评分
        quality_score = self._compute_quality_score(
            peak_factor_db, peak_to_rms_ratio, zcr, 
            energy_concentration, peak_position
        )
        
        return {
            'peak_factor_db': peak_factor_db,
            'peak_to_rms_ratio': peak_to_rms_ratio,
            'zero_crossing_rate': zcr,
            'energy_concentration': energy_concentration,
            'peak_position': peak_position,
            'quality_score': quality_score
        }
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """计算过零率"""
        signs = np.sign(audio)
        zcr = np.sum(np.abs(np.diff(signs))) / 2 / len(audio)
        return zcr
    
    def _compute_energy_concentration(self, audio: np.ndarray) -> float:
        """
        计算能量集中度（前20%时间窗口的能量占比）
        Click应该有高集中度
        """
        total_energy = np.sum(audio**2)
        if total_energy == 0:
            return 0
        
        # 中心20%窗口
        center = len(audio) // 2
        window_size = len(audio) // 5
        start = max(0, center - window_size // 2)
        end = min(len(audio), center + window_size // 2)
        
        center_energy = np.sum(audio[start:end]**2)
        concentration = center_energy / total_energy
        
        return concentration
    
    def _compute_peak_position(self, audio: np.ndarray) -> float:
        """
        计算峰值位置（归一化到0-1）
        0.5表示在中心，偏离中心说明可能切割不佳
        """
        peak_idx = np.argmax(np.abs(audio))
        normalized_position = peak_idx / len(audio)
        return normalized_position
    
    def _compute_quality_score(self, 
                               peak_factor_db: float,
                               peak_to_rms_ratio: float,
                               zcr: float,
                               energy_concentration: float,
                               peak_position: float) -> float:
        """
        综合质量评分（0-1）
        
        好的click特征：
        - 高峰值因子（> 15dB）
        - 高能量集中度（> 0.3）
        - 峰值在中心附近（0.4-0.6）
        - 过零率适中（0.1-0.3）
        """
        score = 0.0
        
        # 峰值因子（25%权重）
        if peak_factor_db > 20:
            score += 0.25
        elif peak_factor_db > 15:
            score += 0.15
        elif peak_factor_db > 10:
            score += 0.05
        
        # 能量集中度（30%权重）
        if energy_concentration > 0.4:
            score += 0.30
        elif energy_concentration > 0.25:
            score += 0.20
        elif energy_concentration > 0.15:
            score += 0.10
        
        # 峰值位置（20%权重）
        peak_deviation = abs(peak_position - 0.5)
        if peak_deviation < 0.1:  # 在中心±10%范围内
            score += 0.20
        elif peak_deviation < 0.2:
            score += 0.10
        
        # 过零率（15%权重）
        if 0.1 <= zcr <= 0.3:
            score += 0.15
        elif 0.05 <= zcr <= 0.4:
            score += 0.08
        
        # Peak-to-RMS比值（10%权重）
        if peak_to_rms_ratio > 5:
            score += 0.10
        elif peak_to_rms_ratio > 3:
            score += 0.05
        
        return score
    
    def check_directory(self, 
                       input_dir: Path,
                       quality_threshold: float = 0.4,
                       visualize: int = 0) -> Tuple[list, list]:
        """
        检查整个目录的click片段
        
        Args:
            input_dir: 输入目录
            quality_threshold: 质量阈值（低于此值视为低质量）
            visualize: 可视化多少个低质量样本
            
        Returns:
            (good_files, bad_files)
        """
        input_dir = Path(input_dir)
        wav_files = list(input_dir.rglob('*.wav'))
        
        print(f"检查 {len(wav_files)} 个click片段...")
        
        good_files = []
        bad_files = []
        quality_scores = []
        
        for wav_file in tqdm(wav_files, desc="质量检查"):
            try:
                audio, sr = sf.read(wav_file)
                
                # 重采样
                if sr != self.sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                
                # 单声道
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                
                # 检查质量
                quality = self.check_quality(audio)
                quality_scores.append(quality['quality_score'])
                
                if quality['quality_score'] >= quality_threshold:
                    good_files.append((wav_file, quality))
                else:
                    bad_files.append((wav_file, quality))
                    
            except Exception as e:
                print(f"处理 {wav_file} 时出错: {e}")
                bad_files.append((wav_file, {'quality_score': 0.0}))
        
        # 打印统计
        print(f"\n{'='*60}")
        print(f"质量检查完成")
        print(f"{'='*60}")
        print(f"总数: {len(wav_files)}")
        print(f"高质量: {len(good_files)} ({len(good_files)/len(wav_files)*100:.1f}%)")
        print(f"低质量: {len(bad_files)} ({len(bad_files)/len(wav_files)*100:.1f}%)")
        print(f"平均质量分数: {np.mean(quality_scores):.3f}")
        print(f"质量分数中位数: {np.median(quality_scores):.3f}")
        
        # 可视化低质量样本
        if visualize > 0 and bad_files:
            self._visualize_bad_samples(bad_files[:visualize])
        
        return good_files, bad_files
    
    def _visualize_bad_samples(self, bad_files: list):
        """可视化低质量样本"""
        n_samples = len(bad_files)
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        
        if n_samples == 1:
            axes = [axes]
        
        for i, (wav_file, quality) in enumerate(bad_files):
            audio, sr = sf.read(wav_file)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            time_axis = np.arange(len(audio)) / sr * 1000
            axes[i].plot(time_axis, audio, linewidth=0.8)
            axes[i].set_title(
                f"{wav_file.name}\nQuality: {quality['quality_score']:.3f}, "
                f"PF: {quality.get('peak_factor_db', 0):.1f}dB, "
                f"Conc: {quality.get('energy_concentration', 0):.2f}",
                fontsize=10
            )
            axes[i].set_xlabel('Time (ms)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def move_bad_files(self, 
                      bad_files: list,
                      output_dir: Path,
                      action: str = 'move'):
        """
        移动或删除低质量文件
        
        Args:
            bad_files: 低质量文件列表
            output_dir: 输出目录（移动时使用）
            action: 'move' 或 'delete'
        """
        output_dir = Path(output_dir)
        
        if action == 'move':
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n移动 {len(bad_files)} 个低质量文件到 {output_dir}...")
            
            for wav_file, quality in tqdm(bad_files):
                shutil.move(wav_file, output_dir / wav_file.name)
                
        elif action == 'delete':
            print(f"\n删除 {len(bad_files)} 个低质量文件...")
            for wav_file, quality in tqdm(bad_files):
                wav_file.unlink()
        
        print(f"✅ 完成")


def main():
    parser = argparse.ArgumentParser(description='Click质量检查工具')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Click片段目录')
    parser.add_argument('--quality-threshold', type=float, default=0.4,
                       help='质量阈值（0-1，默认0.4）')
    parser.add_argument('--visualize', type=int, default=5,
                       help='可视化多少个低质量样本（默认5）')
    parser.add_argument('--action', type=str, choices=['report', 'move', 'delete'],
                       default='report',
                       help='处理动作：report(仅报告), move(移动), delete(删除)')
    parser.add_argument('--bad-output-dir', type=str, default='data/bad_clicks',
                       help='低质量文件输出目录（action=move时使用）')
    parser.add_argument('--sample-rate', type=int, default=44100)
    
    args = parser.parse_args()
    
    checker = ClickQualityChecker(sample_rate=args.sample_rate)
    
    # 检查质量
    good_files, bad_files = checker.check_directory(
        Path(args.input_dir),
        quality_threshold=args.quality_threshold,
        visualize=args.visualize
    )
    
    # 保存报告
    report_path = Path(args.input_dir).parent / 'quality_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Click质量检查报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"总数: {len(good_files) + len(bad_files)}\n")
        f.write(f"高质量: {len(good_files)}\n")
        f.write(f"低质量: {len(bad_files)}\n\n")
        f.write("低质量文件列表:\n")
        for wav_file, quality in bad_files:
            f.write(f"  {wav_file.name}: {quality['quality_score']:.3f}\n")
    
    print(f"\n报告已保存到: {report_path}")
    
    # 执行动作
    if args.action != 'report' and bad_files:
        checker.move_bad_files(
            bad_files,
            Path(args.bad_output_dir),
            action=args.action
        )


if __name__ == '__main__':
    main()