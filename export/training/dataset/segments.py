"""
Dataset construction for training samples (增强版).
支持：
1. 单个120ms click片段
2. Click train序列（500ms，包含2-5个click）
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import soundfile as sf
import json
from tqdm import tqdm
import random

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.augment.pipeline import AugmentationPipeline


class DatasetBuilder:
    """构建训练样本：单click + click train序列（统一长度到500ms）"""
    
    def __init__(self,
                 sample_rate: int = 44100,
                 window_ms: float = 120.0,
                 random_offset_ms: float = 10.0,
                 unified_length_ms: float = 500.0):  # 新增：统一长度
        """
        Initialize dataset builder.
        
        Args:
            sample_rate: Sample rate (Hz)
            window_ms: Window duration for single clicks (ms)
            random_offset_ms: Random time offset range (ms)
            unified_length_ms: 统一所有样本到此长度（毫秒）
        """
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms * sample_rate / 1000)  # 5292样本
        self.offset_samples = int(random_offset_ms * sample_rate / 1000)
        self.unified_samples = int(unified_length_ms * sample_rate / 1000)  # 22050样本
        
    # ... [其他方法保持不变] ...
    
    def _pad_to_unified_length(self, segment: np.ndarray) -> np.ndarray:
        """
        将任意长度的片段padding到统一长度（500ms）
        
        Args:
            segment: 输入片段（可能是120ms或500ms）
            
        Returns:
            Padding后的片段（500ms）
        """
        if len(segment) >= self.unified_samples:
            return segment[:self.unified_samples]
        
        # 中心padding（将原片段放在中间）
        pad_total = self.unified_samples - len(segment)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # 使用常数padding（填充0）
        padded = np.pad(segment, (pad_left, pad_right), mode='constant')
        
        return padded
        
    # ========== 原有方法保持不变 ==========
    
    def build_positive_samples(self,
                              audio: np.ndarray,
                              candidates: List[ClickCandidate],
                              file_id: str) -> List[Dict[str, Any]]:
        """Build positive samples centered on detected clicks."""
        samples = []
        
        for i, candidate in enumerate(candidates):
            offset = random.randint(-self.offset_samples, self.offset_samples)
            center_idx = candidate.peak_idx + offset
            segment = self._extract_centered_window(audio, center_idx)
            
            if segment is not None:
                sample = {
                    'waveform': segment,
                    'label': 1,
                    'file_id': file_id,
                    'candidate_idx': i,
                    'peak_time': candidate.peak_time,
                    'confidence': candidate.confidence_score
                }
                samples.append(sample)
                
        return samples
        
    def build_negative_samples(self,
                      noise_audio: np.ndarray,
                      file_id: str,
                      n_samples: int) -> List[Dict[str, Any]]:
        """Build negative samples from noise(统一长度到500ms)."""
        samples = []
        
        # 🔧 修改1: 如果噪音太短，padding到所需长度
        if len(noise_audio) < self.unified_samples:
            repeats = int(np.ceil(self.unified_samples / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)[:self.unified_samples]
        
        max_start = len(noise_audio) - self.unified_samples
        
        # 🔧 修改2: 即使 max_start == 0 也生成样本
        if max_start < 0:  # 只有真正太短才跳过
            return samples
            
        for i in range(n_samples):
            # 如果噪音刚好等于所需长度，直接使用整个片段
            if max_start == 0:
                start_idx = 0
            else:
                start_idx = random.randint(0, max_start)
                
            segment = noise_audio[start_idx:start_idx + self.unified_samples]
            
            # 🔧 关键修复：注释掉_normalize_segment调用
            # 因为noise_audio已经在main.py中进行了RMS归一化
            # segment = self._normalize_segment(segment)
            
            # 🔧 添加：简单的峰值限制（防止极端异常值）
            peak = np.max(np.abs(segment))
            if peak > 1.0:  # 如果超过1.0（理论上不应该），进行clip
                segment = np.clip(segment, -1.0, 1.0)
            
            sample = {
                'waveform': segment.astype(np.float32),
                'label': 0,
                'file_id': file_id,
                'candidate_idx': -1,
                'peak_time': start_idx / self.sample_rate,
                'confidence': 0.0
            }
            samples.append(sample)
            
        return samples
    
    # ========== 新增：Click Train序列生成 ==========
    
    def build_click_train_samples(self,
                          click_files: List[Path],
                          n_train_samples: int,
                          train_length_ms: float = 500.0,
                          min_clicks: int = 2,
                          max_clicks: int = 5,
                          ici_range_ms: Tuple[float, float] = (10.0, 80.0),
                          sample_rate: int = None,
                          noise_pool: List[np.ndarray] = None,
                          augmenter: 'AugmentationPipeline' = None) -> List[Dict[str, Any]]:
        """
        构建Click Train序列样本（修复版 - 持续背景噪音）
        
        Args:
            click_files: Click片段文件列表（.wav）
            n_train_samples: 要生成的train样本数量
            train_length_ms: Train序列长度（毫秒）
            min_clicks: 最少click数
            max_clicks: 最多click数
            ici_range_ms: ICI范围（毫秒）
            sample_rate: 采样率（如果None则使用self.sample_rate）
            noise_pool: 噪音池（用于SNR混合）
            augmenter: 增强器对象
            
        Returns:
            Train样本字典列表
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        train_samples_total = int(train_length_ms * sample_rate / 1000)
        train_samples = []
        
        # 检查文件数量
        if len(click_files) < min_clicks:
            print(f"⚠️ Click文件数({len(click_files)})少于最小clicks数({min_clicks})")
            return []
        
        # 如果文件数少于max_clicks，调整max_clicks
        actual_max_clicks = min(max_clicks, len(click_files))
        if actual_max_clicks < max_clicks:
            print(f"⚠️ Click文件数不足，将max_clicks调整为 {actual_max_clicks}")
        
        for train_idx in tqdm(range(n_train_samples), desc="生成Click Train序列"):
            try:
                # 1. 随机选择click数量和文件
                n_clicks = random.randint(min_clicks, actual_max_clicks)
                selected_files = random.sample(click_files, n_clicks)
                
                # 2. 加载click音频
                clicks = []
                for cf in selected_files:
                    audio, sr = sf.read(cf)
                    
                    # 重采样
                    if sr != sample_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                    
                    # 转单声道
                    if audio.ndim == 2:
                        audio = audio.mean(axis=1)
                    
                    # 🔧 修复：RMS归一化（关键步骤）
                    rms = np.sqrt(np.mean(audio**2))
                    if rms > 1e-8:
                        target_rms = 0.1
                        audio = audio * (target_rms / rms)
                    
                    # 峰值裁剪
                    peak = np.max(np.abs(audio))
                    if peak > 0.95:
                        audio = audio / peak * 0.95
                    
                    clicks.append(audio)
                
                # 3. 放置clicks（叠加到零初始化的数组）
                train_audio = self._place_clicks_with_realistic_ici(
                    clicks, train_samples_total, ici_range_ms, sample_rate
                )
                
                if train_audio is None:
                    continue
                
                # 4. SNR混合：模拟真实海洋环境（持续背景噪声 + clicks）
                if noise_pool is not None and augmenter is not None:
                    #if random.random() < augmenter.apply_prob:
                    if True:  # 始终应用SNR混合
                        # 4.1 选择噪声并提取等长片段
                        noise = random.choice(noise_pool)
                        
                        if len(noise) > len(train_audio):
                            start = random.randint(0, len(noise) - len(train_audio))
                            background_noise = noise[start:start + len(train_audio)]
                        else:
                            # 如果噪声太短，重复填充
                            repeats = int(np.ceil(len(train_audio) / len(noise)))
                            background_noise = np.tile(noise, repeats)[:len(train_audio)]
                        
                        # 🔧 修复：改进功率计算（考虑稀疏性）
                        # 计算clicks的RMS功率（基于整体信号）
                        signal_rms = np.sqrt(np.mean(train_audio**2))
                        
                        # 计算噪声RMS功率
                        noise_rms = np.sqrt(np.mean(background_noise**2))
                        
                        # 根据目标SNR计算噪声缩放因子
                        target_snr = random.uniform(*augmenter.snr_range)  # -5 到 5 dB
                        snr_linear = 10**(target_snr / 10)
                        
                        if noise_rms > 1e-10:
                            # SNR = RMS_signal / RMS_noise
                            # noise_scale = RMS_signal / (SNR * RMS_noise_original)
                            noise_scale = signal_rms / (snr_linear * noise_rms)
                        else:
                            noise_scale = 0
                        
                        # 🔧 限制噪声缩放因子（防止噪声过大）
                        noise_scale = np.clip(noise_scale, 0, 5.0)  # 最大5倍
                        
                        # 叠加持续的背景噪声（关键步骤）
                        train_audio = train_audio + noise_scale * background_noise

                # 5. 最终峰值归一化（避免削波）
                peak = np.max(np.abs(train_audio))
                if peak > 0:
                    train_audio = train_audio / peak * 0.95
                
                # 6. 保存样本
                train_samples.append({
                    'waveform': train_audio,
                    'label': 1,
                    'file_id': f'click_train_{train_idx:04d}',
                    'candidate_idx': -1,
                    'peak_time': -1,
                    'confidence': 1.0,
                    'n_clicks': n_clicks
                })
                
            except Exception as e:
                print(f"生成train {train_idx} 时出错: {e}")
                continue
        
        return train_samples
    
    def _place_clicks_with_realistic_ici(self,
                                     clicks: List[np.ndarray],
                                     train_samples_total: int,
                                     ici_range_ms: Tuple[float, float],
                                     sample_rate: int) -> Optional[np.ndarray]:
        """
        使用更真实的ICI分布放置clicks
        
        策略：
        1. 从正态分布中采样ICI（均值=(min+max)/2，标准差=(max-min)/6）
        2. 确保ICI在允许范围内
        3. 如果放不下，重新尝试
        
        Args:
            clicks: Click音频片段列表
            train_samples_total: 总样本数
            ici_range_ms: ICI范围（毫秒）
            sample_rate: 采样率
            
        Returns:
            放置好的train音频，如果失败则返回None
        """
        import random

        # 初始化为背景噪音（而不是0）
        # 注意：这里不添加噪音，在外部SNR混合时添加
        train_audio = np.zeros(train_samples_total, dtype=np.float32)
        
        # 计算ICI分布参数（正态分布）
        ici_mean_ms = (ici_range_ms[0] + ici_range_ms[1]) / 2
        ici_std_ms = (ici_range_ms[1] - ici_range_ms[0]) / 6  # 99.7%落在范围内
        
        current_pos = 0
        placements = []
        
        for i, click in enumerate(clicks):
            click_len = len(click)
            
            # 确保不会越界
            if current_pos + click_len > train_samples_total:
                # 尝试从头重新放置（给些随机偏移）
                offset = random.randint(0, int(0.1 * sample_rate))  # 最多100ms偏移
                if offset + click_len > train_samples_total:
                    return None  # 放不下，放弃此train
                current_pos = offset
            
            # 放置click（叠加，避免覆盖）
            end_pos = min(current_pos + click_len, train_samples_total)
            train_audio[current_pos:end_pos] += click[:end_pos-current_pos]
            placements.append(current_pos / sample_rate * 1000)
            
            # 计算下一个click的位置（如果不是最后一个）
            if i < len(clicks) - 1:
                # 从正态分布采样ICI
                for attempt in range(10):  # 最多尝试10次
                    ici_ms = np.random.normal(ici_mean_ms, ici_std_ms)
                    # 限制在范围内
                    ici_ms = np.clip(ici_ms, ici_range_ms[0], ici_range_ms[1])
                    ici_samples = int(ici_ms * sample_rate / 1000)
                    
                    next_pos = current_pos + click_len + ici_samples
                    
                    # 检查是否有足够空间
                    if next_pos + len(clicks[i+1]) <= train_samples_total:
                        current_pos = next_pos
                        break
                else:
                    # 尝试失败，返回None
                    return None
        
        return train_audio
    
    # ========== 辅助方法 ==========
    
    def _extract_centered_window(self,
                                audio: np.ndarray,
                                center_idx: int) -> Optional[np.ndarray]:
        """Extract window centered on index."""
        half_window = self.window_samples // 2
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        if start_idx < 0 or end_idx > len(audio):
            segment = self._extract_with_padding(audio, center_idx)
        else:
            segment = audio[start_idx:end_idx]
            
        if len(segment) != self.window_samples:
            return None
            
        segment = self._normalize_segment(segment)
        return segment
        
    def _extract_with_padding(self,
                             audio: np.ndarray,
                             center_idx: int) -> np.ndarray:
        """Extract window with reflection padding if needed."""
        half_window = self.window_samples // 2
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        pad_left = max(0, -start_idx)
        pad_right = max(0, end_idx - len(audio))
        
        extract_start = max(0, start_idx)
        extract_end = min(len(audio), end_idx)
        
        segment = audio[extract_start:extract_end]
        
        if pad_left > 0 or pad_right > 0:
            segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
            
        return segment
        
    def _normalize_segment(self, segment: np.ndarray) -> np.ndarray:
        """Normalize segment to zero mean and unit variance."""
        segment = segment - np.mean(segment)
        
        mad = np.median(np.abs(segment - np.median(segment)))
        if mad > 1e-10:
            segment = segment / (1.4826 * mad)
        else:
            rms = np.sqrt(np.mean(segment**2))
            if rms > 1e-10:
                segment = segment / rms
                
        return segment.astype(np.float32)
    
    # ========== 保存/加载方法保持不变 ==========
    
    def save_dataset(self,
                    samples: List[Dict[str, Any]],
                    output_dir: Path,
                    split: str = 'train') -> Path:
        """Save dataset to disk."""
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        waveforms = np.array([s['waveform'] for s in samples])
        labels = np.array([s['label'] for s in samples])
        
        np.save(split_dir / 'waveforms.npy', waveforms)
        np.save(split_dir / 'labels.npy', labels)
        
        metadata = []
        for s in samples:
            meta = {k: v for k, v in s.items() if k != 'waveform'}
            metadata.append(meta)
            
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved {len(samples)} samples to {split_dir}")
        print(f"  Sample shape: {waveforms.shape}")
        print(f"  Positive: {np.sum(labels == 1)}")
        print(f"  Negative: {np.sum(labels == 0)}")
        
        return split_dir
        
    def load_dataset(self, dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load dataset from disk."""
        dataset_dir = Path(dataset_dir)
        
        waveforms = np.load(dataset_dir / 'waveforms.npy')
        labels = np.load(dataset_dir / 'labels.npy')
        
        with open(dataset_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return waveforms, labels, metadata
        
    def balance_dataset(self,
                       samples: List[Dict[str, Any]],
                       balance_ratio: float = 1.0) -> List[Dict[str, Any]]:
        """Balance positive and negative samples."""
        positive = [s for s in samples if s['label'] == 1]
        negative = [s for s in samples if s['label'] == 0]
        
        n_positive = len(positive)
        n_negative_target = int(n_positive * balance_ratio)
        
        if len(negative) > n_negative_target:
            negative = random.sample(negative, n_negative_target)
        elif len(negative) < n_negative_target:
            negative = random.choices(negative, k=n_negative_target)
            
        balanced = positive + negative
        random.shuffle(balanced)
        
        return balanced