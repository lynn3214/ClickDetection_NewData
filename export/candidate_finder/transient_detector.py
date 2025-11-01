"""
瞬态特征检测器，用于区分海豚Click和Snapping Shrimp。
提取上升时间、能量集中度、峰值锐度等关键瞬态特征。
"""

import numpy as np
from typing import Dict, Optional
from scipy.signal import find_peaks

from utils.dsp.envelope import compute_hilbert_envelope


class TransientDetector:
    """瞬态特征检测器"""
    
    def __init__(self, sample_rate: int = 44100):
        """
        初始化瞬态检测器
        
        Args:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        
    def analyze_transient(self, 
                         audio: np.ndarray, 
                         peak_idx: int,
                         window_ms: float = 20.0) -> Dict[str, float]:
        """
        分析峰值周围的瞬态特征
        
        Args:
            audio: 完整音频信号
            peak_idx: 峰值位置索引
            window_ms: 分析窗口长度(毫秒)
            
        Returns:
            瞬态特征字典
        """
        window_samples = int(window_ms * self.sample_rate / 1000)
        half_window = window_samples // 2
        
        # 提取窗口
        start = max(0, peak_idx - half_window)
        end = min(len(audio), peak_idx + half_window)
        segment = audio[start:end]
        
        if len(segment) < 10:  # 窗口太小
            return self._default_features()
        
        # 调整峰值在窗口内的位置
        peak_in_window = peak_idx - start
        peak_in_window = max(0, min(len(segment) - 1, peak_in_window))
        
        # 计算各项瞬态特征
        features = {}
        features['attack_time_ms'] = self._compute_attack_time(segment, peak_in_window)
        features['decay_time_ms'] = self._compute_decay_time(segment, peak_in_window)
        features['peak_sharpness'] = self._compute_peak_sharpness(segment, peak_in_window)
        features['energy_concentration'] = self._compute_energy_concentration(segment, peak_in_window)
        features['rise_slope'] = self._compute_rise_slope(segment, peak_in_window)
        features['envelope_symmetry'] = self._compute_envelope_symmetry(segment, peak_in_window)
        
        return features
    
    def _compute_attack_time(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算上升时间(Attack Time)：从10%峰值到90%峰值的时间
        
        Args:
            segment: 音频片段
            peak_idx: 峰值在片段中的位置
            
        Returns:
            上升时间(毫秒)
        """
        if peak_idx < 1:
            return 0.0
        
        # 计算包络
        envelope = compute_hilbert_envelope(segment)
        peak_value = envelope[peak_idx]
        
        if peak_value <= 0:
            return 0.0
        
        # 寻找10%和90%位置
        threshold_10 = peak_value * 0.1
        threshold_90 = peak_value * 0.9
        
        # 从峰值向左搜索
        idx_90 = peak_idx
        idx_10 = peak_idx
        
        # 找90%点
        for i in range(peak_idx, -1, -1):
            if envelope[i] <= threshold_90:
                idx_90 = i
                break
        
        # 找10%点
        for i in range(idx_90, -1, -1):
            if envelope[i] <= threshold_10:
                idx_10 = i
                break
        
        # 计算时间差
        attack_samples = idx_90 - idx_10
        attack_time_ms = (attack_samples / self.sample_rate) * 1000
        
        return max(0.0, attack_time_ms)
    
    def _compute_decay_time(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算衰减时间(Decay Time)：从峰值到10%峰值的时间
        
        Args:
            segment: 音频片段
            peak_idx: 峰值位置
            
        Returns:
            衰减时间(毫秒)
        """
        if peak_idx >= len(segment) - 1:
            return 0.0
        
        envelope = compute_hilbert_envelope(segment)
        peak_value = envelope[peak_idx]
        
        if peak_value <= 0:
            return 0.0
        
        threshold_10 = peak_value * 0.1
        
        # 从峰值向右搜索10%点
        idx_10 = len(segment) - 1
        for i in range(peak_idx, len(segment)):
            if envelope[i] <= threshold_10:
                idx_10 = i
                break
        
        decay_samples = idx_10 - peak_idx
        decay_time_ms = (decay_samples / self.sample_rate) * 1000
        
        return max(0.0, decay_time_ms)
    
    def _compute_peak_sharpness(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算峰值锐度：峰值与周围平均值的比值
        Shrimp的峰值极其尖锐，比值通常>20
        
        Args:
            segment: 音频片段
            peak_idx: 峰值位置
            
        Returns:
            峰值锐度
        """
        envelope = compute_hilbert_envelope(segment)
        peak_value = envelope[peak_idx]
        
        # 计算峰值附近±5个样本的平均值（排除峰值本身）
        window_size = 5
        start = max(0, peak_idx - window_size)
        end = min(len(envelope), peak_idx + window_size + 1)
        
        # 排除峰值点
        neighbors = np.concatenate([envelope[start:peak_idx], envelope[peak_idx+1:end]])
        
        if len(neighbors) == 0:
            return 0.0
        
        mean_neighbors = np.mean(neighbors)
        
        if mean_neighbors > 0:
            sharpness = peak_value / mean_neighbors
        else:
            sharpness = 0.0
        
        return sharpness
    
    def _compute_energy_concentration(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算能量集中度：峰值附近小窗口的能量占比
        Shrimp的能量极度集中，通常>0.95
        
        Args:
            segment: 音频片段
            peak_idx: 峰值位置
            
        Returns:
            能量集中度(0-1)
        """
        # 计算总能量
        total_energy = np.sum(segment**2)
        
        if total_energy <= 0:
            return 0.0
        
        # 峰值附近±2ms的能量
        window_samples = int(0.002 * self.sample_rate)  # 2ms
        start = max(0, peak_idx - window_samples)
        end = min(len(segment), peak_idx + window_samples)
        
        peak_energy = np.sum(segment[start:end]**2)
        
        concentration = peak_energy / total_energy
        
        return min(1.0, concentration)
    
    def _compute_rise_slope(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算上升斜率：峰值前的平均斜率
        
        Args:
            segment: 音频片段
            peak_idx: 峰值位置
            
        Returns:
            上升斜率(归一化)
        """
        if peak_idx < 2:
            return 0.0
        
        envelope = compute_hilbert_envelope(segment)
        
        # 计算前半部分的一阶差分
        rise_portion = envelope[:peak_idx]
        
        if len(rise_portion) < 2:
            return 0.0
        
        slopes = np.diff(rise_portion)
        mean_slope = np.mean(slopes)
        
        # 归一化
        peak_value = envelope[peak_idx]
        if peak_value > 0:
            normalized_slope = mean_slope / peak_value
        else:
            normalized_slope = 0.0
        
        return normalized_slope
    
    def _compute_envelope_symmetry(self, segment: np.ndarray, peak_idx: int) -> float:
        """
        计算包络对称性：上升时间与衰减时间的比值
        对称click的比值接近1，Shrimp通常<0.5（上升极快）
        
        Args:
            segment: 音频片段
            peak_idx: 峰值位置
            
        Returns:
            对称性比值
        """
        attack_time = self._compute_attack_time(segment, peak_idx)
        decay_time = self._compute_decay_time(segment, peak_idx)
        
        if decay_time > 0:
            symmetry = attack_time / decay_time
        else:
            symmetry = 0.0
        
        return symmetry
    
    def _default_features(self) -> Dict[str, float]:
        """返回默认特征值"""
        return {
            'attack_time_ms': 0.0,
            'decay_time_ms': 0.0,
            'peak_sharpness': 0.0,
            'energy_concentration': 0.0,
            'rise_slope': 0.0,
            'envelope_symmetry': 0.0
        }
    
    def compute_dolphin_likelihood(self, transient_features: Dict[str, float]) -> float:
        """
        根据瞬态特征计算海豚可能性评分
        
        Args:
            transient_features: 瞬态特征字典
            
        Returns:
            海豚可能性评分(0.1-2.0)，>1表示更像海豚，<1表示更像Shrimp
        """
        score = 1.0
        
        # 特征1：峰值锐度（Shrimp极端尖锐）
        peak_sharpness = transient_features.get('peak_sharpness', 0)
        if peak_sharpness > 20:  # Shrimp典型值>20
            score *= 0.3  # 大幅降权
        elif peak_sharpness > 12:
            score *= 0.6
        elif peak_sharpness < 8:  # 理想的海豚范围
            score *= 1.2
        
        # 特征2：上升时间（Shrimp<0.5ms，海豚1-5ms）
        attack_time_ms = transient_features.get('attack_time_ms', 0)
        if attack_time_ms < 0.5:  # 极短 → 可能是Shrimp
            score *= 0.4
        elif 1.0 <= attack_time_ms <= 5.0:  # 理想范围
            score *= 1.2
        elif attack_time_ms > 10.0:  # 太长，可能是其他噪音
            score *= 0.7
        
        # 特征3：能量集中度（Shrimp极高>0.95）
        energy_conc = transient_features.get('energy_concentration', 0)
        if energy_conc > 0.95:  # 过于集中
            score *= 0.5
        elif 0.75 <= energy_conc <= 0.90:  # 适中
            score *= 1.1
        
        # 特征4：包络对称性
        symmetry = transient_features.get('envelope_symmetry', 0)
        if symmetry < 0.3:  # 极不对称（Shrimp特征）
            score *= 0.6
        elif 0.5 <= symmetry <= 2.0:  # 相对对称
            score *= 1.1
        
        # 特征5：上升斜率
        rise_slope = transient_features.get('rise_slope', 0)
        if rise_slope > 0.5:  # 斜率极陡
            score *= 0.7
        
        return np.clip(score, 0.1, 2.0)