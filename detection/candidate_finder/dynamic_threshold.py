"""
增强的自适应检测器，集成瞬态特征分析。
在原有多特征检测基础上增加瞬态验证层。
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from detection.rules.features import FeatureExtractor
from detection.candidate_finder.transient_detector import TransientDetector
from utils.dsp.envelope import compute_hilbert_envelope, measure_envelope_width
from utils.dsp.teager import TeagerKaiserOperator
from scipy.signal import find_peaks


@dataclass
class DetectionParams:
    """Parameters for adaptive detection."""
    tkeo_threshold: float = 6.0  # Robust z-score threshold
    ste_threshold: float = 6.0
    hfc_threshold: float = 4.0
    high_low_ratio_threshold: float = 1.2
    envelope_width_min: float = 0.2  # ms
    envelope_width_max: float = 1.8  # ms
    spectral_centroid_min: float = 8500  # Hz
    refractory_ms: float = 1.5  # Minimum time between clicks
    # 新增：瞬态过滤参数
    enable_transient_filter: bool = True
    min_dolphin_likelihood: float = 0.3  # 最小海豚可能性阈值


@dataclass
class ClickCandidate:
    """Detected click candidate with transient features."""
    peak_idx: int
    peak_time: float
    tkeo_value: float
    ste_value: float
    hfc_value: float
    spectral_centroid: float
    high_low_ratio: float
    envelope_width: float
    confidence_score: float
    transient_features: Dict[str, float] = field(default_factory=dict)  # 瞬态特征
    dolphin_likelihood: float = 1.0  # 海豚可能性评分
    spectral_features: Dict[str, float] = field(default_factory=dict)  # 频谱特征


class EnhancedAdaptiveDetector:
    """
    增强的自适应检测器，融合瞬态检测
    三层过滤策略：
    1. 规则层（高召回）- 多特征检测
    2. 瞬态验证层（区分Shrimp）- 瞬态特征分析
    3. 置信度过滤层（最终筛选）
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 params: DetectionParams = None):
        """
        Initialize enhanced detector.
        
        Args:
            sample_rate: Sample rate in Hz
            params: Detection parameters
        """
        self.sample_rate = sample_rate
        self.params = params or DetectionParams()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            window_ms=1.0,
            step_ms=0.25
        )
        
        # TKEO算子
        self.tkeo_operator = TeagerKaiserOperator(smooth_window=5)
        
        # 瞬态检测器（新增）
        self.transient_detector = TransientDetector(sample_rate)
        
    def detect_clicks(self, audio: np.ndarray) -> List[ClickCandidate]:
        """
        三层过滤策略检测clicks
        
        Args:
            audio: Input audio signal
            
        Returns:
            List of click candidates with transient analysis
        """
        # === 第1层：规则层多特征检测（高召回） ===
        features = self.feature_extractor.extract_all_features(audio)
        envelope = compute_hilbert_envelope(audio)
        
        # 找到候选峰值
        tkeo_peaks = self._find_tkeo_peaks(features['tkeo_z'])
        
        # 评估每个候选
        rule_candidates = []
        for peak_idx in tkeo_peaks:
            candidate = self._evaluate_candidate(
                peak_idx, audio, features, envelope
            )
            if candidate is not None:
                rule_candidates.append(candidate)
        
        if not rule_candidates:
            return []
        
        # === 第2层：瞬态验证层（区分Shrimp） ===
        if self.params.enable_transient_filter:
            enhanced_candidates = self._apply_transient_verification(
                audio, rule_candidates
            )
        else:
            enhanced_candidates = rule_candidates
        
        # === 第3层：置信度过滤 ===
        filtered_candidates = [
            c for c in enhanced_candidates 
            if c.dolphin_likelihood >= self.params.min_dolphin_likelihood
        ]
        
        # 应用refractory period
        filtered_candidates = self._apply_refractory_period(filtered_candidates)
        
        return filtered_candidates
    
    def _apply_transient_verification(self,
                                     audio: np.ndarray,
                                     candidates: List[ClickCandidate]) -> List[ClickCandidate]:
        """
        应用瞬态验证，计算每个候选的海豚可能性
        
        Args:
            audio: 完整音频信号
            candidates: 规则检测得到的候选列表
            
        Returns:
            增强后的候选列表（带有瞬态特征和海豚评分）
        """
        enhanced = []
        
        for candidate in candidates:
            # 提取瞬态特征
            transient_features = self.transient_detector.analyze_transient(
                audio, candidate.peak_idx
            )
            
            # 计算海豚可能性评分
            dolphin_score = self._compute_dolphin_likelihood(
                candidate, transient_features
            )
            
            # 更新候选对象
            candidate.transient_features = transient_features
            candidate.dolphin_likelihood = dolphin_score
            
            # 调整置信度（结合原始置信度和海豚评分）
            candidate.confidence_score = candidate.confidence_score * dolphin_score
            
            enhanced.append(candidate)
        
        return enhanced
    
    def _compute_dolphin_likelihood(self,
                                   candidate: ClickCandidate,
                                   transient_features: Dict[str, float]) -> float:
        """
        计算海豚可能性评分，区分Shrimp
        
        关键区分特征：
        1. 瞬态锐度（Shrimp极端尖锐）
        2. 上升时间（Shrimp极短）
        3. 能量集中度（Shrimp极高）
        4. 频谱特性（海豚更宽带）
        5. 频谱峰值位置（海豚>30kHz常见）
        
        Args:
            candidate: Click候选
            transient_features: 瞬态特征字典
            
        Returns:
            海豚可能性评分(0.1-2.0)
        """
        score = 1.0
        
        # === 特征1：瞬态锐度（Shrimp极端尖锐） ===
        peak_sharpness = transient_features.get('peak_sharpness', 0)
        if peak_sharpness > 20:  # Shrimp典型值>20
            score *= 0.3  # 大幅降权
        elif peak_sharpness > 12:
            score *= 0.6
        elif peak_sharpness < 8:  # 理想海豚范围
            score *= 1.2
        
        # === 特征2：上升时间（Shrimp<0.5ms，海豚1-5ms） ===
        attack_time_ms = transient_features.get('attack_time_ms', 0)
        if attack_time_ms < 0.5:  # 极短 → Shrimp
            score *= 0.4
        elif 1.0 <= attack_time_ms <= 5.0:  # 理想范围
            score *= 1.2
        elif attack_time_ms > 10.0:  # 太长
            score *= 0.7
        
        # === 特征3：能量集中度（Shrimp极高>0.95） ===
        energy_conc = transient_features.get('energy_concentration', 0)
        if energy_conc > 0.95:  # 过于集中
            score *= 0.5
        elif 0.75 <= energy_conc <= 0.90:  # 适中
            score *= 1.1
        
        # === 特征4：包络对称性 ===
        symmetry = transient_features.get('envelope_symmetry', 0)
        if symmetry < 0.3:  # 极不对称（Shrimp）
            score *= 0.6
        elif 0.5 <= symmetry <= 2.0:  # 相对对称
            score *= 1.1
        
        # === 特征5：频谱中心（海豚高频特征） ===
        if candidate.spectral_centroid > 35000:  # 超高频
            score *= 1.4  # 强烈暗示海豚
        elif candidate.spectral_centroid < 15000:  # 中低频
            score *= 0.5  # 可能是Shrimp或噪音
        
        # === 特征6：频谱宽带特性 ===
        # 通过高低频比判断宽带特性
        if candidate.high_low_ratio > 1.5:  # 高频丰富
            score *= 1.2
        elif candidate.high_low_ratio < 0.8:  # 低频为主
            score *= 0.6
        
        return np.clip(score, 0.1, 2.0)
    
    def _find_tkeo_peaks(self, tkeo_z: np.ndarray) -> np.ndarray:
        """
        Find peaks in normalized TKEO that exceed threshold.
        
        Args:
            tkeo_z: Normalized TKEO values
            
        Returns:
            Array of peak frame indices
        """
        above_threshold = tkeo_z > self.params.tkeo_threshold
        peaks = []
        
        for i in range(1, len(tkeo_z) - 1):
            if (above_threshold[i] and 
                tkeo_z[i] > tkeo_z[i-1] and 
                tkeo_z[i] > tkeo_z[i+1]):
                peaks.append(i)
                
        return np.array(peaks)
    
    def _evaluate_candidate(self,
                           frame_idx: int,
                           audio: np.ndarray,
                           features: Dict[str, np.ndarray],
                           envelope: np.ndarray) -> Optional[ClickCandidate]:
        """
        Evaluate a candidate peak using all features.
        
        Args:
            frame_idx: Frame index of candidate
            audio: Full audio signal
            features: Extracted features
            envelope: Hilbert envelope
            
        Returns:
            ClickCandidate if valid, None otherwise
        """
        # Convert frame index to sample index
        step_samples = self.feature_extractor.step_samples
        sample_idx = frame_idx * step_samples
        
        # Check bounds
        if sample_idx >= len(audio) or frame_idx >= len(features['tkeo_z']):
            return None
            
        # Get feature values
        tkeo_z = features['tkeo_z'][frame_idx]
        ste_z = features['ste_z'][frame_idx]
        hfc_z = features['hfc_z'][frame_idx]
        centroid = features['spectral_centroid'][frame_idx]
        hl_ratio = features['high_low_ratio'][frame_idx]
        
        # Check primary thresholds
        if tkeo_z < self.params.tkeo_threshold:
            return None
        if ste_z < self.params.ste_threshold:
            return None
            
        # Check secondary criteria
        secondary_pass = (
            hfc_z >= self.params.hfc_threshold or
            hl_ratio >= self.params.high_low_ratio_threshold
        )
        if not secondary_pass:
            return None
            
        # Measure envelope width
        env_width = measure_envelope_width(
            envelope, sample_idx, self.sample_rate, db_threshold=-10
        )
        
        # Check envelope width
        if not (self.params.envelope_width_min <= env_width <= self.params.envelope_width_max):
            return None
            
        # Optional: check spectral centroid
        if centroid < self.params.spectral_centroid_min:
            return None
            
        # Calculate confidence score
        confidence = self._calculate_confidence(
            tkeo_z, ste_z, hfc_z, hl_ratio, env_width
        )
        
        # Create candidate
        peak_time = sample_idx / self.sample_rate
        candidate = ClickCandidate(
            peak_idx=sample_idx,
            peak_time=peak_time,
            tkeo_value=tkeo_z,
            ste_value=ste_z,
            hfc_value=hfc_z,
            spectral_centroid=centroid,
            high_low_ratio=hl_ratio,
            envelope_width=env_width,
            confidence_score=confidence,
            transient_features={},
            dolphin_likelihood=1.0,
            spectral_features={}
        )
        
        return candidate
    
    def _calculate_confidence(self,
                             tkeo_z: float,
                             ste_z: float,
                             hfc_z: float,
                             hl_ratio: float,
                             env_width: float) -> float:
        """Calculate confidence score for candidate."""
        # Normalize each feature contribution
        tkeo_contrib = min(tkeo_z / 10, 1.0)
        ste_contrib = min(ste_z / 10, 1.0)
        hfc_contrib = min(hfc_z / 8, 1.0)
        ratio_contrib = min(hl_ratio / 2, 1.0)
        
        # Envelope width: ideal around 0.5-1.0 ms
        width_contrib = 1.0 - abs(env_width - 0.8) / 1.0
        width_contrib = max(0, min(width_contrib, 1.0))
        
        # Weighted average
        confidence = (
            0.3 * tkeo_contrib +
            0.25 * ste_contrib +
            0.2 * hfc_contrib +
            0.15 * ratio_contrib +
            0.1 * width_contrib
        )
        
        return confidence
    
    def _apply_refractory_period(self,
                                candidates: List[ClickCandidate]) -> List[ClickCandidate]:
        """Apply refractory period: merge candidates too close together."""
        if len(candidates) <= 1:
            return candidates
            
        # Sort by time
        candidates = sorted(candidates, key=lambda c: c.peak_time)
        
        refractory_s = self.params.refractory_ms / 1000
        filtered = []
        
        i = 0
        while i < len(candidates):
            current = candidates[i]
            
            # Look ahead for candidates within refractory period
            j = i + 1
            group = [current]
            while j < len(candidates):
                if candidates[j].peak_time - current.peak_time < refractory_s:
                    group.append(candidates[j])
                    j += 1
                else:
                    break
                    
            # Keep the one with highest confidence
            best = max(group, key=lambda c: c.confidence_score)
            filtered.append(best)
            
            i = j
            
        return filtered
    
    def batch_detect(self,
                    audio: np.ndarray,
                    chunk_duration: float = 60.0,
                    overlap: float = 0.5) -> List[ClickCandidate]:
        """
        Detect clicks in long audio using chunks.
        
        Args:
            audio: Input audio signal
            chunk_duration: Chunk duration in seconds
            overlap: Overlap duration in seconds
            
        Returns:
            List of all detected candidates
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        all_candidates = []
        
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Detect in chunk
            candidates = self.detect_clicks(chunk)
            
            # Adjust times to global coordinates
            for candidate in candidates:
                candidate.peak_idx += start
                candidate.peak_time = candidate.peak_idx / self.sample_rate
                
            all_candidates.extend(candidates)
            
            if end >= len(audio):
                break
                
        # Remove duplicates from overlaps
        all_candidates = self._remove_duplicates(all_candidates, overlap)
        
        return all_candidates
    
    def _remove_duplicates(self,
                          candidates: List[ClickCandidate],
                          overlap: float) -> List[ClickCandidate]:
        """Remove duplicate detections from overlapping chunks."""
        if len(candidates) <= 1:
            return candidates
            
        candidates = sorted(candidates, key=lambda c: c.peak_time)
        filtered = []
        
        i = 0
        while i < len(candidates):
            current = candidates[i]
            
            # Find duplicates (within 10ms)
            duplicates = [current]
            j = i + 1
            while j < len(candidates):
                if abs(candidates[j].peak_time - current.peak_time) < 0.01:
                    duplicates.append(candidates[j])
                    j += 1
                else:
                    break
                    
            # Keep best (highest confidence)
            best = max(duplicates, key=lambda c: c.confidence_score)
            filtered.append(best)
            
            i = j
            
        return filtered


# 为了向后兼容，保留原始类名的别名
AdaptiveDetector = EnhancedAdaptiveDetector