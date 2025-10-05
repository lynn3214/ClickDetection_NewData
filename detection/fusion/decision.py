"""
Fusion decision logic combining rules, model predictions, and train consistency.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from detection.train_builder.cluster import ClickTrain, TrainBuilder


@dataclass
class FusionConfig:
    """Configuration for fusion decision."""
    high_confidence_threshold: float = 0.95
    medium_confidence_threshold: float = 0.60
    train_consistency_required: bool = True
    min_train_clicks: int = 3
    max_ici_cv: float = 0.4
    doublet_min_ici_ms: float = 8.0
    doublet_max_ici_ms: float = 80.0
    doublet_min_confidence: float = 0.85


class FusionDecider:
    """Makes final detection decisions using multiple criteria."""
    
    def __init__(self, config: FusionConfig = None):
        """
        Initialize fusion decider.
        
        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig()
        self.train_builder = TrainBuilder(
            min_ici_ms=5.0,
            max_ici_ms=150.0,
            min_train_clicks=2
        )
        
    def apply_fusion(self,
                    candidates: List[ClickCandidate],
                    model_scores: np.ndarray = None) -> Tuple[List[int], Dict[str, Any]]:
        """
        Apply fusion logic to filter candidates.
        
        Args:
            candidates: List of click candidates
            model_scores: Optional model probability scores (0-1) for each candidate
            
        Returns:
            Tuple of (accepted_indices, decision_info)
        """
        if not candidates:
            return [], {'total': 0, 'accepted': 0, 'rejected': 0}
            
        n_candidates = len(candidates)
        
        # Use model scores if provided, otherwise use rule-based confidence
        if model_scores is None:
            scores = np.array([c.confidence_score for c in candidates])
        else:
            scores = np.array(model_scores)
            
        # Build trains for consistency checking
        trains = self.train_builder.build_trains(candidates)
        
        # Create candidate to train mapping
        candidate_to_train = self._map_candidates_to_trains(candidates, trains)
        
        # Decision logic
        accepted = []
        decision_reasons = []
        
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            reason, accept = self._decide_single(
                candidate, score, i, candidate_to_train, trains
            )
            decision_reasons.append(reason)
            if accept:
                accepted.append(i)
                
        # Compile decision info
        decision_info = {
            'total': n_candidates,
            'accepted': len(accepted),
            'rejected': n_candidates - len(accepted),
            'high_confidence': sum(1 for s in scores if s >= self.config.high_confidence_threshold),
            'medium_confidence': sum(1 for s in scores if self.config.medium_confidence_threshold <= s < self.config.high_confidence_threshold),
            'low_confidence': sum(1 for s in scores if s < self.config.medium_confidence_threshold),
            'n_trains': len(trains),
            'decision_reasons': decision_reasons
        }
        
        return accepted, decision_info
        
    def _decide_single(self,
                      candidate: ClickCandidate,
                      score: float,
                      idx: int,
                      candidate_to_train: Dict[int, int],
                      trains: List[ClickTrain]) -> Tuple[str, bool]:
        """
        Make decision for single candidate.
        
        Args:
            candidate: Click candidate
            score: Model/confidence score
            idx: Candidate index
            candidate_to_train: Mapping from candidate index to train ID
            trains: List of trains
            
        Returns:
            Tuple of (reason, accept)
        """
        # High confidence: always accept
        if score >= self.config.high_confidence_threshold:
            return "high_confidence_direct", True
            
        # Low confidence: reject
        if score < self.config.medium_confidence_threshold:
            return "low_confidence_reject", False
            
        # Medium confidence: check train consistency
        if not self.config.train_consistency_required:
            return "medium_confidence_no_train_check", True
            
        # Check if in train
        if idx not in candidate_to_train:
            return "medium_confidence_not_in_train", False
            
        train_id = candidate_to_train[idx]
        train = trains[train_id]
        
        # Check train consistency
        if train.n_clicks >= self.config.min_train_clicks and train.ici_cv <= self.config.max_ici_cv:
            return "medium_confidence_consistent_train", True
            
        # Check doublet special case
        if train.n_clicks == 2:
            if self._check_doublet(train, score):
                return "medium_confidence_consistent_doublet", True
            else:
                return "medium_confidence_inconsistent_doublet", False
                
        return "medium_confidence_inconsistent_train", False
        
    def _check_doublet(self, train: ClickTrain, score: float) -> bool:
        """
        Check if doublet meets special criteria.
        
        Args:
            train: Click train with 2 clicks
            score: Model score
            
        Returns:
            True if doublet is acceptable
        """
        ici_ok = (self.config.doublet_min_ici_ms <= train.ici_median <= 
                  self.config.doublet_max_ici_ms)
        conf_ok = score >= self.config.doublet_min_confidence
        
        return ici_ok and conf_ok
        
    def _map_candidates_to_trains(self,
                                 candidates: List[ClickCandidate],
                                 trains: List[ClickTrain]) -> Dict[int, int]:
        """
        Create mapping from candidate index to train ID.
        
        Args:
            candidates: List of candidates
            trains: List of trains
            
        Returns:
            Dictionary mapping candidate index to train ID
        """
        mapping = {}
        for train in trains:
            for click_idx in train.click_indices:
                mapping[click_idx] = train.train_id
        return mapping
        
    def get_statistics(self, decision_info: Dict[str, Any]) -> str:
        """
        Generate human-readable statistics string.
        
        Args:
            decision_info: Decision information dictionary
            
        Returns:
            Formatted statistics string
        """
        stats = [
            f"Total candidates: {decision_info['total']}",
            f"Accepted: {decision_info['accepted']} ({decision_info['accepted']/decision_info['total']*100:.1f}%)",
            f"Rejected: {decision_info['rejected']} ({decision_info['rejected']/decision_info['total']*100:.1f}%)",
            f"High confidence: {decision_info['high_confidence']}",
            f"Medium confidence: {decision_info['medium_confidence']}",
            f"Low confidence: {decision_info['low_confidence']}",
            f"Trains detected: {decision_info['n_trains']}"
        ]
        return "\n".join(stats)