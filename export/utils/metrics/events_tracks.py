"""
Event and track-level evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from detection.train_builder.cluster import ClickTrain


@dataclass
class EventMetrics:
    """Event-level evaluation metrics."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass
class TrainMetrics:
    """Train-level evaluation metrics."""
    true_trains: int
    detected_trains: int
    matched_trains: int
    precision: float
    recall: float
    f1_score: float
    mean_ici_error: float
    mean_click_count_error: float


class EventEvaluator:
    """Evaluates event-level detections against ground truth."""
    
    def __init__(self, tolerance_ms: float = 5.0):
        """
        Initialize evaluator.
        
        Args:
            tolerance_ms: Time tolerance for matching (ms)
        """
        self.tolerance_s = tolerance_ms / 1000
        
    def evaluate(self,
                predictions: List[ClickCandidate],
                ground_truth: List[ClickCandidate]) -> EventMetrics:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Predicted candidates
            ground_truth: Ground truth candidates
            
        Returns:
            EventMetrics object
        """
        if not ground_truth:
            return EventMetrics(0, len(predictions), 0, 0.0, 0.0, 0.0)
            
        if not predictions:
            return EventMetrics(0, 0, len(ground_truth), 0.0, 0.0, 0.0)
            
        # Match predictions to ground truth
        matched_pred, matched_gt = self._match_events(predictions, ground_truth)
        
        tp = len(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EventMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1
        )
        
    def _match_events(self,
                     predictions: List[ClickCandidate],
                     ground_truth: List[ClickCandidate]) -> Tuple[List[int], List[int]]:
        """
        Match predictions to ground truth using greedy nearest-neighbor.
        
        Args:
            predictions: Predicted candidates
            ground_truth: Ground truth candidates
            
        Returns:
            Tuple of (matched_pred_indices, matched_gt_indices)
        """
        pred_times = np.array([c.peak_time for c in predictions])
        gt_times = np.array([c.peak_time for c in ground_truth])
        
        matched_pred = []
        matched_gt = set()
        
        for i, pred_time in enumerate(pred_times):
            # Find nearest ground truth
            diffs = np.abs(gt_times - pred_time)
            nearest_idx = np.argmin(diffs)
            
            # Check if within tolerance and not already matched
            if diffs[nearest_idx] <= self.tolerance_s and nearest_idx not in matched_gt:
                matched_pred.append(i)
                matched_gt.add(nearest_idx)
                
        return matched_pred, list(matched_gt)


class TrainEvaluator:
    """Evaluates train-level detections."""
    
    def __init__(self,
                 overlap_threshold: float = 0.5,
                 ici_tolerance_ms: float = 10.0):
        """
        Initialize train evaluator.
        
        Args:
            overlap_threshold: Minimum temporal overlap for matching (0-1)
            ici_tolerance_ms: Tolerance for ICI comparison (ms)
        """
        self.overlap_threshold = overlap_threshold
        self.ici_tolerance_ms = ici_tolerance_ms
        
    def evaluate(self,
                predictions: List[ClickTrain],
                ground_truth: List[ClickTrain]) -> TrainMetrics:
        """
        Evaluate predicted trains against ground truth.
        
        Args:
            predictions: Predicted trains
            ground_truth: Ground truth trains
            
        Returns:
            TrainMetrics object
        """
        if not ground_truth:
            return TrainMetrics(0, len(predictions), 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        if not predictions:
            return TrainMetrics(len(ground_truth), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        # Match trains
        matches = self._match_trains(predictions, ground_truth)
        
        matched_count = len(matches)
        precision = matched_count / len(predictions) if predictions else 0.0
        recall = matched_count / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate ICI and click count errors for matched trains
        ici_errors = []
        click_count_errors = []
        
        for pred_idx, gt_idx in matches:
            pred_train = predictions[pred_idx]
            gt_train = ground_truth[gt_idx]
            
            ici_error = abs(pred_train.ici_median - gt_train.ici_median)
            ici_errors.append(ici_error)
            
            count_error = abs(pred_train.n_clicks - gt_train.n_clicks)
            click_count_errors.append(count_error)
            
        mean_ici_error = float(np.mean(ici_errors)) if ici_errors else 0.0
        mean_count_error = float(np.mean(click_count_errors)) if click_count_errors else 0.0
        
        return TrainMetrics(
            true_trains=len(ground_truth),
            detected_trains=len(predictions),
            matched_trains=matched_count,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_ici_error=mean_ici_error,
            mean_click_count_error=mean_count_error
        )
        
    def _match_trains(self,
                     predictions: List[ClickTrain],
                     ground_truth: List[ClickTrain]) -> List[Tuple[int, int]]:
        """
        Match predicted trains to ground truth using temporal overlap.
        
        Args:
            predictions: Predicted trains
            ground_truth: Ground truth trains
            
        Returns:
            List of (pred_idx, gt_idx) tuples
        """
        matches = []
        matched_gt = set()
        
        for i, pred in enumerate(predictions):
            best_overlap = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                    
                overlap = self._compute_temporal_overlap(pred, gt)
                
                if overlap > best_overlap and overlap >= self.overlap_threshold:
                    best_overlap = overlap
                    best_gt_idx = j
                    
            if best_gt_idx >= 0:
                matches.append((i, best_gt_idx))
                matched_gt.add(best_gt_idx)
                
        return matches
        
    def _compute_temporal_overlap(self,
                                 train1: ClickTrain,
                                 train2: ClickTrain) -> float:
        """
        Compute temporal overlap ratio between two trains.
        
        Args:
            train1: First train
            train2: Second train
            
        Returns:
            Overlap ratio (0-1)
        """
        # Find overlap interval
        overlap_start = max(train1.start_time, train2.start_time)
        overlap_end = min(train1.end_time, train2.end_time)
        
        if overlap_end <= overlap_start:
            return 0.0
            
        overlap_duration = overlap_end - overlap_start
        
        # Compute as ratio of shorter train
        min_duration = min(train1.duration, train2.duration)
        
        if min_duration <= 0:
            return 0.0
            
        return overlap_duration / min_duration