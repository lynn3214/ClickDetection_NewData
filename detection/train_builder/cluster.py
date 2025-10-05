"""
Click train builder using temporal clustering.
Computes ICI statistics for train-level filtering.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import csv
from pathlib import Path

from detection.candidate_finder.dynamic_threshold import ClickCandidate


@dataclass
class ClickTrain:
    """Click train with ICI statistics."""
    train_id: int
    click_indices: List[int]
    start_time: float
    end_time: float
    duration: float
    n_clicks: int
    ici_median: float
    ici_mean: float
    ici_std: float
    ici_cv: float  # Coefficient of variation
    ici_min: float
    ici_max: float
    mean_confidence: float
    

class TrainBuilder:
    """Builds click trains from detected events."""
    
    def __init__(self,
                 min_ici_ms: float = 5.0,
                 max_ici_ms: float = 150.0,
                 min_train_clicks: int = 2):
        """
        Initialize train builder.
        
        Args:
            min_ici_ms: Minimum inter-click interval (ms)
            max_ici_ms: Maximum inter-click interval (ms)
            min_train_clicks: Minimum clicks per train
        """
        self.min_ici_s = min_ici_ms / 1000
        self.max_ici_s = max_ici_ms / 1000
        self.min_train_clicks = min_train_clicks
        
    def build_trains(self, candidates: List[ClickCandidate]) -> List[ClickTrain]:
        """
        Build click trains from candidates.
        
        Args:
            candidates: List of click candidates (sorted by time)
            
        Returns:
            List of click trains
        """
        if len(candidates) < self.min_train_clicks:
            return []
            
        # Sort by time
        candidates = sorted(candidates, key=lambda c: c.peak_time)
        
        # Cluster into trains
        trains = []
        current_train = [0]  # Start with first click
        
        for i in range(1, len(candidates)):
            ici = candidates[i].peak_time - candidates[i-1].peak_time
            
            if self.min_ici_s <= ici <= self.max_ici_s:
                # Continue current train
                current_train.append(i)
            else:
                # End current train if valid
                if len(current_train) >= self.min_train_clicks:
                    train = self._create_train(
                        len(trains), current_train, candidates
                    )
                    trains.append(train)
                    
                # Start new train
                current_train = [i]
                
        # Handle last train
        if len(current_train) >= self.min_train_clicks:
            train = self._create_train(len(trains), current_train, candidates)
            trains.append(train)
            
        return trains
        
    def _create_train(self,
                     train_id: int,
                     click_indices: List[int],
                     candidates: List[ClickCandidate]) -> ClickTrain:
        """
        Create ClickTrain object with statistics.
        
        Args:
            train_id: Train identifier
            click_indices: Indices of clicks in this train
            candidates: Full list of candidates
            
        Returns:
            ClickTrain object
        """
        # Get times
        times = [candidates[i].peak_time for i in click_indices]
        start_time = times[0]
        end_time = times[-1]
        duration = end_time - start_time
        
        # Calculate ICIs
        icis = np.diff(times) * 1000  # Convert to ms
        
        # ICI statistics
        ici_median = float(np.median(icis))
        ici_mean = float(np.mean(icis))
        ici_std = float(np.std(icis))
        ici_cv = ici_std / ici_mean if ici_mean > 0 else 0
        ici_min = float(np.min(icis))
        ici_max = float(np.max(icis))
        
        # Mean confidence
        confidences = [candidates[i].confidence_score for i in click_indices]
        mean_confidence = float(np.mean(confidences))
        
        return ClickTrain(
            train_id=train_id,
            click_indices=click_indices,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            n_clicks=len(click_indices),
            ici_median=ici_median,
            ici_mean=ici_mean,
            ici_std=ici_std,
            ici_cv=ici_cv,
            ici_min=ici_min,
            ici_max=ici_max,
            mean_confidence=mean_confidence
        )
        
    def filter_trains(self,
                     trains: List[ClickTrain],
                     max_ici_cv: float = 0.4,
                     min_clicks: int = 3) -> List[ClickTrain]:
        """
        Filter trains based on consistency criteria.
        
        Args:
            trains: List of trains
            max_ici_cv: Maximum ICI coefficient of variation
            min_clicks: Minimum clicks per train
            
        Returns:
            Filtered trains
        """
        filtered = []
        
        for train in trains:
            if train.n_clicks >= min_clicks and train.ici_cv <= max_ici_cv:
                filtered.append(train)
                
        return filtered
        
    def check_train_consistency(self,
                               train: ClickTrain,
                               max_ici_cv: float = 0.4,
                               min_clicks: int = 3) -> bool:
        """
        Check if train meets consistency criteria.
        
        Args:
            train: Click train
            max_ici_cv: Maximum ICI CV
            min_clicks: Minimum clicks
            
        Returns:
            True if consistent
        """
        return train.n_clicks >= min_clicks and train.ici_cv <= max_ici_cv
        
    def check_doublet_consistency(self,
                                 train: ClickTrain,
                                 min_ici_ms: float = 8.0,
                                 max_ici_ms: float = 80.0,
                                 min_confidence: float = 0.85) -> bool:
        """
        Check if 2-click train (doublet) is consistent.
        
        Args:
            train: Click train with 2 clicks
            min_ici_ms: Minimum ICI (ms)
            max_ici_ms: Maximum ICI (ms)
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if doublet is consistent
        """
        if train.n_clicks != 2:
            return False
            
        ici_ok = min_ici_ms <= train.ici_median <= max_ici_ms
        conf_ok = train.mean_confidence >= min_confidence
        
        return ici_ok and conf_ok


def save_trains_csv(trains: List[ClickTrain],
                   output_path: Path) -> None:
    """
    Save trains to CSV file.
    
    Args:
        trains: List of click trains
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not trains:
        return
        
    fieldnames = ['train_id', 'start_time', 'end_time', 'duration',
                  'n_clicks', 'ici_median', 'ici_mean', 'ici_std',
                  'ici_cv', 'ici_min', 'ici_max', 'mean_confidence']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for train in trains:
            row = asdict(train)
            # Remove click_indices from output
            row.pop('click_indices', None)
            writer.writerow(row)