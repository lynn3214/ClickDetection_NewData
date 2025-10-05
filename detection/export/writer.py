"""
Export detected events and trains to files.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import soundfile as sf
import csv
from dataclasses import asdict

from detection.candidate_finder.dynamic_threshold import ClickCandidate
from detection.train_builder.cluster import ClickTrain


class ExportWriter:
    """Handles exporting detection results."""
    
    def __init__(self, output_dir: Path, sample_rate: int = 44100):
        """
        Initialize export writer.
        
        Args:
            output_dir: Output directory for exports
            sample_rate: Audio sample rate
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
    def export_events(self,
                     candidates: List[ClickCandidate],
                     audio: np.ndarray,
                     file_id: str,
                     export_audio: bool = True) -> Dict[str, Path]:
        """
        Export detected events.
        
        Args:
            candidates: List of click candidates
            audio: Full audio signal
            file_id: Identifier for source file
            export_audio: Whether to export audio segments
            
        Returns:
            Dictionary of exported file paths
        """
        events_dir = self.output_dir / 'events' / file_id
        events_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export CSV
        csv_path = events_dir / 'events.csv'
        self._export_events_csv(candidates, csv_path)
        exported['csv'] = csv_path
        
        # Export audio segments
        if export_audio:
            audio_dir = events_dir / 'audio'
            audio_dir.mkdir(exist_ok=True)
            
            for i, candidate in enumerate(candidates):
                segment = self._extract_segment(audio, candidate.peak_idx)
                filename = f"click_{i:04d}_{int(candidate.peak_time*1000):08d}ms.wav"
                filepath = audio_dir / filename
                sf.write(str(filepath), segment, self.sample_rate)
                
            exported['audio_dir'] = audio_dir
            
        return exported
        
    def export_trains(self,
                     trains: List[ClickTrain],
                     candidates: List[ClickCandidate],
                     audio: np.ndarray,
                     file_id: str,
                     export_audio: bool = True) -> Dict[str, Path]:
        """
        Export detected click trains.
        
        Args:
            trains: List of click trains
            candidates: List of all candidates
            audio: Full audio signal
            file_id: Identifier for source file
            export_audio: Whether to export audio segments
            
        Returns:
            Dictionary of exported file paths
        """
        trains_dir = self.output_dir / 'trains' / file_id
        trains_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export trains CSV
        csv_path = trains_dir / 'trains.csv'
        self._export_trains_csv(trains, csv_path)
        exported['csv'] = csv_path
        
        # Export train-level audio
        if export_audio:
            audio_dir = trains_dir / 'audio'
            audio_dir.mkdir(exist_ok=True)
            
            for train in trains:
                # Extract full train segment
                train_segment = self._extract_train_segment(
                    audio, train, candidates
                )
                filename = f"train_{train.train_id:04d}.wav"
                filepath = audio_dir / filename
                sf.write(str(filepath), train_segment, self.sample_rate)
                
            exported['audio_dir'] = audio_dir
            
        return exported
        
    def _export_events_csv(self,
                          candidates: List[ClickCandidate],
                          output_path: Path) -> None:
        """Export candidates to CSV."""
        if not candidates:
            return
            
        fieldnames = list(asdict(candidates[0]).keys())
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for candidate in candidates:
                writer.writerow(asdict(candidate))
                
    def _export_trains_csv(self,
                          trains: List[ClickTrain],
                          output_path: Path) -> None:
        """Export trains to CSV."""
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
                row.pop('click_indices', None)
                writer.writerow(row)
                
    def _extract_segment(self,
                        audio: np.ndarray,
                        peak_idx: int,
                        pre_ms: float = 2.0,
                        post_ms: float = 10.0) -> np.ndarray:
        """Extract short segment around peak."""
        pre_samples = int(pre_ms * self.sample_rate / 1000)
        post_samples = int(post_ms * self.sample_rate / 1000)
        
        start = max(0, peak_idx - pre_samples)
        end = min(len(audio), peak_idx + post_samples)
        
        segment = audio[start:end]
        
        # Normalize
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak
            
        return segment
        
    def _extract_train_segment(self,
                              audio: np.ndarray,
                              train: ClickTrain,
                              candidates: List[ClickCandidate],
                              margin_ms: float = 50.0) -> np.ndarray:
        """Extract segment containing entire train with margins."""
        # Get first and last click indices
        first_idx = candidates[train.click_indices[0]].peak_idx
        last_idx = candidates[train.click_indices[-1]].peak_idx
        
        # Add margins
        margin_samples = int(margin_ms * self.sample_rate / 1000)
        start = max(0, first_idx - margin_samples)
        end = min(len(audio), last_idx + margin_samples)
        
        segment = audio[start:end]
        
        # Normalize
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak
            
        return segment
        
    def create_summary_report(self,
                            file_id: str,
                            stats: Dict[str, Any]) -> Path:
        """
        Create summary report text file.
        
        Args:
            file_id: File identifier
            stats: Statistics dictionary
            
        Returns:
            Path to report file
        """
        report_path = self.output_dir / 'reports' / f"{file_id}_summary.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"Detection Summary Report: {file_id}\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
                
        return report_path