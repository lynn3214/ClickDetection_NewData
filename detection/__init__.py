"""
Detection module for dolphin click detection.
"""

from .candidate_finder.dynamic_threshold import (
    AdaptiveDetector,
    DetectionParams,
    ClickCandidate
)
from .segmenter.cropper import ClickSegmenter
from .features_event.event_stats import (
    EventStatsExtractor,
    save_event_stats_csv
)
from .train_builder.cluster import (
    TrainBuilder,
    ClickTrain,
    save_trains_csv
)
from .fusion.decision import (
    FusionDecider,
    FusionConfig
)
from .export.writer import ExportWriter

__all__ = [
    # Candidate detection
    'AdaptiveDetector',
    'DetectionParams',
    'ClickCandidate',
    
    # Segmentation
    'ClickSegmenter',
    
    # Event statistics
    'EventStatsExtractor',
    'save_event_stats_csv',
    
    # Train building
    'TrainBuilder',
    'ClickTrain',
    'save_trains_csv',
    
    # Fusion
    'FusionDecider',
    'FusionConfig',
    
    # Export
    'ExportWriter',
]