"""
Candidate finder module for click detection.
"""

from .dynamic_threshold import (
    AdaptiveDetector,
    DetectionParams,
    ClickCandidate
)
from .peak_merge import PeakMerger

__all__ = [
    'AdaptiveDetector',
    'DetectionParams',
    'ClickCandidate',
    'PeakMerger',
]