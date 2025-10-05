"""
Click segmentation module.
"""

from .cropper import (
    ClickSegmenter,
    save_candidates_csv
)

__all__ = [
    'ClickSegmenter',
    'save_candidates_csv',
]
