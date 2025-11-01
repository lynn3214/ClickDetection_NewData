"""1D CNN models for click classification."""
from .model import ClickClassifier1D, LightweightClickClassifier, create_model
from .inference import ClickDetectorInference

__all__ = [
    'ClickClassifier1D',
    'LightweightClickClassifier',
    'create_model',
    'ClickDetectorInference'
]