"""Model definitions."""
from models.cnn1d.inference import ClickDetectorInference
from models.cnn1d.model import ClickClassifier1D, LightweightClickClassifier, create_model

__all__ = [
    'ClickDetectorInference',
    'ClickClassifier1D',
    'LightweightClickClassifier',
    'create_model'
]