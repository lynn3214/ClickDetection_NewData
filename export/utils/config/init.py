"""
Configuration management module.
Handles YAML configuration loading and validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Data and output directory paths configuration."""
    data_dir: Path
    training_dir: Path
    test_dir: Path
    noise_dir: Path
    snippets_dir: Path
    reports_dir: Path


@dataclass
class DetectionConfig:
    """Detection parameters configuration."""
    sample_rate: int
    bandpass_low: float
    bandpass_high: float
    window_ms: float
    step_ms: float
    refractory_ms: float
    tkeo_threshold: float
    ste_threshold: float
    hfc_threshold: float
    envelope_width_range: tuple
    spectral_centroid_min: float


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    early_stopping_patience: int
    optimizer: str
    loss_function: str
    augmentation_enabled: bool


@dataclass
class InferenceConfig:
    """Inference thresholds and export options."""
    high_confidence_threshold: float
    medium_confidence_threshold: float
    train_consistency_required: bool
    min_train_clicks: int
    max_ici_cv: float
    export_audio: bool
    export_features: bool


class ConfigManager:
    """Manages all configuration loading and access."""
    
    def __init__(self, config_dir: Path):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration YAML files
        """
        self.config_dir = Path(config_dir)
        self.paths: Optional[PathConfig] = None
        self.detection: Optional[DetectionConfig] = None
        self.training: Optional[TrainingConfig] = None
        self.inference: Optional[InferenceConfig] = None
        
    def load_all(self) -> None:
        """Load all configuration files."""
        self.paths = self._load_paths()
        self.detection = self._load_detection()
        self.training = self._load_training()
        self.inference = self._load_inference()
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            filename: Name of YAML file
            
        Returns:
            Configuration dictionary
        """
        filepath = self.config_dir / filename
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_paths(self) -> PathConfig:
        """Load paths configuration."""
        config = self._load_yaml("paths.yaml")
        return PathConfig(**config)
        
    def _load_detection(self) -> DetectionConfig:
        """Load detection configuration."""
        config = self._load_yaml("detection.yaml")
        return DetectionConfig(**config)
        
    def _load_training(self) -> TrainingConfig:
        """Load training configuration."""
        config = self._load_yaml("training.yaml")
        return TrainingConfig(**config)
        
    def _load_inference(self) -> InferenceConfig:
        """Load inference configuration."""
        config = self._load_yaml("inference.yaml")
        return InferenceConfig(**config)


def load_config(config_dir: Path) -> ConfigManager:
    """
    Load all configurations.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Loaded configuration manager
    """
    manager = ConfigManager(config_dir)
    manager.load_all()
    return manager