"""
Configuration management module.
Handles YAML configuration loading and validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file and return as dictionary.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


class ConfigNamespace:
    """
    Namespace wrapper for configuration dictionary.
    Allows accessing config values via dot notation.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize namespace from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dictionary-style get with default."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config_namespace(config_path: Union[str, Path]) -> ConfigNamespace:
    """
    Load configuration and return as namespace object.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration namespace
    """
    config_dict = load_config(config_path)
    return ConfigNamespace(config_dict)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    
    return merged


# For backward compatibility, keep the original class structure
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
        return load_config(self.config_dir / filename)
            
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


def load_config_manager(config_dir: Path) -> ConfigManager:
    """
    Load all configurations using ConfigManager.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Loaded configuration manager
    """
    manager = ConfigManager(config_dir)
    manager.load_all()
    return manager