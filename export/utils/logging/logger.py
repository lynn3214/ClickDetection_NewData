"""
Unified logging utility with progress tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from tqdm import tqdm


class ProjectLogger:
    """Centralized logger for the project."""
    
    def __init__(self,
                 name: str = "DolphinClickDetection",
                 log_dir: Optional[Path] = None,
                 level: int = logging.INFO):
        """
        Initialize project logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging to file: {log_file}")
            
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
        

class ProgressTracker:
    """Progress tracking with tqdm."""
    
    def __init__(self, total: int, desc: str = "Processing", unit: str = "it"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description
            unit: Unit name
        """
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        
    def update(self, n: int = 1):
        """Update progress."""
        self.pbar.update(n)
        
    def set_postfix(self, **kwargs):
        """Set postfix statistics."""
        self.pbar.set_postfix(**kwargs)
        
    def close(self):
        """Close progress bar."""
        self.pbar.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()