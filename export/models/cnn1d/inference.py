"""
Model inference wrapper.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List

from models.cnn1d.model import ClickClassifier1D


class ClickDetectorInference:
    """Inference wrapper for click detection model."""
    
    def __init__(self,
                 model: ClickClassifier1D,
                 device: str = 'cpu',
                 batch_size: int = 32):
        """
        Initialize inference wrapper.
        
        Args:
            model: Trained model
            device: Device ('cpu' or 'cuda')
            batch_size: Batch size for inference
        """
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        self.model.to(self.device)
        self.model.eval()
        
    @classmethod
    def from_checkpoint(cls,
                       checkpoint_path: Union[str, Path],
                       device: str = 'cpu',
                       batch_size: int = 32) -> 'ClickDetectorInference':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device
            batch_size: Batch size
            
        Returns:
            ClickDetectorInference instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Recreate model
        model_config = checkpoint.get('model_config', {})
        model = ClickClassifier1D(**model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device, batch_size)
        
    def predict_single(self, waveform: np.ndarray) -> float:
        """
        Predict probability for single waveform.
        
        Args:
            waveform: Input waveform [length]
            
        Returns:
            Click probability (0-1)
        """
        with torch.no_grad():
            # Convert to tensor
            x = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, length]
            x = x.to(self.device)
            
            # Predict
            probs = self.model.predict_proba(x)
            click_prob = probs[0, 1].cpu().item()  # Class 1 probability
            
        return click_prob
        
    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for batch of waveforms.
        
        Args:
            waveforms: Input waveforms [batch, length]
            
        Returns:
            Click probabilities [batch]
        """
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(waveforms), self.batch_size):
                batch = waveforms[i:i + self.batch_size]
                
                # Convert to tensor
                x = torch.from_numpy(batch).float()
                x = x.to(self.device)
                
                # Predict
                probs = self.model.predict_proba(x)
                click_probs = probs[:, 1].cpu().numpy()  # Class 1 probabilities
                
                all_probs.append(click_probs)
                
        return np.concatenate(all_probs)
        
    def predict_list(self, waveform_list: List[np.ndarray]) -> np.ndarray:
        """
        Predict probabilities for list of waveforms (possibly different lengths).
        
        Args:
            waveform_list: List of waveforms
            
        Returns:
            Click probabilities [len(waveform_list)]
        """
        # Pad to same length
        max_len = max(len(w) for w in waveform_list)
        padded = np.zeros((len(waveform_list), max_len), dtype=np.float32)
        
        for i, waveform in enumerate(waveform_list):
            padded[i, :len(waveform)] = waveform
            
        return self.predict_batch(padded)