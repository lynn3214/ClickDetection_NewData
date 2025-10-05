"""
1D CNN model for click classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock1D(nn.Module):
    """1D residual block."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Stride
            dilation: Dilation rate
        """
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        """Forward pass."""
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class ClickClassifier1D(nn.Module):
    """1D CNN classifier for dolphin clicks."""
    
    def __init__(self,
                 input_length: int = 8820,  # 0.2s at 44.1kHz
                 num_classes: int = 2,
                 base_channels: int = 32,
                 num_blocks: int = 4,
                 dropout: float = 0.3):
        """
        Initialize classifier.
        
        Args:
            input_length: Input signal length
            num_classes: Number of output classes
            base_channels: Base number of channels
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv_init = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn_init = nn.BatchNorm1d(base_channels)
        
        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        channels = base_channels
        
        for i in range(num_blocks):
            out_channels = channels * 2 if i % 2 == 1 else channels
            dilation = 2 ** (i % 3)  # 1, 2, 4, 1, 2, 4, ...
            stride = 2 if i % 2 == 1 else 1
            
            block = ResidualBlock1D(
                channels, out_channels,
                kernel_size=3, stride=stride, dilation=dilation
            )
            self.blocks.append(block)
            channels = out_channels
            
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, length] or [batch, 1, length]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, length]
            
        # Initial conv
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
            
        # Global pooling
        x = self.global_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # [batch, channels]
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def predict_proba(self, x):
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities [batch, num_classes]
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


def create_model(config: dict) -> ClickClassifier1D:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        ClickClassifier1D model
    """
    return ClickClassifier1D(
        input_length=config.get('input_length', 8820),
        num_classes=config.get('num_classes', 2),
        base_channels=config.get('base_channels', 32),
        num_blocks=config.get('num_blocks', 4),
        dropout=config.get('dropout', 0.3)
    )