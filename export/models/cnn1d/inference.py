"""
Model inference wrapper (修复模型加载兼容性).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List

from models.cnn1d.model import ClickClassifier1D, LightweightClickClassifier, create_model


class ClickDetectorInference:
    """Inference wrapper for click detection model."""
    
    def __init__(self,
                 model,
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
        Load model from checkpoint (自动检测模型架构).
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device
            batch_size: Batch size
            
        Returns:
            ClickDetectorInference instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # ========== 🔧 自动推断模型配置 ==========
        model_config = checkpoint.get('model_config', {})
        
        # 1. 检测模型类型
        has_fc1 = 'fc1.weight' in state_dict
        has_fc = 'fc.weight' in state_dict
        use_lightweight = has_fc and not has_fc1
        
        # 2. 检测通道数
        base_channels = state_dict['conv_init.weight'].shape[0]
        
        # 3. 检测block数量
        num_blocks = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and '.conv1.weight' in k)
        
        # 4. 获取输入长度
        input_length = model_config.get('input_length', 22050)
        num_classes = model_config.get('num_classes', 2)
        
        # 构建完整配置
        full_config = {
            'input_length': input_length,
            'num_classes': num_classes,
            'use_lightweight': use_lightweight,
            'base_channels': base_channels,
            'num_blocks': num_blocks,
            'dropout': 0.3
        }
        
        print(f"[INFO] 检测到模型配置:")
        print(f"  - 模型类型: {'Lightweight' if use_lightweight else 'Full'}")
        print(f"  - 基础通道: {base_channels}")
        print(f"  - Block数量: {num_blocks}")
        print(f"  - 输入长度: {input_length}")
        
        # ========== 创建模型 ==========
        try:
            model = create_model(full_config)
            model.load_state_dict(state_dict)
            print(f"[SUCCESS] 模型加载成功!")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise
        
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