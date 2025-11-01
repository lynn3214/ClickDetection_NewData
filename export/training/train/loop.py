"""
Training loop with early stopping and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

from models.cnn1d.model import ClickClassifier1D
from utils.logging.logger import ProjectLogger


class Trainer:
    """Handles model training with early stopping."""
    
    def __init__(self,
                 model: ClickClassifier1D,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: Weight decay
            class_weights: Optional class weights for loss
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Logger
        self.logger = ProjectLogger()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 100,
             early_stopping_patience: int = 10,
             checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Log
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / 'best_model.pt',
                        epoch, val_loss
                    )
                    self.logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Save periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt',
                    epoch, val_loss
                )
                
        self.logger.info("Training completed")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history
        
    def save_checkpoint(self,
                       path: Path,
                       epoch: int,
                       val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Checkpoint path
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'model_config': {
                'input_length': self.model.input_length,
                'num_classes': self.model.num_classes
            }
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


def create_dataloaders(waveforms: np.ndarray,
                      labels: np.ndarray,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        waveforms: Waveform array [N, length]
        labels: Label array [N]
        batch_size: Batch size
        val_split: Validation split ratio
        shuffle: Whether to shuffle
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data
    n_samples = len(waveforms)
    n_val = int(n_samples * val_split)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
        
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(waveforms[train_indices]).float(),
        torch.from_numpy(labels[train_indices]).long()
    )
    
    val_dataset = TensorDataset(
        torch.from_numpy(waveforms[val_indices]).float(),
        torch.from_numpy(labels[val_indices]).long()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader