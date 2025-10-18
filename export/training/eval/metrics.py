"""
Evaluation metrics for model performance.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, f1_score
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class ModelEvaluator:
    """Evaluates model performance with various metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        
    def compute_metrics(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels [N]
            y_pred: Predicted labels [N]
            y_proba: Prediction probabilities [N, num_classes] or [N]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics['confusion_matrix'] = cm
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_negatives'] = int(tn)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # If probabilities provided, compute ROC and PR curves
        if y_proba is not None:
            # Handle both binary and multi-class
            if y_proba.ndim == 2:
                y_proba_positive = y_proba[:, 1]  # Probability of positive class
            else:
                y_proba_positive = y_proba
                
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba_positive)
            roc_auc = auc(fpr, tpr)
            
            metrics['roc_fpr'] = fpr
            metrics['roc_tpr'] = tpr
            metrics['roc_thresholds'] = roc_thresholds
            metrics['roc_auc'] = roc_auc
            
            # PR curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true, y_proba_positive
            )
            pr_auc = auc(recall_curve, precision_curve)
            
            metrics['pr_precision'] = precision_curve
            metrics['pr_recall'] = recall_curve
            metrics['pr_thresholds'] = pr_thresholds
            metrics['pr_auc'] = pr_auc
            
        self.metrics = metrics
        return metrics
        
    def find_optimal_threshold(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities for positive class
            metric: Metric to optimize ('f1', 'recall', 'precision')
            
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0, 1, 101)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'precision':
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                score = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            scores.append(score)
            
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return optimal_threshold, optimal_score
        
    def plot_confusion_matrix(self,
                             save_path: Optional[Path] = None,
                             class_names: list = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save figure
            class_names: List of class names
            
        Returns:
            Matplotlib figure
        """
        if 'confusion_matrix' not in self.metrics:
            raise ValueError("Metrics not computed yet")
            
        cm = self.metrics['confusion_matrix']
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
            
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
                
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    def plot_roc_curve(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'roc_fpr' not in self.metrics:
            raise ValueError("ROC metrics not computed")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(self.metrics['roc_fpr'], self.metrics['roc_tpr'],
               label=f"ROC curve (AUC = {self.metrics['roc_auc']:.3f})",
               linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    def plot_pr_curve(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'pr_precision' not in self.metrics:
            raise ValueError("PR metrics not computed")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(self.metrics['pr_recall'], self.metrics['pr_precision'],
               label=f"PR curve (AUC = {self.metrics['pr_auc']:.3f})",
               linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    def plot_threshold_analysis(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot how metrics vary with threshold.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        thresholds = np.linspace(0, 1, 101)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        ax.axvline(thresholds[optimal_idx], color='red', linestyle='--',
                  label=f'Optimal threshold = {thresholds[optimal_idx]:.2f}')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Metrics vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig