"""
Evaluation metrics for model performance.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, f1_score
)
from typing import Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
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
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tn'] = int(tn)
        
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
        
    def plot_confusion_matrix(self,
                             save_path: Optional[Path] = None,
                             class_names: list = None) -> plt.Figure:
        """Plot confusion matrix."""
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
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
                
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    def plot_roc_curve(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot ROC curve."""
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
            plt.close(fig)
            
        return fig
        
    def plot_pr_curve(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
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
            plt.close(fig)
            
        return fig


# ⚠️ 新增: 兼容 inference.py 的函数接口
def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Wrapper function for computing metrics (compatible with inference.py).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.compute_metrics(y_true, y_pred, y_proba)


def plot_results(metrics: Dict,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                y_proba: np.ndarray,
                output_dir: Path) -> None:
    """
    Generate and save all evaluation plots.
    
    Args:
        metrics: Metrics dictionary
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator()
    evaluator.metrics = metrics
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # ROC curve
    if 'roc_fpr' in metrics:
        evaluator.plot_roc_curve(
            save_path=output_dir / 'roc_curve.png'
        )
    
    # PR curve
    if 'pr_precision' in metrics:
        evaluator.plot_pr_curve(
            save_path=output_dir / 'pr_curve.png'
        )