"""
Generate evaluation reports with plots and tables.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd

from training.eval.metrics import ModelEvaluator
from utils.logging.logger import ProjectLogger


class EvaluationReporter:
    """Generates comprehensive evaluation reports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize reporter.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ProjectLogger()
        
    def generate_report(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: np.ndarray,
                       metadata: Optional[Dict] = None,
                       report_name: str = 'evaluation_report') -> Dict[str, Path]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            metadata: Optional metadata dictionary
            report_name: Name for the report
            
        Returns:
            Dictionary of generated file paths
        """
        self.logger.info(f"Generating evaluation report: {report_name}")
        
        # Create report directory
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        generated_files = {}
        
        # Compute metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        # Save metrics as JSON
        metrics_file = report_dir / 'metrics.json'
        self._save_metrics_json(metrics, metrics_file, metadata)
        generated_files['metrics_json'] = metrics_file
        
        # Save metrics as text
        text_file = report_dir / 'metrics.txt'
        self._save_metrics_text(metrics, text_file, metadata)
        generated_files['metrics_text'] = text_file
        
        # Generate plots
        cm_plot = report_dir / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(save_path=cm_plot)
        generated_files['confusion_matrix'] = cm_plot
        
        if 'roc_fpr' in metrics:
            roc_plot = report_dir / 'roc_curve.png'
            evaluator.plot_roc_curve(save_path=roc_plot)
            generated_files['roc_curve'] = roc_plot
            
        if 'pr_precision' in metrics:
            pr_plot = report_dir / 'pr_curve.png'
            evaluator.plot_pr_curve(save_path=pr_plot)
            generated_files['pr_curve'] = pr_plot
            
        # Threshold analysis
        threshold_plot = report_dir / 'threshold_analysis.png'
        proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        evaluator.plot_threshold_analysis(y_true, proba_positive, save_path=threshold_plot)
        generated_files['threshold_analysis'] = threshold_plot
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
            y_true, proba_positive, metric='f1'
        )
        
        threshold_info = {
            'optimal_threshold': float(optimal_threshold),
            'f1_at_optimal': float(optimal_f1)
        }
        
        threshold_file = report_dir / 'optimal_threshold.json'
        with open(threshold_file, 'w') as f:
            json.dump(threshold_info, f, indent=2)
        generated_files['optimal_threshold'] = threshold_file
        
        self.logger.info(f"Report generated at {report_dir}")
        self.logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.3f})")
        
        return generated_files
        
    def _save_metrics_json(self,
                          metrics: Dict,
                          output_path: Path,
                          metadata: Optional[Dict] = None) -> None:
        """Save metrics as JSON (excluding numpy arrays)."""
        json_metrics = {}
        
        # Convert to JSON-serializable format
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                continue  # Skip arrays
            elif isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
                
        if metadata:
            json_metrics['metadata'] = metadata
            
        with open(output_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
            
    def _save_metrics_text(self,
                          metrics: Dict,
                          output_path: Path,
                          metadata: Optional[Dict] = None) -> None:
        """Save metrics as human-readable text."""
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            
            if metadata:
                f.write("Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
            f.write("Classification Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n\n")
            
            if 'roc_auc' in metrics:
                f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")
            if 'pr_auc' in metrics:
                f.write(f"  PR AUC:    {metrics['pr_auc']:.4f}\n")
                
            f.write("\nConfusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"  TN: {metrics['true_negatives']:<6} FP: {metrics['false_positives']}\n")
            f.write(f"  FN: {metrics['false_negatives']:<6} TP: {metrics['true_positives']}\n")
            
    def generate_per_file_report(self,
                                results_by_file: Dict[str, Dict],
                                report_name: str = 'per_file_analysis') -> Path:
        """
        Generate per-file analysis report.
        
        Args:
            results_by_file: Dictionary mapping file_id to metrics
            report_name: Report name
            
        Returns:
            Path to generated CSV
        """
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        # Create DataFrame
        rows = []
        for file_id, metrics in results_by_file.items():
            row = {'file_id': file_id}
            row.update(metrics)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save CSV
        csv_path = report_dir / 'per_file_metrics.csv'
        df.to_csv(csv_path, index=False)
        
        # Save summary statistics
        summary_path = report_dir / 'summary_statistics.txt'
        with open(summary_path, 'w') as f:
            f.write("Per-File Statistics Summary\n")
            f.write("="*60 + "\n\n")
            f.write(df.describe().to_string())
            
        self.logger.info(f"Per-file report saved to {report_dir}")
        
        return csv_path