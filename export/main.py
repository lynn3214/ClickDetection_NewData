#!/usr/bin/env python3
"""
Dolphin Click Detector - Inference Script
For model inference and evaluation on new datasets
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import pandas as pd

from models.cnn1d.inference import ClickDetectorInference
from training.eval import ModelEvaluator, EvaluationReporter
from utils.logging import ProjectLogger


def load_config(config_path: Path) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_audio_segments(data_dir: Path, sample_rate: int = 44100) -> tuple:
    """
    Load audio segments
    
    Args:
        data_dir: Data directory
        sample_rate: Sample rate
        
    Returns:
        (waveforms, file_ids)
    """
    audio_files = list(data_dir.glob('*.wav'))
    
    waveforms = []
    file_ids = []
    
    for audio_file in tqdm(audio_files, desc=f"Loading {data_dir.name}"):
        try:
            audio, sr = sf.read(audio_file)
            
            # Convert to mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Resample (if needed)
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            waveforms.append(audio)
            file_ids.append(audio_file.stem)
            
        except Exception as e:
            print(f"Failed to load {audio_file.name}: {e}")
            continue
    
    return waveforms, file_ids


def evaluate_model(config_path: Path,
                   checkpoint_path: Path,
                   positive_dir: Path,
                   negative_dir: Path,
                   output_dir: Path):
    """
    Evaluate model performance
    
    Args:
        config_path: Configuration file path
        checkpoint_path: Model checkpoint path
        positive_dir: Positive samples directory
        negative_dir: Negative samples directory
        output_dir: Output directory
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ProjectLogger(log_dir=output_dir)
    logger.info("="*70)
    logger.info("Dolphin Click Detector - Model Evaluation")
    logger.info("="*70)
    
    # Load config
    config = load_config(config_path)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Model checkpoint: {checkpoint_path}")
    
    # Inference parameters
    inference_config = config.get('inference', {})
    batch_size = inference_config.get('batch_size', 32)
    device = inference_config.get('device', 'cpu')
    sample_rate = config.get('data', {}).get('sample_rate', 44100) or 44100
    
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    
    # Load model
    logger.info("\nLoading model...")
    model = ClickDetectorInference.from_checkpoint(
        checkpoint_path,
        device=device,
        batch_size=batch_size
    )
    logger.info("✓ Model loaded successfully")
    
    # Load data
    logger.info("\nLoading test data...")
    logger.info(f"Positive samples directory: {positive_dir}")
    logger.info(f"Negative samples directory: {negative_dir}")
    
    pos_waveforms, pos_ids = load_audio_segments(positive_dir, sample_rate)
    neg_waveforms, neg_ids = load_audio_segments(negative_dir, sample_rate)
    
    logger.info(f"✓ Positive samples: {len(pos_waveforms)}")
    logger.info(f"✓ Negative samples: {len(neg_waveforms)}")
    
    # Combine data
    all_waveforms = pos_waveforms + neg_waveforms
    all_ids = pos_ids + neg_ids
    y_true = np.array([1]*len(pos_waveforms) + [0]*len(neg_waveforms))
    
    # Pad to uniform length
    logger.info("\nPreprocessing audio segments...")
    target_length = model.model.input_length
    padded_waveforms = []
    
    for waveform in tqdm(all_waveforms, desc="Padding audio"):
        if len(waveform) < target_length:
            # Center padding
            pad_total = target_length - len(waveform)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = np.pad(waveform, (pad_left, pad_right), mode='constant')
        else:
            # Truncate
            padded = waveform[:target_length]
        
        padded_waveforms.append(padded)
    
    X = np.array(padded_waveforms, dtype=np.float32)
    logger.info(f"✓ Data shape: {X.shape}")
    
    # Inference
    logger.info("\nStarting inference...")
    y_proba = model.predict_batch(X)
    
    # Apply threshold
    threshold = config.get('thresholds', {}).get('confidence_threshold', 0.5)
    y_pred = (y_proba >= threshold).astype(int)
    
    logger.info(f"✓ Inference complete")
    logger.info(f"Confidence threshold: {threshold}")
    
    # Evaluation
    logger.info("\nComputing evaluation metrics...")
    evaluator = ModelEvaluator()
    
    # Convert probabilities to 2D array (for compatibility with evaluation code)
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba_2d)
    
    logger.info("\n" + "="*70)
    logger.info("Evaluation Results")
    logger.info("="*70)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC:            {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        logger.info(f"PR AUC:             {metrics['pr_auc']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Negative (TN): {metrics['tn']:<6} False Positive (FP): {metrics['fp']}")
    logger.info(f"  False Negative (FN): {metrics['fn']:<6} True Positive (TP): {metrics['tp']}")
    
    # Generate report
    output_config = config.get('output', {})
    
    if output_config.get('save_predictions', True):
        logger.info("\nSaving predictions...")
        predictions_df = pd.DataFrame({
            'file_id': all_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'confidence': y_proba
        })
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"✓ Predictions saved: {predictions_path}")
    
    if output_config.get('save_misclassified_files', True):
        logger.info("\nSaving list of misclassified files...")
        misclassified_mask = (y_true != y_pred)
        misclassified_df = pd.DataFrame({
            'file_id': np.array(all_ids)[misclassified_mask],
            'true_label': y_true[misclassified_mask],
            'predicted_label': y_pred[misclassified_mask],
            'confidence': y_proba[misclassified_mask]
        })
        misclassified_path = output_dir / 'misclassified.csv'
        misclassified_df.to_csv(misclassified_path, index=False)
        logger.info(f"✓ Misclassified list saved: {misclassified_path}")
        logger.info(f"  Misclassified count: {misclassified_mask.sum()}")
    
    if output_config.get('save_confusion_matrix', True):
        logger.info("\nGenerating confusion matrix plot...")
        cm_path = output_dir / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(save_path=cm_path)
        logger.info(f"✓ Confusion matrix saved: {cm_path}")
    
    if output_config.get('save_roc_curve', True) and 'roc_fpr' in metrics:
        logger.info("\nGenerating ROC curve...")
        roc_path = output_dir / 'roc_curve.png'
        evaluator.plot_roc_curve(save_path=roc_path)
        logger.info(f"✓ ROC curve saved: {roc_path}")
    
    if output_config.get('save_pr_curve', True) and 'pr_precision' in metrics:
        logger.info("\nGenerating PR curve...")
        pr_path = output_dir / 'pr_curve.png'
        evaluator.plot_pr_curve(save_path=pr_path)
        logger.info(f"✓ PR curve saved: {pr_path}")
    
    if output_config.get('generate_detailed_report', True):
        logger.info("\nGenerating detailed report...")
        reporter = EvaluationReporter(output_dir)
        
        metadata = {
            'checkpoint': str(checkpoint_path),
            'positive_samples': len(pos_waveforms),
            'negative_samples': len(neg_waveforms),
            'threshold': threshold,
            'sample_rate': sample_rate
        }
        
        reporter.generate_report(
            y_true, y_pred, y_proba_2d,
            metadata=metadata,
            report_name='evaluation'
        )
        logger.info(f"✓ Detailed report saved: {output_dir / 'evaluation'}")
    
    logger.info("\n" + "="*70)
    logger.info("Evaluation complete!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Dolphin Click Detector - Model Inference and Evaluation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_wav.yaml',
        help='Config file path (default: configs/eval_wav.yaml)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best.pt',
        help='Model checkpoint path (default: checkpoints/best.pt)'
    )
    
    parser.add_argument(
        '--positive-dir',
        type=str,
        required=True,
        help='Positive samples directory (audio segments with dolphin clicks)'
    )
    
    parser.add_argument(
        '--negative-dir',
        type=str,
        required=True,
        help='Negative samples directory (noise segments)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir)
    output_dir = Path(args.output_dir)
    
    # Check file existence
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if not positive_dir.exists():
        raise FileNotFoundError(f"Positive samples directory not found: {positive_dir}")
    
    if not negative_dir.exists():
        raise FileNotFoundError(f"Negative samples directory not found: {negative_dir}")
    
    # Run evaluation
    evaluate_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
