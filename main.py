"""
Main CLI entry point for dolphin click detection pipeline.
"""

import argparse
from pathlib import Path
import sys

from utils.config import load_config
from utils.logging.logger import ProjectLogger
from utils.audio_io.manifest import scan_audio_files
#from utils.preprocessing.resample_and_filter import preprocess_audio_file
from detection.candidate_finder.dynamic_threshold import AdaptiveDetector, DetectionParams
from detection.segmenter.cropper import ClickSegmenter
from detection.features_event.event_stats import EventStatsExtractor, save_event_stats_csv
from detection.train_builder.cluster import TrainBuilder, save_trains_csv
from detection.fusion.decision import FusionDecider, FusionConfig
from detection.export.writer import ExportWriter
from training.dataset.segments import DatasetBuilder
from training.augment.pipeline import AugmentationPipeline
from models.cnn1d.model import create_model
from models.cnn1d.inference import ClickDetectorInference
from training.train.loop import Trainer, create_dataloaders
from training.eval.report import EvaluationReporter

import numpy as np
import torch
import soundfile as sf
import pandas as pd


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Dolphin Click Detection Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan audio files')
    scan_parser.add_argument('--input-dir', type=str, required=True,
                            help='Input directory to scan')
    scan_parser.add_argument('--output', type=str, required=True,
                            help='Output manifest file')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect click candidates')
    detect_parser.add_argument('--input', type=str, required=True,
                              help='Input audio file')
    detect_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory')
    detect_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                              help='Detection config file')
    
    # Batch detect command
    batch_detect_parser = subparsers.add_parser(
        'batch-detect', 
        help='Batch detect clicks in directory'
    )
    batch_detect_parser.add_argument('--input-dir', type=str, required=True,
                                    help='Input directory containing wav files')
    batch_detect_parser.add_argument('--output-dir', type=str, required=True,
                                    help='Output directory for results')
    batch_detect_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                                    help='Detection config file')
    batch_detect_parser.add_argument('--save-audio', action='store_true',
                                    help='Save extracted click segments')
    batch_detect_parser.add_argument('--recursive', action='store_true',
                                    help='Search recursively for wav files')
    
    # Trains command
    trains_parser = subparsers.add_parser('trains', help='Build click trains')
    trains_parser.add_argument('--events-csv', type=str, required=True,
                              help='Events CSV file')
    trains_parser.add_argument('--output', type=str, required=True,
                              help='Output trains CSV')
    trains_parser.add_argument('--config', type=str, default='configs/detection.yaml',
                              help='Detection config file')
    
    # Build dataset command
    dataset_parser = subparsers.add_parser('build-dataset',
                                          help='Build training dataset')
    dataset_parser.add_argument('--events-dir', type=str, required=True,
                               help='Directory containing detected events')
    dataset_parser.add_argument('--noise-dir', type=str, required=True,
                               help='Directory containing noise samples')
    dataset_parser.add_argument('--output-dir', type=str, required=True,
                               help='Output dataset directory')
    dataset_parser.add_argument('--config', type=str, default='configs/training.yaml',
                               help='Training config file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train CNN model')
    train_parser.add_argument('--dataset-dir', type=str, required=True,
                             help='Dataset directory')
    train_parser.add_argument('--output-dir', type=str, required=True,
                             help='Output directory for checkpoints')
    train_parser.add_argument('--config', type=str, default='configs/training.yaml',
                             help='Training config file')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Model checkpoint path')
    eval_parser.add_argument('--dataset-dir', type=str, required=True,
                            help='Test dataset directory')
    eval_parser.add_argument('--output-dir', type=str, required=True,
                            help='Output directory for reports')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export final detections')
    export_parser.add_argument('--input', type=str, required=True,
                              help='Input audio file')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Model checkpoint')
    export_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory')
    export_parser.add_argument('--config', type=str, default='configs/inference.yaml',
                              help='Inference config file')
    
    return parser


def cmd_scan(args):
    """Execute scan command."""
    logger = ProjectLogger()
    logger.info(f"Scanning directory: {args.input_dir}")
    
    manifest = scan_audio_files(
        Path(args.input_dir),
        extensions=['.wav'],
        recursive=True
    )
    
    manifest.to_csv(args.output, index=False)
    logger.info(f"Manifest saved to {args.output}")
    logger.info(f"Found {len(manifest)} audio files")


def cmd_detect(args):
    """Execute detect command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info(f"Detecting clicks in: {args.input}")
    
    # Load audio
    audio, sr = sf.read(args.input)
    
    # Initialize detector
    params = DetectionParams(
        tkeo_threshold=config['thresholds']['tkeo_z'],
        ste_threshold=config['thresholds']['ste_z'],
        hfc_threshold=config['thresholds']['hfc_z'],
        high_low_ratio_threshold=config['thresholds']['high_low_ratio'],
        envelope_width_min=config['envelope']['width_min_ms'],
        envelope_width_max=config['envelope']['width_max_ms'],
        spectral_centroid_min=config['thresholds']['spectral_centroid_min'],
        refractory_ms=config['refractory_ms']
    )
    
    detector = AdaptiveDetector(sample_rate=sr, params=params)
    
    # Detect
    candidates = detector.batch_detect(
        audio,
        chunk_duration=config['batch']['chunk_duration_s'],
        overlap=config['batch']['overlap_s']
    )
    
    logger.info(f"Detected {len(candidates)} candidates")
    
    # Extract event statistics
    stats_extractor = EventStatsExtractor(sample_rate=sr)
    stats_list = []
    
    for candidate in candidates:
        stats = stats_extractor.extract_event_stats(audio, candidate)
        stats_list.append(stats)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    events_csv = output_dir / 'events.csv'
    save_event_stats_csv(stats_list, events_csv)
    
    logger.info(f"Events saved to {events_csv}")

def cmd_batch_detect(args):
    """Execute batch-detect command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for wav files
    logger.info(f"Scanning for wav files in: {input_dir}")
    if args.recursive:
        wav_files = list(input_dir.rglob('*.wav'))
    else:
        wav_files = list(input_dir.glob('*.wav'))
    
    logger.info(f"Found {len(wav_files)} wav files")
    
    # Initialize detector
    params = DetectionParams(
        tkeo_threshold=config['thresholds']['tkeo_z'],
        ste_threshold=config['thresholds']['ste_z'],
        hfc_threshold=config['thresholds']['hfc_z'],
        high_low_ratio_threshold=config['thresholds']['high_low_ratio'],
        envelope_width_min=config['envelope']['width_min_ms'],
        envelope_width_max=config['envelope']['width_max_ms'],
        spectral_centroid_min=config['thresholds']['spectral_centroid_min'],
        refractory_ms=config['refractory_ms']
    )
    
    # Process each file
    from tqdm import tqdm
    all_stats = []
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        try:
            # Load audio
            audio, sr = sf.read(wav_file)
            file_id = wav_file.stem
            
            # Detect
            detector = AdaptiveDetector(sample_rate=sr, params=params)
            candidates = detector.batch_detect(
                audio,
                chunk_duration=config['batch']['chunk_duration_s'],
                overlap=config['batch']['overlap_s']
            )
            
            logger.info(f"{wav_file.name}: {len(candidates)} candidates")
            
            # Extract stats
            stats_extractor = EventStatsExtractor(sample_rate=sr)
            for candidate in candidates:
                stats = stats_extractor.extract_event_stats(audio, candidate)
                stats['file_id'] = file_id
                stats['source_file'] = str(wav_file)
                all_stats.append(stats)
            
            # Optional: Save audio segments
            if args.save_audio and candidates:
                segmenter = ClickSegmenter(sample_rate=sr)
                audio_dir = output_dir / 'audio' / file_id
                segmenter.extract_and_save(audio, candidates, audio_dir, file_id)
                
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {str(e)}")
            continue
    
    # Save combined results
    if all_stats:
        csv_path = output_dir / 'all_events.csv'
        save_event_stats_csv(all_stats, csv_path)
        logger.info(f"Saved {len(all_stats)} events to {csv_path}")
    else:
        logger.warning("No events detected in any files")


def cmd_trains(args):
    """Execute trains command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info(f"Building trains from: {args.events_csv}")
    
    # Load events (this is simplified - in practice you'd reconstruct ClickCandidate objects)
    events_df = pd.read_csv(args.events_csv)
    
    # For this example, we'll need to import the candidates from somewhere
    # This is a placeholder - actual implementation would need to deserialize candidates
    logger.warning("Train building from CSV requires candidate objects - not fully implemented")
    
    # Initialize train builder
    train_builder = TrainBuilder(
        min_ici_ms=config['train']['min_ici_ms'],
        max_ici_ms=config['train']['max_ici_ms'],
        min_train_clicks=config['train']['min_train_clicks']
    )
    
    # Build trains (placeholder)
    # trains = train_builder.build_trains(candidates)
    
    logger.info("Train building command - implementation depends on serialization format")


def cmd_build_dataset(args):
    """Execute build-dataset command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info("Building training dataset")
    
    # Initialize dataset builder
    dataset_config = config['dataset']
    sample_rate = config.get('sample_rate', 44100)
    builder = DatasetBuilder(
        sample_rate=sample_rate,
        window_ms=dataset_config['window_ms'],
        random_offset_ms=dataset_config['random_offset_ms']
    )
    
    events_dir = Path(args.events_dir)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)
    
    all_positive_samples = []
    all_negative_samples = []
    
    # Process positive samples (from detected events)
    logger.info(f"Processing positive samples from {events_dir}")
    
    # Method 1: If events are saved as individual wav files
    event_audio_files = list(events_dir.rglob('*.wav'))
    if event_audio_files:
        logger.info(f"Found {len(event_audio_files)} event audio files")
        for audio_file in event_audio_files:
            try:
                audio, sr = sf.read(audio_file)
                if sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                # Assume click is centered, create 0.2s window
                file_id = audio_file.stem
                positive_sample = builder._extract_centered_window(
                    audio, 
                    peak_idx=len(audio)//2  # Assume center
                )
                
                all_positive_samples.append({
                    'waveform': positive_sample,
                    'label': 1,
                    'file_id': file_id
                })
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
    
    # Method 2: If events.csv exists with full audio
    events_csv = events_dir / 'all_events.csv'
    if events_csv.exists() and not event_audio_files:
        logger.info(f"Building from events CSV: {events_csv}")
        import pandas as pd
        events_df = pd.read_csv(events_csv)
        
        # Group by source file
        for source_file in events_df['source_file'].unique():
            try:
                audio, sr = sf.read(source_file)
                if sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                file_events = events_df[events_df['source_file'] == source_file]
                file_id = Path(source_file).stem
                
                for _, event in file_events.iterrows():
                    peak_idx = int(event['peak_idx'])
                    window = builder._extract_centered_window(audio, peak_idx)
                    
                    if window is not None:
                        all_positive_samples.append({
                            'waveform': window,
                            'label': 1,
                            'file_id': file_id
                        })
            except Exception as e:
                logger.error(f"Error processing {source_file}: {e}")
                continue
    
    logger.info(f"Collected {len(all_positive_samples)} positive samples")
    
    # Process negative samples (from noise)
    logger.info(f"Processing negative samples from {noise_dir}")
    noise_files = list(noise_dir.rglob('*.wav'))
    
    n_negative_per_file = max(1, len(all_positive_samples) // max(1, len(noise_files)))
    
    for noise_file in noise_files:
        try:
            audio, sr = sf.read(noise_file)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            file_id = noise_file.stem
            negative_samples = builder.build_negative_samples(
                audio, file_id, n_negative_per_file
            )
            all_negative_samples.extend(negative_samples)
            
        except Exception as e:
            logger.error(f"Error processing {noise_file}: {e}")
            continue
    
    logger.info(f"Collected {len(all_negative_samples)} negative samples")
    
    # Combine and balance
    all_samples = all_positive_samples + all_negative_samples
    
    if not all_samples:
        logger.error("No samples collected! Check input directories.")
        return
    
    balanced_samples = builder.balance_dataset(
        all_samples,
        balance_ratio=dataset_config.get('balance_ratio', 1.0)
    )
    
    logger.info(f"Total balanced samples: {len(balanced_samples)}")
    
    # Split into train/val
    val_split = dataset_config.get('val_split', 0.2)
    n_val = int(len(balanced_samples) * val_split)
    
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(balanced_samples)
    
    train_samples = balanced_samples[n_val:]
    val_samples = balanced_samples[:n_val]
    
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    
    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    builder.save_dataset(train_samples, output_dir, split='train')
    builder.save_dataset(val_samples, output_dir, split='val')
    
    logger.info("Dataset building completed")


def cmd_train(args):
    """Execute train command."""
    logger = ProjectLogger()
    config = load_config(args.config)
    
    logger.info("Starting model training")
    
    # Load dataset
    dataset_dir = Path(args.dataset_dir)
    builder = DatasetBuilder()
    
    train_waveforms, train_labels, _ = builder.load_dataset(dataset_dir / 'train')
    val_waveforms, val_labels, _ = builder.load_dataset(dataset_dir / 'val')
    
    logger.info(f"Training samples: {len(train_waveforms)}")
    logger.info(f"Validation samples: {len(val_waveforms)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_waveforms).float(),
            torch.from_numpy(train_labels).long()
        ),
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_waveforms).float(),
            torch.from_numpy(val_labels).long()
        ),
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    model = create_model(config['model'])
    
    # Calculate class weights
    unique, counts = np.unique(train_labels, return_counts=True)
    class_weights = torch.FloatTensor(len(counts) / counts)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=config['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        class_weights=class_weights
    )
    
    # Train
    output_dir = Path(args.output_dir)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        checkpoint_dir=output_dir
    )
    
    logger.info("Training completed")


def cmd_eval(args):
    """Execute eval command."""
    logger = ProjectLogger()
    
    logger.info("Evaluating model")
    
    # Load model
    inference = ClickDetectorInference.from_checkpoint(
        args.checkpoint,
        device='cpu',
        batch_size=32
    )
    
    # Load test dataset
    dataset_dir = Path(args.dataset_dir)
    builder = DatasetBuilder()
    test_waveforms, test_labels, _ = builder.load_dataset(dataset_dir)
    
    logger.info(f"Test samples: {len(test_waveforms)}")
    
    # Predict
    y_proba = inference.predict_batch(test_waveforms)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Generate report
    reporter = EvaluationReporter(Path(args.output_dir))
    
    # Convert to 2D probabilities
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])
    
    generated_files = reporter.generate_report(
        y_true=test_labels,
        y_pred=y_pred,
        y_proba=y_proba_2d,
        metadata={
            'checkpoint': args.checkpoint,
            'dataset': args.dataset_dir
        }
    )
    
    logger.info(f"Evaluation report generated at {args.output_dir}")


def cmd_export(args):
    """Execute export command."""
    logger = ProjectLogger()
    detection_config = load_config('configs/detection.yaml')
    inference_config = load_config(args.config)
    
    logger.info(f"Processing file: {args.input}")
    
    # Load audio
    audio, sr = sf.read(args.input)
    file_id = Path(args.input).stem
    
    # Step 1: Rule-based detection
    params = DetectionParams(
        tkeo_threshold=detection_config['thresholds']['tkeo_z'],
        ste_threshold=detection_config['thresholds']['ste_z'],
        hfc_threshold=detection_config['thresholds']['hfc_z'],
        high_low_ratio_threshold=detection_config['thresholds']['high_low_ratio'],
        envelope_width_min=detection_config['envelope']['width_min_ms'],
        envelope_width_max=detection_config['envelope']['width_max_ms'],
        spectral_centroid_min=detection_config['thresholds']['spectral_centroid_min'],
        refractory_ms=detection_config['refractory_ms']
    )
    
    detector = AdaptiveDetector(sample_rate=sr, params=params)
    candidates = detector.batch_detect(audio)
    
    logger.info(f"Rule-based detection: {len(candidates)} candidates")
    
    # Step 2: Extract 0.2s windows for model inference
    builder = DatasetBuilder(sample_rate=sr)
    windows = []
    
    for candidate in candidates:
        window = builder._extract_centered_window(audio, candidate.peak_idx)
        if window is not None:
            windows.append(window)
        else:
            windows.append(np.zeros(builder.window_samples))
    
    windows = np.array(windows)
    
    # Step 3: Model inference
    inference = ClickDetectorInference.from_checkpoint(
        args.checkpoint,
        device='cpu',
        batch_size=inference_config['batch_size']
    )
    
    model_scores = inference.predict_batch(windows)
    logger.info(f"Model inference completed")
    
    # Step 4: Fusion decision
    fusion_cfg = FusionConfig(
        high_confidence_threshold=inference_config['fusion']['high_confidence_threshold'],
        medium_confidence_threshold=inference_config['fusion']['medium_confidence_threshold'],
        train_consistency_required=inference_config['fusion']['train_consistency_required'],
        min_train_clicks=inference_config['fusion']['min_train_clicks'],
        max_ici_cv=inference_config['fusion']['max_ici_cv'],
        doublet_min_ici_ms=inference_config['doublet']['min_ici_ms'],
        doublet_max_ici_ms=inference_config['doublet']['max_ici_ms'],
        doublet_min_confidence=inference_config['doublet']['min_confidence']
    )
    
    decider = FusionDecider(config=fusion_cfg)
    accepted_indices, decision_info = decider.apply_fusion(candidates, model_scores)
    
    accepted_candidates = [candidates[i] for i in accepted_indices]
    
    logger.info(f"Fusion decision: {len(accepted_candidates)} accepted")
    logger.info(decider.get_statistics(decision_info))
    
    # Step 5: Build trains
    train_builder = TrainBuilder(
        min_ici_ms=detection_config['train']['min_ici_ms'],
        max_ici_ms=detection_config['train']['max_ici_ms'],
        min_train_clicks=detection_config['train']['min_train_clicks']
    )
    
    trains = train_builder.build_trains(accepted_candidates)
    logger.info(f"Built {len(trains)} click trains")
    
    # Step 6: Export results
    output_dir = Path(args.output_dir)
    exporter = ExportWriter(output_dir, sample_rate=sr)
    
    if inference_config['export']['export_events']:
        event_files = exporter.export_events(
            accepted_candidates,
            audio,
            file_id,
            export_audio=inference_config['export']['export_audio']
        )
        logger.info(f"Events exported to {event_files['csv']}")
    
    if inference_config['export']['export_trains']:
        train_files = exporter.export_trains(
            trains,
            accepted_candidates,
            audio,
            file_id,
            export_audio=inference_config['export']['export_audio']
        )
        logger.info(f"Trains exported to {train_files['csv']}")
    
    if inference_config['export']['create_summary']:
        summary_stats = {
            'total_candidates': len(candidates),
            'accepted_clicks': len(accepted_candidates),
            'rejection_rate': 1 - len(accepted_candidates) / len(candidates) if candidates else 0,
            'num_trains': len(trains),
            **decision_info
        }
        
        report_path = exporter.create_summary_report(file_id, summary_stats)
        logger.info(f"Summary report: {report_path}")
    
    logger.info("Export completed successfully")


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    commands = {
        'scan': cmd_scan,
        'detect': cmd_detect,
        'batch-detect': cmd_batch_detect,
        'trains': cmd_trains,
        'build-dataset': cmd_build_dataset,
        'train': cmd_train,
        'eval': cmd_eval,
        'export': cmd_export
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        logger = ProjectLogger()
        logger.error(f"Command failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()