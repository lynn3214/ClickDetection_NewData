"""
Simplified: Quick Visualization Tool for Click Samples
Features:
- Read waveforms.npy and labels.npy from a given path
- Randomly sample 10 positive samples (label = 1)
- Plot the waveform and spectrum for each sample
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random

class ClickSampleViewer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def load_data(self, dataset_dir: Path):
        """Load the dataset"""
        dataset_dir = Path(dataset_dir)
        waveforms = np.load(dataset_dir / 'waveforms.npy')
        labels = np.load(dataset_dir / 'labels.npy')
        print(f"✅ Data loaded successfully: {len(waveforms)} samples")
        print(f"Number of positive samples: {np.sum(labels == 1)}, Number of negative samples: {np.sum(labels == 0)}")
        return waveforms, labels

    def plot_waveform_and_spectrum(self, waveform: np.ndarray, sample_idx: int):
        """Plot the waveform and spectrum of a single sample"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # === Waveform ===
        time_axis = np.arange(len(waveform)) / self.sample_rate * 1000
        axes[0].plot(time_axis, waveform, color='steelblue', linewidth=0.8)
        axes[0].set_title(f'Click Sample #{sample_idx} - Waveform', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # === Spectrum ===
        fft = np.fft.rfft(waveform * np.hanning(len(waveform)))
        magnitude = np.abs(fft)
        freq = np.fft.rfftfreq(len(waveform), 1/self.sample_rate)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        axes[1].plot(freq / 1000, magnitude_db, color='darkgreen', linewidth=1)
        axes[1].set_title('Spectrum (FFT)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Frequency (kHz)')
        axes[1].set_ylabel('Magnitude (dB)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, self.sample_rate / 2000])

        plt.tight_layout()
        plt.show()

    def visualize_positive_samples(self, waveforms: np.ndarray, labels: np.ndarray, n_samples: int = 10):
        """Randomly sample and display positive samples"""
        pos_indices = np.where(labels == 1)[0]
        if len(pos_indices) == 0:
            print("❌ No positive samples found!")
            return

        selected = random.sample(list(pos_indices), min(n_samples, len(pos_indices)))
        print(f"🎯 Randomly selecting {len(selected)} positive samples for visualization...")

        for i, idx in enumerate(selected):
            self.plot_waveform_and_spectrum(waveforms[idx], idx)


def main():
    parser = argparse.ArgumentParser(description='Quick Visualization Tool for Click Samples')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Directory containing waveforms.npy and labels.npy')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate (default: 44100)')
    parser.add_argument('--num', type=int, default=10, help='Number of samples to visualize (default: 10)')
    args = parser.parse_args()

    viewer = ClickSampleViewer(sample_rate=args.sample_rate)
    waveforms, labels = viewer.load_data(Path(args.dataset_dir))
    viewer.visualize_positive_samples(waveforms, labels, n_samples=args.num)


if __name__ == '__main__':
    main()