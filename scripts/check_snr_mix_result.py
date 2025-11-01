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
        """加载数据集并解析元数据"""
        dataset_dir = Path(dataset_dir)
        waveforms = np.load(dataset_dir / 'waveforms.npy')
        labels = np.load(dataset_dir / 'labels.npy')
        
        # 加载元数据以区分样本类型
        import json
        with open(dataset_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ 成功加载数据: {len(waveforms)} 个样本")
        print(f"正样本数: {np.sum(labels == 1)}, 负样本数: {np.sum(labels == 0)}")
        
        return waveforms, labels, metadata

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

    def visualize_samples_by_type(self, waveforms: np.ndarray, labels: np.ndarray, 
                              metadata: list, n_samples: int = 5):
        """分别可视化单个click和click序列"""
        # 获取所有正样本索引
        pos_indices = np.where(labels == 1)[0]
        
        # 提取 file_ids
        file_ids = [entry['file_id'] for entry in metadata]  # 从列表中提取 file_id
        
        # 区分单个click和序列
        single_clicks = []
        click_trains = []
        
        for idx in pos_indices:
            file_id = file_ids[idx]  # 正确的索引方式
            if 'train_' in file_id:
                click_trains.append(idx)
            else:
                single_clicks.append(idx)
                    
        print(f"\n找到:")
        print(f"- 单个click: {len(single_clicks)} 个")
        print(f"- Click序列: {len(click_trains)} 个")
        
        # 可视化单个click
        if single_clicks:
            print("\n🎯 随机抽样单个click:")
            selected = random.sample(single_clicks, min(n_samples, len(single_clicks)))
            for idx in selected:
                self.plot_waveform_and_spectrum(waveforms[idx], f"Single Click #{idx}")
                    
        # 可视化click序列
        if click_trains:
            print("\n🎯 随机抽样click序列:")
            selected = random.sample(click_trains, min(n_samples, len(click_trains)))
            for idx in selected:
                self.plot_waveform_and_spectrum(waveforms[idx], f"Click Train #{idx}")


def main():
    parser = argparse.ArgumentParser(description='Click样本可视化工具')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='包含waveforms.npy和labels.npy的目录')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='采样率 (默认: 44100)')
    parser.add_argument('--num', type=int, default=5,
                       help='每种类型要可视化的样本数 (默认: 5)')
    args = parser.parse_args()

    viewer = ClickSampleViewer(sample_rate=args.sample_rate)
    waveforms, labels, metadata = viewer.load_data(Path(args.dataset_dir))
    viewer.visualize_samples_by_type(waveforms, labels, metadata, n_samples=args.num)


if __name__ == '__main__':
    main()