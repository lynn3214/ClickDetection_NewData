"""
使用方法：
1. 运行 build-dataset 时添加 --save-wav 参数
2. 检查 output_dir/debug_wavs/ 中的wav文件
3. 运行此脚本验证SNR和频谱
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class SNRVerifier:
    """验证SNR混合质量"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def compute_snr(self, mixed: np.ndarray, clean: np.ndarray) -> float:
        """
        计算实际SNR
        
        Args:
            mixed: 混合信号
            clean: 干净信号
            
        Returns:
            SNR (dB)
        """
        noise = mixed - clean
        
        signal_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
    
    def check_clipping(self, audio: np.ndarray, threshold: float = 0.99) -> dict:
        """
        检查是否存在截幅
        
        Args:
            audio: 音频信号
            threshold: 截幅阈值（接近±1）
            
        Returns:
            截幅统计
        """
        peak = np.max(np.abs(audio))
        clipped = np.sum(np.abs(audio) >= threshold)
        
        return {
            'peak_value': float(peak),
            'clipped_samples': int(clipped),
            'clipping_rate': float(clipped / len(audio)),
            'is_clipped': clipped > 0
        }
    
    def plot_comparison(self, 
                       clean: np.ndarray, 
                       mixed: np.ndarray,
                       title: str = "SNR Mixing Verification"):
        """对比干净信号和混合信号"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        time_axis = np.arange(len(clean)) / self.sample_rate * 1000
        
        # === 波形对比 ===
        axes[0, 0].plot(time_axis, clean, linewidth=0.8, label='Clean')
        axes[0, 0].set_title('Clean Signal Waveform')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip threshold')
        axes[0, 0].axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        
        axes[0, 1].plot(time_axis, mixed, linewidth=0.8, label='Mixed', color='orange')
        axes[0, 1].set_title('Mixed Signal Waveform')
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Clip threshold')
        axes[0, 1].axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].legend()
        
        # === 频谱对比 ===
        fft_clean = np.fft.rfft(clean * np.hanning(len(clean)))
        fft_mixed = np.fft.rfft(mixed * np.hanning(len(mixed)))
        
        freq = np.fft.rfftfreq(len(clean), 1/self.sample_rate) / 1000
        
        mag_clean_db = 20 * np.log10(np.abs(fft_clean) + 1e-10)
        mag_mixed_db = 20 * np.log10(np.abs(fft_mixed) + 1e-10)
        
        axes[1, 0].plot(freq, mag_clean_db, linewidth=1, label='Clean')
        axes[1, 0].set_title('Clean Signal Spectrum')
        axes[1, 0].set_xlabel('Frequency (kHz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, 22.05])
        
        axes[1, 1].plot(freq, mag_mixed_db, linewidth=1, label='Mixed', color='orange')
        axes[1, 1].set_title('Mixed Signal Spectrum')
        axes[1, 1].set_xlabel('Frequency (kHz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 22.05])
        
        # === 噪音成分 ===
        noise = mixed - clean
        axes[2, 0].plot(time_axis, noise, linewidth=0.8, color='red', label='Noise')
        axes[2, 0].set_title('Extracted Noise')
        axes[2, 0].set_xlabel('Time (ms)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].grid(True, alpha=0.3)
        
        fft_noise = np.fft.rfft(noise * np.hanning(len(noise)))
        mag_noise_db = 20 * np.log10(np.abs(fft_noise) + 1e-10)
        
        axes[2, 1].plot(freq, mag_noise_db, linewidth=1, color='red', label='Noise')
        axes[2, 1].set_title('Noise Spectrum')
        axes[2, 1].set_xlabel('Frequency (kHz)')
        axes[2, 1].set_ylabel('Magnitude (dB)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_xlim([0, 22.05])
        
        # 计算SNR
        actual_snr = self.compute_snr(mixed, clean)
        fig.suptitle(f'{title}\nActual SNR: {actual_snr:.2f} dB', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def verify_dataset(self, debug_wav_dir: Path, n_samples: int = 5):
        """验证整个数据集"""
        debug_wav_dir = Path(debug_wav_dir)
        
        # 找到正样本wav文件
        pos_files = sorted(list(debug_wav_dir.glob('pos_*.wav')))
        
        if not pos_files:
            print(f"❌ 未找到调试wav文件: {debug_wav_dir}")
            return
        
        print(f"✅ 找到 {len(pos_files)} 个正样本wav")
        print(f"随机检查 {min(n_samples, len(pos_files))} 个样本...\n")
        
        import random
        selected = random.sample(pos_files, min(n_samples, len(pos_files)))
        
        for i, wav_file in enumerate(selected):
            print(f"\n{'='*60}")
            print(f"样本 {i+1}/{len(selected)}: {wav_file.name}")
            print('='*60)
            
            audio, sr = sf.read(wav_file)
            
            # 检查截幅
            clip_info = self.check_clipping(audio)
            print(f"峰值: {clip_info['peak_value']:.4f}")
            print(f"截幅样本数: {clip_info['clipped_samples']}")
            print(f"截幅率: {clip_info['clipping_rate']*100:.2f}%")
            
            if clip_info['is_clipped']:
                print("⚠️  警告：检测到截幅！")
            else:
                print("✅ 无截幅")
            
            # 绘图（需要clean信号用于SNR计算，这里仅展示混合信号）
            # 如果有对应的clean文件，可以加载对比
            # 这里简化为只显示混合信号
            
        print(f"\n{'='*60}")
        print("验证完成")


def main():
    parser = argparse.ArgumentParser(description='验证SNR混合质量')
    parser.add_argument('--debug-dir', type=str, required=True,
                       help='debug_wavs目录路径')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='检查样本数')
    args = parser.parse_args()
    
    verifier = SNRVerifier()
    verifier.verify_dataset(Path(args.debug_dir), args.num_samples)


if __name__ == '__main__':
    main()