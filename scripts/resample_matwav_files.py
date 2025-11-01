#!/usr/bin/env python3
"""
WAV文件重采样工具
功能：
专门用来处理mat转换成的wav文件，
将WAV文件重采样至44.1kHz并替换原文件
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from scipy.signal import resample_poly
import shutil
import tempfile

class AudioResampler:
    """音频重采样器"""
    
    def __init__(self, target_sr: int = 44100):
        self.target_sr = target_sr
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("AudioResampler")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def resample_file(self, input_path: Path) -> bool:
        """重采样单个文件并替换原文件"""
        try:
            # 读取音频
            audio, sr = sf.read(input_path)
            
            # 如果已经是目标采样率，跳过处理
            if sr == self.target_sr:
                self.logger.debug(f"{input_path.name} 已经是目标采样率，跳过处理")
                return True
                
            # 单声道处理
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 计算重采样参数
            gcd = np.gcd(self.target_sr, sr)
            up = self.target_sr // gcd
            down = sr // gcd
            
            # 重采样
            resampled = resample_poly(audio, up, down)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
            # 保存到临时文件
            sf.write(str(tmp_path), resampled, self.target_sr)
            
            # 替换原文件
            shutil.move(str(tmp_path), str(input_path))
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件 {input_path.name} 时出错: {str(e)}")
            # 确保清理临时文件
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)
            return False
    
    def process_directory(self, input_dir: Path) -> tuple[int, int]:
        """处理目录中的所有文件"""
        input_dir = Path(input_dir)
        
        # 获取所有wav文件
        wav_files = list(input_dir.glob("*.wav"))
        
        if not wav_files:
            self.logger.warning(f"在 {input_dir} 中未找到WAV文件！")
            return 0, 0
        
        self.logger.info(f"找到 {len(wav_files)} 个WAV文件")
        
        # 处理文件
        success_count = 0
        fail_count = 0
        
        for wav_file in tqdm(wav_files, desc="重采样进度"):
            if self.resample_file(wav_file):
                success_count += 1
            else:
                fail_count += 1
                
        return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='WAV文件重采样工具')
    parser.add_argument('--input-dir', type=str, 
                       default='data/training_resampled/mat_files',
                       help='输入目录（文件将被原地替换）')
    parser.add_argument('--target-sr', type=int, default=44100,
                       help='目标采样率（默认44100Hz）')
    
    args = parser.parse_args()
    
    # 创建重采样器
    resampler = AudioResampler(target_sr=args.target_sr)
    
    # 处理文件
    success_count, fail_count = resampler.process_directory(
        Path(args.input_dir)
    )
    
    # 打印统计信息
    resampler.logger.info("\n" + "="*60)
    resampler.logger.info("重采样完成")
    resampler.logger.info("="*60)
    resampler.logger.info(f"成功: {success_count}")
    resampler.logger.info(f"失败: {fail_count}")
    
    if success_count > 0:
        resampler.logger.info(f"\n✅ 已完成文件重采样，文件已更新在原目录: {args.input_dir}")

if __name__ == '__main__':
    main()