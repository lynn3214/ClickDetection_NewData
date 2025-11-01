import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import soundfile as sf
import h5py
from tqdm import tqdm

def _read_mat_v73_h5(file_path: Path, dataset_key: str = 'newFiltDat', fs_key: str = 'fs') -> tuple[np.ndarray, int]:
    """使用 h5py 读取 v7.3 mat 文件."""
    with h5py.File(file_path, 'r') as f:
        # 1. 读取数据
        if dataset_key not in f:
            raise KeyError(f'Key "{dataset_key}" not found in {file_path}')
        dset = f[dataset_key]
        data = dset[()]
        
        # 2. 读取采样率
        if fs_key in f:
            sr = int(f[fs_key][()])
        elif fs_key in dset.attrs:
            sr = int(dset.attrs[fs_key])
        else:
            raise KeyError(f'Cannot find sampling rate "{fs_key}" in file')
    return np.asarray(data), sr

def convert_mat_to_wav(mat_path: Path, output_path: Path, channel: int = 10) -> None:
    """转换单个 .mat 文件到 .wav 格式."""
    try:
        # 尝试读取旧格式 mat 文件
        try:
            mat = sio.loadmat(mat_path, squeeze_me=True)
            data = mat['newFiltDat']
            sr = int(mat['fs'])
        except NotImplementedError:
            # v7.3 格式需要使用 h5py
            data, sr = _read_mat_v73_h5(mat_path)
            
        # 处理数据维度
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T
            
        # 选择通道
        if data.ndim == 2:
            ch_idx = min(channel, data.shape[1] - 1)
            audio = data[:, ch_idx]
        else:
            audio = data
            
        # 保存为 wav
        sf.write(str(output_path), audio.astype(np.float32), sr)
        print(f"转换成功: {mat_path.name} -> {output_path.name}")
        
    except Exception as e:
        print(f"转换失败 {mat_path.name}: {str(e)}")

def main():
    # 设置输入输出路径
    input_dir = Path("data/raw/training_sources/mat_files")
    output_dir = Path("data/training_resampled/mat_files")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 .mat 文件
    mat_files = list(input_dir.glob("*.mat"))
    
    if not mat_files:
        print("未找到 .mat 文件！")
        return
        
    print(f"找到 {len(mat_files)} 个 .mat 文件")
    
    # 转换每个文件
    for mat_file in tqdm(mat_files, desc="转换进度"):
        output_path = output_dir / (mat_file.stem + ".wav")
        convert_mat_to_wav(mat_file, output_path)
    
    print("\n转换完成!")

if __name__ == "__main__":
    main()