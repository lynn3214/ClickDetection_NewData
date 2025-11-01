#!/usr/bin/env python3
"""训练前完整性检查脚本"""
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from utils.config import load_config
from training.dataset.segments import DatasetBuilder
from models.cnn1d.model import create_model, count_parameters

def check_training_readiness():
    """检查是否准备好进行训练"""
    
    print("=" * 70)
    print("训练前检查")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 1. 检查配置文件
    print("\n[1/6] 检查配置文件...")
    try:
        config = load_config('configs/training.yaml')
        print(f"  ✅ 配置文件加载成功")
        print(f"  - 样本率: {config['sample_rate']}")
        print(f"  - 模型输入长度: {config['model']['input_length']}")
        print(f"  - 批次大小: {config['training']['batch_size']}")
        print(f"  - 学习率: {config['training']['learning_rate']}")
    except Exception as e:
        errors.append(f"配置文件加载失败: {e}")
        return False
    
    # 2. 检查数据集
    print("\n[2/6] 检查数据集...")
    dataset_dir = Path('data/training_dataset')
    
    if not dataset_dir.exists():
        errors.append(f"数据集目录不存在: {dataset_dir}")
        return False
    
    try:
        builder = DatasetBuilder()
        
        # 加载训练集
        train_dir = dataset_dir / 'train'
        if not train_dir.exists():
            errors.append(f"训练集目录不存在: {train_dir}")
            return False
        
        train_waveforms = np.load(train_dir / 'waveforms.npy')
        train_labels = np.load(train_dir / 'labels.npy')
        
        print(f"  ✅ 训练集加载成功")
        print(f"  - 样本数: {len(train_waveforms)}")
        print(f"  - 样本形状: {train_waveforms.shape}")
        print(f"  - 标签分布: {np.bincount(train_labels)}")
        
        # 加载验证集
        val_dir = dataset_dir / 'val'
        if not val_dir.exists():
            errors.append(f"验证集目录不存在: {val_dir}")
            return False
        
        val_waveforms = np.load(val_dir / 'waveforms.npy')
        val_labels = np.load(val_dir / 'labels.npy')
        
        print(f"  ✅ 验证集加载成功")
        print(f"  - 样本数: {len(val_waveforms)}")
        print(f"  - 样本形状: {val_waveforms.shape}")
        
        # 检查数据形状匹配
        expected_length = config['model']['input_length']
        if train_waveforms.shape[1] != expected_length:
            errors.append(
                f"训练集样本长度({train_waveforms.shape[1]}) "
                f"与配置不符({expected_length})"
            )
        
        if val_waveforms.shape[1] != expected_length:
            errors.append(
                f"验证集样本长度({val_waveforms.shape[1]}) "
                f"与配置不符({expected_length})"
            )
        
        # 检查数据范围
        train_min, train_max = train_waveforms.min(), train_waveforms.max()
        if train_max > 10 or train_min < -10:
            warnings.append(
                f"训练集幅度范围异常: [{train_min:.2f}, {train_max:.2f}]"
            )
        
    except Exception as e:
        errors.append(f"数据集加载失败: {e}")
        return False
    
    # 3. 检查模型创建
    print("\n[3/6] 检查模型...")
    try:
        model = create_model(config['model'])
        n_params = count_parameters(model)
        
        print(f"  ✅ 模型创建成功")
        print(f"  - 模型类型: {'Lightweight' if config['model'].get('use_lightweight', True) else 'Full'}")
        print(f"  - 参数数量: {n_params:,}")
        print(f"  - 输入长度: {config['model']['input_length']}")
        print(f"  - 输出类别: {config['model']['num_classes']}")
        
        # 测试前向传播
        test_input = torch.randn(2, config['model']['input_length'])
        with torch.no_grad():
            test_output = model(test_input)
        
        if test_output.shape != (2, config['model']['num_classes']):
            errors.append(
                f"模型输出形状错误: {test_output.shape}, "
                f"期望: (2, {config['model']['num_classes']})"
            )
        else:
            print(f"  ✅ 模型前向传播测试通过")
        
    except Exception as e:
        errors.append(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 检查设备
    print("\n[4/6] 检查计算设备...")
    device = config.get('device', 'cpu')
    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            warnings.append("配置为CUDA但CUDA不可用，将使用CPU")
            config['device'] = 'cpu'
    else:
        print(f"  ℹ️  使用CPU训练")
    
    # 5. 检查输出目录
    print("\n[5/6] 检查输出目录...")
    output_dir = Path('checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ 检查点保存目录: {output_dir}")
    
    # 6. 预估训练时间
    print("\n[6/6] 训练时间预估...")
    batch_size = config['training']['batch_size']
    n_epochs = config['training']['num_epochs']
    n_train = len(train_waveforms)
    n_batches = n_train // batch_size
    
    print(f"  - 每轮批次数: {n_batches}")
    print(f"  - 最大训练轮数: {n_epochs}")
    print(f"  - 早停耐心: {config['training']['early_stopping_patience']}")
    
    if device == 'cpu':
        est_time_per_epoch = n_batches * 0.5  # 假设每批次0.5秒
        est_total_hours = est_time_per_epoch * n_epochs / 3600
        print(f"  - 预估每轮时间: {est_time_per_epoch:.1f}秒")
        print(f"  - 预估总时间: {est_total_hours:.1f}小时 (不考虑早停)")
    
    # 输出总结
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)
    
    if errors:
        print("\n❌ 发现错误:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    if warnings:
        print("\n⚠️  警告:")
        for warn in warnings:
            print(f"  - {warn}")
    
    print("\n✅ 所有检查通过，可以开始训练!")
    print("\n推荐训练命令:")
    print("  python main.py train \\")
    print("    --dataset-dir data/training_dataset \\")
    print("    --output-dir checkpoints \\")
    print("    --config configs/training.yaml")
    
    return True

if __name__ == "__main__":
    success = check_training_readiness()
    sys.exit(0 if success else 1)