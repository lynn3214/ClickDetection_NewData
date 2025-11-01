#!/usr/bin/env python3
"""检查测试流程所需的文件和功能"""
import sys
from pathlib import Path

def check_test_pipeline():
    """检查测试流程完整性"""
    
    print("=" * 70)
    print("测试流程完整性检查")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 1. 检查数据目录
    print("\n[1] 检查数据目录...")
    
    test_resampled = Path("data/test_resampled")
    if not test_resampled.exists():
        errors.append(f"测试数据目录不存在: {test_resampled}")
    else:
        wav_files = list(test_resampled.rglob("*.wav"))
        print(f"  ✅ 测试集目录存在")
        print(f"  - 找到 {len(wav_files)} 个wav文件")
        if len(wav_files) == 0:
            warnings.append("测试集目录为空")
    
    noise_test = Path("data/noise_test_segs")
    if not noise_test.exists():
        errors.append(f"测试噪音目录不存在: {noise_test}")
    else:
        noise_files = list(noise_test.glob("*.wav"))
        print(f"  ✅ 测试噪音目录存在")
        print(f"  - 找到 {len(noise_files)} 个噪音片段")
        if len(noise_files) == 0:
            warnings.append("测试噪音目录为空")
    
    # 2. 检查配置文件
    print("\n[2] 检查配置文件...")
    
    detection_config = Path("configs/detection_enhanced.yaml")
    if not detection_config.exists():
        errors.append(f"检测配置文件不存在: {detection_config}")
    else:
        print(f"  ✅ {detection_config}")
    
    eval_config = Path("configs/eval_wav.yaml")
    if not eval_config.exists():
        errors.append(f"评估配置文件不存在: {eval_config}")
    else:
        print(f"  ✅ {eval_config}")
        # 检查配置内容
        import yaml
        try:
            with open(eval_config) as f:
                config = yaml.safe_load(f)
            
            window_ms = config.get('windowing', {}).get('window_ms')
            if window_ms and window_ms != 500:
                warnings.append(
                    f"eval_wav.yaml中window_ms={window_ms}ms，"
                    f"与训练样本500ms不符"
                )
        except Exception as e:
            warnings.append(f"无法读取配置: {e}")
    
    # 3. 检查模型文件
    print("\n[3] 检查模型文件...")
    
    checkpoint = Path("checkpoints/best_model.pt")
    if not checkpoint.exists():
        errors.append(f"模型文件不存在: {checkpoint}")
        print(f"  ❌ {checkpoint}")
    else:
        print(f"  ✅ {checkpoint}")
        import torch
        try:
            ckpt = torch.load(checkpoint, map_location='cpu')
            model_config = ckpt.get('model_config', {})
            input_length = model_config.get('input_length')
            print(f"  - 模型输入长度: {input_length} 样本")
            if input_length != 22050:
                warnings.append(
                    f"模型输入长度为{input_length}，"
                    f"应该是22050 (500ms@44.1kHz)"
                )
        except Exception as e:
            warnings.append(f"无法读取模型: {e}")
    
    # 4. 检查关键脚本
    print("\n[4] 检查关键脚本...")
    
    main_py = Path("main.py")
    if not main_py.exists():
        errors.append("main.py 不存在")
    else:
        print(f"  ✅ main.py")
        
        # 检查是否有必要的命令
        with open(main_py) as f:
            content = f.read()
        
        if 'cmd_batch_detect' not in content:
            errors.append("main.py 缺少 batch-detect 命令")
        else:
            print("  ✅ batch-detect 命令存在")
        
        if 'cmd_eval_wav' not in content:
            warnings.append("main.py 可能缺少 eval-wav 命令")
        else:
            print("  ✅ eval-wav 命令存在")
    
    # 5. 检查缺失的脚本
    print("\n[5] 检查可选脚本...")
    
    extract_script = Path("scripts/extract_verified_segments.py")
    if not extract_script.exists():
        warnings.append(f"缺少脚本: {extract_script}")
        print(f"  ⚠️  {extract_script} (需要创建)")
    else:
        print(f"  ✅ {extract_script}")
    
    # 6. 检查eval_wav_files.py
    eval_wav_py = Path("eval_wav_files.py")
    if not eval_wav_py.exists():
        warnings.append(f"缺少脚本: {eval_wav_py}")
        print(f"  ⚠️  {eval_wav_py} (main.py可能会用到)")
    else:
        print(f"  ✅ {eval_wav_py}")
    
    # 输出总结
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)
    
    if errors:
        print("\n❌ 发现错误:")
        for err in errors:
            print(f"  - {err}")
    
    if warnings:
        print("\n⚠️  警告:")
        for warn in warnings:
            print(f"  - {warn}")
    
    if not errors and not warnings:
        print("\n✅ 所有检查通过!")
    
    # 给出建议
    print("\n" + "=" * 70)
    print("建议的测试流程")
    print("=" * 70)
    
    print("\n步骤1: 使用detector提取候选片段 (500ms)")
    print("  python main.py batch-detect \\")
    print("    --input-dir data/test_resampled \\")
    print("    --output-dir data/test_detection_results \\")
    print("    --config configs/detection_enhanced.yaml \\")
    print("    --save-audio \\")
    print("    --segment-ms 500 \\  # ✅ 改为500ms")
    print("    --recursive")
    
    print("\n步骤2: CNN模型初筛 (过滤低置信度)")
    print("  python scripts/filter_by_confidence.py \\")
    print("    --input data/test_detection_results/audio \\")
    print("    --checkpoint checkpoints/best_model.pt \\")
    print("    --output data/test_high_confidence \\")
    print("    --threshold 0.7")
    print("  ⚠️  需要创建此脚本")
    
    print("\n步骤3: 人工验证 (audacity)")
    print("  - 打开 data/test_high_confidence/ 下的wav文件")
    print("  - 标记为正样本/负样本")
    print("  - 保存到 data/test_labels_verified.txt")
    
    print("\n步骤4: 提取验证样本")
    print("  python scripts/extract_verified_segments.py \\")
    print("    --audio data/test_resampled \\")
    print("    --labels data/test_labels_verified.txt \\")
    print("    --output data/test_positive_segments")
    print("  ⚠️  需要创建此脚本")
    
    print("\n步骤5: 构建测试数据集")
    print("  python scripts/build_test_dataset.py \\")
    print("    --positive data/test_positive_segments \\")
    print("    --negative data/noise_test_segs \\")
    print("    --output data/test_dataset")
    print("  ⚠️  需要创建此脚本")
    
    print("\n步骤6: 评估模型")
    print("  python main.py eval \\")
    print("    --checkpoint checkpoints/best_model.pt \\")
    print("    --dataset-dir data/test_dataset \\")
    print("    --output-dir reports/test_results")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = check_test_pipeline()
    sys.exit(0 if success else 1)