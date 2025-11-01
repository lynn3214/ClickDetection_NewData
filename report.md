# 步骤1.1: 组织原始数据（分离训练/测试集）
python prepare_data.py \
  --raw_dir data/raw \
  --output_dir data \
  --verbose

# 预期结果：
# ├── data/raw/training_sources/  (高SNR训练数据)
# ├── data/test_raw/              (低SNR测试数据)
# └── data/raw/noise/             (保持原位)

# 步骤2.1: 处理训练数据(不处理mat_files，只处理wav文件)
python preprocessing/resample_and_filter.py \
  --input data/raw/training_sources \
  --output data/training_resampled \
  --sr_target 44100 \
  --hp_cutoff 1000 \
  --verbose

# 步骤2.2: 转换mat文件，并重采样
python scripts/convert_mat_to_wav.py 
# --input-dir data/raw/training_sources/mat_files \
# --output-dir data/training_resampled/mat_files
# 在convert_mat_to_wav.py中已经设置了default路径，不需要手动设置。

# 用resample_matwav_files.py重采样到44.1kHz，替换原本的文件保存在data/training_resampled/mat_files目录下
python scripts/resample_matwav_files.py --input-dir data/training_resampled/mat_files


# 步骤2.3: 处理测试数据
python preprocessing/resample_and_filter.py \
  --input data/test_raw \
  --output data/test_resampled \
  --verbose

# 步骤2.4: 处理噪音数据
python preprocessing/resample_and_filter.py \
  --input data/raw/noise \
  --output data/noise_resampled \
  --verbose

# 步骤3.1: 从训练数据中检测并提取click片段
python main.py batch-detect \
  --input-dir data/training_resampled \
  --output-dir data/detection_results \
  --config configs/detection_enhanced.yaml \
  --save-audio \
  --segment-ms 120 \
  --recursive


# 预期结果：
# data/detection_results/
# ├── all_events.csv           (所有检测到的click统计)
# └── audio/                   (提取的click片段，按文件分组)
#     ├── file1/
#     │   ├── click_00000_12345ms.wav
#     │   └── ...
#     └── file2/


# 步骤3.2: 收集Click片段 
python scripts/collect_clicks.py \
  --input data/detection_results/audio \
  --output data/augmented_clicks \
  --verbose

# 步骤3.3: 检查提取出的click质量，可视化低质量的click片段，可以选择将低质量的片段move到其他位置，只使用高质量的片段进行后续步骤
python scripts/check_click_quality.py \  --input-dir data/detection_results/audio \       
  --quality-threshold 0.4 \                     
  --visualize 10 \                  
  --action report  

# 步骤4.1: 切割噪音并划分训练/测试集
python preprocessing/segment_noise.py \
  --input data/noise_resampled \
  --output-train data/noise_train_segs \
  --output-test data/noise_test_segs \
  --segment-ms 500 \
  --train-ratio 0.8 \
  --verbose

# ========== C4: 构建训练数据集（SNR混合） ==========
python main.py build-dataset \
  --events-dir data/augmented_clicks \
  --noise-dir data/noise_train_segs \
  --output-dir data/training_dataset \
  --config configs/training.yaml \
  --save-wav \
  --verbose

从波形图可以看到：
持续背景噪音 
整个500ms片段都有可见的背景噪音
click之间的区域不再是静音
SNR [-5,15]dB
问题1：
噪音看起来像"窄条带"
但不是削波/截幅了，因为放大能看见随机的峰值变化，而不是被切平了

# 问题2：是否有必要在单个样本内实现SNR随机 （噪音稀疏程度变化+单个样本内SNR随机，而不是固定值）
## 噪音稀疏程度变化
方案A：添加噪音调制（Amplitude Modulation）
# 使噪音密集度随时间变化
modulation = np.random.uniform(0.8, 1.2, len(background_noise))
background_noise_modulated = background_noise * modulation
增加复杂度，可能干扰CNN学习（引入额外变量）

方案B：混合多个噪音源
python# 叠加不同噪音（船只、波浪、生物）
noise1 = random.choice(noise_pool)
noise2 = random.choice(noise_pool)
background_noise = 0.7 * noise1 + 0.3 * noise2
评估：
增加多样性
更真实（海洋是多源噪音）
实现较复杂

## 单个样本内SNR随机变化
方案：
for i in range(4):  # 分成4段
    segment_start = i * (len(train_audio) // 4)
    segment_end = (i + 1) * (len(train_audio) // 4)
    
    # 每段独立SNR
    snr = random.uniform(-5, 15)
    noise_scale = calculate_noise_scale(snr)
    
    train_audio[segment_start:segment_end] += noise_scale * background_noise[segment_start:segment_end]
  
破坏了SNR定义的一致性
可能让模型难以学习（信号统计特性不稳定）

# ========== D1: 检查截幅问题 ==========
python scripts/verify_snr_mix.py \
  --debug-dir data/training_dataset/debug_wavs \
  --num-samples 10

# ========== D2: 可视化检查 ==========
python check_snr_mix_result.py \
  --dataset-dir data/training_dataset/train \
  --num 10

# ========== D3: 手动检查（推荐） ==========
# 用Audacity打开几个 debug_wavs/pos_*.wav 文件
# 检查：
# - 波形是否在 [-1, 1] 范围内
# - 是否有明显截幅（波形顶部被削平）
# - 频谱是否自然（无突变、无谐波失真）

# ========== E1: 训练 ==========
python main.py train \
  --dataset-dir data/training_dataset \
  --output-dir models/checkpoints \
  --config configs/training.yaml \
  --verbose

如果可以，我计划使用这个命令开始训练：

python main.py train \   --dataset-dir data/training_dataset \   --output-dir models/checkpoints \   --config configs/training.yaml \   --verbose

我的训练模型代码如下，main.py和training.yaml跟提供给你的版本一致，没有进行修改。请你帮我检查是否与前面构成训练集的逻辑一致，是否可以正常运行。如果有需要修改的地方请指出并给出修改意见。
# 监控训练：
tail -f logs/training/*.log  # 如果有日志文件


# ========== E2: 评估（验证集） ==========
python main.py eval \
  --checkpoint models/checkpoints/best_model.pt \
  --dataset-dir data/training_dataset/val \
  --output-dir reports/eval_results \
  --verbose

# 查看结果：
cat reports/eval_results/evaluation_report/metrics.txt

# ========== F1: 准备测试集正样本（半自动标注） ==========
# F1.1 用detector粗筛
python main.py batch-detect \
  --input-dir data/test_resampled \
  --output-dir data/test_detection_results \
  --config configs/detection_enhanced.yaml \
  --save-audio \
  --segment-ms 120 \
  --recursive \
  --verbose

# F1.2 导出Audacity标签（需要创建此脚本）
python scripts/export_for_audacity.py \
  --input data/test_detection_results/all_events.csv \
  --output data/test_labels.txt

# F1.3 人工验证
# - 在Audacity中打开测试音频
# - 导入标签文件
# - 逐个听检并删除误检
# - 导出验证后的标签为 test_labels_verified.txt

# F1.4 提取验证后的片段（需要创建此脚本）
python scripts/extract_verified_segments.py \
  --audio data/test_resampled \
  --labels data/test_labels_verified.txt \
  --output data/test_positive_segments


# ========== F2: 文件级评估 ==========
python main.py eval-wav \
  --checkpoint models/checkpoints/best_model.pt \
  --positive-dir data/test_resampled \
  --negative-dir data/noise_test_segs \
  --output-dir reports/eval_wav_results \
  --config configs/eval_wav.yaml \
  --verbose

# 查看结果：
cat reports/eval_wav_results/metrics.txt


# 邮件中的问题： 叠加的噪音似乎被截幅了
问题分析：
# 1. 多个click叠加（振幅可能很大）
train_audio = self._place_clicks_with_realistic_ici(...)  # 多个click直接相加！

# 2. SNR混合
if noise_pool is not None and augmenter is not None:
    if random.random() < augmenter.apply_prob:
        noise = random.choice(noise_pool)
        train_audio = augmenter.snr_mix(train_audio, noise)  # ⚠️ 问题在这里
        

多个click叠加后，train_audio 的功率比单个click大得多（可能是2-5倍）
snr_mix 使用以下公式计算噪音缩放因子：

pythonsignal_power = np.mean(train_audio**2)  # ⚠️ 这个功率很大！
noise_power = np.mean(noise_segment**2)

snr_linear = 10**(target_snr / 10)  # 例如 SNR=0dB -> snr_linear=1
noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

mixed = signal + noise_scale * noise_segment  # ⚠️ noise_scale可能很大

如果 signal_power 是单click的5倍，那么 noise_scale 也会大约5倍
导致噪音电平过高，最终峰值归一化后，click反而变小了


真实高噪音场景：
- 噪音是持续的、均匀的背景噪音
- 即使SNR=-5dB，click仍然可见，只是被噪音部分掩盖
- 噪音电平在整个片段中是一致的

当前错误场景：
- 噪音只出现在click附近（见你的观察2）
- 噪音电平不稳定，有的片段极高，有的正常
- 这不符合物理规律

问题根源
看 _place_clicks_with_realistic_ici 函数：
pythondef _place_clicks_with_realistic_ici(self, clicks, train_samples_total, ...):
    train_audio = np.zeros(train_samples_total, dtype=np.float32)  # ⚠️ 初始化为0
    
    # ...
    
    for i, click in enumerate(clicks):
        # 放置click（叠加）
        train_audio[current_pos:end_pos] += click[:end_pos-current_pos]  # ⚠️ 只在click位置有信号
    
    return train_audio
问题所在：

train_audio 初始化为全0
只在click的位置叠加了信号
click之间的区域仍然是0（或接近0）

然后在 snr_mix 中：
python# train_audio: [0, 0, 0, ..., click1, 0, 0, ..., click2, 0, 0, ...]
# noise: [n, n, n, n, n, n, n, n, ...]

# 计算信号功率
signal_power = np.mean(train_audio**2)  # ⚠️ 被大量的0拉低了

# 计算噪音缩放
noise_scale = sqrt(signal_power / (snr_linear * noise_power))  # ⚠️ 很小

# 混合
mixed = train_audio + noise_scale * noise  # ⚠️ 噪音被严重衰减
```

**结果**：
- 噪音在**整个片段**中都被添加了，但缩放因子很小
- click部分：`click + tiny_noise` → 看起来像有噪音
- 空白部分：`0 + tiny_noise` → 几乎看不到噪音
- 峰值归一化后，click被放大到0.95，噪音也被同步放大，但**相对比例不变**

## 🎯 正确的实现思路

真实海洋环境中的click train应该是：
```
[background_noise + click1 + background_noise + click2 + background_noise + ...]
```

而不是当前的：
```
[0 + click1 + 0 + click2 + 0 + ...]  然后叠加微弱噪音
解决方案
方案A：先添加背景噪音，再叠加clicks（推荐）
这是最符合物理规律的方法：
pythondef _place_clicks_with_realistic_ici(self, clicks, train_samples_total, ...):
    # 🔧 改变策略：先创建背景噪音，再叠加clicks
    
    # 1. 初始化为背景噪音（而不是0）
    # 注意：这里不添加噪音，在外部SNR混合时添加
    train_audio = np.zeros(train_samples_total, dtype=np.float32)
    
    # 2. 叠加clicks
    for i, click in enumerate(clicks):
        train_audio[current_pos:end_pos] += click[:end_pos-current_pos]
    
    return train_audio
然后修改 build_click_train_samples：
python# 放置clicks
train_audio = self._place_clicks_with_realistic_ici(...)

if train_audio is None:
    continue

# 🔧 关键修改：先生成纯背景噪音作为基底
if noise_pool is not None and augmenter is not None:
    if random.random() < augmenter.apply_prob:
        noise = random.choice(noise_pool)
        
        # 提取等长噪音段
        if len(noise) > len(train_audio):
            start = random.randint(0, len(noise) - len(train_audio))
            background_noise = noise[start:start + len(train_audio)]
        else:
            repeats = int(np.ceil(len(train_audio) / len(noise)))
            background_noise = np.tile(noise, repeats)[:len(train_audio)]
        
        # 计算噪音缩放（基于click train的功率）
        signal_power = np.mean(train_audio**2)
        noise_power = np.mean(background_noise**2)
        
        # 随机SNR
        target_snr = random.uniform(*augmenter.snr_range)
        snr_linear = 10**(target_snr / 10)
        
        if noise_power > 0:
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        else:
            noise_scale = 0
        
        # 直接叠加背景噪音
        train_audio = train_audio + noise_scale * background_noise
    else:
        # 不添加噪音（纯净样本）
        pass
else:
    # 如果没有噪音池，保持原样
    pass

# 最终峰值归一化
peak = np.max(np.abs(train_audio))
if peak > 0:
    train_audio = train_audio / peak * 0.95


# 🔧 完全重写SNR混合逻辑：模拟📊 预期效果对比
修改前（当前）
波形特征：
- Click区域：有一些噪音纹理
- Click之间：几乎完全安静（0附近）
- 不符合真实海洋环境
```

### 修改后（预期）
```
波形特征：
- 整个片段：持续的背景噪音（振幅取决于SNR）
- Click区域：click脉冲叠加在噪音上
- Click之间：可见的持续噪音
- 符合真实海洋环境（类似你的原始录音）

修改后，生成的click train样本应该是：
```
时间轴: [0ms -------- 250ms -------- 500ms]
波形:   [噪音+click1 + 噪音 + click2 + 噪音]
        ^^^^^^^^^^^^   ^^^^^   ^^^^^^^   ^^^^^
        持续的背景噪音贯穿整个片段
对比
特征修改前修改后Click区域click + 微弱噪音click + 明显噪音Click之间几乎静音持续噪音（与click区域相同水平）噪音分布不均匀均匀持续真实性❌ 不符合✅ 符合海洋环境



## 实验结果

### 测试集设计
本研究采用两阶段检测架构：
1. **Stage 1: Detector（规则过滤）**
   - 使用TKEO、包络分析、高频比等特征
   - 从测试音频中提取click候选片段
   
2. **Stage 2: CNN分类器（精细分类）**
   - 对detector输出的候选进行二分类
   - 区分真实click和残留噪音

### 测试集构成
- **正样本**: 3358个片段，来自X个独立测试文件（总时长Y秒）
  - 提取方式：Detector筛选后的候选
  - **注意**: 代表"detector能检测到的click"，不代表所有真实click
  
- **负样本**: 3990个片段，来自独立海洋噪音录音
  - 随机分割500ms片段

### CNN分类性能（在detector输出上）
```
准确率:  97.84%
精确率:  98.25%
召回率:  96.99%
F1分数:  97.62%
ROC AUC: 99.05%

混淆矩阵:
  TN: 3932   FP: 58
  FN: 101    TP: 3257
```

### 性能解读
- CNN在detector筛选的候选中，能以98.3%的精确率识别真实click
- 对噪音的拒识率为98.5%（FPR=1.5%）
- **系统整体性能取决于detector的召回率**（待评估）

### 局限性
1. 测试集正样本由detector提取，存在**选择偏差**
2. 未评估detector漏检情况（低SNR、重叠click）
3. 未测试极端困难样本（如SNR<-5dB）