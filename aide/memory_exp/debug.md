
## DEBUG 阶段性经验总结
- 🐛 **高频错误模式 (Frequent Error Patterns)**：
  **1. 文件路径构建错误（最高频，出现 4 次）**
- 错误模式：使用 `filename.replace(".wav", ".bmp")` 构建谱图路径，当源文件名不含 `.wav` 后缀时替换失效，导致路径缺少 `.bmp` 扩展名
- 典型报错：`FileNotFoundError: './input/supplemental_data/filtered_spectrograms/PC2_20090513_050000_0010'`（路径末尾无扩展名）

**2. CSV 文件 Header 解析错误（出现 3 次）**
- 错误模式：直接对 CSV 文件逐行读取并转换 `int(parts[0])`，未跳过或验证 header 行（"rec_id"）
- 典型报错：`ValueError: invalid literal for int() with base 10: 'rec_id'`

**3. 验证指标 NaN/模型不学习（出现 3 次）**
- 错误模式：多标签分类使用 `StratifiedKFold` 导致验证集某些类别无正样本，`roc_auc_score` 返回 NaN；或谱图文件缺失时使用占位灰图训练
- 典型报错：`Val AUC: nan` 或 `Val AUC=0.5000`（所有 fold 相同，模型未学习）

**4. 库/环境变量不匹配（出现 2 次）**
- 错误模式：使用特定预训练标签（如 `efficientnet_b3.ra_in1k`）在当前 `timm` 版本不可用；或变量未实例化直接使用
- 典型报错：`RuntimeError: Invalid pretrained tag` 或 `NameError: name 'skf' is not defined`

**5. 提交文件 KeyError（出现 1 次）**
- 错误模式：测试预测字典的 key 类型（int/str）与遍历时的 rec_id 类型不匹配
- 典型报错：`KeyError: 1`
- 🛠️ **标准修复方案 (Standard Fixes)**：
  **1. 文件路径构建 - 治本方案**
```python
# 错误做法（治标）：
img_path = os.path.join(SPEC_DIR, filename.replace(".wav", ".bmp"))

# 正确做法（治本）：
base_name = os.path.splitext(filename)[0]  # 移除任意扩展名
img_path = os.path.join(SPEC_DIR, base_name + ".bmp")
# 并添加文件存在性检查与回退机制
if not os.path.exists(img_path):
    img_path = os.path.join(FALLBACK_SPEC_DIR, base_name + ".bmp")
if not os.path.exists(img_path):
    # 动态生成谱图或记录警告
```

**2. CSV Header 处理 - 治本方案**
```python
# 错误做法（治标）：
next(f)  # 假设第一行是 header

# 正确做法（治本）：
for line in f:
    parts = line.strip().split(",")
    if len(parts) < 2 or not parts[0].isdigit():  # 验证首字段是否为数字
        continue
    rec_id = int(parts[0])
```

**3. 多标签验证指标 NaN - 治本方案**
```python
# 错误做法（治标）：
try:
    val_auc = roc_auc_score(labels, preds, average="macro")
except ValueError:
    val_auc = 0.5

# 正确做法（治本）：
# (1) 使用 MultilabelStratifiedKFold 替代 StratifiedKFold
# (2) 在计算前检查类别分布
if all_labels.sum() == 0 or all_labels.sum() == len(all_labels) * NUM_CLASSES:
    val_auc = 0.5  # 单类别情况
else:
    val_auc = roc_auc_score(all_labels, all_preds, average="macro")
# (3) 处理 NaN 返回值
val_auc = 0.5 if np.isnan(val_auc) else val_auc
```

**4. 库版本兼容性 - 治本方案**
```python
# 错误做法：
model = timm.create_model("efficientnet_b3.ra_in1k", pretrained=True)

# 正确做法：
model = timm.create_model("efficientnet_b3", pretrained=True)  # 使用基础变体
# 或先验证可用变体：available_models = timm.list_models()
```

**5. 字典 Key 类型一致性 - 治本方案**
```python
# 确保 key 类型统一
for i, rid in enumerate(rec_ids):
    fold_test_preds[int(rid)] = probs[i]  # 统一转换为 int

# 提交时同样转换
for rid in test_rec_ids:
    probs = final_test_preds[int(rid)]
```
- 🛡️ **防御性编程规范 (Defensive Coding Rules)**：
  **1. 文件路径操作规范**
- 禁止使用 `str.replace()` 处理文件扩展名，必须使用 `os.path.splitext()`
- 所有文件读取前必须添加 `os.path.exists()` 检查
- 关键数据文件缺失时必须有回退机制（fallback）或明确报错（而非静默使用占位数据）

**2. CSV/文本文件解析规范**
- 禁止假设文件无 header，必须验证首行内容
- 禁止直接访问 `parts[n]`，必须先检查 `len(parts) >= n+1`
- 禁止直接 `int()` 转换，必须先验证 `str.isdigit()` 或使用 try-except 捕获

**3. 机器学习指标计算规范**
- 多标签分类必须使用 `MultilabelStratifiedKFold`，禁止使用 `StratifiedKFold`
- `roc_auc_score` 等指标计算前必须检查标签分布（是否存在单类别情况）
- 必须处理 NaN 返回值：`metric = 0.5 if np.isnan(metric) else metric`
- 禁止将不可比指标直接相减（如 Train Loss - Val AUC）

**4. 第三方库使用规范**
- 使用预训练模型时必须验证模型变体在当前环境可用
- 优先使用基础模型名（如 `efficientnet_b3`），避免特定预训练标签（如 `.ra_in1k`）
- 关键对象（如 `skf = StratifiedKFold(...)`）必须在使用前显式实例化

**5. 数据结构一致性规范**
- 字典 key 类型必须统一（全部 int 或全部 str），在存入和取出时显式转换
- 多标签预测的 rec_id 必须与提交文件 ID 格式一致（验证 `rec_id * 100 + species_id` 计算）

**6. Debug 记忆强制检查清单**
- 每次新代码执行前必须检查：文件路径构建、CSV header 处理、验证集类别分布、模型预训练标签兼容性
- 相同错误模式出现 2 次以上必须写入防御性规范，禁止后续 Agent 重复
----------------------------------------

## DEBUG 阶段性经验总结
- 🐛 **高频错误模式 (Frequent Error Patterns)**：
  1. **模型架构不匹配 (Model Architecture Mismatch)**: 保存 checkpoint 时使用一种架构，加载时使用不同架构，导致 `Unexpected key(s) in state_dict` 和 `size mismatch` 错误。高频出现在冗余的 save/reload 循环中。

2. **张量操作错误 (Tensor Operation Errors)**: 
   - 缺少 `.detach()` 直接调用 `.cpu().numpy()` 导致 `Can't call numpy() on Tensor that requires grad`
   - Attention Pooling 维度不匹配导致 `size of tensor a must match size of tensor b`

3. **数据加载解析错误 (Data Loading Parsing Errors)**:
   - CSV 文件头未验证直接 `int()` 转换导致 `ValueError: invalid literal for int()`
   - 字典中 None 值未处理导致 `TypeError: 'NoneType' object is not iterable`
   - 文件路径构建使用 `.replace()` 而非 `os.path.splitext()` 导致扩展名丢失

4. **多标签分类折叠策略错误 (Multi-label Stratification Error)**: 使用 `StratifiedKFold` 而非 `MultilabelStratifiedKFold`，导致验证集某些类别无正样本，AUC 恒为 0.5000（随机猜测）

5. **网络依赖模型加载失败 (Network-dependent Model Loading)**: `torch.hub.load()` 在受限网络环境下超时，导致模型初始化失败

6. **领域特定配置错误 (Domain-specific Configuration)**: 
   - 对频谱图使用 ImageNet 归一化参数（应使用频谱图专用统计量）
   - 缺失音频专用增强（SpecAugment、背景噪声注入）
- 🛠️ **标准修复方案 (Standard Fixes)**：
  1. **模型保存/加载标准化**:
   - 移除同一执行周期内的冗余 save/reload 循环，直接使用 `train_fold()` 返回的已加载最佳权重的 model 对象
   - 如必须模拟离线提交，确保 save 后立即 load 且架构完全一致
   - 添加 `weights_only=False` 参数兼容 PyTorch 2.4+

2. **张量操作规范**:
   - 所有从计算图衍生的预测张量，在 `.cpu().numpy()` 前必须调用 `.detach()`
   - Attention Pooling 实现前验证输入维度：`features.shape[1]` 必须 > 1 才有意义

3. **数据加载防御**:
   - CSV 解析时使用 `parts[0].isdigit()` 验证后再 `int()` 转换
   - 字典取值使用 `labels_dict.get(rid) or [0]` 处理 None 值
   - 文件路径使用 `os.path.splitext(filename)[0] + '.bmp'` 替代 `.replace()`

4. **多标签分类正确实践**:
   - 必须使用 `MultilabelStratifiedKFold`（从 iterative-stratification 包）
   - 回退方案：若包不可用，使用普通 `KFold` 而非 `StratifiedKFold`
   - `roc_auc_score` 计算时添加 `average='micro'` 或逐类别计算后过滤 NaN

5. **模型加载去网络化**:
   - 优先使用 `timm.create_model()` 等本地缓存模型
   - 避免 `torch.hub.load()` 等运行时网络依赖
   - 预下载权重到本地缓存目录

6. **领域适配配置**:
   - 频谱图归一化使用 `[0.5, 0.5, 0.5]` 或数据集统计量，禁用 ImageNet 参数
   - 音频任务添加 SpecAugment（Time/Freq Masking）和背景噪声注入
   - 10 秒固定长度片段无需滑动窗口，但建议使用 Attention Pooling 处理非平稳信号
- 🛡️ **防御性编程规范 (Defensive Coding Rules)**：
  1. **【模型一致性检查】**: 在 `load_state_dict()` 前添加断言验证架构匹配：
   ```python
   model_state = model.state_dict()
   checkpoint = torch.load(path, weights_only=False)
   assert set(model_state.keys()) == set(checkpoint.keys()), "Architecture mismatch detected"
   ```

2. **【张量分离强制】**: 创建工具函数强制 detach：
   ```python
   def tensor_to_numpy(t):
       return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()
   ```

3. **【CSV 解析安全】**: 统一使用验证函数：
   ```python
   def safe_int(s):
       return int(s) if s.isdigit() else None
   ```

4. **【多标签折叠强制】**: 在 CV 初始化时添加类型检查：
   ```python
   if labels.ndim > 1:  # multi-label
       from iterative_stratification import MultilabelStratifiedKFold
       cv = MultilabelStratifiedKFold(n_splits=3)
   else:
       cv = StratifiedKFold(n_splits=3)
   ```

5. **【模型加载回退】**: 实现网络故障回退机制：
   ```python
   try:
       model = torch.hub.load(repo, model_name)
   except (URLError, HTTPError):
       model = timm.create_model(fallback_name, pretrained=True)
   ```

6. **【领域配置验证】**: 在 Dataset 初始化时验证归一化参数：
   ```python
   if input_type == 'spectrogram':
       assert mean != [0.485, 0.456, 0.406], "ImageNet stats invalid for spectrograms"
   ```

7. **【None 值防御】**: 所有字典取值操作添加默认值：
   ```python
   value = data_dict.get(key, default_value) or default_value
   ```

8. **【AUC 计算健壮性】**: 处理类别缺失情况：
   ```python
   try:
       auc = roc_auc_score(y_true, y_pred, average='micro')
   except ValueError:  # only one class present
       auc = 0.5
   ```
----------------------------------------

## DEBUG 阶段性经验总结
- 🐛 **高频错误模式 (Frequent Error Patterns)**：
  1. **张量梯度追踪错误**：Tensor requiring grad直接调用.cpu().numpy()导致RuntimeError，出现3次（节点2、3、4均涉及）

2. **多标签分类验证指标破坏**：使用StratifiedKFold+argmax()处理多标签数据，导致验证集某些类别无正样本，roc_auc_score返回NaN或0.5000，出现2次（节点1、7）

3. **频域/时域维度混淆**：生成粉红噪声时未先将时域信号转换到频域就直接除以频率bins，导致张量维度不匹配(1024 vs 513)，出现1次（节点5）

4. **文件路径处理不可靠**：使用str.replace()处理文件扩展名，当文件名无.wav后缀时导致路径错误，出现2次（节点1、6）

5. **预训练模型标签不兼容**：使用特定pretrained tag（如convnext_large.in12k_ft_in1k）在当前timm版本不可用，出现1次（节点10）

6. **提交格式不匹配**：输出列名/行数与sample_submission.csv不一致，导致提交被拒，出现2次（节点1、5）

7. **NaN值未妥善处理**：roc_auc_score在单类验证集返回NaN时未检查，导致最终分数为0.0000，出现2次（节点6、9）
- 🛠️ **标准修复方案 (Standard Fixes)**：
  1. **张量detach规范**：所有预测张量在.cpu().numpy()前必须调用.detach()，标准修复链路：outputs.detach().cpu().numpy()

2. **多标签交叉验证**：使用KFold替代StratifiedKFold，或实现MultilabelStratifiedKFold；验证指标计算采用per-class AUC并跳过单类情况：if len(np.unique(y_true)) < 2: continue

3. **频域信号处理**：粉红噪声生成标准流程：white_noise → torch.fft.rfft() → 除以sqrt(freqs) → torch.fft.irfft()，确保频域操作在频域进行

4. **文件路径安全处理**：使用os.path.splitext(filename)[0] + ".bmp"替代filename.replace(".wav", ".bmp")，并添加文件存在性检查和fallback机制

5. **预训练模型兼容**：使用基础模型名称（如"convnext_base"）而非特定pretrained tag，添加模型加载fallback机制

6. **提交格式校验**：严格对照sample_submission.csv的列名和行数，添加assert检查：assert len(submission) == expected_rows

7. **NaN防御处理**：AUC计算函数封装NaN检查，best_auc初始化避免NaN比较：if val_auc is not None and val_auc > best_auc
- 🛡️ **防御性编程规范 (Defensive Coding Rules)**：
  1. **张量操作铁律**：任何从计算图提取的Tensor在调用.numpy()、.item()或传递给numpy/scikit-learn函数前，必须显式调用.detach()

2. **多标签任务规范**：
   - 禁止对多标签向量使用argmax()进行折划分
   - 验证指标必须逐类别计算并过滤单类情况
   - 使用try-except包裹roc_auc_score并检查返回值

3. **信号处理规范**：
   - 频域操作（如1/f缩放）必须在FFT变换后进行
   - 始终检查张量形状匹配：assert tensor_a.shape[-1] == tensor_b.shape[-1]

4. **文件路径规范**：
   - 禁止使用str.replace()处理文件扩展名
   - 统一使用os.path.splitext()
   - 所有文件读取操作前添加os.path.exists()检查

5. **模型加载规范**：
   - 预训练模型使用基础名称，避免版本特定tag
   - 添加模型加载try-except和fallback机制
   - 检查预训练权重维度与当前模型匹配

6. **提交完整性规范**：
   - 生成提交文件后立即校验行数和列名
   - 添加assert len(df) == expected_count和assert list(df.columns) == expected_columns

7. **数值稳定性规范**：
   - 所有指标计算函数返回前检查NaN/Inf
   - 最佳模型选择逻辑处理NaN情况：if val_auc is not None and not np.isnan(val_auc)

8. **GPU内存管理规范**：
   - 每个epoch结束后调用torch.cuda.empty_cache()
   - DataLoader设置num_workers=0, pin_memory=False避免共享内存问题
   - 大张量操作后显式删除引用：del tensor; torch.cuda.empty_cache()
----------------------------------------

## DEBUG 阶段性经验总结
- 🐛 **高频错误模式 (Frequent Error Patterns)**：
  1. **维度不匹配错误 (Dimension Mismatch)**：频域操作中最常见，FFT 变换后返回 N//2+1 个频点 bins，但代码试图在长度为 N 的时域张量上直接应用频域缩放。典型错误：`torch.fft.rfftfreq(1024)` 返回 513 个 bins，却与 1024 长度的张量运算。

2. **模型完全不学习 (Val AUC=0.5000)**：出现频率最高的致命错误 (5/10 节点)。根本原因包括：(a) 数据文件加载失败导致训练零填充静音数据；(b) 注意力机制使用 sigmoid 而非 softmax 导致权重分布错误；(c) 预训练权重未正确加载；(d) 归一化参数域不匹配（ImageNet 统计量用于频谱图）。

3. **字典键类型不一致 (Key Type Mismatch)**：tensor 类型与 int 类型混用导致 KeyError。典型场景：DataLoader 返回 tensor 类型的 rec_id，但预测字典使用 int 类型键。

4. **设备不一致 (Device Mismatch)**：CPU/GPU 张量混用，缺少 `.detach()` 导致计算图累积或 RuntimeError。

5. **提交格式错误 (Submission Format)**：列数/列名与 sample_submission.csv 不匹配，导致 Kaggle 提交被拒。
- 🛠️ **标准修复方案 (Standard Fixes)**：
  1. **频域操作修复链路**：
   - 确认操作对象是频域张量还是时域张量
   - 频谱图增强时，1/f 缩放应直接沿频率维度 (dim=1) 应用，形状广播为 (1, -1, 1) 而非 (1, 1, -1)
   - 避免对已计算的频谱图重复进行 FFT 变换

2. **数据加载修复链路**：
   - 优先使用预计算特征（BMP 频谱图）而非原始 WAV 文件
   - 添加文件存在性检查：`if not os.path.exists(path): raise FileNotFoundError(...)`
   - 提供 fallback 目录机制：filtered_spectrograms → spectrograms

3. **类型统一修复链路**：
   - 字典键在存入时统一转换：`key = int(tensor_item.item()) if isinstance(tensor_item, torch.Tensor) else int(tensor_item)`
   - 在预测循环中标准化：`rid_int = int(rid.item()) if isinstance(rid, torch.Tensor) else int(rid)`

4. **模型学习修复链路**：
   - 验证预训练权重加载：添加 `print(f"Pretrained weights loaded: {state_dict is not None}")`
   - 注意力机制使用 softmax：`F.softmax(attention_scores / temperature, dim=-1)`
   - 归一化参数匹配数据域：频谱图使用 [0.5, 0.5, 0.5] 而非 ImageNet 统计量

5. **提交格式验证链路**：
   - 生成前读取 sample_submission.csv 验证列名：`expected_cols = pd.read_csv(sample_path).columns.tolist()`
   - 行数断言：`assert len(submission) == n_test_samples * n_classes`
- 🛡️ **防御性编程规范 (Defensive Coding Rules)**：
  1. **频域操作黄金法则**：
   ```python
   # 禁止：在时域张量上直接应用频域缩放
   # 错误：white / torch.sqrt(freqs).view(1, 1, -1)  # freqs 长度 513, white 长度 1024
   
   # 正确：确认维度匹配后再运算
   assert white.shape[-1] == freqs.shape[0], f"Dim mismatch: {white.shape[-1]} vs {freqs.shape[0]}"
   ```

2. **数据加载防御性检查**：
   ```python
   # 禁止：静默 fallback 到零填充
   # 错误：waveform = np.zeros(MAX_AUDIO_LEN) if file not found
   
   # 正确：显式报错或跳过
   if not os.path.exists(wav_path):
       raise FileNotFoundError(f"WAV file not found: {wav_path}")
   # 或记录并跳过该样本
   ```

3. **字典键类型规范**：
   ```python
   # 禁止：混用 tensor 和 int 作为键
   # 错误：preds[rid] = prob  # rid 可能是 tensor(1) 或 int(1)
   
   # 正确：统一转换
   def normalize_key(key):
       if isinstance(key, torch.Tensor):
           return int(key.item())
       return int(key)
   ```

4. **设备一致性检查**：
   ```python
   # 禁止：跨设备张量运算
   # 错误：loss = criterion(output, labels)  # output 在 GPU, labels 在 CPU
   
   # 正确：统一设备
   labels = labels.to(output.device)
   # 或转换前 detach
   probs = output.detach().cpu().numpy()
   ```

5. **提交格式前置验证**：
   ```python
   # 禁止：生成后才发现格式错误
   # 正确：生成前验证
   sample = pd.read_csv('sample_submission.csv')
   assert submission.columns.tolist() == sample.columns.tolist()
   assert len(submission) == expected_rows
   ```

6. **模型学习状态监控**：
   ```python
   # 每个 epoch 检查学习信号
   if epoch > 5 and val_auc < 0.55:
       print(f"WARNING: Model not learning after {epoch} epochs")
       # 检查梯度范数
       grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       print(f"Gradient norm: {grad_norm}")
   ```
----------------------------------------

## DEBUG 阶段性经验总结
- 🐛 **高频错误模式 (Frequent Error Patterns)**：
  **1. 张量操作类型错误（最高频，出现 5+ 次）**
- `.copy()` vs `.clone()`：在 PyTorch Tensor 上错误调用 NumPy 的.copy() 方法，应使用.clone()
- `.detach()` 缺失：从计算图提取 Tensor 调用.cpu().numpy()/.item() 前未显式.detach()
- 维度不匹配：FFT 频域维度混淆（如 rfftfreq(1024) 返回 513 bins 却与 1024 长度张量广播）
- 设备不一致：增强操作在 CPU 执行但模型在 GPU，导致后续运算设备冲突

**2. 数值稳定性问题（出现 3+ 次）**
- FocalLoss 实现导致 NaN：`torch.exp(-ce_loss)` 在 ce_loss 较大时数值下溢
- 梯度爆炸：微调大型 Transformer 时缺少梯度裁剪（clip_grad_norm_）
- 预测值含 NaN：未在前向传播后验证输出，直接传入 roc_auc_score 导致崩溃

**3. 域适配错误（出现 4+ 次）**
- ImageNet 归一化用于频谱图：[0.485,0.456,0.406] 应改为 [0.5,0.5,0.5]
- 语音模型用于鸟鸣分类：Wav2Vec2（语音预训练）vs PANNs/AST（音频预训练）
- Pooling 与 backbone 不匹配：Attention Pooling 期望序列输入但 ConvNeXt 输出已全局池化

**4. 训练配置问题（出现 6+ 次）**
- Val AUC stuck at 0.5000：模型未学习（学习率过低 2e-5、epoch 不足 8、层冻结过度）
- Scheduler 配置错误：OneCycleLR 的 steps_per_epoch 错误除以 accumulation_steps
- Scheduler 步进时机：OneCycleLR 应每 batch 步进而非每 epoch

**5. 文件路径处理（出现 3+ 次）**
- 使用 str.replace() 处理扩展名：源文件名无扩展名时失败，应使用 os.path.splitext()
- 缺少文件存在性检查：关键数据文件缺失时无回退机制
- 🛠️ **标准修复方案 (Standard Fixes)**：
  **1. 张量操作修复链路**
```python
# 错误：spec = spec.copy()  # Tensor 无.copy() 方法
# 正确：spec = spec.clone()  # PyTorch Tensor 深拷贝

# 错误：numpy_array = tensor.cpu().numpy()  # 未.detach()
# 正确：numpy_array = tensor.detach().cpu().numpy()

# 错误：image_tensor = self.spec_augment(image_tensor)  # CPU 增强
#        image = image.to(DEVICE)  # 后移设备
# 正确：image = image.to(DEVICE)
#        image_tensor = self.spec_augment(image_tensor)  # GPU 增强
```

**2. 数值稳定性修复链路**
```python
# 错误：自定义 FocalLoss 使用 torch.exp(-ce_loss)
# 正确：criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# 错误：无梯度裁剪
# 正确：torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 错误：直接传入预测值
# 正确：assert not torch.isnan(predictions).any(), "Predictions contain NaN"
```

**3. 域适配修复链路**
```python
# 错误：transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# 正确：transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

# 错误：Wav2Vec2 用于鸟鸣分类（语音域）
# 正确：PANNs(CNN14) 或 AST（AudioSet 音频预训练）

# 错误：Attention Pooling 用于已全局池化的 ConvNeXt 输出
# 正确：移除 Attention Pooling，直接使用 backbone 输出接分类头
```

**4. 训练配置修复链路**
```python
# 错误：epochs=8, lr=2e-5, freeze 大部分层
# 正确：epochs=15-30, lr=5e-5, 仅冻结前 2-3 个 transformer 块

# 错误：steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
# 正确：steps_per_epoch = len(train_loader)

# 错误：scheduler.step() 在 epoch 级别调用
# 正确：scheduler.step() 在每 batch 迭代后调用（OneCycleLR 设计）
```

**5. 文件路径修复链路**
```python
# 错误：wav_path = path.replace('.bmp', '.wav')
# 正确：base, _ = os.path.splitext(path); wav_path = base + '.wav'

# 错误：直接 open(file_path)
# 正确：if os.path.exists(file_path): ... else: 使用备用路径
```

**成功链路验证**：节点 f09996837c034951ab21af4f5ee99100 实现 AUC=0.8476，采用 ConvNeXt + Focal Loss + Label Smoothing + TTA + 正确归一化 + 梯度累积 + 梯度检查点
- 🛡️ **防御性编程规范 (Defensive Coding Rules)**：
  **1. 张量操作铁律（必须遵守）**
- 任何从计算图提取的 Tensor 在调用.numpy()、.item() 或传递给 numpy/scikit-learn 函数前，必须显式调用.detach()
- PyTorch Tensor 深拷贝使用.clone()，禁止使用.copy()（NumPy 方法）
- 数据增强流水线必须在 tensor.to(DEVICE) 之后执行，确保所有操作在 GPU 进行
- 始终打印张量形状：`print(f"Shape: {tensor.shape}")` 在进行矩阵运算前

**2. 数值稳定性规范（必须遵守）**
- 优先使用 PyTorch 内置损失函数（BCEWithLogitsLoss、CrossEntropyLoss），避免自定义含 exp/log 的实现
- 微调大型 Transformer 时必须添加梯度裁剪：`clip_grad_norm_(model.parameters(), max_norm=1.0)`
- 在传入评估指标函数前验证：`assert not torch.isnan(predictions).any()`
- 启用 AMP 混合精度训练时使用 GradScaler 包装 optimizer.step()

**3. 域适配检查清单（必须验证）**
- 频谱图归一化：使用 [0.5,0.5,0.5] 而非 ImageNet 参数
- 模型预训练域：音频任务使用 AudioSet 预训练模型（PANNs/AST），非 ImageNet 或语音模型
- Pooling 层匹配：确认 backbone 输出形状与 pooling 层期望输入一致
- 时频参数：鸟鸣高频特征需 n_mels=128, hop_length=256-512, n_fft=1024-2048

**4. 训练配置验证（必须检查）**
- 学习率范围：微调 Transformer 使用 1e-5 到 5e-5，非 2e-5 以下
- Epoch 数量：至少 15-30 epoch，8 epoch 不足以收敛
- 层冻结策略：仅冻结前 2-3 个块，非冻结 90% 以上参数
- Scheduler 配置：OneCycleLR 的 steps_per_epoch = len(train_loader)，且每 batch 调用 step()

**5. 文件路径处理规范（必须遵守）**
- 扩展名处理：使用 os.path.splitext() 而非 str.replace()
- 文件读取前：必须添加 os.path.exists() 检查
- 关键数据缺失：实现备用路径或默认值回退机制
- 路径构建：使用 os.path.join() 而非字符串拼接

**6. 提交格式验证（必须断言）**
- 提交行数：assert len(submission) == n_test_samples * n_classes
- 提交 ID 格式：验证 Id = rec_id * 100 + species_id
- 概率范围：assert predictions.min() >= 0 and predictions.max() <= 1
- 概率多样性：assert predictions.max() - predictions.min() > 0.1（避免退化输出）
----------------------------------------
