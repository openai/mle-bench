
## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  **核心有效策略组合（AUC 0.7419→0.7551）：**
1. **ConvNeXt-Large + Focal Loss**：作为基础架构，比AST transformer（0.5000）和CRNN更稳定可靠
2. **Weight Decay 0.01**：从0.02降低至0.01有效缓解过拟合（节点627fa087对比8c35b121）
3. **Label Smoothing 0.1**：改善多标签分类校准，与focal loss协同工作
4. **OneCycleLR调度器**：比固定学习率收敛更快，最终AUC提升0.0132
5. **SpecAugment（时频掩码）**：在ConvNeXt基础上额外提升0.0066（节点595e266对比156963f5）
6. **显存优化组合**：梯度累积4步 + 梯度检查点 + AMP混合精度，确保大模型在20GB显存下可训练

**关键组合效益：**
- ConvNeXt + Focal Loss + Weight Decay 0.01 + Label Smoothing + OneCycleLR = 0.7551（最佳）
- SpecAugment仅在ConvNeXt稳定架构上有效，单独尝试新架构（AST/CRNN/PANNs）均失败
- ❌ **无效/负面策略 (Negative Constraints)**：
  **明确避坑指南：**
1. **AST Transformer架构**：节点aabf6e160显示AUC=0.5000（随机猜测），原因是架构实现错误（直接访问内部层而非使用ASTForAudioClassification的logits），且AST对显存要求过高
2. **PANNs ResNet38**：节点49e44d81因注意力机制张量维度不匹配而崩溃，实现复杂度高
3. **CRNN + LSTM**：节点ef584263因tensor.detach()缺失导致运行时错误，且训练效率低
4. **单纯调参无增强**：节点156963f5和b51521529仅调整weight decay/dropout但无数据增强，AUC从0.7419降至0.7161/0.7189
5. **BatchNorm1d + 小batch**：节点b1cd8770因batch size=1导致BatchNorm崩溃，需添加drop_last=True
6. **Generalization Gap计算错误**：多个节点混淆loss与1-AUC量纲，导致错误的underfitting/overfitting判断
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  **组合效益观察：**
1. **正则化组合**：Weight Decay 0.01 + Label Smoothing 0.1 + Focal Loss三者协同，单独使用任一效果有限
2. **数据增强依赖基础架构**：SpecAugment在ConvNeXt上有效（+0.0066），但在不稳定架构上无法验证
3. **学习率调度器+正则化**：OneCycleLR与降低weight decay配合，避免过早收敛到次优解
4. **显存优化链**：梯度累积4步 + 梯度检查点 + AMP必须同时启用，缺一可能导致OOM或训练不稳定
5. **提交格式陷阱**：多个节点因提交格式（2列vs3列，Id计算方式）导致合规性问题，与模型性能无关但影响最终得分
- 🎯 **下一步探索建议 (Next Steps)**：
  **下一步探索建议（优先级排序）：**
1. **AudioSet预训练模型正确集成**：使用PANNs-CNN14或AST的官方预训练权重（非随机初始化），注意正确调用forward()返回logits
2. **背景噪声增强**：添加pink noise和场地特定底噪注入，针对任务描述中的wind/rain挑战
3. **Attention Pooling替换全局池化**：处理10秒音频中鸟鸣位置可变的问题，提升时序特征聚合
4. **5-Fold Stratified CV**：当前3-Fold方差较大（0.67-0.79），5-Fold可提供更稳定评估
5. **类别平衡采样**：针对长尾物种分布，实施class-weighted sampling或focal loss参数调优
6. **Mel频谱参数调优**：若从原始WAV处理，尝试n_mels=256、hop_length=256以更好捕捉2-8kHz鸟鸣频率
7. **测试时增强（TTA）**：在推理阶段应用时频掩码多版本预测平均
----------------------------------------

## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  **已验证有效的核心策略组合：**

1. **损失函数与正则化组合**：Focal Loss + Label Smoothing (0.1) + Weight Decay (0.01-0.008) 是处理多标签鸟类分类类别不平衡的最佳组合，历史最佳AUC 0.7551由此产生。

2. **交叉验证策略**：对多标签数据必须使用简单KFold而非StratifiedKFold。使用`argmax(axis=1)`进行分层会导致2/3折的AUC卡在0.5000（随机猜测），因为多标签样本的单一标签无法代表真实分布。

3. **数据增强**：SpecAugment（时频遮蔽）对频谱图有效，但必须配合正确的dtype处理（显式float32），否则AMP autocast会因double类型冲突而崩溃。

4. **模型架构**：ConvNeXt-Large在预计算频谱图上表现稳定（AUC 0.71-0.75范围），优于ConvNeXt-Tiny。ImageNet预训练虽非音频专用，但在有限数据下比尝试加载AudioSet模型更可靠。

5. **提交格式**：必须严格遵循"Id=rec_id*100+species, Probability"或"rec_id, species, Probability"格式，格式错误会导致评估失败。
- ❌ **无效/负面策略 (Negative Constraints)**：
  **明确避坑指南（导致metric严重下降或执行失败）：**

1. **多标签分层采样陷阱**：严禁使用`StratifiedKFold(train_labels_array.argmax(axis=1))`。此操作将多标签退化为单标签，导致验证集类别分布极度不均衡，2/3折完全无法学习（AUC=0.5000）。

2. **网络依赖型模型加载**：避免使用`torch.hub.load()`或`transformers.AutoModel.from_pretrained()`直接下载预训练权重。在隔离环境中会导致9小时超时失败。必须实现本地权重fallback机制。

3. **数据类型隐式转换**：`np.array / 255.0`会将float32提升为float64，与AMP autocast的Half类型冲突。必须显式`.astype(np.float32)`并在训练循环中`images = images.float()`。

4. **架构维度硬编码**：在`__init__`中硬编码Linear层维度（如`nn.Linear(1792, 1792)`）会导致forward时维度不匹配。必须通过dummy forward pass动态计算或使用AdaptiveAvgPool2d固定输出。

5. **过度追求音频专用模型**：AST/PANNs理论上更优，但实现复杂度高（维度匹配、梯度检查点、离线权重），在当前约束下ConvNeXt+正确CV策略更稳定可靠。

6. **学习率调度器替换**：OneCycleLR替代CosineAnnealingLR在此任务上导致AUC从0.5959降至0.5252，不建议随意更换已验证的调度器。
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  **组合效益观察：**

1. **Focal Loss + Label Smoothing协同**：单独使用Focal Loss时AUC约0.59-0.66，但配合Label Smoothing 0.1后达到0.7551。Focal Loss处理难易样本权重，Label Smoothing防止过度自信，两者互补提升校准能力。

2. **SpecAugment + 正确CV策略**：SpecAugment单独使用时因CV bug导致效果被掩盖（0.5252），但修复KFold后配合SpecAugment达到0.7180。说明数据增强的效果依赖于正确的评估框架。

3. **Weight Decay调优边际效应**：从0.01调整到0.008仅带来-0.0168的轻微下降，说明在0.01附近存在局部最优，大幅调整收益有限，应优先优化其他组件。

4. **梯度累积+检查点+小batch**：batch_size=8 + accumulation_steps=4 + gradient_checkpointing的组合使ConvNeXt-Large能在20GB VRAM内训练，这是实现高性能的必要条件而非可选优化。

5. **音频增强缺失的瓶颈**：当前所有成功节点都缺少背景噪声注入（风雨声），这是任务描述明确提到的挑战。SpecAugment+背景噪声的组合可能是突破0.7551的关键。
- 🎯 **下一步探索建议 (Next Steps)**：
  **下一步探索建议（按优先级排序）：**

1. **背景噪声注入增强**：在SpecAugment基础上添加Pink Noise或真实环境底噪（风雨、昆虫声）混叠。这是任务描述明确提到的挑战，且当前所有成功尝试都缺失此增强。实现时注意噪声强度控制在-20dB至-10dB范围。

2. **注意力池化替换全局平均池化**：在ConvNeXt末端添加Attention Pooling层，使模型能关注有声片段、忽略静音区域。这对10秒固定长度音频中的非平稳鸟鸣信号尤为重要。

3. **5-Fold CV提升评估稳定性**：当前3-Fold CV方差较大（Fold间AUC差异0.02-0.28）。切换到5-Fold可获得更可靠的验证信号，但需相应减少epochs或增加gradient accumulation steps以控制总训练时间。

4. **离线预训练音频模型**：如果环境允许预下载PANNs CNN14权重到本地，可尝试AudioSet预训练模型。必须实现完整的offline fallback机制，并添加梯度检查点以控制VRAM。

5. **训练轮数增加**：当前15 epochs可能不足，分析显示存在underfitting（训练损失持续下降但验证AUC plateau）。尝试25-30 epochs配合early stopping（patience=5）。

6. **集成策略**：训练2-3个不同种子或不同Weight Decay的ConvNeXt-Large模型，对预测概率取平均。集成通常能带来+0.01-0.02的AUC提升。
----------------------------------------

## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  **核心有效组合（AUC 0.7501，接近历史最佳0.7551）**：
1. **Label Smoothing 0.1 + Focal Loss + OneCycleLR**：节点b1efec证明此三元组合是关键，单独移除任一组件都会导致性能下降（节点507f7降至0.6252）
2. **ConvNeXt-Large backbone**：相比AST Transformer更稳定，在频谱图分类任务上表现优异
3. **简单KFold（非StratifiedKFold）**：多标签数据场景下，避免argmax-based stratification导致的类别分布扭曲
4. **TTA（Test-Time Augmentation）**：节点07107c证明在推理阶段应用水平翻转+时间平移可提升约1.7% AUC，且不增加训练成本
5. **适度正则化**：dropout (0.3/0.2/0.2) + weight_decay=0.01 比过度降低正则化（节点544ce降至0.7491）效果更好
- ❌ **无效/负面策略 (Negative Constraints)**：
  **明确避坑指南**：
1. **MixUp augmentation**：节点1037f从0.7364降至0.6673（-6.9%），多标签场景下插值样本可能破坏物种标签的稀疏性
2. **Attention Pooling替换全局平均池化**：节点92820从0.7501降至0.7336（-1.7%），在145样本小数据集上引入额外参数导致欠拟合加剧
3. **过度降低正则化**：节点544ce将dropout从0.3→0.2、weight_decay从0.01→0.008，性能从0.7501降至0.7491，表明当前正则化强度已接近最优
4. **StratifiedKFold + argmax**：节点507f7等多次尝试证明在多标签数据上会导致AUC≈0.5的灾难性失败
5. **AST Transformer backbone**：节点798d3等尝试显示实现复杂度高、内存问题频发，不如ConvNeXt稳定
6. **提交格式错误**：节点e8352f、7669d等多次因Id列格式问题导致submission invalid，浪费迭代机会
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  **组合效益观察**：
1. **Label Smoothing + Focal Loss协同**：历史记忆显示Focal Loss单独使用AUC仅0.59-0.66，但与Label Smoothing 0.1结合后达到0.7551，两者在多标签校准上存在强协同
2. **OneCycleLR + 20 epochs**：节点b1efec证明此组合能充分利用训练周期，而CosineAnnealingWarmRestarts（节点e8352f之前）效果较差
3. **背景噪声注入的实现陷阱**：节点7669d、96641c两次尝试均因tensor维度不匹配bug失败，说明音频增强需要更谨慎的维度处理（unsqueeze/squeeze）
4. **TTA的低成本增益**：节点07107c在较低基线(0.6442)上仍能提升1.7%，表明TTA可与任何训练配置组合使用
5. **欠拟合信号识别**：多个节点（1037f、544ce、07107c）分析均指出训练loss持续下降但val AUC未饱和，建议增加epochs至25-30
- 🎯 **下一步探索建议 (Next Steps)**：
  **下一步探索建议**：
1. **音频专用预训练模型**：尝试PANNs-CNN14或AST（AudioSet预训练），历史分析指出可比ImageNet预训练提升5-10% AUC
2. **背景噪声注入（修复后）**：在确保tensor维度正确前提下，添加pink noise + 环境噪声（风/雨）混合，SNR控制在10-20dB
3. **增加训练轮数**：基于欠拟合信号，将epochs从20增至25-30，配合early stopping（patience=5）
4. **SpecAugment参数调优**：当前time_mask=40/freq_mask=20，可尝试更大mask比例或添加cutout
5. **5-Fold CV**：145训练样本下3-Fold方差较大（节点b1efec折间方差0.8025/0.7013/0.7465），5-Fold可提高评估稳定性
6. **集成策略**：ConvNeXt-Large + ResNet backbone集成，或不同augmentation配置的模型平均
----------------------------------------

## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  **核心有效策略（按收益排序）：**

1. **移除MixUp + 添加TTA**：最显著的收益组合（+0.1051 AUC，0.6673→0.7724）。MixUp在多标签鸟声分类中有害，TTA（水平翻转+固定时间平移25-30像素）在推理阶段提供稳定增益。

2. **增加训练容量**：epochs从20增至30配合early stopping（patience=5）带来+0.0589 AUC提升（0.7501→0.8090）。模型存在欠拟合信号，需要更多训练步数收敛。

3. **Global Average Pooling优于Attention Pooling**：在小数据集（145训练样本）上，标准池化比Attention Pooling稳定（+0.0429 AUC），避免过拟合风险。

4. **稳定架构组合**：ConvNeXt-Large + Focal Loss(γ=2.0) + Label Smoothing(0.1) + OneCycleLR + Gradient Accumulation(4步) 是已验证的基线，提供0.73-0.77 AUC的稳定表现。

5. **SpecAugment时频遮蔽**：TIME_MASK_PARAM=40, FREQ_MASK_PARAM=20的时频遮蔽增强对频谱图鲁棒性有效。
- ❌ **无效/负面策略 (Negative Constraints)**：
  **明确避坑指南（按危害程度排序）：**

1. **背景噪声增强（pink/white noise）严重有害**：添加SNR -15~-10dB的pink noise导致-0.1301 AUC暴跌（0.7724→0.6423）。尽管理论上应对环境噪声，但在此任务中破坏频谱图特征，绝对禁止。

2. **MixUp augmentation在多标签场景有害**：历史数据显示-6.9% AUC下降。鸟声多标签分类中，音频混合会混淆物种特征，必须移除。

3. **Attention Pooling在小数据集上不稳定**：历史尝试显示-1.7% AUC下降。145样本的训练集不足以学习attention权重，改用Global Average Pooling。

4. **TTA实现陷阱**：
   - 时间平移必须用固定值（25或30像素），随机值导致推理不一致
   - rec_id必须从tensor转换为int（rid.item()），否则KeyError
   - 验证DataLoader不能设drop_last=True，否则AUC计算样本缺失

5. **OneCycleLR调度器步频错误**：必须每batch调用scheduler.step()，每epoch调用会导致学习率曲线失效。

6. **标签平滑一致性**：训练用label smoothing=0.1，验证时必须用原始labels（smoothing=0.0）计算loss，否则train/val分布不匹配。
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  **组合效益观察：**

1. **TTA增益依赖稳定架构**：TTA单独使用效果有限，但与ConvNeXt-Large + Focal Loss + Label Smoothing + OneCycleLR组合时产生协同效应。TTA在已收敛模型上放大泛化能力。

2. **训练容量×正则化平衡**：增加epochs（20→30）只有在Focal Loss + Label Smoothing + Dropout(0.2-0.3)的正则化保护下才有效，否则过拟合。early stopping(patience=5)是关键安全网。

3. **梯度累积×大模型**：ConvNeXt-Large在20GB VRAM限制下必须配合Gradient Accumulation(4步) + Gradient Checkpointing + AMP混合精度，三者缺一不可。

4. **确定性TTA×可复现性**：TTA的所有变换（翻转、平移）必须确定性，随机变换会导致多次推理结果不一致，影响模型集成和调试。

5. **频谱图预处理×ImageNet预训练**：使用预计算频谱图（128×1024 BMP）绕过librosa参数调优，但牺牲了n_mels/hop_length的精细控制。若改用AudioSet预训练模型（PANNs/AST），需重新评估预处理策略。
- 🎯 **下一步探索建议 (Next Steps)**：
  **下一步探索建议（按优先级排序）：**

1. **音频专用预训练模型迁移**：当前ConvNeXt是ImageNet预训练，建议尝试PANNs-CNN14或AST（Audio Spectrogram Transformer）在AudioSet上的预训练权重，预期+3-5% AUC提升。

2. **原始音频特征工程**：从预计算频谱图切换到librosa/torchaudio动态生成mel-spectrogram，优化n_mels=128, hop_length=256, n_fft=512参数，针对鸟叫高频特性调整。

3. **分层K-Fold交叉验证**：当前3-fold CV存在类别分布不均（Fold方差0.12），改用Stratified K-Fold确保稀有物种在各折均匀分布。

4. **类别平衡权重**：针对长尾分布，在Focal Loss基础上添加per-class权重，提升稀有物种检测率。

5. **推理优化**：当前TTA仅3种变换，可探索添加频率遮蔽、轻微缩放等额外变换，但需验证收益/计算成本比。

6. **模型集成**：在单模型0.8090 AUC基础上，尝试3-5个不同种子/架构的模型集成，预期+1-2% AUC提升。

**禁止尝试**：背景噪声注入、MixUp、Attention Pooling、随机TTA变换。
----------------------------------------

## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  1. **5-Fold CV + 类别加权采样**：这是最成功的组合（0.8451 AUC），通过WeightedRandomSampler解决长尾物种分布问题，同时5折交叉验证显著降低折间方差（从3-Fold的0.69-0.83范围稳定到0.84-0.87范围）。2. **训练容量充足**：30-40 epochs配合早停（patience=5-7）是必要条件，减少epochs会导致欠拟合（0.67-0.70范围）。3. **简单TTA优于复杂TTA**：水平翻转+固定时间移位（28像素）的确定性TTA稳定有效，而Multi-Position Temporal Cropping（5位置）反而降低性能。4. **SpecAugment参数**：TIME_MASK=40, FREQ_MASK=20是历史验证值，但必须配合稳定训练配置单独添加会导致崩溃。5. **验证集完整性**：val_loader必须设置drop_last=False，否则排除样本导致AUC计算不准确。
- ❌ **无效/负面策略 (Negative Constraints)**：
  1. **避免复杂TTA变体**：Multi-Position Temporal Cropping（5个时间位置裁剪）导致AUC从0.7192降至0.6981，增加推理复杂度但无收益。2. **SpecAugment不可单独添加**：在欠稳定配置下单独添加SpecAugment导致AUC从0.679暴跌至0.5682，必须与充足训练容量和正确TTA配合使用。3. **避免微调dropout**：仅调整dropout（0.3→0.25）效果有限（0.809→0.785），架构微调不如数据策略改进。4. **提交格式严格校验**：多次因列名错误（Id vs rec_id,species,Probability）导致提交失败，必须在代码中添加断言验证。5. **避免不必要依赖**：torchlibrosa导入导致ModuleNotFoundError，使用CPU端librosa即可满足频谱图计算需求。
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  1. **5-Fold CV + 类别加权采样 + 简单TTA**：这是突破性组合，5-Fold解决评估稳定性，类别加权解决长尾分布，简单TTA提供推理时增益，三者协同达到0.8451 AUC。2. **SpecAugment + 充足epochs + 正确TTA**：SpecAugment单独使用有害，但与30+ epochs和简单TTA组合时达到0.76-0.78稳定范围。3. **梯度累积+混合精度+梯度检查点**：这三者组合使ConvNeXt-Large能在20GB VRAM配额下训练，是高性能模型的前提条件。4. **早停机制+验证集完整性**：早停（patience=5-7）防止过拟合，但必须配合drop_last=False确保验证AUC计算准确，两者缺一不可。
- 🎯 **下一步探索建议 (Next Steps)**：
  1. **音频专用模型迁移**：当前ConvNeXt（ImageNet预训练）已达性能瓶颈，建议迁移至PANNs（Cnn14）或AST（Audio Spectrogram Transformer）等AudioSet预训练模型，预期可提升5-10% AUC。2. **音频原生增强**：添加Background Noise Injection（Pink Noise + HJA场地风雨底噪）和MixUp，当前仅使用SpecAugment，音频特定增强缺失。3. **时频参数优化**：当前使用预计算频谱图（128×1024），建议直接用librosa控制n_mels=128-256、hop_length=256-512以优化鸟叫高频（2-8kHz）分辨率。4. **注意力池化**：模型末端用Attention Pooling替代全局平均池化，处理非平稳鸟鸣信号中的静音片段。5. **保持5-Fold CV + 类别加权**：这是当前最成功的评估和采样策略，后续实验必须保留此基础配置。
----------------------------------------

## IMPROVE 阶段性经验总结
- ✅ **有效优化策略 (Positive Guidance)**：
  **已验证有效的优化策略：**

1. **5-Fold CV + Class-Weighted Sampling (WeightedRandomSampler)**：节点6 (+0.0901) 和节点9 (+0.0703) 证明该组合能有效处理长尾物种分布，但需配合强正则化防止过拟合。

2. **ConvNeXt-Large + 强正则化组合**：节点6成功的关键是 ConvNeXt-Large + Label Smoothing(0.1) + AMP + Gradient Accumulation(4步) + 30 epochs，在小数据集上实现良好泛化。

3. **Global Average Pooling 优于 Attention Pooling**：节点5 (+0.0527) 证明在145训练样本的小数据集上，Attention Pooling导致性能下降(-1.7%)，Global Average Pooling更稳定。

4. **Focal Loss + Label Smoothing 双正则化**：节点7和节点9显示该组合能有效处理多标签类别不平衡，训练/验证损失差距控制在0.02-0.08范围内。

5. **TTA 需与其他策略组合使用**：节点1单独TTA导致性能下降(-0.0367)，但节点9中TTA与5-Fold CV+Class-Weighted Sampling组合时贡献正向收益。
- ❌ **无效/负面策略 (Negative Constraints)**：
  **明确避坑指南：**

1. **避免在小数据集上使用 Attention Pooling**：节点5历史数据显示Attention Pooling使AUC从0.7501降至0.7336(-1.7%)，Global Average Pooling更稳定。

2. **5-Fold CV 必须配合强正则化**：节点3仅使用5-Fold CV+Class-Weighted Sampling导致严重过拟合(AUC从0.6045暴跌至0.5528)，必须同时使用ConvNeXt-Large+Label Smoothing+Dropout。

3. **AST模型实现陷阱**：节点8和10两次尝试均因API不匹配失败——(a) AST需要`input_values`而非`pixel_values`参数；(b) AST期望128×1024 mel频谱图而非224×224 RGB图像；(c) 归一化参数(mean=-4.27, std=4.57)仅适用于log-mel值而非[0,1]图像像素。

4. **TTA Tensor维度陷阱**：节点2和节点7的TTA实现导致shape mismatch——测试时图像已含batch维度，不应再`unsqueeze(0)`；`apply_tta`应返回flatten后的1D数组以匹配预测缓冲区。

5. **避免单独依赖TTA**：节点1证明TTA单独使用无法提升性能，必须与5-Fold CV+Class-Weighted Sampling组合。

6. **ImageNet预训练模型用于频谱图的局限**：所有ConvNeXt节点使用ImageNet归一化(mean=[0.485,0.456,0.406])处理频谱图，非音频最优但可接受；AudioSet预训练模型(PANNs/AST)理论上更优但实现复杂度高。
- 🔗 **潜在的组合效益 (Synergy Observations)**：
  **组合效益观察：**

1. **5-Fold CV + Class-Weighted Sampling + ConvNeXt-Large + Label Smoothing**：节点6证明该四重组合产生显著协同效应(+0.0901)，单独使用5-Fold CV(节点3)会导致过拟合，但加入ConvNeXt-Large容量和Label Smoothing正则化后性能大幅提升。

2. **TTA + 5-Fold CV + Class-Weighted Sampling**：节点9显示TTA在该组合中贡献正向收益(+0.0703)，而节点1单独TTA导致性能下降，证明TTA需要稳定的基础模型和均衡采样才能发挥作用。

3. **Gradient Accumulation + Gradient Checkpointing + AMP**：节点6、9、10证明该三重组合使ConvNeXt-Large和AST等大模型能在共享H100(20GB VRAM)上训练，是高性能的必要条件而非可选优化。

4. **Focal Loss + Label Smoothing 双正则化**：节点7和9显示该组合比单独使用任一方法更能控制泛化差距，尤其适用于19物种多标签分类的长尾分布。

5. **3-Fold vs 5-Fold 的权衡**：节点7(3-Fold, 0.7289) vs 节点9(5-Fold, 0.7413)显示5-Fold在配合Class-Weighted Sampling时略优，但3-Fold更稳定且训练时间短40%。
- 🎯 **下一步探索建议 (Next Steps)**：
  **下一步探索建议：**

1. **优先修复AST实现并验证音频专用模型**：基于节点8/10的bug分析，正确实现AST需要：(a) 从原始WAV使用librosa计算mel频谱图(n_mels=128, hop_length=160)；(b) 使用`input_values`参数；(c) 应用正确的AudioSet归一化。若成功，AST的AudioSet预训练权重可能超越ConvNeXt的ImageNet权重。

2. **尝试PANNs (CNN14) 作为中间方案**：PANNs比AST实现简单(纯CNN架构)，同样使用AudioSet预训练，可作为AST的替代验证音频预训练的价值。

3. **添加背景噪声增强**：所有节点均缺失Background Noise Injection(Pink Noise/环境底噪)，任务明确提及雨声风声干扰，建议添加该增强提升野外录音鲁棒性。

4. **优化频谱图参数**：当前使用预计算BMP频谱图(128×1024)，建议尝试从原始WAV重新计算，优化n_mels=128, hop_length=256-512以更好匹配鸟叫高频特性(2-8kHz)。

5. **尝试温度校准优化概率输出**：节点分析多次提及概率校准需求，建议在推理阶段添加temperature scaling优化最终提交概率。

6. **验证分层K-Fold确保类别分布均衡**：当前K-Fold可能未保证每折物种分布均衡，建议实现StratifiedKFold for multi-label以减小折叠方差(当前Fold方差达0.15+)。
----------------------------------------
