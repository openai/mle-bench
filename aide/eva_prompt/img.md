## General Image 任务代码进化专家评价

**【任务背景】**：本任务为通用计算机视觉分类（涵盖基础分类、细粒度FGVC、隐写及雷达图识别）。核心策略依赖于现代 CNN/ViT 架构、数据增强及度量学习。


**【领域专家深度评价指令】**
请作为Kaggle CV Grandmaster，深入分析上述代码及运行结果，制定改进策略：
1. **Backbone与分辨率匹配**：评估当前选择的模型（如 ConvNeXt, EfficientNetV2, Swin/ViT）是否契合当前图像的分辨率和特征粒度。对于细粒度任务（FGVC），是否采用了足够大的分辨率或裁切策略？
2. **数据增强机制 (Data Augmentation)**：审查 Albumentations 的配置。是否充分利用了 MixUp, CutMix 缓解过拟合并平滑标签？对于雷达/隐写这种极其依赖底层像素分布的特殊任务，是否误用了破坏像素结构的增强（如过度缩放、JPEG压缩）？
3. **Loss函数与类别不平衡**：结合运行结果中的类别得分，评价是否需要引入 Focal Loss, Class-Balanced Loss，或者在细粒度识别中引入 ArcFace/CosFace 等度量学习Loss。
4. **推理与后处理**：评估是否引入了 TTA (Test Time Augmentation) 策略提升鲁棒性。
5.视觉表征增强方案 (Representation Enhancement)：数据增强算子：列出具体的 Albumentations 组合（如：增加 RandomGamma 或 CLAHE 处理光照不均）。Backbone 缩放计划：建议从 EfficientNet-B0 升级至 B4 的逻辑逻辑及对应的 batch_size 调整建议。采样策略：针对混淆矩阵中的弱势类别，提出 WeightedRandomSampler 的具体权重分配方案。