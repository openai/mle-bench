## Detection & Segmentation 任务代码进化专家评价

**【任务背景】**：本任务涉及像素级的密集预测（图像分割）或包围盒回归（目标检测）。对内存管理、边界处理和小目标识别要求极高。

**【领域专家深度评价指令】**
作为图像密集预测专家，请从以下技术栈对当前代码进行诊断并升级：
1. **架构与感受野**：对于分割任务，U-Net/FPN 等解码器的特征融合是否充分（如是否引入了 ASPP, CBAM等注意力模块）？对于检测任务，Anchor配置是否契合数据集中目标的尺寸分布（特别是小目标）？
2. **损失函数设计 (Loss Tuning)**：评价当前的组合Loss。分割任务是否合理结合了 BCE + Dice Loss 或引入了 Lovasz Loss 优化IoU？检测任务是否使用了 GIoU/CIoU 和 Focal Loss 处理正负样本极度不均衡？
3. **数据拼接与边界处理**：对于如2.5D切片分割或卫星图分割，评估是否采用了 2.5D Channel Stacking 或无缝拼接（Seamless ensembling）策略。
4. **后处理提分**：是否实现了检测的 WBF (Weighted Boxes Fusion) 替代传统 NMS？分割是否添加了 TTA 或连通域形态学后处理？
5. **改进输出**：几何与像素精度修正方案 (Geometry Refinement)：损失函数配比：给出 $Dice$ 与 $BCE$ 的比例建议（如：$1:1$ 平衡），或针对小目标引入 $Focal\ Loss$。后处理逻辑：详细说明 WBF (Weighted Boxes Fusion) 的阈值设置思路，或分割掩码的空洞填充（Morphology）策略。推理切片 (Tiling)：建议测试时的滑动窗口步长及重叠率（Overlap），以解决大图边缘效应。