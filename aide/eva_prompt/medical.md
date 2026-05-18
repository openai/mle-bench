## Medical Imaging 任务代码进化专家评价

**【任务背景】**：本任务属于医疗影像分析，特点是图像尺寸极大（如WSI全切片）、标注成本高、通常存在极端的类别不平衡或严重的Domain Shift。

**【领域专家深度评价指令】**
请作为医疗AI领域的资深专家，审查上述方案并提出专业优化建议：
1. **高分辨率处理与多实例学习 (MIL)**：对于病理切片等超大图像，评估当前代码是否采用了合理的 Tiling（切块）策略。如果是基于Bag层面的预测，MIL聚合函数（如 Attention pooling, Gated Attention）设计是否科学？
2. **医疗先验与预处理**：评价代码是否进行了必要的医疗域预处理（如病理图的颜色反卷积、H&E染色归一化，或X光的CLAHE直方图均衡化，3D医疗图像的窗宽窗位调整）。
3. **指标导向的优化**：医疗指标通常不是简单的Accuracy。评估Loss函数是否与赛题Metric（如 Quadratic Weighted Kappa, Macro AUC）直接对齐或存在平滑逼近策略。
4. **泛化与稳定性**：分析交叉验证策略，是否采用了严格的 Patient-level (GroupKFold) 划分以防止数据穿越？
5. **改进输出**：临床先验优化方案 (Clinical Prior Optimization)：预处理管线：明确物理维度的预处理建议（如：统一重采样至 $1.0 \text{mm} \times 1.0 \text{mm}$ 的体素间距）。多实例学习 (MIL) 逻辑：如果 Bag 预测效果差，建议具体的注意力池化（Attention-based Pooling）改进逻辑。验证一致性：要求强制执行 GroupKFold(groups=patient_id) 并解释其在防穿越中的必要性。