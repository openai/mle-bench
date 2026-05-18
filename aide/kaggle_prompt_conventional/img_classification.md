Image Classification Competition: Expert-Level Workflow and Technical Framework

1. Data Asset Management and Preprocessing Standards  
1.1 Data Mapping and Structural Validation  

In competition environments, datasets are typically stored under the `./input` directory. The primary task is to establish a robust mapping between metadata (CSV) and physical image files.  

File Access Standards: Load metadata using pandas. Focus on verifying label distribution to inform subsequent loss function selection.  

Structural Integrity Check: Validate the validity of image paths. Avoid time-consuming visual explorations; instead, concentrate on automating data pipeline integration.  

```python
import pandas as pd
# Used solely to verify data pipeline entry point
train_df = pd.read_csv('./input/train.csv')
print(f"Dataset integration: {train_df.shape[0]} samples found.")
```  

1.2 Preprocessing and Augmentation Pipeline  

Preprocessing is the critical stage that transforms raw pixels into features digestible by models.  

Standardization Logic: Apply corresponding mean and standard deviation values based on the source of the selected pretrained model (e.g., ImageNet).  

Augmentation Strategy Selection:  

- Basic Transforms: Resizing, random cropping, horizontal flipping.  
- Advanced Transforms: For tasks with high texture complexity, consider incorporating AutoAugment or CutMix strategies to improve generalization boundaries.  

2. Deep Learning Architecture Selection Logic  
Training models from scratch is prohibited. Instead, select the optimal backbone from the Hugging Face mirror site using a decision matrix aligned with the physical properties of the current task.  

2.1 Architecture Decision Matrix  

The agent must match image characteristics described in the task specification against the table below:  

| Task Scenario | Architecture Recommendation Logic | Core Advantage |  
|----------------|-------------------------------------|----------------|  
| Small-scale data / natural images | Prioritize validation of ResNet (e.g., resnet50) | Extremely fast convergence—ideal for establishing a baseline. |  
| High-resolution / fine-grained classification | Explore EfficientNet series (b0 to b7) | Compound scaling mechanism effectively captures subtle local features. |  
| Ultra-large-scale data / complex backgrounds | Introduce Swin Transformer or ViT | Global attention mechanism handles long-range feature dependencies. |  

2.2 Transfer Learning and Weight Configuration  

Environment Setup: Must configure the mirror via `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` to ensure compliant weight downloads.  

Phased Training: Recommend a three-stage strategy—"freeze backbone → train head → full fine-tuning"—to preserve pretrained features from corruption.  

3. Model Construction and Robustness Evaluation  
3.1 Objective Function and Optimizer Pairing  

Classification Loss: CrossEntropyLoss is preferred for multiclass tasks; if severe class imbalance exists, validate the performance gain offered by Focal Loss.  

Optimizer Logic: Default to AdamW, which outperforms conventional Adam in handling weight decay.  

3.2 Validation Framework  

Stratified K-Fold Cross-Validation: To ensure high correlation between local cross-validation (CV) scores and Leaderboard results, stratified sampling is mandatory.  

Early Stopping: Monitor validation loss to prevent overfitting under limited computational resources.  

3.3 Performance Quantification Metrics  

Beyond accuracy, introduce F1-score or AUC-ROC according to task characteristics. For multi-label tasks, mAP (mean Average Precision) serves as the core reference metric.  

4. Ensemble Learning and Submission Standards  
4.1 Model Ensembling  

TTA (Test-Time Augmentation): During inference, apply minor augmentations (e.g., flipping or resizing) to the test set and average predictions—typically yielding a stable 0.5%–1% improvement.  

Weighted Voting: If multiple architectures are trained (e.g., CNN and Transformer fusion), assign weights based on CV scores and perform soft voting.  

4.2 Submission File Generation  

Path Constraint: Output results must strictly conform to the `./submission.csv` format requirement.  

Inference Efficiency: Due to resource constraints (`nvidia_a100`), maximize `batch_size` during inference on large test sets to optimize runtime.