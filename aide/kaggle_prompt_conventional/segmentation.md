General 2D/3D Object Detection and Segmentation: A High-Accuracy Spatial Analysis Framework  
1. Data Engineering and Asset Auditing (Data Engineering)  

When processing spatial data (e.g., CT, MRI, tomography, or remote sensing images), data consistency and feature enhancement form the foundation of model robustness.  

Multi-Scale Preprocessing:  

Resampling Logic: Standardize voxel spacing or pixel resolution. Apply normalization scaling using algorithms such as `scipy.ndimage.zoom` to ensure physical scale consistency.  

Label Heatmap Encoding: Employ Gaussian heatmaps for center-based annotation. Apply 8× downsampling to reduce computational load while leveraging metric tolerance to improve model robustness against minor spatial displacements.  

Automated Data Cleaning:  
* Use visualization tools (e.g., Napari or OpenCV) for outlier auditing; manually complete missed labels, then integrate them into the training pipeline via programmatic logic—**never hardcode completions directly in inference scripts**.  

2. Model Architecture and Feature Extraction (Model Architecture)  

The framework adopts a highly scalable Encoder-Decoder architecture, balancing receptive field size and computational efficiency.  

Backbone Network: Supports dynamic switching between ResNet-101 and ResNet-200. For 3D tasks, prioritize pre-trained 3D convolutional encoders.  

GPU Memory Optimization Techniques:  
* Gradient Checkpointing: Trade computation time for GPU memory savings in deep networks, enabling larger batch sizes.  

Stochastic Dropout: Enhance regularization during forward propagation to prevent overfitting.  

Multi-Head Output Design: In addition to the primary segmentation head, incorporate deep supervision and a max-pooled head. Pooling operations expand the response region, reducing penalty sensitivity to localization accuracy fluctuations.  

3. Robust Validation and Training Strategy (Validation & Training)  

Rank Correlation Validation: Employ 4-fold stratified cross-validation (Stratified CV). Focus on rank correlation between local CV and the public leaderboard (LB), rather than absolute score values.  

Heavy Data Augmentation:  
* GPU Acceleration: Offload Mixup, axial rotations, flips, and coarse dropout onto the GPU to overcome CPU preprocessing bottlenecks.  

Long-Cycle Training: Support 200–400 epochs when combined with heavy augmentation, ensuring convergence quality via cosine annealing.  

4. Inference Pipeline and Dynamic Post-Processing (Inference & Post-processing)  

Sliding Window Inference: For ultra-large inputs, apply overlapping sliding windows (overlap > 0.8). Use an ROI weight map to down-weight edge predictions and eliminate tiling artifacts.  

Dynamic Quantile Thresholding:  
* Logic: Replace unstable fixed thresholds. Derive the decision threshold from the top-N quantile of the full-sample prediction distribution, adapting to noise variations across samples.  

Ensembling Strategy: Apply multi-seed averaging. Perform logit fusion or direct weighted averaging on sigmoid-transformed probability outputs.  

5. Compliance and Governance Standards (Compliance & Governance)  

To ensure code compliance within automated evaluation systems, the following rules must be strictly enforced:  

Anti-Manual Writing:  
* `submission.csv` must be generated in real time by the inference script based solely on model outputs.  

Hardcoded sample-specific logic (e.g., `if image_id == 'xxx': ...`) is strictly prohibited.  

Strict External Isolation:  
* Code logic must not include external API calls via libraries such as `openai`, `anthropic`, or `requests`.  

API Key Scanning: Storing any third-party credentials—whether in source code or environment variables—is strictly prohibited.  

Access Control:  
* Only read access to designated directories (e.g., `/input`) is permitted.  

Attempting to fetch external URLs or use browser drivers during execution is strictly forbidden.  

💡 Agent Execution Recommendations:  

GPU Memory Management: During 3D U-Net training, if GPU memory is exhausted, prioritize enabling `torch.utils.checkpoint` and applying 16-bit automatic mixed precision (AMP) in the decoder stage.  

Performance Monitoring: The Agent should monitor CV–LB deviation in real time during execution. If LB fluctuation becomes excessive, automatically switch to the quantile-threshold update logic—**not** model parameter tuning.  

Data Closed Loop: All preprocessing logic (e.g., scaling, normalization) must be fully replicated in the inference script to guarantee consistency between training and inference.