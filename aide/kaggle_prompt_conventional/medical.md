Medical Imaging Analysis: Cascaded Detection, Multi-Task Segmentation, and a 2.5D Spatiotemporal Feature Framework (Medical CV)

1. Data Engineering and Spatial Topology Construction  
In medical imaging (CT/MRI/X-ray) competitions, efficiently compressing 3D voxel information into 2D networks while precisely localizing regions of interest (ROIs) is the decisive factor.

1.1 2.5D Slice Stacking and Spatiotemporal Context  

2.5D Topology Strategy: Do not input only a single slice. Instead, center on the target slice and stack adjacent slices before and after it (e.g., slice=3 or 5) as input channels. This enables 2D convolutions to extract pseudo-3D spatially continuous features.  

Windowing Transformation: Raw medical data (DICOM/HU values) exhibits an extremely wide dynamic range. Apply different windowing parameters according to the task (e.g., lung, bone, soft tissue), or feed multiple windowed versions as separate input channels.  

Python  
# Build 2.5D slice stacking and normalization logic  
def load_2_5d_slice(image_paths, target_idx, window_params, slice_num=5):  
    """  
    image_paths: List of paths for the entire sequence  
    target_idx: Index of the target slice  
    window_params: (width, level) window width and level  
    """  
    half_slice = slice_num // 2  
    indices = np.clip(  
        np.arange(target_idx - half_slice, target_idx + half_slice + 1),  
        0, len(image_paths) - 1  
    )  

    slices = []  
    for idx in indices:  
        img = load_dicom(image_paths[idx])  
        # Apply windowing transformation  
        img = apply_windowing(img, *window_params)   
        slices.append(img)  

    return np.stack(slices, axis=-1) # (H, W, C)  

1.2 Cascaded Region Cropping (ROI Detection)  

Detection-Guided Cropping: For high-resolution images, first train a lightweight detector (e.g., EfficientDet-D0) to localize organ regions. Then feed only the cropped ROIs into subsequent high-resolution segmentation/classification models—significantly reducing background noise and improving computational efficiency.  

2. First-Level Model Architecture (Level-1 Models)  
2.1 Multi-Task Backbone Network Optimization  

Encoder Selection: Prioritize empirical validation of EfficientNet-V2 or ConvNeXt families. In medical domains, CNNs’ translation invariance typically outperforms conventional Transformers (unless data volume is extremely large).  

Multi-Task Head Design: Attach two parallel branches after a shared encoder:  

Classification Branch (Cls): Global average pooling followed by a fully connected layer to determine whether the slice contains pathology (for positive/negative sample filtering).  

Segmentation Branch (Seg): A U-Net-style decoder for pixel-level annotation.  

Loss Function Weighting: Use Binary Cross-Entropy (BCE) loss for classification; for segmentation, combine Dice, BCE, and Lovász losses—leveraging classification branch signals to guide segmentation boundary learning.  

2.2 Training Stability and Convergence Techniques  

SWA and EMA: Medical image samples exhibit high variability; incorporating SWA (Stochastic Weight Averaging) or EMA (Exponential Moving Average) is critical for improving single-fold stability.  

GroupKFold: Folds must be stratified by Patient_ID. Absolutely prohibit splitting slices from the same patient across training and validation sets (to avoid data leakage).  

3. Second-Level Models and Accuracy Refinement (Level-2 Fine-tuning)  
3.1 Hard Sample Resampling and Segmentation Fine-tuning  

Positive-Sample Fine-tuning: After multi-task pretraining, freeze the classification branch and perform a second-stage fine-tuning of the segmentation model exclusively on Positive Samples (i.e., slices containing pathology).  

Resolution Upscaling: During second-stage fine-tuning, increase input resolution further (e.g., from 320 to 512), combined with mixed-precision training (AMP).  

3.2 Offline Pseudo-labeling  

Use the trained model to predict on unlabeled test set or external data, then extract high-confidence masks as pseudo-labels for inclusion in training.  

4. Inference Ensemble and Post-processing  
4.1 Sequence Consistency Constraints  

In medical video/time-series slices, lesion appearance is temporally continuous.  

Sliding-Window Filtering: For example, retain a segmentation result only if the model classifies three consecutive frames as positive.  

Connected-Component Filtering: Remove isolated segmentation fragments smaller than a threshold (e.g., < 25 pixels).  

4.2 Test-Time Augmentation (TTA)  

Leverage symmetry inherent in medical images by enforcing TTA with HorizontalFlip and VerticalFlip.  

🛡️ Audit-Compliant Guidelines  
Data Isolation Protocol: All inference scripts must run in closed-loop environments. Hardcoding any patient privacy information or external API keys within inference code is strictly prohibited.  

Inference Performance Constraint: Medical competitions impose strict inference time limits (e.g., processing tens of thousands of slices within 9 hours). Avoid inefficient per-pixel for-loops; vectorized operations are mandatory.  

Weight Source Declaration: Only pre-trained weights loaded via official channels (e.g., Timm, SMP) are permitted. Pulling models from non-mirrored sources (e.g., private cloud storage) is prohibited.