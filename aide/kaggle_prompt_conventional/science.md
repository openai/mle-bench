Scientific Discovery and Structural Modeling: A Physics-Constrained and Domain-Knowledge-Enhanced Deep Learning Framework (Science & Modeling)  
1. Data Engineering and Geometric/Biological Topology Construction  
In natural science tasks (e.g., RNA structure prediction, seismic wave analysis, materials science), raw data inherently obeys stringent physical laws. Models must perceive geometric or biological topology—not merely numerical values.  

1.1 Physics-Constrained Enhancement and Structured Mapping  
Geometric Encoding: For 3D structures, in addition to sequence information, the Euclidean distance map between atoms/nucleotides must be explicitly extracted. For disordered regions, geometric interpolation (e.g., sinusoidal perturbation for curvature modeling) is applied instead of zero-filling.  
Variant Mapping: Given the complexity of scientific data (e.g., 93 types of RNA-modified bases), a comprehensive dictionary mapping must be established to align heterogeneous, long-tailed data (e.g., modified bases) into a standardized feature space.  

```python  
# Implement geometric backbone reconstruction logic  
def reconstruct_backbone(coords, gap_mask):  
    """  
    Reconstruct missing 3D coordinates based on geometric principles.  
    coords: known coordinates (N, 3)  
    gap_mask: boolean mask indicating missing positions  
    """  
    # 1. Extract backbone direction vectors  
    direction = compute_backbone_direction(coords)  
    # 2. Linear interpolation + sinusoidal perturbation (to preserve RNA/protein backbone curvature)  
    for i in range(len(gap_mask)):  
        if gap_mask[i]:  
            offset = np.sin(i * np.pi / gap_len) * perpendicular_vec  
            coords[i] = linear_interp(coords[i-1], coords[i+1]) + offset  
    return coords  
```  

1.2 Evaluation-Metric-Driven Feature Selection  
Scale-Invariant Processing: Metrics such as TM-score—or similar normalized measures—require normalization of samples with varying lengths/scales during feature engineering, ensuring model accuracy across a broad range.  

2. Level-1 Model Architecture  
2.1 Hybrid Architecture Optimization Logic (TBM + DL)  
Architecture Selection: In scientific domains, Template-Based Modeling (TBM) and Deep Learning (DL) should operate in parallel. DL models (e.g., DRfold2, RoseTTAFold) capture nonlinear interactions, while TBM provides a stable physical foundation.  
Computational Accuracy Enhancement: Scientific computing is highly sensitive to numerical error. We recommend enforcing float64 arithmetic in the scoring and selection module, and leveraging GPU acceleration for matrix distance computations (e.g., `torch.cdist`).  

2.2 Physics-Inspired Head Design  
Energy Function Head: Instead of directly predicting numerical values, this head predicts energy terms. Output optimization proceeds by minimizing physical energy (e.g., steric clash, base-pairing constraints).  
Confidence-Adaptive Constraint Weighting: Physical constraint weights in the loss function are dynamically adjusted according to the model’s predicted confidence:  
$Constraint\_Weight = \lambda \times (1 - \min(Confidence, 0.8))$  

3. Level-2 Models and Optimization Modules  
3.1 Iterative Automatic Differentiation Optimization  
Second-Order Optimizer Integration: After model output, integrate PyTorch’s native LBFGS optimizer for structural refinement. Compared to Adam, LBFGS offers superior convergence on scientific-computing landscapes.  
External Knowledge Base Integration (Hybrid Inference): During inference, incorporate high-performance external tools (e.g., Boltz-1 or AlphaFold3 conformations) as additional potential energy terms, enabling “inference-side stacking” of heterogeneous models.  

4. Ensemble Strategy and Graceful Degradation (Hybrid Ensemble)  
4.1 Deterministic and Probabilistic Fusion  
Fallback Mechanism: Route tasks based on complexity. For simple tasks/short sequences: prioritize computationally lightweight template-based methods (TBM). For complex tasks/novel structures: activate deep learning models, augmented by ranking algorithms.  
GBDT-Based Structural Scoring: Use LightGBM to re-rank multiple candidate models. Features include not only model-predicted probabilities but also:  
- Physical features: atomic clash rate (“Clash score”), bond length/angle deviations.  
- Evolutionary features: homologous sequence coverage, multiple sequence alignment (MSA) depth.  

🛡️ Audit-Safe Compliance Guidelines  
Computational Resource Boundaries: Precomputed tables exceeding large-scale thresholds (e.g., >10 GB PDB databases) must never be hardcoded into inference code; instead, access must occur via incremental indexing techniques.  
Determinism Guarantee: Because scientific computing involves extensive floating-point operations, `torch.use_deterministic_algorithms(True)` must be enforced at the top of all code to prevent result jitter caused by GPU parallelism.  
Local Execution Principle: All optimization procedures (e.g., structural refinement) must execute entirely on local nodes; invoking remote HPC center REST APIs is strictly prohibited.