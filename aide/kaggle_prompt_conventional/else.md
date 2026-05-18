General Kaggle Competition: Structured Expert Workflow and End-to-End Governance  
1. Data Asset Audit and Integrity-Preserving Feature Engineering  

In any Kaggle task, auditing the quality of raw data and ensuring compliance in feature generation form the foundation for a robust solution.  

1.1 Label Distribution Audit and Completeness Check  

Distribution Consistency Assessment: Audit distributional shifts (Adversarial Validation) across core features between the training and test sets. If abnormally aligned sample counts per category are detected, flag them as potential collection bias.  

Outlier Cleansing: For continuous variables, identify outliers using Z-Score or IQR methods; manual deletion of corresponding outlier records in the test set is strictly prohibited.  

Multi-Source Fusion (Compliant Version): Public external datasets explicitly authorized on the competition page may be incorporated. Compliance Warning: Retrieving test-set labels via API or directly using LLM-predicted outputs as training supplements is strictly forbidden.  

1.2 Automated Feature Derivation (No Prior Injection)  

High-Order Interactions: Apply automated feature-generation operators (e.g., Target Encoding, Polynomial Features); all statistical features must be generated using K-Fold Out-of-Fold (OOF) logic to prevent label leakage.  

Temporal Window Engineering: For time-series–related tasks, construct sliding-window statistics (Mean, Std, Max); using “future data” to predict the current state is strictly prohibited.  

2. Robust Validation Framework (Validation Architecture)  

This serves as the critical barrier preventing score divergence between local validation (CV) and leaderboard (LB) scores.  

2.1 Cross-Validation Strategy Optimization  

Stratified K-Fold: Ensure label distribution consistency across all folds relative to the full dataset. For multi-label tasks, adopt Iterative Stratification.  

Group Strategy: If natural clustering exists in the data (e.g., multiple records per user), GroupKFold must be used—ensuring entities in the validation set never appear in the training set—to realistically simulate generalization capability.  

2.2 Evaluation Metric Consistency  

Surrogate Loss Function: When the competition metric (e.g., QuadraticWeightedKappa) is non-differentiable, design a smooth surrogate loss function (e.g., weighted CrossEntropy), and guarantee 1:1 alignment between CV evaluation logic and the official scoring script.  

3. Model Architecture and Training Strategy  

Pursue a balance between accuracy and efficiency while mitigating compliance risks through strict code governance.  

3.1 Heterogeneous Model Ensemble Pathway  

Baseline Construction: First validate the capability of GBDT-family models (LightGBM, XGBoost, CatBoost) on structured data.  

Deep Learning Embedding: For unstructured data, select mainstream backbones (e.g., Transformer variants or ResNet series).  

Compliance Detection Logic: > [Compliance Self-Check: Code Audit Points]  

Prohibited: `openai.*`, `anthropic.*`, `requests.get(external_api)`.  

Strictly Forbidden: Manual assignment such as `df.loc[test_idx, 'target'] = [...]`.  

Logging Monitoring: Verify inference logs for hard-coded submission vectors.  

3.2 Training Hyperparameter Optimization  

Automated Hyperparameter Tuning: Use Optuna for Bayesian optimization; search space must be constrained within physically reasonable bounds.  

Learning Rate Scheduling: Apply Cosine Annealing with Warmup; initial learning rate is recommended between 1e−3 and 5e−5.  

4. Model Ensemble and Inference Engineering (Ensemble & Inference)  

4.1 Multi-Level Stacking  

Meta-Model Design: Use Ridge Regression or simple linear weighting as the second-level fusion model to prevent overfitting to OOF noise.  

Feature Pruning: During inference, eliminate features contributing less than 0.001 to prediction performance using Permutation Importance, optimizing inference latency.  

4.2 Inference Optimization Under Environmental Constraints  

Memory Management: Under Kaggle’s memory constraints, invoke `gc.collect()` and `torch.cuda.empty_cache()`.  

Operator Acceleration: Convert model weights to FP16 or half-precision mode—significantly boosting inference throughput without sacrificing critical accuracy.  

5. Failure Paths and Pitfall Summary (Negative Results)  

Based on extensive experimentation, the following strategies typically yield “high effort, low return” and are not recommended as primary avenues:  

Blindly Increasing Model Depth: On small-to-medium datasets, excessively deep networks often cause severe overfitting.  

Overreliance on Pseudo-Labeling: If the base model’s LB score is below 0.7, pseudo-labeling tends to inject more noise than gain.  

Manual Fine-Tuning of Specific Test-Set Samples: Such “targeted optimization” easily inflates Public LB scores while causing drastic oscillation (Shake-down) in Private LB scores.  

💡 Agent Execution Recommendations:  

Compliance-Silent Audit: Before generating `submission.csv`, the Agent should automatically scan the current environment for sensitive strings (e.g., API Keys) and ensure all `print` outputs reflect only training metrics—not raw data content.  

Deterministic Experiments: Set a unified global random seed for all stochastic processes (e.g., `seed`) to guarantee reproducibility—an industry-standard requirement for expert-level Kaggle solutions.