"""Implementation guideline."""

import time

import humanize


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    exec_timeout = int(agent.cfg.exec.timeout)
    return get_impl_guideline(
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
    )


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    steps_remaining: int,
    exec_timeout: int,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "Crucial Rules for PyTorch Multiprocessing:All Dataset and Model definitions MUST be at the top level of the script (not inside functions).You MAY use num_workers == 8 for DataLoaders.ALL execution logic (training loops, data loading, inference) MUST be wrapped strictly inside if __name__ == __main__"
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        f"📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development.",
        "",
        "⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`",
        "• AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`",
        "",
        "📈 **Logging (CRITICAL)**: You MUST explicitly print training loss and validation metric at the end of EVERY epoch/fold. ",
        "Format strictly as: `Epoch [X/Y] | Train Loss: [val] | Val Metric: [val]`. Disable default spammy logs (no `verbose=1` for sklearn/LightGBM). ",
        "The absolute last line of stdout MUST be `print(f'Final Validation Score: {score}')`.",
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "□ Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "□ Did I generate submission.csv in correct path with ALL test predictions?",
        "□ Did I print validation metric as the last line?",
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",

                # "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        # "**PyTorch Multiprocessing**: All `Dataset` and `nn.Module` definitions MUST be at module top level (not inside functions). "
        # "Set `num_workers=8` in all DataLoaders. ALL execution logic (training, data loading, inference) MUST be wrapped inside "
        # "`if __name__ == '__main__':`.",
        # "",
        # "**1. Real Model Inference for ALL Predictions**",
        # "• EVERY validation and test prediction MUST come from a trained model's forward pass.",
        # "• Pipeline: Load data -> Preprocess -> `model.eval(); model(...)` -> Save predictions.",
        # "• FORBIDDEN: constants, placeholders, dummy values, empty arrays, summary statistics, random numbers.",
        # "• FORBIDDEN: fake/mock metric functions — use real `sklearn.metrics` or a correct manual implementation.",
        # "• Why: shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE).",
        # "",
        # "**2. Generate submission.csv**",
        # "• Path: `./submission/submission.csv` (NOT `./working/submission.csv`).",
        # "• Content: model predictions on ALL test samples.",
        # "• Format: follow task description exactly (column names + dtypes).",
        # "",
        # "**3. Print Validation Metric**",
        # "• MUST print exactly: `print(f'Final Validation Score: {score}')` as the LAST line of execution.",
        # "• Score MUST be computed on a hold-out validation set using the task's official metric formula.",
        # "• CONSISTENCY: validation and test inference paths MUST share the same preprocessing / post-processing. "
        # "Any divergence (e.g. different aug, different reconstruction, different rounding) causes large val->test gaps.",
        # "",
        # "🎯 **Training Recipe Checklist** (apply unless task description explicitly contradicts):",
        # "• **FULL fine-tuning is the default** — DO NOT freeze the backbone or do linear-probe-only. "
        # "Freezing is acceptable only when training samples <500 AND must be justified in the plan.",
        # "• **Backbone tier**: prefer Premium > Strong > Baseline. Choose the LARGEST variant that fits GPU budget "
        # "(DINOv3-L/G > ViT-Base; ConvNeXtV2-L/H > ResNet50; DeBERTa-v3-large > BERT-base; ModernBERT-large > BERT-base; "
        # "Whisper-large-v3 > whisper-base). If OOM, drop batch size before dropping model size.",
        # "• **Schedule**: cosine LR with linear warmup (5-10% of total steps). Backbone LR = 1e-5 ~ 5e-5; head LR = 10-100x backbone LR. "
        # "Optimizer = AdamW, weight_decay = 0.05.",
        # "• **Epoch count**: image classification typically 10-30 epochs full FT; NLP fine-tune 3-10 epochs; "
        # "small datasets (<2000 samples) need MORE epochs (15-50) with stronger regularization. 1-2 epoch fine-tunes "
        # "are almost never enough for full FT to converge.",
        # "• **Augmentation is mandatory for vision/audio**: at least RandAugment + Mixup + CutMix for image classification; "
        # "SpecAugment for audio spectrograms. Document which augmentations were chosen and why.",
        # "• **Fine-grained signals**: if the task involves >50 classes OR visually-similar categories OR FGVC-style "
        # "(species/products/landmarks), you MUST use ArcFace / sub-center ArcFace / large-margin softmax instead of plain CE. "
        # "Plain CE softmax under-performs by 5-15% on FGVC.",
        # "• **Regularization**: label_smoothing=0.1 for classification; consider EMA on weights for full FT runs >=10 epochs.",
        # "• **TTA**: at inference, average predictions over >=4 augmentations (flips + crops). Skip only if the metric is order-sensitive.",
        # "",
        # "🚫 **Anti-patterns to AVOID**:",
        # "• Frozen backbone + tiny linear head ('feature extraction only').",
        # "• 1-2 epoch fine-tuning that cannot possibly converge full FT.",
        # "• No augmentation on image / audio tasks.",
        # "• Plain CE softmax on fine-grained data with >50 classes.",
        # "• ResNet50 / ViT-base / EfficientNet-B0 / BERT-base when a Strong/Premium tier alternative is listed.",
        # "",
        # "📁 **Directories**: input data in `./input/`, submission output in `./submission/`, temp files in `./working/`.",
        # "",
        # "📦 **Packages**: `numpy`, `pandas`, `sklearn`, `torch`, `transformers`, `timm`, `xgboost`, `lightgbm` are pre-installed. "
        # "`torch.hub.load(...)` and HuggingFace `from_pretrained(...)` are available.",
        # "",
        # "⚠️ **API Compatibility Pitfalls**:",
        # "• LightGBM: `fit(..., early_stopping_rounds=...)` -> `fit(..., callbacks=[lgb.early_stopping(...)])`.",
        # "• XGBoost: `fit(..., early_stopping_rounds=...)` -> `XGBClassifier(early_stopping_rounds=...)`.",
        # "• AdamW: `from transformers import AdamW` (deprecated) -> `from torch.optim import AdamW`.",
        # "• Albumentations Resize: `Resize(size=(H, W))` -> `Resize(height=H, width=W)`.",
        # "• Albumentations RandomResizedCrop: `RandomResizedCrop(H, W, scale=...)` -> `RandomResizedCrop(size=(H, W), scale=...)`.",
        # "",
        # "📈 **Logging (CRITICAL)**: You MUST explicitly print training loss and validation metric at the end of EVERY epoch/fold. ",
        # "Format strictly as: `Epoch [X/Y] | Train Loss: [val] | Val Metric: [val]`. Disable default spammy logs (no `verbose=1` for sklearn/LightGBM). ",
        # "The absolute last line of stdout MUST be `print(f'Final Validation Score: {score}')`.",
        # "",
        # "⚠️ **Self-Check Before Finalizing**:",
        # "□ Did predictions pass through model's learned weights during inference? (If NO -> INVALID)",
        # "□ Did I generate `./submission/submission.csv` with ALL test predictions, in the required format?",
        # "□ Did I print `Final Validation Score: ...` as the LAST line?",
        # "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
        # "□ Are training and inference using IDENTICAL preprocessing / augmentation logic (modulo train/eval mode)?",
        # "□ Did I fully fine-tune the backbone (not frozen) with cosine schedule + meaningful augmentation?",
        # "□ Did I explicitly print the Train Loss and Validation Metric for EVERY epoch to allow learning curve analysis?",
    ]

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
