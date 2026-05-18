#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for code review in search pipeline.
"""

from typing import Dict, Any
from utils.response import wrap_code

# ============================================================================
# Code Review Prompts
# ============================================================================
def get_code_review_prompt(task_desc: str, code: str) -> Dict[str, Any]:
    """Build full code review prompt dict from task description and code."""
    introduction = (
        "You are a Senior Data Science Code Reviewer. Your goal is to ensure the submission is legally valid and logically sound.\n\n"
        "⚠️ **CRITICAL INSTRUCTION**:\n"
        "You must strictly follow the [Code Review Guidelines] provided below.\n"
        "Do NOT rely on your general knowledge if it conflicts with the Environment Facts listed in the guidelines.\n"
        "Your output must be a structured review focusing ONLY on Data Leakage and Critical Integrity."
        "**STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones (e.g., ResNet, VGG) just to make code executable. Do not question or change the user's model choice.\n"
    )
    prompt = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Code to review": wrap_code(code),
        "Instructions": {},
    }
    prompt["Instructions"]["Code review guidelines"] = get_code_review_guidelines()
    prompt["Instructions"]["Response format"] = get_code_review_response_format()
    return prompt

def get_code_review_guidelines() -> list:
    """Code review guidelines."""
    guidelines = [
        "# 📜 Code Review Guidelines\n",
        "",
        "## ✅ Environment Facts (TRUTH - Do NOT Flag)\n",
        "**Trust these facts absolutely. Overwrite your internal knowledge cutoff:**",
        "  • **Paths**: `./input/`, `./working/`, `./submission/` ALL EXIST. **Don't question the path.**",
        "  • **Submission File Location**: Must save the submission to `./submission/submission.csv`.",
        "  • **Bleeding Edge Environment**: Assume the environment has the LATEST versions of all libraries (transformers, torch, etc.).",
        "  • **Dynamic Dependencies**: Assume necessary `pip install` commands are executed automatically in the background.",
        "  • **Model Availability**: ALL models (including those released after your training data cutoff) are available and compatible.",
        "  • **STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones just to make code executable.\n",
        "  • **Unknown Models are Valid**: If you see a model name you don't recognize or think is too new, assume it is a private or SOTA model that works perfectly.",
        "---\n",
               "## 🚫 STRICTLY FORBIDDEN (Zero Tolerance)\n",
        "**You will be penalized if you violate these:**",
        "  • **NO Model Downgrades**: Never replace a user's chosen model string with an 'older/safer' alternative (e.g., do not change a specific large model to a generic base model).",
        "  • **NO Compatibility Speculation**: Do not flag issues based on 'library version requirements' or 'unknown model names'.",
        "  • **Immutable Variables**: Treat variables defining `model_name`, `backbone`, or `checkpoint` as CONSTANTS. You are NOT allowed to edit them.",
        "  • **Do NOT Question or Change Model**: Treat the user's model/backbone/checkpoint choice as final. Do not suggest alternatives, do not 'fix' model names, do not replace with ResNet/VGG/base. Only fix data leakage and critical logic bugs.",
        "  **Don't question the path.**",
        "",
        "---\n",
        "## 🔴 P0 - Data Leakage (HIGHEST PRIORITY)\n",
        "",
        "### P0.1 Data Leakage - Process Order 🚨\n",
        "",
        "**Check if preprocessing is done BEFORE split** (validation data leaks into training):",
        "",
        "❌ **MUST FIX**:",
        "  • Scaler/PCA fitted on full data then split",
        "  • Feature engineering (Target Encoding, etc.) using full data",
        "  • Upsampling (SMOTE) applied before split",
        "",
        "✅ **Correct**: Split first → fit on train only → transform separately",
        "",
        "### P0.2 Data Leakage - Split Strategy 🚨\n",
        "**Core Logic: Check for I.I.D. Violation**",
        "❌ **Flag ONLY IF**: The chosen split method mathematically violates the data's dependency structure.",
        "",
        "## 🟡 P1 - Critical Correctness\n",
        "",
        "### P1.1 Metric & Logic Correctness",
        "  • Task requires F1 but code uses accuracy?",
        "  • Task requires RMSE but code uses MSE?",
        "",
        "### P1.2 Inference Integrity",
        "  • Test predictions: np.zeros(), np.ones(), train_mean(), np.random()?",
        "  • Val predictions: not from actual model.predict()?",
        "",
        "### P1.3 Best Model Usage",
        "  • Code uses best checkpoint (not last epoch) for test predictions?",
        "",
        "### P1.4 API Compatibility",
        "**Common API Issues to Fix:**",
        "  • LightGBM: Use `callbacks=[lgb.early_stopping(...)]` not `early_stopping_rounds=...` in fit()",
        "  • XGBoost: Use `XGBClassifier(early_stopping_rounds=...)` (correct) not `fit(early_stopping_rounds=...)`",
        "  • AdamW: Use `from torch.optim import AdamW` (not from transformers)",
        "  • NO tqdm, NO verbose=1 in training",
        "",
        "---\n",
        "## 📋 Decision Rule\n",
        "",
        "**needs_revision=True** ONLY IF:",
        "  • P0 data leakage found (MUST FIX)",
        "  • OR P1 critical bug found",
        "",
        "**needs_revision=False** IF:",
        "  • No P0/P1 bugs found",
        "",
        "**Default**: Approve unless concrete logic bugs found"
    ]
    return guidelines


def get_code_review_response_format() -> list:
    """Code review response format."""
    return [
        "🚨 **CRITICAL: OUTPUT REQUIREMENT**",
        "",
        "**Required Fields:**",
        "- `needs_revision` (boolean): true if code has issues that must be fixed, false if code is correct",
        "- `reasoning` (string): EXACTLY 2-4 sentences explaining your decision (NO MORE)",
        "",
        "**Conditional Field:**",
        "- `revised_code` (string): ONLY if needs_revision=true, provide targeted fixes using SEARCH/REPLACE format",
        "",
        "🚫 **If needs_revision=false (code is correct):**",
        "- DO NOT provide revised_code (must be null/omitted)",
        "- Original code will be used as-is",
        "- This prevents accidental modifications to working code",
        "",
        "✅ **If needs_revision=true (code has issues):**",
        "- MUST provide revised_code using SEARCH/REPLACE diff format",
        "- Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks for each fix",
        "- SEARCH block must match original code EXACTLY (character-by-character, same indentation)",
        "- Only include the specific buggy lines that need fixing",
        "- Can provide multiple SEARCH/REPLACE blocks for different issues",
        "- Preserve the solution approach and model architecture",
        "- Fix only the specific issues identified (metric mismatch, data leakage, API errors)",
        "- DO NOT change model architecture, data split method, or metric calculation (unless they are buggy)",
        "",
        "**Reasoning Field Guidelines:**",
        "⚠️ STRICT LENGTH LIMIT: Write EXACTLY 2-4 sentences. Be concise.",
        "Cover: (1) what issues found, (2) why they matter, (3) what will be fixed.",
        "DO NOT write detailed analysis, step-by-step checks, or comprehensive explanations.",
        "",
        "**Why this format matters:**",
        "The JSON schema format ensures that code is ONLY modified when necessary.",
        "When needs_revision=false, it's impossible to accidentally change working code.",
        "⚠️ reasoning MUST be 2-4 sentences only. Do NOT write long analysis or enumerate checks."
    ]
