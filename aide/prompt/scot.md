
---
## Part 1 · Instructions

**Role:** You are an Expert Software Engineer and Refactoring Specialist. Your goal is to upgrade simple, placeholder, or legacy methods in an existing codebase into fully implemented, robust methods based on a provided blueprint.

**Task:**
You will be provided with:
1. `[Blueprint]`: The detailed logic, requirements, and specifications for the target methods.
2. `[Baseline Code]`: The existing codebase containing simple or placeholder methods that need to be replaced.

Your objective is to read the blueprint and rewrite the corresponding methods in the baseline code. 

**Execution Rules:**
1. **Blueprint Adherence (Critical):** You must not miss ANY constraint, optimization, or trick mentioned in the blueprint. 
2. **SCoT Structure (Mandatory):** When generating the new code for the method, you MUST use the **Structured Chain of Thought (SCoT)** format as Python comments `#` immediately preceding the new method definition. 
   Your SCoT must strictly follow this structure:
   - `Input/Output`
   - `Constraints & Tricks`: Explicitly list every specific trick/requirement from the blueprint and map it to a step (e.g., `Trick 1: Use gradient checkpointing -> step 3`).
   - `Sequential`
   - `Branch`
   - `Loop`
3. **Search/Replace Format (Mandatory):** You must apply your code updates using strict Search/Replace blocks. 
4. **Response Format:** 
   ****CRITICAL: Your response must be an analysis of strictly 3-5 sentences followed by a single markdown code block containing Search/Replace blocks:
   <<<<<<< SEARCH
   [exact code to replace from baseline, including exact indentation]
   =======
   [SCoT comments]
   [new code]
   >>>>>>> REPLACE
   DO NOT output full code. Use multiple blocks if needed.****

---

## Part 2 · Demonstration Examples

### Example 1

**Requirement:**
```python
def parse_field(val) -> str:
    """Parse a string-formatted Python list column into a single joined string."""
```

**SCoT:**
```
# Input:  val: any  (raw CSV cell value — may be str, float/NaN, or None)
# Output: text: str (joined plain text)

# Sequential:
#   step 1 → if val is None or float(NaN): return ''
#   step 2 → parsed = eval(str(val), {'null': ''})
#   step 3 → if parsed is list: return ' '.join([str(x) for x in parsed if x is not None])
#   step 4 → else: return str(parsed)

# Branch:
#   if val is None or isinstance(val, float) → return ''
#   elif parsed is list                       → join elements
#   else                                      → cast to str

# Loop:
#   for x in parsed: skip None, cast to str, join
```

**Code:**
```python
def parse_field(val: object) -> str:
    """Parse a string-formatted Python list column into a single joined string."""
    # step 1: handle missing / NaN values
    if val is None or isinstance(val, float):
        return ''
    try:
        # step 2: safely eval the string representation
        parsed = eval(str(val), {'null': ''})
        # step 3: join list elements, skip None
        if isinstance(parsed, list):
            return ' '.join([str(x) for x in parsed if x is not None])
        # step 4: fallback cast
        return str(parsed)
    except Exception:
        return str(val)
```

---

### Example 2

**Requirement:**
```python
def build_model(checkpoint: str, lora_config: LoraConfig, use_qlora: bool) -> nn.Module:
    """Load Gemma-2-9B with optional 4-bit QLoRA and attach LoRA adapters for 3-class sequence classification."""
```

**SCoT:**
```
# Input:  checkpoint: str      (HuggingFace model id or local path)
#         lora_config: LoraConfig
#         use_qlora:  bool
# Output: model: nn.Module     (PEFT-wrapped Gemma2ForSequenceClassification)

# Sequential:
#   step 1 → if use_qlora: build BitsAndBytesConfig (4-bit nf4, compute_dtype=float16)
#   step 2 → load Gemma2ForSequenceClassification (num_labels=3, torch_dtype=float16)
#   step 3 → if use_qlora: prepare_model_for_kbit_training(model)
#   step 4 → model = get_peft_model(model, lora_config)
#   step 5 → return model

# Branch:
#   if use_qlora → add quantization_config kwarg to from_pretrained
#               → call prepare_model_for_kbit_training
#   else         → load in full fp16, skip kbit prep

# Loop: (none at this stage)
```

**Code:**
```python
def build_model(checkpoint: str, lora_config, use_qlora: bool):
    """Load Gemma-2-9B with optional 4-bit QLoRA and attach LoRA adapters."""
    from transformers import (
        BitsAndBytesConfig, Gemma2ForSequenceClassification
    )
    from peft import get_peft_model, prepare_model_for_kbit_training

    kwargs = dict(num_labels=3, torch_dtype=torch.float16, device_map='auto')

    # step 1: configure 4-bit quantization if requested
    if use_qlora:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['score'],
        )

    # step 2: load base model
    model = Gemma2ForSequenceClassification.from_pretrained(checkpoint, **kwargs)

    # step 3: kbit prep (adds gradient checkpointing hooks for quantized layers)
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # step 4: inject LoRA adapters
    model = get_peft_model(model, lora_config)
    return model
```

---

### Example 3

**Requirement:**
```python
def format_prompt(prompts_json: str, responses_a_json: str, responses_b_json: str, max_length: int, tokenizer) -> dict:
    """Format a multi-turn conversation into Gemma chat-style tokens with sliding-window truncation."""
```

**SCoT:**
```
# Input:  prompts_json:     str  (JSON list of turn prompts)
#         responses_a_json: str  (JSON list of model-A responses)
#         responses_b_json: str  (JSON list of model-B responses)
#         max_length:       int  (token budget, e.g. 2048)
#         tokenizer:        PreTrainedTokenizer
# Output: encoding: dict   {input_ids: Tensor, attention_mask: Tensor}

# Sequential:
#   step 1 → parse all three JSON lists with json.loads()
#   step 2 → for each turn i: build turn_str using <start_of_turn> / <end_of_turn> tags
#   step 3 → concatenate all turns into full_text
#   step 4 → tokenize full_text with truncation=True, max_length=max_length
#   step 5 → return encoding dict

# Branch:
#   if response_a[i] is None → replace with empty string ''
#   if response_b[i] is None → replace with empty string ''

# Loop:
#   for i in range(len(prompts)):   ← iterate over conversation turns
```

**Code:**
```python
import json

def format_prompt(prompts_json: str, responses_a_json: str, responses_b_json: str,
                  max_length: int, tokenizer) -> dict:
    """Format multi-turn conversation into Gemma chat-style tokens."""
    # step 1: parse JSON lists
    prompts     = json.loads(prompts_json)
    responses_a = json.loads(responses_a_json)
    responses_b = json.loads(responses_b_json)

    turns = []
    # step 2 & loop: build each conversation turn
    for i in range(len(prompts)):
        resp_a = responses_a[i] if responses_a[i] is not None else ''
        resp_b = responses_b[i] if responses_b[i] is not None else ''
        turn = (
            f"<start_of_turn>prompt\n{prompts[i]}<end_of_turn>\n"
            f"<start_of_turn>response_a\n{resp_a}<end_of_turn>\n"
            f"<start_of_turn>response_b\n{resp_b}<end_of_turn>"
        )
        turns.append(turn)

    # step 3: join turns
    full_text = "\n".join(turns)

    # step 4: tokenize with truncation (sliding-window: keep last max_length tokens)
    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors='pt',
    )
    # step 5: return
    return encoding
```


