# Pretrained Models Usage Guide
## Fetching from HF-mirror
The example Python code is as follows:
```python
import os

# Must be set before importing transformers/huggingface_hub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# Example: Download model to specified directory
model_id = "MODEL_ID"
local_dir = "./DIR/MODEL_ID"

print(f"Pulling model from HF-mirror: {model_id} ...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Recommended to set to False to avoid symlink issues
    resume_download=True
)
print("Download complete!")
```
### Text Models

#### BERT for Text Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = MODEL_ROOT / "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Example usage
texts = ["This is a positive example", "This is a negative example"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

#### RoBERTa for Text Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = MODEL_ROOT / "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Usage similar to BERT
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

#### DeBERTa for Advanced NLP
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = MODEL_ROOT / "microsoft_deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# More powerful than BERT/RoBERTa for complex tasks
```

#### T5 for Text Generation
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = MODEL_ROOT / "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Text generation
input_text = "translate English to French: Hello world"
inputs = tokenizer(input_text, return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, num_beams=4)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Multilingual T5 (mT5)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = MODEL_ROOT / "mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Supports multiple languages
input_text = "translate English to German: Hello world"
inputs = tokenizer(input_text, return_tensors="pt")
```

### Vision Models

#### Vision Transformer (ViT)
```python
from transformers import ViTImageProcessor, ViTForImageClassification

model_path = MODEL_ROOT / "vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)

# Process images
inputs = processor(images=images, return_tensors="pt")  # images should be tensors in [0,1]
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
```

#### Swin Transformer
```python
from transformers import AutoImageProcessor, SwinForImageClassification

model_path = MODEL_ROOT / "swin-base-patch4-window7-224"
processor = AutoImageProcessor.from_pretrained(model_path)
model = SwinForImageClassification.from_pretrained(model_path)

# Usage similar to ViT
inputs = processor(images=images, return_tensors="pt")
```

#### DINO for Feature Extraction
```python
from transformers import ViTImageProcessor, ViTModel

model_path = MODEL_ROOT / "dino-vits8"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTModel.from_pretrained(model_path)

# Extract features
inputs = processor(images=images, return_tensors="pt")
outputs = model(**inputs)
features = outputs.last_hidden_state[:, 0, :]  # CLS token features
```

#### EfficientNet
```python
from transformers import AutoImageProcessor, EfficientNetForImageClassification

model_path = MODEL_ROOT / "efficientnet-b0"
processor = AutoImageProcessor.from_pretrained(model_path)
model = EfficientNetForImageClassification.from_pretrained(model_path)

# Efficient image classification
inputs = processor(images=images, return_tensors="pt")
```

#### TIMM Models (ConvNeXt, EfficientNet)
```python
import timm
import torchvision.transforms as transforms

# Manual normalization for TIMM models
timm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# ConvNeXt
model = timm.create_model('convnext_tiny.fb_in1k', pretrained=False)
state_dict = torch.load(MODEL_ROOT / "timm" / "convnext_tiny.fb_in1k.pth")
model.load_state_dict(state_dict)

# Apply normalization
images = timm_transform(images)  # images should be [0,1] tensors
outputs = model(images)

# EfficientNet-B0
model = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', pretrained=False)
state_dict = torch.load(MODEL_ROOT / "timm" / "tf_efficientnet_b0.ns_jft_in1k.pth")
model.load_state_dict(state_dict)
```

### Audio Models

#### Whisper for Speech Recognition
```python
from transformers import AutoProcessor, WhisperForConditionalGeneration

model_path = MODEL_ROOT / "openai_whisper-base"
processor = AutoProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Process audio
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# Generate transcription
generated_ids = model.generate(**inputs, max_length=50)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

#### Wav2Vec2 for Audio Features
```python
from transformers import AutoProcessor, Wav2Vec2Model

model_path = MODEL_ROOT / "wav2vec2-base"
processor = AutoProcessor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# Extract audio features
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)
audio_features = outputs.last_hidden_state
```

#### Data2Vec Audio
```python
from transformers import AutoProcessor, Data2VecAudioModel

model_path = MODEL_ROOT / "data2vec-audio-base"
processor = AutoProcessor.from_pretrained(model_path)
model = Data2VecAudioModel.from_pretrained(model_path)

# Similar usage to Wav2Vec2
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)
audio_features = outputs.last_hidden_state
```

### Object Detection Models

#### YOLOv8
```python
from ultralytics import YOLO

# Load model
model = YOLO(str(MODEL_ROOT / "yolov8" / "yolov8n.pt"))

# Inference
results = model.predict(source="path/to/image.jpg", save=True)
```

#### YOLOv5
```python
# YOLOv5 directory structure contains training/inference scripts
# Use the detect.py script or import models
import sys
sys.path.append(str(MODEL_ROOT / "yolov5"))

from models.experimental import attempt_load
model = attempt_load(str(MODEL_ROOT / "yolov5" / "weights" / "yolov5s.pt"))
```

## Fine-tuning Examples

### Text Classification Fine-tuning
```python
def finetune_text_model(model, tokenizer, train_dataloader, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model
```

### Image Classification Fine-tuning
```python
def finetune_vision_model(model, processor, train_dataloader, num_epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(num_epochs):
        for images, labels in train_dataloader:
            inputs = processor(images=images, return_tensors="pt")

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model
```

## Important Notes
0. **DOWNLOAD FIRST**: NONE OF THESE MODELS ARE PRE-DOWNLOADED; YOU MUST VERIFY AND DOWNLOAD THEM IF NECESSARY.

1. **Path Configuration**: Always use relative paths from your working directory to `Models/pretrained_models`.

2. **Model Loading**: Different model types require different processors/tokenizers.

3. **Data Preprocessing**:
   - Text: Use appropriate tokenizers.
   - Images: Convert to tensors in [0,1]; let processors handle normalization.
   - Audio: Ensure correct sample rate (typically 16 kHz).

4. **Device Management**: Move models to GPU if available.
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
   ```

5. **Memory Management**: Use `torch.no_grad()` during inference to conserve memory.

6. **Batch Processing**: Adjust batch sizes according to model size and available memory.

This guide provides the essential code patterns for loading and using all pretrained models in the `Models/pretrained_models` directory.