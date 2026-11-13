"""
Download and cache the DeBERTa-v3 model locally
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
print(f"Downloading tokenizer and model for: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("SUCCESS: Download complete and cached locally!")
print("id2label mapping:", model.config.id2label)
print("label2id mapping:", model.config.label2id)

