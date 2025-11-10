# simple_download.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_base_model():
    print("📥 Downloading base T5-small model...")
    
    model_path = "./merged_summarizer"
    os.makedirs(model_path, exist_ok=True)
    
    # Download and save complete base model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("✅ Base model downloaded and saved!")
    print("📁 Files created:")
    for file in os.listdir(model_path):
        print(f"   - {file}")

download_base_model()
