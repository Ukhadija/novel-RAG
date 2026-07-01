import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model 