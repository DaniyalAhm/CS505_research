#!/usr/bin/env python3
# evaluation_aime_batch_fixed.py


import os
import pathlib
from tqdm import tqdm

# set cache dir …
user     = 'daniyala'
cache_dir= os.path.join('/projectnb/cs505aw/students', user, '.hf_cache')
pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = cache_dir


import re
import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


torch.cuda.empty_cache()

def load_model(base_name: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        device = torch.device("cpu")
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
        )
        base.to(device)

    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        device_map="auto" if torch.cuda.is_available() else {"": device},
    )
    model.eval()
    return tokenizer, model, device
import re

def extract_answer(full_text: str) -> str:
    tag_match = re.search(r"<Answer>\s*([^<]+?)\s*</Answer>", full_text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip().rstrip('."\'')
    
    parts = re.split(r"(?i)Answer\s*:\s*", full_text, maxsplit=1)
    if len(parts) > 1:
        after = parts[1].strip()
        return after.split()[0].rstrip('."\'')
    
    lines = [line.strip() for line in full_text.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1].split()[0].rstrip('."\'')
    
    return ""

def main():
    BASE    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ADAPTER = "../checkpoint-epoch0"     print("Loading model…")
    tokenizer, model, device = load_model(BASE, ADAPTER)
    print(f"Model loaded on {device}.")

    ds = load_dataset("HuggingFaceH4/MATH-500")
    split = ds.get("test" )

    problems  = [ex["problem"] for ex in split]
    solutions = ["<Answer>"] * len(problems)

    enc = tokenizer(
        problems,
        solutions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    records = []
    model.eval()
    with torch.inference_mode():
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(loader, desc="AIME Eval")):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prefix_len = input_ids.shape[1]
            gens = [out[prefix_len:] for out in outputs]
            texts = tokenizer.batch_decode(gens, skip_special_tokens=True)

            start = batch_idx * loader.batch_size
            for i, full in enumerate(texts):
                idx          = start + i
                problem_text = split[idx]["problem"]
                true_ans     = str(split[idx]["answer"])

                pred    = extract_answer(full)
                correct = (pred == true_ans)
                records.append({
                    "problem":      problem_text,
                    "true_answer":  true_ans,
                    "prediction":   pred,
                    "correct":      correct
                })

    df       = pd.DataFrame(records)
    accuracy = df["correct"].mean()
    print(f"AIME accuracy: {accuracy:.2%}")
    df.to_csv("math500_results.csv", index=False)
    print("Results saved to aime_results.csv")

if __name__ == "__main__":
    main()
