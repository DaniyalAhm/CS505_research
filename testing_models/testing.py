#!/usr/bin/env python3
# evaluation_aime_batch_fixed.py

import re
import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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

def extract_answer(full_text: str) -> str:
    # take the very first whitespace-separated token of the continuation
    # assume the solution begins immediately after the input prompt
    tok = full_text.strip().split()
    return tok[0].rstrip(".") if tok else ""

def main():
    BASE    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ADAPTER = "./open220k"  # adjust to your adapter path

    print("Loading model…")
    tokenizer, model, device = load_model(BASE, ADAPTER)
    print(f"Model loaded on {device}.")

    # 1) load the AIME dataset split
    ds = load_dataset("Maxwell-Jia/AIME_2024")
    split = ds.get("test", ds.get("validation", ds["train"]))

    # 2) prepare pairs (problem, "") exactly as in training
    problems  = [ex["Problem"] for ex in split]
    solutions = [""] * len(problems)

    # 3) batch‐tokenize into tensors
    enc = tokenizer(
        problems,
        solutions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # 4) wrap in TensorDataset & DataLoader
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    records = []
    model.eval()
    with torch.inference_mode():
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(loader, desc="AIME Eval")):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 5) generate just the downstream tokens
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # 6) decode only the *generated* part:
            #    slice off the prefix length
            prefix_len = input_ids.shape[1]
            gens = [out[prefix_len:] for out in outputs]
            texts = tokenizer.batch_decode(gens, skip_special_tokens=True)

            # 7) map back to original examples
            start = batch_idx * loader.batch_size
            for i, full in enumerate(texts):
                idx          = start + i
                problem_text = split[idx]["Problem"]
                true_ans     = str(split[idx]["Answer"])

                pred    = extract_answer(full)
                correct = (pred == true_ans)
                records.append({
                    "problem":      problem_text,
                    "true_answer":  true_ans,
                    "prediction":   pred,
                    "correct":      correct
                })

    # 8) aggregate & save
    df       = pd.DataFrame(records)
    accuracy = df["correct"].mean()
    print(f"AIME accuracy: {accuracy:.2%}")
    df.to_csv("aime_results.csv", index=False)
    print("Results saved to aime_results.csv")

if __name__ == "__main__":
    main()
