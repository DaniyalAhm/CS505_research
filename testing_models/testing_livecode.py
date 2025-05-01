#!/usr/bin/env python3
"""
evaluate_livebench_hf.py

Run pass@1 evaluation of a LoRA-tuned 8-bit Qwen-1.5B model on the
HuggingFace LiveBench coding dataset (“livebench/coding”).

Everything is configured via the constants below.
"""
import os
import pathlib
from tqdm import tqdm

# set cache dir …
user     = 'daniyala'
cache_dir= os.path.join('/projectnb/cs505aw/students', user, '.hf_cache')
pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

import os
import tempfile
import subprocess

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft       import PeftModel

# ─── CONFIGURE EVERYTHING HERE ───────────────────────────────────────────────

# Either a local folder or HF repo ID:
BASE_MODEL    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Path to your LoRA adapter folder:
ADAPTER_DIR   = "../CODE_PARROT_RESULTS"

# How many tokens to generate per prompt:
MAX_NEW_TOKENS = 128

# If True, run on CPU even if a GPU is available:
FORCE_CPU      = False

# ─── END OF CONFIG ──────────────────────────────────────────────────────────

def load_lora_model(base_name: str, adapter_dir: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto"
    )
    base.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base, adapter_dir, device_map="auto").eval()
    model.to(device)
    return tokenizer, model

def run_pytest(code: str, tests: str) -> bool:
    """
    Write code→submission.py and tests→test_submission.py,
    run pytest, return True if exit code == 0.
    """
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "submission.py"), "w") as cf:
            cf.write(code + "\n")
        with open(os.path.join(td, "test_submission.py"), "w") as tf:
            tf.write(tests + "\n")
        res = subprocess.run(
            ["pytest", "--maxfail=1", "--disable-warnings", "-q", "test_submission.py"],
            cwd=td,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return (res.returncode == 0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Using device: {device}\nLoading model…")
    tokenizer, model = load_lora_model(BASE_MODEL, ADAPTER_DIR, device)
    print("Model ready.\nLoading LiveBench coding dataset…")

    ds = load_dataset("livecodebench/code_generation")
    split = ds.get("test") or ds.get("validation") or ds["train"]

    passes = 0
    for ex in tqdm(split, desc="LiveBench Eval"):
        prompt = ex["question_content"]   # if your dataset uses a different key, rename it
        tests  = ex["private_test_cases"]    # likewise adjust if needed

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        if run_pytest(gen, tests):
            passes += 1

    total = len(split)
    print(f"\npass@1 = {passes}/{total} = {passes/total:.2%}")

if __name__ == "__main__":
    main()
