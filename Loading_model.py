#!/usr/bin/env python3
# train_ddp_fixed.py

import os
import pathlib
from tqdm import tqdm

# set cache dir …
user     = 'daniyala'
cache_dir= os.path.join('/projectnb/cs505aw/students', user, '.hf_cache')
pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    get_scheduler
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.optim import Adam8bit
from functools import partial
import re

# ————————————————————————————————————————————————————
# 1) Formatting & tokenization functions
# ————————————————————————————————————————————————————

def format_examples(ex):
    ex["input_text"] = f"Q: {ex['problem'].strip()}\nA:"
    ex["target_text"] = (
        f"<think> {ex['solution'].strip()}\n </think>"      # chain‐of‐thought on its own lines
        f"<Answer> {ex['answer'].strip()} </Answer>"  # exact casing & no leading comma
    )
    return ex


def tokenize_fn(examples, tokenizer, max_length=1024):
    tok = tokenizer(
        examples["input_text"],
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

def get_dataset():
    # load & format
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train[:10000]")
    ds = ds.map(format_examples, batched=False)

    # split
    splits = ds.train_test_split(test_size=0.1, seed=42)
    train_raw, test_raw = splits["train"], splits["test"]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    tok = partial(tokenize_fn, tokenizer=tokenizer, max_length=1024)

    # batch‐tokenize & remove original columns
    train_tok = train_raw.map(tok, batched=True, remove_columns=train_raw.column_names)
    test_tok  = test_raw .map(tok, batched=True, remove_columns=test_raw .column_names)

    # switch to torch tensors
    train_tok.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    test_tok .set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    return train_tok, test_tok

# ————————————————————————————————————————————————————
# 2) Main DDP training loop
# ————————————————————————————————————————————————————

def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    # load 8-bit quant + LoRA
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    base = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantization_config=bnb
    )
    base.gradient_checkpointing_enable()
    base = prepare_model_for_kbit_training(base)
    model = get_peft_model(
        base,
        LoraConfig(r=8, target_modules=["q_proj","v_proj"],
                   lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    )
    model.to(rank)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    # data & collator
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    collator  = DataCollatorWithPadding(tokenizer)

    train_ds, test_ds = get_dataset()
    train_sampler = DistributedSampler(train_ds)
    test_sampler  = DistributedSampler(test_ds)

    train_loader = DataLoader(
        train_ds, sampler=train_sampler,
        batch_size=2, collate_fn=collator, num_workers=4, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  sampler=test_sampler,
        batch_size=2, collate_fn=collator, num_workers=4, pin_memory=True
    )

    # optimizer + scheduler
    num_epochs   = 3
    optim        = Adam8bit(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_scheduler(
        "linear", optimizer=optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # train
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            batch = {k: v.to(rank) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if rank == 0 and step % 100 == 0:
                lr = optim.param_groups[0]["lr"]
                print(f"[{epoch}/{num_epochs}] Step {step}/{total_steps} — lr={lr:.2e} loss={loss:.4f}")

        if rank == 0:
            model.module.save_pretrained(f"./checkpoint-epoch{epoch}")

        # eval
        model.eval()
        total_eval_loss = 0.0
        count = 0
        test_sampler.set_epoch(epoch)
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(rank) for k, v in batch.items()}
                total_eval_loss += model(**batch).loss.item()
                count += 1
        if rank == 0:
            print(f"Epoch {epoch} eval loss: {total_eval_loss/count:.4f}")

        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
