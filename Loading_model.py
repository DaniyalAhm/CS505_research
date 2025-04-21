# Use a pipeline as a high-level helper

import os
import pathlib

# Set the huggingface cache to a location
# in the class project.

user = 'daniyala'
cache_dir = os.path.join('/projectnb/cs505aw/students',
                         user,'.hf_cache')
# Make the directory, if needed
pathlib.Path(cache_dir).mkdir(parents=True,
                              exist_ok=True)

# Configure the environment to cache HF
# stuff in cache_dir
os.environ['HF_HOME']=cache_dir 
import torch
from transformers import pipeline

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from transformers import DataCollatorWithPadding

# train_ddp.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

def get_dataset(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train[:4000]")
    def tok(ex):
        t = tokenizer(ex["problem"], ex["solution"],
                      padding="max_length", truncation=True)
        t["labels"] = t["input_ids"].copy()
        return t
    ds = ds.map(tok, batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","labels"])
    return ds.train_test_split(test_size=0.1, seed=42)["train"]

def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    collator = DataCollatorWithPadding(tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantization_config=bnb,
        device_map='cpu'
        
    )
    # apply LoRA
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(r=8, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    )
    model.gradient_checkpointing_enable()
    model.to(rank)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    train_ds = get_dataset(tokenizer)
    sampler  = DistributedSampler(train_ds)
    loader   = DataLoader(train_ds, sampler=sampler,
                          batch_size=1, collate_fn=collator,
                          num_workers=1, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    for epoch in range(1):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
        if dist.get_rank() == 0:
            model.module.save_pretrained(f"./checkpoint-epoch{epoch}")

    dist.destroy_process_group()

if __name__=="__main__":
    main()
