# Use a pipeline as a high-level helper

import os
import pathlib
from tqdm import tqdm
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
from bitsandbytes.optim import Adam8bit

from transformers import DataCollatorWithPadding
from transformers import get_scheduler
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

### TRAINED OPen221k on max lenth 512 and 20000 samples
### Trained Code parent on 10000 samples and length 1024, 3 epochs with 10000 samples
### Trained Code Parrot GIthub on 10000 samples with max length of 1024


def get_dataset(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/apps", split="train[:10000]")
    def tok(ex):
        t = tokenizer(ex["question"], ex["solutions"],
                      padding="max_length", truncation=True, max_length=1024)
        t["labels"] = t["input_ids"].copy()
        return t
    ds = ds.map(tok, batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","labels"])
    return ds.train_test_split(test_size=0.1, seed=42)

def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantization_config=bnb,
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

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    collator = DataCollatorWithPadding(tokenizer)

    dataset = get_dataset(tokenizer)
    train_ds = dataset['train']
    test_ds = dataset['test']

    train_sampler  = DistributedSampler(train_ds)
    test_sampler = DistributedSampler(test_ds)
    train_loader   = DataLoader(train_ds, sampler=train_sampler,
                          batch_size=2, collate_fn=collator,
                          num_workers=4, pin_memory=True)

    test_loader   = DataLoader(test_ds, sampler=test_sampler,
                          batch_size=2, collate_fn=collator,
                          num_workers=4, pin_memory=True)



    num_epochs=3
    optim = Adam8bit(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optim,
    start_factor=1.0, end_factor=0.0,
    total_iters=total_steps)



    for epoch in  tqdm(range(num_epochs),desc="training"):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(rank) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            scheduler.step()        

            optim.zero_grad()

            
            if rank == 0 and step % 100 == 0:
                lr = optim.param_groups[0]["lr"]
                print(f"[Epoch {epoch} Step {step}/{total_steps}] lr = {lr:.3e}  loss = {loss:.4f}")


        if dist.get_rank() == 0:
            model.module.save_pretrained(f"./checkpoint-epoch{epoch}")


        # evaluation
        model.eval()
        total_eval_loss = 0.0
        count = 0
        test_sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch in tqdm(test_loader,desc="Evaluating"):
                batch = {k: v.to(rank) for k, v in batch.items()}
                loss = model(**batch).loss
                total_eval_loss += loss.item()
                count += 1

        # clear GPU cache
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
