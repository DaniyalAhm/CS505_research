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




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=bnb_config,
    device_map="auto",       # <-- put the entire model (sharded 4‑bit) on GPU 0
)

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(r=8, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)


dataset = load_dataset("open-r1/OpenR1-Math-220k",split="train[:4000]")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def tokenize(examples):
    tokens = tokenizer(examples["problem"], examples['solution'], padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True)
#print(dataset)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_raw, test_raw = split["train"], split["test"]
train_ds = train_raw.map(tokenize, batched=True)
test_ds  = test_raw.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,     # This is per GPU
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,                         # Mixed precision for memory savings
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
)



trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

trainer.train()
