# inference.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

def load_model(base_name: str, adapter_dir: str):
    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_name)

    # 2) Decide device
    if torch.cuda.is_available():
        device = "cuda"
        # 3a) GPU + bitsandbytes 4-bit quant
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        device = "cpu"
        # 3b) CPU FP16 fallback
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
        )
        base.to(device)

    # 4) Wrap with LoRA adapters
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        device_map="auto" if torch.cuda.is_available() else {"": device}
    )

    model.eval()
    return tokenizer, model, device

def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens=32):
    # Build the prompt in the same format you trained on
    q = prompt.replace(" ", "")
    input_text = f"Q: {q}\nA:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy
            pad_token_id=tokenizer.pad_token_id
        )
    # Decode only the newly generated part
    decoded = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()
    return decoded


def generate_cot_answer(model, tokenizer, problem, device,
                        max_new_tokens=128, cot_steps=5):
    # remove spaces so arithmetic tokens don't get split
    q = problem.replace(" ", "")

    # 1) build a CoT‐style prompt
    prompt  = (
        f"Q: {q}\n"
        "Let's work this out step by step:\n"
        "1."
    )
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)

    # 2) generate with a longer budget to allow room for reasoning
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    full_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    # 3) split out the chain of thought and the final answer
    #    we assume the model numbers steps as "1.", "2.", … then writes "Answer: X"
    cot, _, rest = full_text.partition("Answer:")
    answer = rest.strip().split()  # first token after "Answer:"

    return cot, answer

if __name__ == "__main__":
    BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ADAPTER = "./open220k"  # path where you saved your LoRA

    print("Loading model…")
    tokenizer, model, device = load_model(BASE, ADAPTER)
    print(f"Model loaded on {device}.")


# 2) build your problem
    problem = (
        "A construction company was building a tunnel. "
        "When 1/3 of the tunnel was completed at the original speed, "
        "they started using new equipment, which increased the construction "
        "speed by 20% and reduced the working hours to 80% of the original. "
        "As a result, it took a total of 185 days to complete the tunnel. "
        "If they had not used the new equipment and continued at the original "
        "speed, it would have taken ____ days to complete the tunnel."
    )

# 3) ask the model
    cot, answer = generate_cot_answer(model, tokenizer, problem, device)
    print("\n--- Chain of thought ---")
    print(cot)
    print("\nAnswer:", answer)
