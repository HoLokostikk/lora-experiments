import torch, json
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

import sys
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT = f"out/lora-r{RANK}"

ds_train = load_dataset("json", data_files="data/train_chat.jsonl", split="train")
ds_val = load_dataset("json", data_files="data/val_chat.jsonl", split="train")

peft_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK * 2,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    max_length=512,
    report_to="none",
    seed=0,
    assistant_only_loss=True,
)

trainer = SFTTrainer(
    model=MODEL,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    peft_config=peft_config,
)

trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(OUT + "/final")
print("saved:", OUT + "/final")
