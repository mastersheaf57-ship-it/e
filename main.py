from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# 🔹 Model
model_name = "mistralai/Mistral-7B-v0.1"

# 🔹 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# 🔹 Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

# 🔥 LoRA (REQUIRED)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# 🔹 Load dataset
dataset = load_dataset("json", data_files="data.json")

# 🔹 Tokenize + FIX labels
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True)

# 🔹 Format
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 🔹 Training args
training_args = TrainingArguments(
    output_dir="./jar-ai",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
)

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# 🔹 Train
trainer.train()

# 🔹 Save
trainer.save_model("./jar-ai")
tokenizer.save_pretrained("./jar-ai")
