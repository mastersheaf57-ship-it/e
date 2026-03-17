from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

model_name = "mistralai/Mistral-7B-v0.1"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ NEW 4-bit config (fixed)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# dataset
dataset = load_dataset("json", data_files="data.jsonl")

def format(example):
    return {"text": example["instruction"] + "\n" + example["response"]}

dataset = dataset.map(format)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)

# training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
    fp16=True
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# train
trainer.train()

# save
model.save_pretrained("./jar-ai")
