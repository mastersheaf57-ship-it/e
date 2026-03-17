from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 🔹 Model name (you can change this)
model_name = "mistralai/Mistral-7B-v0.1"

# 🔹 Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ FIX: add pad token (VERY IMPORTANT)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 Load model (4-bit for lower VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# 🔹 Load dataset (your data.json file)
dataset = load_dataset("json", data_files="data.json")

# 🔹 Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 🔹 Apply tokenization
dataset = dataset.map(tokenize, batched=True)

# 🔹 Set format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 🔹 Training settings
training_args = TrainingArguments(
    output_dir="./jar-ai",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True
)

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# 🔹 Train!
trainer.train()

# 🔹 Save model
trainer.save_model("./jar-ai")
tokenizer.save_pretrained("./jar-ai")
