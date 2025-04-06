import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

import dotenv
import os

from train_helper import get_dataset, format_prompt

dotenv.load_dotenv()
login(os.getenv("HF_TOKEN"))


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )


model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset_name = "stoic_qa"
train_dataset, val_dataset = get_dataset(dataset_name, from_hf=False)

tokenized_train = train_dataset.map(
    format_prompt,
    remove_columns=train_dataset.column_names
).map(
    tokenize_function,
    batched=True
)

tokenized_val = val_dataset.map(
    format_prompt,
    remove_columns=val_dataset.column_names
).map(
    tokenize_function,
    batched=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./stoic-career-gemma-3-4b",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
)

# Train the model
trainer.train()

trainer.save_model("./stoic-career-gemma-3-4b-final")
tokenizer.save_pretrained("./stoic-career-gemma-3-4b-final")
