import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from accelerate import dispatch_model, infer_auto_device_map
import dotenv
import os
from tqdm import tqdm

dotenv.load_dotenv()
login(os.getenv("HF_TOKEN"))


def process_dataset(dataset_split, split_name, output_file):
    valid_passages = [
        example for example in dataset_split
        if 40 <= len(example["text"]) <= 300
    ]

    if not valid_passages:
        print(f"No valid passages found in {split_name} split")
        return

    instruction_template = """### Instruction:
You are a helpful assistant trained on Stoic philosophy.

Given a short Stoic passage, generate a realistic question someone might ask in a modern context for career advice, and answer it using the ideas from the passage. Do not refer to the passage in the question or the answer, assume the person asking the question and answering it do not have access to the passage.

### Passage:
{passage}

### Output:
Q:"""

    for i in tqdm(range(0, len(valid_passages), BATCH_SIZE), desc=f"Processing {split_name}"):
        batch = valid_passages[i:i + BATCH_SIZE]
        prompts = [instruction_template.format(
            passage=example["text"].strip()) for example in batch]

        inputs = tokenizer(
            prompts,
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        if inputs["input_ids"].size(0) == 0:
            print(f"Warning: Empty inputs in batch {i}")
            continue

        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )
            except Exception as e:
                print(f"Error generating batch {i}: {str(e)}")
                continue

        for output in outputs:
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            result = output_text.split("Q:")[-1].strip()
            if "A:" in result:
                question, answer = result.split("A:", 1)
                qa_pair = {
                    "prompt": question.strip(),
                    "response": answer.strip()
                }
                output_file.write(json.dumps(qa_pair) + "\n")
                output_file.flush()  # Ensure the line is written immediately


model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Memory for a server with 4 NVIDIA A10 GPUs
max_memory = {i: "22GB" for i in range(4)}
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)

BATCH_SIZE = 16  # 4 batches per GPU
MAX_LENGTH = 512

dataset = load_dataset("eliwill/Stoic-dataset")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

with open("stoic_qa_train.jsonl", "w") as train_file:
    process_dataset(train_dataset, "train", train_file)

with open("stoic_qa_val.jsonl", "w") as val_file:
    process_dataset(val_dataset, "validation", val_file)
