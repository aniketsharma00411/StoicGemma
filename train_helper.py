
from datasets import load_dataset, Dataset
import json


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_dataset(name, from_hf=True):
    if from_hf:
        dataset = load_dataset(name)

        return dataset["train"], dataset["validation"]
    else:
        train_data = load_jsonl(f"{name}_train.jsonl")
        val_data = load_jsonl(f"{name}_val.jsonl")

        train_dataset = Dataset.from_dict({
            "prompt": [item["prompt"] for item in train_data],
            "response": [item["response"] for item in train_data]
        })

        val_dataset = Dataset.from_dict({
            "prompt": [item["prompt"] for item in val_data],
            "response": [item["response"] for item in val_data]
        })

        return train_dataset, val_dataset


def format_prompt(example):
    return {
        "text": f"""###
### Question:
{example['prompt']}

### Answer:
{example['response']}

"""
    }
