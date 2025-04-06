from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
import dotenv

dotenv.load_dotenv()
login(os.getenv("HF_TOKEN"))


def upload_dataset():
    dataset = load_dataset('json', data_files={
        'train': 'stoic_qa_train.jsonl',
        'validation': 'stoic_qa_val.jsonl'
    })

    dataset.push_to_hub("aniketsharma00411/stoic-career-qa-dataset")


def upload_model():
    base_model_name = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    adapter_path = "stoic-career-gemma-3-4b-final"
    model = PeftModel.from_pretrained(model, adapter_path)

    model = model.merge_and_unload()

    model.push_to_hub("aniketsharma00411/stoic-career-gemma-3-4b")
    tokenizer.push_to_hub("aniketsharma00411/stoic-career-gemma-3-4b")


upload_dataset()

upload_model()
