# StoicGemma

A fine-tuned version of Google's Gemma 3 4B model specialized in providing Stoic career advice and guidance.

## Model Details

- **Base Model**: [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Fine-tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- **Specialization**: Career advice with Stoic philosophy principles

## Dataset

The model was fine-tuned on a custom dataset of Stoic career-related Q&A pairs. The dataset is available at:
- [aniketsharma00411/stoic-career-qa-dataset](https://huggingface.co/datasets/aniketsharma00411/stoic-career-qa-dataset)

## Usage

### Installation

1. Clone this repository:
```bash
git clone git@github.com:aniketsharma00411/StoicGemma.git
cd StoicGemma
```

2. Create and activate the conda environment:
```bash
conda env create --name stoicgemma --file environment.yml
conda activate stoicgemma
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "aniketsharma00411/stoic-career-gemma-3-4b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

## Key Details

The model was fine-tuned using the following process:

1. **Dataset Creation**:
   - Generated using [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to create diverse career-related questions
   - Questions and Answers were designed based on Stoic passages from the dataset [eliwill/Stoic-dataset](https://huggingface.co/datasets/eliwill/Stoic-dataset)

2. **Base Model**: [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

3. **Fine-tuning Method**: PEFT (Parameter-Efficient Fine-Tuning) using LoRA

## Contact

For questions or feedback, please contact me at aniketsharma00411@gmail.com
