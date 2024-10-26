"""
### Pre-training
Train on unstructured data
"""

# !pip install -U bitsandbytes accelerate transformers datasets peft mlflow safetensors
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator
from torch import nn
from google.colab import userdata

DATASET_PATH = '/content/runescape_corpus.txt'
MODEL_PATH = 'meta-llama/Llama-3.2-1B'

class TrainingUtils:
    def __init__(self):
        pass

    def load_dataset(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            data_list = f.readlines()
            data_list = [line.strip() for line in data_list]
        data_dict = {"text": data_list}
        dataset = Dataset.from_dict(data_dict)
        return dataset


class LlamaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            labels = inputs.get("input_ids")
            outputs = model(**inputs)
            # torch.cuda.synchronize()
            logits = outputs.get("logits")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # torch.cuda.synchronize()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # torch.cuda.synchronize()
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            print(e)

utils = TrainingUtils()
dataset = utils.load_dataset(DATASET_PATH)

access_token = userdata.get('HF_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=access_token)
EOS_TOKEN = tokenizer.eos_token
tokenizer.pad_token = EOS_TOKEN

def add_eos_token_func(text):
    texts = text['text']
    texts_formatted =  [text + EOS_TOKEN for text in texts]
    return {"text": texts_formatted}
dataset_formatted = dataset.map(add_eos_token_func, batched=True)
dataset_tokenized = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding=True, return_tensors="pt"), batched=True)
dataset_tokenized.set_format(type='torch', columns = ['input_ids', 'attention_mask'])

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=access_token)
model = prepare_model_for_kbit_training(model)
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    inference_mode=False,
    r=4,
    lora_alpha=4,
    # for fine-tuning
    # target_modules=["q_proj", "k_proj", "v_proj"],
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
accelerator = Accelerator()
model = accelerator.prepare(model)
model = model.to(accelerator.device)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=False,
    report_to="none",
    fp16=True,
    gradient_accumulation_steps=2,
)

trainer = LlamaTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized,
)

trainer.args._n_gpu = 1
print(f"TRAINING WITH {torch.cuda.device_count()} GPUs...")
print(trainer.train())
OUTPUT_DIR = "content/trained_model_safetensors"

model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)