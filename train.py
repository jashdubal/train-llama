# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from kaggle_secrets import UserSecretsClient

# Setup secrets and paths
user_secrets = UserSecretsClient()
access_token = user_secrets.get_secret("HF_TOKEN")

MODEL_PATH = "meta-llama/Llama-3.2-1B"
DATASET_PATH = '/kaggle/input/runescape-corpus/runescape_corpus.txt'

# Function to load the dataset
def load_dataset(path):
    with open(path, mode='r', encoding='utf-8') as f:
        data_list = f.readlines()
        data_list = [line.strip() for line in data_list]
    data_dict = {"text": data_list}
    dataset = Dataset.from_dict(data_dict)
    return dataset

dataset = load_dataset(DATASET_PATH)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=access_token)
EOS_TOKEN = tokenizer.eos_token
tokenizer.pad_token = EOS_TOKEN

# Tokenize and add labels
def tokenize_and_add_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

dataset_tokenized = dataset.map(tokenize_and_add_labels, batched=True)
dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoader with num_workers=0 for debugging
train_dataloader = DataLoader(dataset_tokenized, batch_size=1, num_workers=0)

# Initialize FSDP plugin and Accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    auto_wrap_policy="TRANSFORMER_BASED_WRAP_POLICY",
)
accelerator = Accelerator(mixed_precision="fp16", fsdp_plugin=fsdp_plugin)

# Load the model and prepare for LoRA
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, use_auth_token=access_token)
model = prepare_model_for_kbit_training(model)
model.resize_token_embeddings(len(tokenizer))

# LoRA configuration
lora_config = LoraConfig(
    inference_mode=False,
    r=4,
    lora_alpha=4,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
        "up_proj", "down_proj", "embed_tokens", "lm_head"
    ],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare model, optimizer, and data loader with accelerator
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

# Training loop
num_epochs = 3
gradient_accumulation_steps = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(train_dataloader):
        assert isinstance(batch, dict), "Batch must be a dictionary"

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.update(1)
    
    if accelerator.is_main_process:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Save model and tokenizer
if accelerator.is_main_process:
    OUTPUT_DIR = "./trained_model_safetensors"
    accelerator.save_state(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
