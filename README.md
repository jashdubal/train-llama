# Training Llama

### Dependencies:
```bash
pip install -U bitsandbytes accelerate transformers datasets peft mlflow
```

### Files:
- `pre_train.py` - Pre-training with FSDP, QLoRA
- `fine-tune.py` - Fine-tuning with FSDP, QLoRA
- `rlhf.py` - Training with RLHF
