# Training Llama

### Dependencies:
```bash
pip install -U bitsandbytes accelerate transformers datasets peft mlflow
```

### Files:
- `src/pre_train.py` - Pre-training with FSDP, QLoRA    
- `src/fine-tune.py` - Fine-tuning with FSDP, QLoRA
- `src/rlhf.py` - Training with RLHF
