# Training Llama

### Dependencies:
```bash
pip install -U bitsandbytes accelerate transformers datasets peft mlflow
```

### Files:
- `src/pretrain.py` - Pre-training with FSDP, LoRA    
- `src/finetune.py` - Fine-tuning with FSDP, LoRA
- `src/rlhf.py` - Training with RLHF
- `src/inference.py` - Inference using trained model
