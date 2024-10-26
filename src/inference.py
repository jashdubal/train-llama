from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "<PATH TO MODEL DIR>"

model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
