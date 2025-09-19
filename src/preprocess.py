import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import os

def format_prompt(sample):
    """A simple formatter for various instruction dataset structures."""
    if 'instruction' in sample and 'output' in sample:
        instruction = sample.get('instruction', '')
        inp = sample.get('input', '')
        output = sample.get('output', '')
        if inp:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    elif 'text' in sample:
        return sample['text']
    elif 'conversations' in sample and isinstance(sample['conversations'], list):
        return "\n".join([f"### {turn.get('from', 'user')}:\n{turn.get('value', '')}" for turn in sample['conversations']])
    else:
        # Fallback for unforeseen structures
        return " ".join(str(v) for v in sample.values())

def load_and_prepare_data(config):
    """Loads, preprocesses, and tokenizes datasets for training and validation."""
    token = os.getenv("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_auth_token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Fatal: Could not load tokenizer for '{config['model_name']}'. Error: {e}")
        raise

    max_seq_length = config["dataset_config"]["max_seq_length"]

    def process_and_tokenize(examples):
        texts = [format_prompt(ex) for ex in examples]
        model_inputs = tokenizer(texts, max_length=max_seq_length, truncation=True, padding=False)

        # SFTTrainer expects a 'text' field by default if not specified otherwise
        model_inputs['text'] = texts
        # Prepare labels for standard language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        # Add dummy fields for HYDRA loss, which will be padded by the DataCollator
        for i in range(len(examples)):
            seq_len = len(model_inputs["input_ids"][i])
            model_inputs["edit_mask"] = model_inputs.get("edit_mask", []) + [np.zeros(seq_len, dtype=np.float32).tolist()]
            model_inputs["risk_mask"] = model_inputs.get("risk_mask", []) + [np.zeros(seq_len, dtype=np.float32).tolist()]
            model_inputs["freq_bucket"] = model_inputs.get("freq_bucket", []) + [(np.random.randint(0, 10, seq_len) / 10.0).astype(np.float32).tolist()]
            model_inputs["position"] = model_inputs.get("position", []) + [(np.arange(0, seq_len, dtype=np.float32) / seq_len).tolist()]
        
        return model_inputs

    train_datasets = []
    for name in config["dataset_config"]["train_datasets"]:
        try:
            ds = load_dataset(name, split='train', use_auth_token=token)
            train_datasets.append(ds)
        except Exception as e:
            print(f"Warning: Failed to load dataset '{name}'. Skipping. Error: {e}")

    if not train_datasets:
        raise ValueError("Fatal: No training datasets could be loaded.")

    full_train_ds = concatenate_datasets(train_datasets).shuffle(seed=config.get("seed", 42))

    num_samples = config["dataset_config"].get("train_samples", -1)
    if num_samples > 0 and num_samples < len(full_train_ds):
        full_train_ds = full_train_ds.select(range(num_samples))

    # Map processing function
    processed_ds = full_train_ds.map(process_and_tokenize, batched=True, batch_size=100, remove_columns=full_train_ds.column_names)

    # Create a validation split
    if 'val_samples' in config["dataset_config"] and len(processed_ds) > config["dataset_config"]['val_samples']:
        split_data = processed_ds.train_test_split(test_size=0.05, seed=config.get("seed", 42))
        train_ds = split_data["train"]
        val_ds = split_data["test"]
    else: # Use the whole dataset for training if it's too small or no validation is specified
        train_ds = processed_ds
        val_ds = None 

    print(f"Training dataset size: {len(train_ds)}")
    if val_ds:
        print(f"Validation dataset size: {len(val_ds)}")

    return train_ds, val_ds, tokenizer
