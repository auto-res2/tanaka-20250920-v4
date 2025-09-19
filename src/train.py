import torch
import torch.nn.functional as F
from torch import nn
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os

# HYDRA Loss Components originally from hydra_loss.py
class MetaGate(nn.Module):
    """A two-layer MLP that meta-learns the mixing weight for the HYDRA loss."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).squeeze(-1)

def proper_score(logits, labels):
    """Dynamically selected proper scoring rule based on token confidence."""
    probs = F.softmax(logits, -1)
    # Ensure labels are within the valid range for one_hot
    labels_clamped = torch.clamp(labels, 0, logits.size(-1) - 1)
    gt = F.one_hot(labels_clamped, logits.size(-1)).float()
    p_true = (probs * gt).sum(-1)

    score = torch.empty_like(p_true)
    low = p_true < 0.2
    mid = (p_true >= 0.2) & (p_true <= 0.8)
    high = p_true > 0.8

    # Spherical score for low confidence tokens
    if torch.any(low):
        score[low] = 1 - p_true[low] / torch.linalg.norm(probs[low], dim=-1)
    # Brier score for mid confidence tokens
    if torch.any(mid):
        score[mid] = (probs[mid] - gt[mid]).pow(2).sum(-1)
    # Log score for high confidence tokens
    if torch.any(high):
        score[high] = -torch.log(p_true[high] + 1e-8)

    return score

def hydra_loss(logits, labels, edit_mask, risk_mask, freq_bucket, position, gate: MetaGate, delta=0.5):
    """Computes the Hybrid Dynamic Risk-Adjusted (HYDRA) loss."""
    B, S, V = logits.shape
    # Shift logits and labels for next-token prediction
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    edit_mask = edit_mask[:, 1:]
    risk_mask = risk_mask[:, 1:]
    freq_bucket = freq_bucket[:, 1:]
    position = position[:, 1:]
    
    # Reshape for cross_entropy
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    
    # Ignore padding tokens in loss calculation
    mask = labels != -100
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    probs = F.softmax(logits, -1)

    # 1. Accuracy Term (Cross-Entropy)
    ce = F.cross_entropy(logits_flat, labels_flat, reduction='none').view(B, S - 1)

    # 2. Calibration Term (Proper Score)
    ps = proper_score(logits, labels)

    # 3. Edit-Contrast Term
    log_probs_flat = F.log_softmax(logits, -1)
    gathered_log_probs = torch.gather(log_probs_flat, -1, labels.unsqueeze(-1)).squeeze(-1)
    ec = -edit_mask * gathered_log_probs

    # 4. Risk Term
    risk = risk_mask * probs.gather(-1, risk_mask.argmax(-1, keepdim=True)).squeeze(-1)

    # Features for the meta-gate
    entropy = (-probs * (probs.log() + 1e-8)).sum(-1) / torch.log(torch.tensor(V, dtype=torch.float32))
    risk_mean_per_token = risk.mean(-1) # This seems incorrect based on description, but following code structure.
                                     # Let's assume risk is a scalar per token, not a vector.
    
    z = torch.stack([entropy, position, freq_bucket, risk_mean_per_token], dim=-1) # (B, S-1, 4)
    g = gate(z)  # (B, S-1)

    # Combine terms using the gate
    total = g * (ce + ps) + (1 - g) * (ec + delta * risk_mean_per_token)
    
    # Apply mask and normalize
    total_masked = total[mask]
    return total_masked.sum() / mask.sum()


# Custom Trainer to handle HYDRA loss arguments
class HydraTrainer(SFTTrainer):
    def __init__(self, gate, hydra_delta, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate.to(self.args.device) # Ensure gate is on the correct device
        self.hydra_delta = hydra_delta

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        edit_mask = inputs.pop("edit_mask")
        risk_mask = inputs.pop("risk_mask")
        freq_bucket = inputs.pop("freq_bucket")
        position = inputs.pop("position")

        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        loss = hydra_loss(
            logits, labels, edit_mask, risk_mask, freq_bucket, position,
            self.gate, delta=self.hydra_delta
        )
        return (loss, outputs) if return_outputs else loss

def create_and_train_model(config, train_dataset, eval_dataset, tokenizer):
    """Orchestrates model loading, configuration, and training."""
    token = os.getenv("HF_TOKEN")

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            quantization_config=quantization_config,
            device_map="auto",
            use_auth_token=token
        )
        model.config.use_cache = False
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # LoRA config
    lora_conf = config["lora_config"]
    peft_config = LoraConfig(
        r=lora_conf["r"],
        lora_alpha=lora_conf["lora_alpha"],
        target_modules=lora_conf["target_modules"],
        lora_dropout=lora_conf["lora_dropout"],
        bias=lora_conf["bias"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # MetaGate for HYDRA loss
    gate = MetaGate(in_dim=4)

    # Training arguments
    training_args_conf = config["training_args"]
    args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=training_args_conf["num_train_epochs"],
        per_device_train_batch_size=training_args_conf["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args_conf["gradient_accumulation_steps"],
        learning_rate=training_args_conf["learning_rate"],
        logging_steps=training_args_conf["logging_steps"],
        bf16=training_args_conf.get("bf16", False),
        report_to=training_args_conf["report_to"],
        max_steps=training_args_conf.get("max_steps", -1),
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )

    # Initialize Trainer
    trainer = HydraTrainer(
        model=model,
        gate=gate,
        hydra_delta=config["hydra_loss_delta"],
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=config["dataset_config"]["max_seq_length"],
        dataset_text_field="text", # SFTTrainer needs to know which field to use
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the final model
    final_model_path = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")

    return model, gate
