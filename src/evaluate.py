import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams
from sklearn.calibration import calibration_curve

def calculate_ece(probs, labels, n_bins=15):
    """Calculates Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_model(model, tokenizer, config):
    """Runs a comprehensive evaluation of the fine-tuned model."""
    print("\n--- Starting Evaluation ---")
    model.eval()
    device = next(model.parameters()).device

    # For this example, we generate text from a few prompts and calculate metrics.
    # A full evaluation would use specific datasets like MMLU, TruthfulQA, etc.
    prompts = [
        "What is the capital of France?",
        "Write a short story about a robot who discovers music.",
        "Explain the theory of relativity in simple terms."
    ]
    
    all_tokens = []
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=True, top_p=0.9, temperature=0.7
            )
        decoded_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        outputs.append(decoded_output)
        all_tokens.extend(generation_output[0].cpu().numpy())

    print("\nGenerated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Output {i+1}: {output}\n")

    # Metric 1: Diversity (Unique 4-grams)
    four_grams = list(ngrams(all_tokens, 4))
    unique_4_grams_ratio = len(set(four_grams)) / len(four_grams) if len(four_grams) > 0 else 0
    
    # Metric 2: ECE (mock calculation)
    # In a real scenario, this would run on a dataset like MMLU.
    mock_probs = np.random.rand(1000)
    mock_labels = (mock_probs > np.random.rand(1000)).astype(int)
    ece_score = calculate_ece(mock_probs, mock_labels)
    
    # Metric 3: Toxicity (mock calculation)
    # A real implementation would use a classifier like Detoxify.
    mock_toxicity_score = np.random.uniform(0.01, 0.05)

    results = {
        "run_name": config["run_name"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "ece_mock": round(ece_score, 4),
            "toxicity_mock": round(mock_toxicity_score, 4),
            "unique_4_grams_ratio": round(unique_4_grams_ratio, 4)
        },
        "generated_samples": outputs
    }

    # Print final results to stdout
    print("\n--- Evaluation Results ---")
    results_json_str = json.dumps(results, indent=2)
    print(results_json_str)

    # Save results to JSON file
    output_dir = config['output_dir']
    results_path = os.path.join(output_dir, f"evaluation_results_{config['run_name']}.json")
    try:
        with open(results_path, 'w') as f:
            f.write(results_json_str)
        print(f"\nResults saved to {results_path}")
    except IOError as e:
        print(f"Error saving results to file: {e}")

    # Generate and save a calibration plot
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    try:
        prob_true, prob_pred = calibration_curve(mock_labels, mock_probs, n_bins=10)
        plt.figure(figsize=(8, 8))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Mock Data)")
        plt.legend()
        plot_path = os.path.join(image_dir, f"calibration_curve_{config['run_name']}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Calibration plot saved to {plot_path}")
    except Exception as e:
        print(f"Error generating plot: {e}")

    return results
