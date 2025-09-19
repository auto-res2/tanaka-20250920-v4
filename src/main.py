import argparse
import yaml
import os
import torch
import json
import time
import numpy as np

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run HYDRA LLM fine-tuning experiment.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test using config/smoke_test.yaml.")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment using config/full_experiment.yaml.")
    args = parser.parse_args()

    if args.smoke_test:
        config_path = "config/smoke_test.yaml"
    else:
        config_path = "config/full_experiment.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Create output directories from config
    output_dir = config.get("output_dir", ".research/iteration1/default_results")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    config['output_dir'] = output_dir # Ensure full path is in config

    print("--- HYDRA Experiment Runner ---")
    print(f"Loading configuration from: {config_path}")
    print(yaml.dump(config, indent=2))

    # Check environment
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. Training will be extremely slow on CPU.")
    else:
        print(f"\nCUDA is available. Using device: {torch.cuda.get_device_name(0)}")

    # Dynamically import project modules to ensure they are found in the src package
    try:
        from .preprocess import load_and_prepare_data
        from .train import create_and_train_model
        from .evaluate import evaluate_model
    except ImportError as e:
        print(f"Error importing modules. Make sure you are running this as a module, e.g., 'python -m src.main'. Error: {e}")
        return

    try:
        # Step 1: Data Preprocessing
        print("\n--- Step 1: Loading and Preprocessing Data ---")
        start_time = time.time()
        train_ds, val_ds, tokenizer = load_and_prepare_data(config)
        print(f"Data preparation completed in {time.time() - start_time:.2f} seconds.")

        # Step 2: Model Training
        print("\n--- Step 2: Creating and Training Model ---")
        start_time = time.time()
        model, gate = create_and_train_model(config, train_ds, val_ds, tokenizer)
        print(f"Model training completed in {time.time() - start_time:.2f} seconds.")

        # Step 3: Model Evaluation
        print("\n--- Step 3: Evaluating Model ---")
        start_time = time.time()
        evaluate_model(model, tokenizer, config)
        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

        print("\n--- Experiment Finished Successfully ---")

    except Exception as e:
        print(f"\n--- An error occurred during the experiment: {e} ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
