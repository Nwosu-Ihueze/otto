# main.py
import os
from data.analyzer import DataAnalyzer
from data.processor import DataProcessor
from training.resource_tuner import ResourceTuner
from training.hyperopt_manager import HyperoptManager
from inference.generator import Generator
from utils.config_utils import load_config, merge_configs

def run_automl_pipeline(dataset_path, target_column, time_limit, output_dir, n_trials):
    print("===== AutoML SLM Pipeline Started =====")
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: Configuration & Setup ---
    print("\n[Phase 1/4] Analyzing Data and Configuring...")
    analyzer = DataAnalyzer(dataset_path)
    dataset_info = analyzer.analyze()

    base_config = load_config('configs/base_config.yaml')
    search_space_config = load_config('configs/search_spaces.yaml')

    if isinstance(base_config.model.params.block_size, str):
        print("Resolving config variable: model.params.block_size -> data.block_size")
        base_config.model.params.block_size = base_config.data.block_size

    # --- ENHANCEMENT: block_size validation ---
    assert base_config.data.block_size == base_config.model.params.block_size, \
        f"Configuration Error: data.block_size ({base_config.data.block_size}) must match " \
        f"model.params.block_size ({base_config.model.params.block_size})."
    print("Configuration validation passed.")

    initial_config = merge_configs(base_config, search_space_config)

    # --- Phase 2: Data Preprocessing ---
    print("\n[Phase 2/4] Processing and Serializing Data...")
    data_processor = DataProcessor(initial_config, output_dir)
    processed_data_info = data_processor.process_and_serialize(dataset_path, target_column)
    
    
    initial_config.model.params.vocab_size = processed_data_info['metadata']['vocab_size']

    print("\n[Pre-HPO] Tuning Search Space for Available Memory...")
    resource_tuner = ResourceTuner()
    tuned_config = resource_tuner.tune_for_memory(initial_config)

    # --- Phase 3: Hyperparameter Optimization ---
    print("\n[Phase 3/4] Running Hyperparameter Optimization...")
    hyperopt_manager = HyperoptManager(tuned_config, processed_data_info, output_dir, n_trials)
    best_trial = hyperopt_manager.run()
    
    # --- Phase 4: Finalization and Inference Example ---
    print("\n[Phase 4/4] Finalizing and Demonstrating Inference...")
    if best_trial is None:
        print("Hyperparameter optimization did not produce a best trial. Exiting.")
        return
    
    print(f"Best trial number: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("\nLoading best model for an inference example...")
    generator = Generator.from_checkpoint(output_dir)
    prompt = "Once upon a time"
    generated_text = generator.generate(prompt=prompt, max_new_tokens=50)
    print(f"\n--- Example Generation ---")
    print(f"Prompt: '{prompt}'")
    print(f"Generated Text: '{generated_text}'")
    
    print("\n===== AutoML SLM Pipeline Finished =====")
