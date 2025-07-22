# AutoML Platform for Small Language Models

This project is a complete, sophisticated AutoML platform for training and fine-tuning small language models (SLMs) based on a GPT-style architecture. It automates the entire pipeline from data processing to hyperparameter optimization and inference.

## Features

- **Automated Configuration**: Analyzes your dataset to generate a robust starting configuration.
- **Resource-Aware Tuning**: Intelligently adjusts the hyperparameter search space to fit within your available GPU memory, preventing OOM errors.
- **Efficient Data Handling**: Uses memory-mapped binary files (`.bin`) to handle datasets much larger than RAM, ensuring high I/O performance during training.
- **Hyperparameter Optimization**: Leverages Optuna to efficiently search for the best model architecture and training parameters.
- **Optimized Training Loop**: Incorporates modern training techniques like Automatic Mixed Precision (AMP), gradient accumulation, and Flash Attention.
- **Modular and Extensible**: The code is organized into logical components for data, models, training, and inference, making it easy to understand and extend.

## Directory Structure

- `configs/`: Default configurations and hyperparameter search spaces.
- `data/`: Data analysis, processing, and loading modules.
- `models/`: The GPT model architecture and its building blocks.
- `training/`: Core logic for the training loop, HPO, and resource management.
- `inference/`: Tools for generating text with a trained model.
- `utils/`: Shared utilities for configuration and common data structures.
- `main.py`: The main orchestrator script for the AutoML pipeline.
- `cli.py`: A command-line interface to run the platform.

## Setup

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The platform is designed to be run from the command line. You need a CSV file with a text column (default name is 'text').

### **Example Usage**

1.  **Create a sample dataset `my_stories.csv`:**

    ```csv
    id,text
    1,"Once upon a time, in a land far away, a brave knight rode a magnificent dragon."
    2,"The wizard waved his wand, and shimmering stars filled the dark night sky."
    3,"A tiny mouse peeked out from its hole, sniffing the air for the scent of cheese."
    ...
    ```

2.  **Run the AutoML pipeline:**

    ```bash
    python cli.py --dataset ./my_stories.csv --output_dir ./automl_results
    ```

### **Command-Line Arguments**

-   `--dataset`: (Required) Path to your training dataset (CSV format).
-   `--target`: (Optional) The name of the text column in your CSV. Defaults to `text`.
-   `--time_limit`: (Optional) The total time limit for hyperparameter optimization in seconds. Defaults to 3600 (1 hour).
-   `--output_dir`: (Optional) The directory where all results, checkpoints, and processed data will be saved. Defaults to `automl_results`.
-   `--n_trials`: (Optional) The number of hyperparameter optimization trials to run. Defaults to 20.

After the run completes, the best model checkpoint and configuration will be saved in the specified output directory.