# cli.py
import argparse
from main import run_automl_pipeline

def main():
    parser = argparse.ArgumentParser(description="AutoML platform for Small Language Models.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the training dataset (CSV).")
    parser.add_argument("--target", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Time limit for HPO in seconds (currently informational).")
    parser.add_argument("--output_dir", type=str, default="automl_results", help="Directory to save results.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of HPO trials to run.")
    
    args = parser.parse_args()
    
    run_automl_pipeline(
        dataset_path=args.dataset,
        target_column=args.target,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        n_trials=args.n_trials
    )

if __name__ == "__main__":
    main()