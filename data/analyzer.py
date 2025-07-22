# data/analyzer.py
import pandas as pd

class DataAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def analyze(self):
        print("Analyzing dataset sample to gather statistics...")
        try:
            # Analyze a sample for speed, especially for large files
            df_sample = pd.read_csv(self.dataset_path, nrows=10000)
            
            # Get total row count
            row_count = sum(1 for row in open(self.dataset_path, 'r')) - 1
            
            if 'text' not in df_sample.columns:
                raise ValueError("Dataset must contain a 'text' column.")
            
            text_lengths = df_sample['text'].astype(str).str.len()
            
            dataset_info = {
                "row_count": row_count,
                "text_stats": {
                    "avg_length": text_lengths.mean(),
                    "max_length": text_lengths.max(),
                    "99th_percentile_length": text_lengths.quantile(0.99)
                }
            }
            print(f"Dataset analysis complete. Rows: {row_count}, Avg text length: {dataset_info['text_stats']['avg_length']:.2f}")
            return dataset_info
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            raise