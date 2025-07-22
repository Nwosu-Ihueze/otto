# data/processor.py
import numpy as np
from datasets import load_dataset
import os
import json
from tqdm import tqdm
from models.registry import tokenizer_registry

class DataProcessor:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.processed_data_dir = os.path.join(output_dir, "processed_data")
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _determine_loader_path(self, dataset_path: str) -> str:
        """Determines the correct path or identifier to pass to `load_dataset`."""
        if dataset_path.startswith("kaggle::"):
            # Format is 'kaggle::user/dataset-name'
            print(f"Detected Kaggle dataset: {dataset_path.replace('kaggle::', '')}")
            return dataset_path.replace('kaggle::', '')
        
        if os.path.exists(dataset_path):

            print(f"Detected local path: {dataset_path}")
            return dataset_path
        
        
        print(f"Assuming Hugging Face Hub dataset: {dataset_path}")
        return dataset_path

    def process_and_serialize(self, dataset_path, target_column='text'):
        print("Starting data processing and serialization...")
        
        try:
           
            loader_path = self._determine_loader_path(dataset_path)
            
           
            if os.path.isfile(loader_path):
                raw_dataset = load_dataset(
                    os.path.splitext(loader_path)[1].strip('.'), 
                    data_files={'train': loader_path, 'validation': loader_path}
                )
            else:
                raw_dataset = load_dataset(loader_path)

        except Exception as e:
            print(f"Failed to load dataset from '{dataset_path}'. Please check the path and format.")
            print(f"Error: {e}")
            raise

        # Ensure a validation split exists.
        if 'validation' not in raw_dataset or len(raw_dataset['validation']) == 0:
            print("No validation split found or validation split is empty. Creating one from the training set (90/10 split).")
            if len(raw_dataset['train']) < 2:
                raise ValueError("Training data must have at least 2 samples to create a validation split.")
            
            split_dataset = raw_dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)
            raw_dataset['train'] = split_dataset['train']
            raw_dataset['validation'] = split_dataset['test']

        tokenizer_builder = tokenizer_registry.get(self.config.data.tokenizer_type)
        tokenizer = tokenizer_builder(self.config)
        
        # Find the actual column names from the loaded dataset
        try:
            column_names = raw_dataset['train'].column_names
            if target_column not in column_names:
                raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {column_names}")
        except Exception as e:
             print(f"Could not determine column names from dataset. Error: {e}")
             raise

        def tokenize_map_function(example):
            if target_column not in example or example[target_column] is None:
                return {'ids': [], 'len': 0}
            
            text = str(example[target_column])
            token_ids = tokenizer.encode_ordinary(text)
            token_ids.append(tokenizer.eot_token)
            return {'ids': token_ids, 'len': len(token_ids)}
        
        print("Tokenizing dataset...")
        tokenized_dataset = raw_dataset.map(
            tokenize_map_function,
            remove_columns=column_names,
            num_proc=os.cpu_count(),
            desc="Tokenizing splits"
        )
        
        for split_name, dset in tokenized_dataset.items():
            if split_name not in ['train', 'validation']: continue 

            file_path = os.path.join(self.processed_data_dir, f"{split_name}.bin")
            dset = dset.filter(lambda x: x['len'] > 1, num_proc=os.cpu_count()) 
            if len(dset) == 0:
                print(f"Warning: Split '{split_name}' is empty after tokenization and filtering. A model cannot be trained.")
                continue

            total_tokens = np.sum(dset['len'], dtype=np.uint64)
            memmap_array = np.memmap(file_path, dtype=np.uint16, mode='w+', shape=(total_tokens,))
            
            write_pointer = 0
            batch_size = 1000 
            total_batches = (len(dset) + batch_size - 1) // batch_size
            
            print(f"Serializing {split_name} split to {file_path}...")
            
            for batch in tqdm(dset.iter(batch_size=batch_size), total=total_batches, desc=f"Writing {split_name}.bin"):
                id_array = np.concatenate(batch['ids'])
                memmap_array[write_pointer : write_pointer + len(id_array)] = id_array
                write_pointer += len(id_array)
            memmap_array.flush()
            print(f"Serialized {split_name} split with {total_tokens} tokens.")

        metadata = {'vocab_size': tokenizer.n_vocab}
        metadata_path = os.path.join(self.processed_data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {
            'data_paths': {
                'train': os.path.join(self.processed_data_dir, 'train.bin'),
                'val': os.path.join(self.processed_data_dir, 'validation.bin')
            },
            'metadata': metadata
        }