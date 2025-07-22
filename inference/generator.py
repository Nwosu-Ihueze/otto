# inference/generator.py
import torch
import tiktoken
import json
import os
from models.gpt import GPT
from utils.common import GPTConfig
from utils.config_utils import create_trial_config

class Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, output_dir):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load best trial info
        best_trial_path = os.path.join(output_dir, "best_trial.json")
        with open(best_trial_path, 'r') as f:
            best_trial_info = json.load(f)
        
        # 2. Load base config and create the best trial config
        from utils.config_utils import load_config # Local import
        base_config = load_config('configs/base_config.yaml')
        metadata_path = os.path.join(output_dir, 'processed_data/metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        base_config.model.params.vocab_size = metadata['vocab_size']
        
        best_config = create_trial_config(base_config, best_trial_info['params'])
        
        # 3. Recreate model with the best hyperparameters
        gpt_config = GPTConfig(**best_config.model.params)
        model = GPT(gpt_config)
        
        # 4. Load the saved weights
        checkpoint_path = os.path.join(output_dir, f"trial_{best_trial_info['trial_number']}_best.pt")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        
        tokenizer = tiktoken.get_encoding(best_config.data.tokenizer_type)
        
        print(f"Loaded best model from trial {best_trial_info['trial_number']} with validation loss {best_trial_info['value (val_loss)']:.4f}")
        return cls(model, tokenizer, device)

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
        start_ids = self.tokenizer.encode_ordinary(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        generated_text = self.tokenizer.decode(y[0].tolist())
        return generated_text