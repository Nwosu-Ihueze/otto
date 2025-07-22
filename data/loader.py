# data/loader.py
import numpy as np
import torch

class TrainingDataLoader:
    def __init__(self, split_file_path, batch_size, block_size, device):
        self.data = np.memmap(split_file_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

    def get_batch(self):
        start_indices = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        
        x_sequences = [torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in start_indices]
        y_sequences = [torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in start_indices]
        
        X = torch.stack(x_sequences)
        Y = torch.stack(y_sequences)

        if self.device_type == 'cuda':
            return X.pin_memory().to(self.device, non_blocking=True), Y.pin_memory().to(self.device, non_blocking=True)
        else:
            return X.to(self.device), Y.to(self.device)