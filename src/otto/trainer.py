import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from dataclasses import dataclass
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from pathlib import Path
import logging
import json
import tiktoken

logger = logging.getLogger(__name__)


from otto.models.gpt import GPT, GPTConfig 


class SLMTrainer:
    """
    Small Language Model trainer that integrates with the preprocessing pipeline.
    
    Uses the exact training configuration and methodology from your training code,
    but integrates cleanly with the DocumentProcessor output.
    """
    
    def __init__(
        self,
        training_data_dir: str,
        model_config: GPTConfig = None,
        output_dir: str = "model_outputs",

        learning_rate: float = 1e-4,
        max_iters: int = 20000,
        warmup_steps: int = 1000,
        min_lr: float = 5e-4,
        eval_iters: int = 500,
        batch_size: int = 32,
        block_size: int = 128,
        gradient_accumulation_steps: int = 32,

        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,

        device: str = None,
        dtype: str = None
    ):
        """
        Initialize the SLM trainer.
        
        Args:
            training_data_dir: Directory containing train.bin and val.bin files
            model_config: GPTConfig object (if None, creates from parameters)
            output_dir: Directory to save model checkpoints and logs
            ... (other args match your training configuration)
        """
        self.training_data_dir = Path(training_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        

        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.block_size = block_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

        if dtype is None:
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        else:
            self.dtype = dtype
            
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.max_token_value + 1
        

        if model_config is None:
            self.model_config = GPTConfig(
                vocab_size=self.vocab_size,
                block_size=block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                dropout=dropout,
                bias=True
            )
        else:
            self.model_config = model_config
            self.model_config.vocab_size = self.vocab_size  
        

        self._verify_training_files()
        

        torch.manual_seed(42)
        torch.set_default_device(self.device)
        
        self.model = GPT(self.model_config).to(self.device)
        

        self.train_loss_list = []
        self.validation_loss_list = []
        self.best_val_loss = float('inf')
        self.best_model_path = self.output_dir / "best_model.pt"
        
        logger.info(f"Initialized SLM Trainer:")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Training device: {self.device}")
        logger.info(f"  Precision: {self.dtype}")
        logger.info(f"  Vocab size: {self.vocab_size}")
    
    def _verify_training_files(self):
        """Verify that required training files exist."""
        train_file = self.training_data_dir / "train.bin"
        val_file = self.training_data_dir / "val.bin"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        logger.info(f"Training files verified: {train_file}, {val_file}")
    
    def get_batch(self, split: str):
        """
        Get a batch of training data (matching your exact implementation).
        """
  
        if split == 'train':
            data = np.memmap(self.training_data_dir / 'train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap(self.training_data_dir / 'val.bin', dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
        
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y
    
    def estimate_loss(self):
        """
        Estimate loss on train and validation sets (matching your implementation).
        """
        out = {}
        self.model.eval()
        
        with torch.inference_mode():
            for split in ['train', 'val']:
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    X, Y = self.get_batch(split)
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        
        self.model.train()
        return out
    
    def setup_training(self):
        """Setup optimizer, schedulers, and scaler."""

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.95), 
            weight_decay=0.1, 
            eps=1e-9
        )
        

        scheduler_warmup = LinearLR(self.optimizer, total_iters=self.warmup_steps)
        scheduler_decay = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_iters - self.warmup_steps, 
            eta_min=self.min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[scheduler_warmup, scheduler_decay], 
            milestones=[self.warmup_steps]
        )
        

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
        
        logger.info("Training setup completed")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'model_config': self.model_config,
            'train_loss': self.train_loss_list,
            'val_loss': self.validation_loss_list,
            'best_val_loss': self.best_val_loss
        }
        

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        

        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self, save_every: int = 1000, log_every: int = 100):
        """
        Main training loop (matching your exact implementation).
        """
        logger.info("Starting training...")
        self.setup_training()
        

        for epoch in tqdm(range(self.max_iters), desc="Training"):

            if epoch % self.eval_iters == 0 and epoch != 0:
                losses = self.estimate_loss()

                print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.5f}")

                self.train_loss_list.append(losses['train'].item())
                self.validation_loss_list.append(losses['val'].item())
  
                is_best = losses['val'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = losses['val']

                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, is_best)
            

            X, y = self.get_batch("train")
            
            with self.ctx:
                logits, loss = self.model(X, y)
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
        
            if ((epoch + 1) % self.gradient_accumulation_steps == 0) or (epoch + 1 == self.max_iters):

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            self.scheduler.step()

            if epoch % log_every == 0 and epoch > 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}: lr={current_lr:.6f}")
        

        final_losses = self.estimate_loss()
        print(f"Final: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
        
        self.save_checkpoint(self.max_iters, final_losses['val'] < self.best_val_loss)
        
        self._save_training_history()
        
        logger.info("Training completed!")
        return {
            'final_train_loss': final_losses['train'].item(),
            'final_val_loss': final_losses['val'].item(),
            'best_val_loss': self.best_val_loss,
            'best_model_path': str(self.best_model_path)
        }
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': [float(x) for x in self.train_loss_list],
            'val_losses': [float(x) for x in self.validation_loss_list],
            'best_val_loss': float(self.best_val_loss),
            'config': {
                'learning_rate': self.learning_rate,
                'max_iters': self.max_iters,
                'warmup_steps': self.warmup_steps,
                'batch_size': self.batch_size,
                'block_size': self.block_size,
                'model_config': {
                    'n_layer': self.model_config.n_layer,
                    'n_head': self.model_config.n_head,
                    'n_embd': self.model_config.n_embd,
                    'vocab_size': self.model_config.vocab_size,
                    'dropout': self.model_config.dropout
                }
            }
        }
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.setup_training() 
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.train_loss_list = checkpoint.get('train_loss', [])
        self.validation_loss_list = checkpoint.get('val_loss', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate_text(self, prompt: str = "", max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = None):
        """Generate text using the trained model."""
        self.model.eval()
        

        if prompt:
            prompt_tokens = self.tokenizer.encode(prompt)
            idx = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:

            idx = torch.randint(0, self.vocab_size, (1, 1), device=self.device)
        

        with torch.no_grad():
            generated = self.model.generate(idx, max_new_tokens, temperature, top_k)
        

        generated_tokens = generated[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        self.model.train()
        return generated_text