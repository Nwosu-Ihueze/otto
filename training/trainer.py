# training/trainer.py
import torch
import os
import optuna 
from torch.amp import GradScaler, autocast
from models.registry import model_registry, optimizer_registry, scheduler_registry
from utils.common import GPTConfig
from data.loader import TrainingDataLoader
from tqdm import tqdm

class TrialTrainer:
    def __init__(self, trial_config, data_info, trial_num, output_dir):
        self.config = trial_config
        self.data_info = data_info
        self.trial_num = trial_num
        self.output_dir = output_dir
        
        self.device = self.config.trainer.device if torch.cuda.is_available() else 'cpu'
        
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.trainer.dtype]
        self.ctx = torch.autocast(device_type=self.device.split(':')[0], dtype=ptdtype)
        self.scaler = GradScaler(enabled=(self.config.trainer.dtype == 'float16' and self.device != 'cpu'))

    def train(self, trial: optuna.Trial): 
        print(f"\n--- Starting Trial {self.trial_num} ---")
        print(f"Hyperparameters: {self.config.model.params} | Batch Size: {self.config.trainer.batch_size} | LR: {self.config.optimizer.params.learning_rate:.2e}")
        
        torch.manual_seed(1337 + self.trial_num)


        self.config.model.params.vocab_size = self.data_info['metadata']['vocab_size']
        gpt_config = GPTConfig(**self.config.model.params)
        
        model_builder = model_registry.get(self.config.model.type)
        model = model_builder(gpt_config).to(self.device)
        
        optimizer_builder = optimizer_registry.get(self.config.optimizer.type)
        static_optimizer_params = self.config.optimizer.params.copy()
        learning_rate = static_optimizer_params.pop('learning_rate')
        optimizer = optimizer_builder(model.parameters(), lr=learning_rate, **static_optimizer_params)

        scheduler_builder = scheduler_registry.get(self.config.scheduler.type)
        scheduler_params = {'max_iters': self.config.trainer.max_iters, **self.config.scheduler.params}
        scheduler = scheduler_builder(optimizer, **scheduler_params)
        
        train_loader = TrainingDataLoader(self.data_info['data_paths']['train'], self.config.trainer.batch_size, gpt_config.block_size, self.device)
        val_loader = TrainingDataLoader(self.data_info['data_paths']['val'], self.config.trainer.batch_size, gpt_config.block_size, self.device)
        
        best_val_loss = float('inf')
        
        pbar = tqdm(range(self.config.trainer.max_iters), desc=f"Trial {self.trial_num}")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(self.config.trainer.gradient_accumulation_steps):
                X, Y = train_loader.get_batch()
                with self.ctx:
                    _, loss = model(X, targets=Y)
                    loss = loss / self.config.trainer.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            
            
            if step > 0 and step % self.config.trainer.eval_interval == 0:
                losses = self._estimate_loss(model, val_loader)
                val_loss = losses['val']
                pbar.set_postfix({'val_loss': f"{val_loss:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
                
               
                trial.report(val_loss, step)
                if trial.should_prune():
                    pbar.close()
                    print(f"--- Trial {self.trial_num} Pruned at step {step} ---")
                    raise optuna.exceptions.TrialPruned()
                
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.output_dir, f"trial_{self.trial_num}_best.pt")
                    torch.save(model.state_dict(), checkpoint_path)
        
        pbar.close()
        print(f"--- Trial {self.trial_num} Finished --- Best Val Loss: {best_val_loss:.4f}")
        return best_val_loss

    @torch.no_grad()
    def _estimate_loss(self, model, val_loader):
        model.eval()
        out = {}
        losses = torch.zeros(self.config.trainer.eval_iters)
        for k in range(self.config.trainer.eval_iters):
            X, Y = val_loader.get_batch()
            with self.ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out['val'] = losses.mean().item()
        model.train()
        return out