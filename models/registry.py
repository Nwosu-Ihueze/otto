# models/registry.py
import tiktoken
from transformers import AutoTokenizer

class Registry:
    """A generic registry to map strings to classes or functions."""
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, key):
        def decorator(func_or_class):
            if key in self._registry:
                raise ValueError(f"Key '{key}' already registered in '{self._name}'.")
            self._registry[key] = func_or_class
            return func_or_class
        return decorator

    def get(self, key):
        if key not in self._registry:
            raise KeyError(f"Key '{key}' not found in '{self._name}' registry. Available keys: {list(self._registry.keys())}")
        return self._registry[key]


model_registry = Registry("models")
optimizer_registry = Registry("optimizers")
scheduler_registry = Registry("schedulers")
tokenizer_registry = Registry("tokenizers")


from .gpt import GPT
from .moe_gpt import MoEGPT  
import torch

@model_registry.register("GPT")
def get_gpt_model(config):
    return GPT(config)

@model_registry.register("MoE-GPT")
def get_moe_gpt_model(config):
    return MoEGPT(config)

@optimizer_registry.register("AdamW")
def get_adamw_optimizer(params, lr, weight_decay, beta1, beta2, eps):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)


@optimizer_registry.register("SGD")
def get_sgd_optimizer(params, lr, momentum=0.9):
    return torch.optim.SGD(params, lr=lr, momentum=momentum)

@scheduler_registry.register("CosineDecayWithWarmup")
def get_cosine_scheduler(optimizer, warmup_iters, max_iters, min_lr):
    from torch.optim.lr_scheduler import LambdaLR
    import math


    initial_lr = optimizer.defaults['lr']

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        
        progress = float(current_step - warmup_iters) / float(max(1, max_iters - warmup_iters))
        

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        

        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay
        

        return decayed_lr / initial_lr

    return LambdaLR(optimizer, lr_lambda)


@tokenizer_registry.register("tiktoken_gpt2")
def get_tiktoken_tokenizer(config):

    return tiktoken.get_encoding("gpt2")

@tokenizer_registry.register("hf_distilbert")
def get_hf_tokenizer(config):

    class HFTokenizerWrapper:
        def __init__(self, model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.eot_token = self.tokenizer.sep_token_id
            self.n_vocab = self.tokenizer.vocab_size

        def encode_ordinary(self, text):

            return self.tokenizer.encode(text, add_special_tokens=False)
        
        def decode(self, tokens):
            return self.tokenizer.decode(tokens)

    return HFTokenizerWrapper("distilbert-base-uncased")