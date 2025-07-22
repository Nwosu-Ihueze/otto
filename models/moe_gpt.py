# models/moe_gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from .blocks import LayerNorm, CausalSelfAttention
from utils.common import GPTConfig

class Expert(nn.Module):
    """A simple MLP expert in a Mixture of Experts layer."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopKGating(nn.Module):
    """Gating network that selects the top-k experts."""
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, num_experts, bias=False)
        self.w_noise = nn.Linear(n_embd, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        clean_logits = self.w_gate(x)
        if self.training:
            raw_noise = self.w_noise(x)
            noise = torch.randn_like(raw_noise) * F.softplus(raw_noise)
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits

        top_k_logits, indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        
        
        gating_scores = F.softmax(top_k_logits, dim=-1)
        
        return indices, gating_scores

class MoEBlock(nn.Module):
    """A Mixture of Experts block."""
    def __init__(self, config: GPTConfig, num_experts=8, top_k=2):
        super().__init__()
        self.gating = NoisyTopKGating(config.n_embd, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(config) for _ in range(num_experts)])
        self.num_experts = num_experts

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C)  

        expert_indices, gating_scores = self.gating(x)
        
        final_output = torch.zeros_like(x)
        
        flat_expert_indices = expert_indices.view(-1)
        flat_x = x.repeat_interleave(self.gating.top_k, dim=0)
        
        
        expert_batch_index = torch.arange(len(x) * self.gating.top_k, device=x.device) // self.gating.top_k
        
        
        for i in range(self.num_experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                tokens_for_expert = flat_x[mask]
                expert_output = self.experts[i](tokens_for_expert)
                
                scores_for_expert = gating_scores.view(-1)[mask]
                weighted_output = expert_output * scores_for_expert.unsqueeze(-1)
                
                final_output.index_add_(0, expert_batch_index[mask], weighted_output)

        return final_output.view(B, T, C)

class MoETransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig, num_experts=8, top_k=2):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoEBlock(config, num_experts=num_experts, top_k=top_k)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x

class MoEGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([MoETransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx