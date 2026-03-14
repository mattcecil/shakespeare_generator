import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for a training run. d_k and d_v are derived from d_model and H."""
    B: int = 32         # batch size
    n: int = 256        # sequence length (context window)
    d_model: int = 384  # embedding / hidden dimension
    d_ff: int = 1536    # feed-forward inner dimension (typically 4 × d_model)
    H: int = 8          # number of attention heads
    epochs: int = 100
    layers: int = 6
    lr: float = 3e-4
    drop_prob: float = 0.1
    vocab_size: int = 1000  # 128 = char-level; >128 = BPE

    def __post_init__(self):
        self.d_k = self.d_model // self.H
        self.d_v = self.d_k


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE positional encoding.

    Uses F.scaled_dot_product_attention (Flash Attention on CUDA) for the inner
    QKV computation. RoPE is applied to Q and K before the attention call.
    Dropout is passed directly to scaled_dot_product_attention rather than
    applied as a separate layer.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Wq = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wk = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wv = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model)

        thetas = 1 / 10000 ** (torch.arange(0, cfg.d_k, 2) / cfg.d_k)
        M = torch.arange(cfg.n)
        angles = torch.outer(M, thetas)  # (n, d_k/2)
        self.register_buffer('sin', angles.sin())
        self.register_buffer('cos', angles.cos())

    def _apply_rope(self, X):
        """Apply Rotary Position Embedding to X ~ (B, H, n, d_k)."""
        n = X.shape[-2]
        x1 = X[..., 0::2]
        x2 = X[..., 1::2]
        x1r = x1 * self.cos[:n] - x2 * self.sin[:n]
        x2r = x1 * self.sin[:n] + x2 * self.cos[:n]
        Xr = torch.stack([x1r, x2r], dim=-1)
        Xr = Xr.flatten(-2)
        return Xr

    def forward(self, X):
        cfg = self.cfg
        B, n, _ = X.shape  # use actual batch size and seq len, not cfg values

        Q = self.Wq(X).view(B, n, cfg.H, cfg.d_k).transpose(-3, -2)
        K = self.Wk(X).view(B, n, cfg.H, cfg.d_k).transpose(-3, -2)
        V = self.Wv(X).view(B, n, cfg.H, cfg.d_v).transpose(-3, -2)

        Q = self._apply_rope(Q)
        K = self._apply_rope(K)

        O = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
                                           dropout_p=cfg.drop_prob if self.training else 0.0)
        O = O.transpose(-3, -2).reshape(B, n, cfg.H * cfg.d_v)
        return self.Wo(O)


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm attention sub-layer + pre-norm FFN sub-layer."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.drop_prob),
        )

    def forward(self, X):
        X = X + self.attn(self.norm1(X))
        X = X + self.ff(self.norm2(X))
        return X


class Transformer(nn.Module):
    """Decoder-only transformer for character/BPE-level language modelling.

    forward() takes a (B, n) token tensor and returns (logits, targets) where
    targets is the input shifted by one position — ready for cross_entropy.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, Xin):
        """Returns (logits, targets). logits shape: (B, n-1, vocab_size)."""
        X = self.embedding(Xin)
        for block in self.blocks:
            X = block(X)
        X = self.output_norm(X)
        X = self.lm_head(X)
        logits = X[:, :-1, :]
        targets = Xin[:, 1:]
        return logits, targets

    def generate(self, prompt_ids, max_new_tokens=200, temperature=1.0, top_k=None):
        """Autoregressively sample max_new_tokens tokens given a list of prompt token IDs."""
        device = next(self.parameters()).device
        X = torch.tensor(prompt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                X_crop = X[:, -self.cfg.n:] if X.shape[1] > self.cfg.n else X
                h = self.embedding(X_crop)
                for block in self.blocks:
                    h = block(h)
                h = self.output_norm(h)
                logits = self.lm_head(h)[0, -1, :]
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        threshold = torch.topk(logits, top_k).values[-1]
                        logits[logits < threshold] = float('-inf')
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                X = torch.cat([X, next_token.unsqueeze(0)], dim=1)

        return X[0, len(prompt_ids):].tolist()
