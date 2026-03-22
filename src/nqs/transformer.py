"""
Transformer-based Neural Quantum State for second quantization.

Inspired by Psiformer (von Glehn et al., 2023), adapted for discrete
occupation-number basis. Uses causal self-attention to autoregressively
model orbital correlations:

    P(σ) = Π_i P(σ_i | σ_1, ..., σ_{i-1})

Each orbital "sees" what all previous orbitals decided, capturing
inter-orbital correlations that product-of-marginals architectures miss.

Key differences from the original Psiformer:
- Second quantization (binary occupation strings) instead of first quantization
- Autoregressive factorization over orbitals instead of electron coordinates
- Separate alpha/beta channels with cross-attention for spin coupling
- KV cache for O(n²) sampling instead of O(n³)

References:
- von Glehn et al. (2023) "A self-attention ansatz for ab-initio quantum chemistry"
- Sharir et al. (2020) "Deep autoregressive models for the efficient variational simulation"
- Barrett et al. (2022) "Autoregressive neural-network wavefunctions for ab initio quantum chemistry"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

try:
    from .base import NeuralQuantumState
except ImportError:
    from nqs.base import NeuralQuantumState


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention with optional KV cache."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Append to KV cache if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask (only needed for full-sequence forward, not cached single-step)
        if past_kv is None and mask is None:
            T_k = k.shape[2]
            mask = torch.triu(torch.ones(T, T_k, device=x.device), diagonal=1).bool()
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        if use_cache:
            return out, new_kv
        return out


class CrossAttention(nn.Module):
    """Multi-head cross-attention (beta attending to alpha) with optional KV cache."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        B, T_q, C = query.shape

        q = self.q_proj(query).reshape(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)

        if cached_kv is not None:
            k, v = cached_kv
        else:
            T_k = context.shape[1]
            kv = self.kv_proj(context).reshape(B, T_k, 2, self.n_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        cross_kv = (k, v) if use_cache else None

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
        out = self.out_proj(out)

        if use_cache:
            return out, cross_kv
        return out


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention + optional cross-attention + KV cache."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.0, has_cross_attn: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = CausalSelfAttention(embed_dim, n_heads, dropout)

        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.ln_cross = nn.LayerNorm(embed_dim)
            self.cross_attn = CrossAttention(embed_dim, n_heads, dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                past_self_kv=None, past_cross_kv=None, use_cache=False):

        # Self-attention (with KV cache)
        if use_cache:
            sa_out, new_self_kv = self.self_attn(
                self.ln1(x), mask, past_kv=past_self_kv, use_cache=True
            )
            x = x + sa_out
        else:
            x = x + self.self_attn(self.ln1(x), mask)
            new_self_kv = None

        # Cross-attention (with KV cache for context)
        new_cross_kv = None
        if self.has_cross_attn and context is not None:
            if use_cache:
                ca_out, new_cross_kv = self.cross_attn(
                    self.ln_cross(x), context, cached_kv=past_cross_kv, use_cache=True
                )
                x = x + ca_out
            else:
                x = x + self.cross_attn(self.ln_cross(x), context)

        # FFN
        x = x + self.ffn(self.ln2(x))

        if use_cache:
            return x, new_self_kv, new_cross_kv
        return x


class AutoregressiveTransformer(nn.Module):
    """
    Autoregressive transformer for generating occupation strings.

    Architecture:
    - Alpha channel: causal transformer over alpha orbitals
    - Beta channel: causal transformer over beta orbitals with
      cross-attention to alpha (sees full alpha configuration)

    Sampling:
    - KV-cached: O(n²) total attention (not O(n³))
    - Constrained: enforce exact particle number (n_alpha, n_beta)
    """

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_qubits = 2 * n_orbitals
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        if ffn_dim is None:
            ffn_dim = 4 * embed_dim

        self.occ_embedding = nn.Embedding(2, embed_dim)
        self.pos_embedding = nn.Embedding(n_orbitals, embed_dim)
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.alpha_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, has_cross_attn=False)
            for _ in range(n_layers)
        ])

        self.beta_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, has_cross_attn=True)
            for _ in range(n_layers)
        ])

        self.alpha_head = nn.Linear(embed_dim, 1)
        self.beta_head = nn.Linear(embed_dim, 1)

        self.alpha_ln = nn.LayerNorm(embed_dim)
        self.beta_ln = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)

    def _alpha_logits(self, alpha_config: torch.Tensor) -> torch.Tensor:
        """Compute logits for alpha orbitals (full-sequence, no cache)."""
        B, T = alpha_config.shape
        device = alpha_config.device

        occ_emb = self.occ_embedding(alpha_config.long())
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        start = self.start_token.expand(B, -1, -1)
        x = torch.cat([start, occ_emb[:, :-1, :]], dim=1) + pos_emb

        for block in self.alpha_blocks:
            x = block(x)

        logits = self.alpha_head(self.alpha_ln(x)).squeeze(-1)
        return logits

    def _beta_logits(self, beta_config: torch.Tensor,
                     alpha_context: torch.Tensor) -> torch.Tensor:
        """Compute logits for beta orbitals (full-sequence, no cache)."""
        B, T = beta_config.shape
        device = beta_config.device

        occ_emb = self.occ_embedding(beta_config.long())
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        start = self.start_token.expand(B, -1, -1)
        x = torch.cat([start, occ_emb[:, :-1, :]], dim=1) + pos_emb

        for block in self.beta_blocks:
            x = block(x, context=alpha_context)

        logits = self.beta_head(self.beta_ln(x)).squeeze(-1)
        return logits

    def _get_alpha_context(self, alpha_config: torch.Tensor) -> torch.Tensor:
        """Get alpha hidden states for beta cross-attention (bidirectional)."""
        B = alpha_config.shape[0]
        device = alpha_config.device

        occ_emb = self.occ_embedding(alpha_config.long())
        pos_idx = torch.arange(self.n_orbitals, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        x = occ_emb + pos_emb

        no_mask = torch.zeros(self.n_orbitals, self.n_orbitals, device=device).bool()
        for block in self.alpha_blocks:
            x = block(x, mask=no_mask)

        return x

    def log_prob(self, config: torch.Tensor) -> torch.Tensor:
        """Compute log P(config) via teacher forcing (full-sequence, no cache)."""
        alpha = config[:, :self.n_orbitals]
        beta = config[:, self.n_orbitals:]

        alpha_logits = self._alpha_logits(alpha)
        alpha_log_prob = -F.binary_cross_entropy_with_logits(
            alpha_logits, alpha.float(), reduction='none'
        ).sum(dim=-1)

        alpha_context = self._get_alpha_context(alpha)
        beta_logits = self._beta_logits(beta, alpha_context)
        beta_log_prob = -F.binary_cross_entropy_with_logits(
            beta_logits, beta.float(), reduction='none'
        ).sum(dim=-1)

        return alpha_log_prob + beta_log_prob

    @torch.no_grad()
    def sample(self, n_samples: int, hard: bool = True,
               temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        KV-cached autoregressive sampling with exact particle number.

        Each step processes only the new token, appending to cached K/V.
        Total attention: O(n_layers * n_orb²) instead of O(n_layers * n_orb³).
        """
        device = next(self.parameters()).device
        n_orb = self.n_orbitals
        B = n_samples

        # ── Alpha channel (KV-cached) ──
        alpha = torch.zeros(B, n_orb, device=device)
        alpha_log_prob = torch.zeros(B, device=device)

        # Initialize cache: one entry per layer
        alpha_kv_cache = [None] * self.n_layers

        for i in range(n_orb):
            # Build input for this step only
            if i == 0:
                x = self.start_token.expand(B, -1, -1)  # (B, 1, embed)
            else:
                prev_occ = alpha[:, i - 1].long().unsqueeze(1)  # (B, 1)
                x = self.occ_embedding(prev_occ)  # (B, 1, embed)

            x = x + self.pos_embedding.weight[i].unsqueeze(0).unsqueeze(0)

            # Forward through alpha blocks with KV cache
            for layer_idx, block in enumerate(self.alpha_blocks):
                x, new_self_kv, _ = block(
                    x, past_self_kv=alpha_kv_cache[layer_idx], use_cache=True
                )
                alpha_kv_cache[layer_idx] = new_self_kv

            logit_i = self.alpha_head(self.alpha_ln(x)).squeeze(-1).squeeze(-1)  # (B,)
            logit_i = logit_i / temperature

            # Particle conservation constraints
            placed = alpha[:, :i].sum(dim=1)
            remaining_slots = n_orb - i
            needed = self.n_alpha - placed

            must_place = (needed >= remaining_slots)
            cannot_place = (placed >= self.n_alpha)
            logit_i = torch.where(must_place, torch.tensor(10.0, device=device), logit_i)
            logit_i = torch.where(cannot_place, torch.tensor(-10.0, device=device), logit_i)

            prob_i = torch.sigmoid(logit_i)
            sample_i = torch.bernoulli(prob_i)
            alpha[:, i] = sample_i

            alpha_log_prob += torch.where(
                sample_i == 1, F.logsigmoid(logit_i), F.logsigmoid(-logit_i),
            )

        # ── Alpha context for beta cross-attention (computed once) ──
        alpha_context = self._get_alpha_context(alpha)

        # Pre-compute cross-attention KV from alpha_context (reused every beta step)
        cross_kv_cache = []
        for block in self.beta_blocks:
            if block.has_cross_attn:
                _, cross_kv = block.cross_attn(
                    # Dummy query just to extract KV from context
                    torch.zeros(B, 1, self.embed_dim, device=device),
                    alpha_context, use_cache=True,
                )
                cross_kv_cache.append(cross_kv)
            else:
                cross_kv_cache.append(None)

        # ── Beta channel (KV-cached, with cross-attention KV pre-computed) ──
        beta = torch.zeros(B, n_orb, device=device)
        beta_log_prob = torch.zeros(B, device=device)

        beta_self_kv_cache = [None] * self.n_layers

        for i in range(n_orb):
            if i == 0:
                x = self.start_token.expand(B, -1, -1)
            else:
                prev_occ = beta[:, i - 1].long().unsqueeze(1)
                x = self.occ_embedding(prev_occ)

            x = x + self.pos_embedding.weight[i].unsqueeze(0).unsqueeze(0)

            for layer_idx, block in enumerate(self.beta_blocks):
                x, new_self_kv, _ = block(
                    x, context=alpha_context,
                    past_self_kv=beta_self_kv_cache[layer_idx],
                    past_cross_kv=cross_kv_cache[layer_idx],
                    use_cache=True,
                )
                beta_self_kv_cache[layer_idx] = new_self_kv

            logit_i = self.beta_head(self.beta_ln(x)).squeeze(-1).squeeze(-1)
            logit_i = logit_i / temperature

            placed = beta[:, :i].sum(dim=1)
            remaining_slots = n_orb - i
            needed = self.n_beta - placed

            must_place = (needed >= remaining_slots)
            cannot_place = (placed >= self.n_beta)
            logit_i = torch.where(must_place, torch.tensor(10.0, device=device), logit_i)
            logit_i = torch.where(cannot_place, torch.tensor(-10.0, device=device), logit_i)

            prob_i = torch.sigmoid(logit_i)
            sample_i = torch.bernoulli(prob_i)
            beta[:, i] = sample_i

            beta_log_prob += torch.where(
                sample_i == 1, F.logsigmoid(logit_i), F.logsigmoid(-logit_i),
            )

        configs = torch.cat([alpha, beta], dim=1)
        log_probs = alpha_log_prob + beta_log_prob

        return configs, log_probs


class TransformerNQS(NeuralQuantumState):
    """
    Bidirectional Transformer NQS for log-amplitude estimation.
    (Not autoregressive — used as a separate network for energy evaluation.)
    """

    def __init__(self, n_orbitals: int, embed_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 4,
                 ffn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.n_orbitals = n_orbitals
        self.n_sites = 2 * n_orbitals

        if ffn_dim is None:
            ffn_dim = 4 * embed_dim

        self.occ_embedding = nn.Embedding(2, embed_dim)
        self.pos_embedding = nn.Embedding(2 * n_orbitals, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, has_cross_attn=False)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)

    def log_prob(self, config: torch.Tensor) -> torch.Tensor:
        B = config.shape[0]
        device = config.device

        occ_emb = self.occ_embedding(config.long())
        pos_idx = torch.arange(self.n_sites, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        x = occ_emb + pos_emb

        no_mask = torch.zeros(self.n_sites, self.n_sites, device=device).bool()
        for block in self.blocks:
            x = block(x, mask=no_mask)

        x = self.ln(x)
        log_amp = self.head(x).squeeze(-1).sum(dim=-1)
        return 2 * log_amp
