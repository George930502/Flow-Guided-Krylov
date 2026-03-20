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

References:
- von Glehn et al. (2023) "A self-attention ansatz for ab-initio quantum chemistry" (Psiformer)
- Sharir et al. (2020) "Deep autoregressive models for the efficient variational simulation"
- Barrett et al. (2022) "Autoregressive neural-network wavefunctions for ab initio quantum chemistry"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .base import NeuralQuantumState
except ImportError:
    from nqs.base import NeuralQuantumState


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask: prevent attending to future positions
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention (beta attending to alpha)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T_q, C = query.shape
        T_k = context.shape[1]

        q = self.q_proj(query).reshape(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, T_k, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention + optional cross-attention."""

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
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.self_attn(self.ln1(x), mask)

        # Cross-attention with residual (beta attending to alpha)
        if self.has_cross_attn and context is not None:
            x = x + self.cross_attn(self.ln_cross(x), context)

        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class AutoregressiveTransformer(nn.Module):
    """
    Autoregressive transformer for generating occupation strings.

    Architecture:
    - Alpha channel: causal transformer over alpha orbitals
    - Beta channel: causal transformer over beta orbitals with
      cross-attention to alpha (sees full alpha configuration)

    Sampling:
    - Sequential: predict P(σ_i=1 | σ_{<i}) at each step
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

        if ffn_dim is None:
            ffn_dim = 4 * embed_dim

        # Embeddings
        # Occupation embedding: 0 or 1 -> embed_dim
        self.occ_embedding = nn.Embedding(2, embed_dim)
        # Positional embedding for orbital index
        self.pos_embedding = nn.Embedding(n_orbitals, embed_dim)
        # Start token (learned)
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Alpha transformer (causal self-attention only)
        self.alpha_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, has_cross_attn=False)
            for _ in range(n_layers)
        ])

        # Beta transformer (causal self-attention + cross-attention to alpha)
        self.beta_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, has_cross_attn=True)
            for _ in range(n_layers)
        ])

        # Output heads: predict logit for occupation at each position
        self.alpha_head = nn.Linear(embed_dim, 1)
        self.beta_head = nn.Linear(embed_dim, 1)

        # Layer norms before output
        self.alpha_ln = nn.LayerNorm(embed_dim)
        self.beta_ln = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)

    def _alpha_logits(self, alpha_config: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for alpha orbitals autoregressively.

        Args:
            alpha_config: (B, T) partial or full alpha config (0/1), T <= n_orbitals

        Returns:
            logits: (B, T) logit for P(σ_i=1) at each position
        """
        B, T = alpha_config.shape
        device = alpha_config.device

        # Embed occupations and add positional encoding
        # Shift right: input at position i is the occupation at position i-1
        # Position 0 gets the start token
        occ_emb = self.occ_embedding(alpha_config.long())  # (B, T, embed_dim)
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos_idx)  # (T, embed_dim)

        # Shift: prepend start token, remove last
        start = self.start_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([start, occ_emb[:, :-1, :]], dim=1) + pos_emb  # (B, T, embed_dim)

        # Causal transformer
        for block in self.alpha_blocks:
            x = block(x)

        logits = self.alpha_head(self.alpha_ln(x)).squeeze(-1)  # (B, T)
        return logits

    def _beta_logits(self, beta_config: torch.Tensor,
                     alpha_context: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for beta orbitals, conditioned on full alpha config.

        Args:
            beta_config: (B, T) partial or full beta config
            alpha_context: (B, n_orbitals, embed_dim) alpha transformer output

        Returns:
            logits: (B, T)
        """
        B, T = beta_config.shape
        device = beta_config.device

        occ_emb = self.occ_embedding(beta_config.long())
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        start = self.start_token.expand(B, -1, -1)
        x = torch.cat([start, occ_emb[:, :-1, :]], dim=1) + pos_emb

        # Causal self-attention + cross-attention to alpha
        for block in self.beta_blocks:
            x = block(x, context=alpha_context)

        logits = self.beta_head(self.beta_ln(x)).squeeze(-1)
        return logits

    def _get_alpha_context(self, alpha_config: torch.Tensor) -> torch.Tensor:
        """Get alpha transformer hidden states for beta cross-attention."""
        B = alpha_config.shape[0]
        device = alpha_config.device

        occ_emb = self.occ_embedding(alpha_config.long())
        pos_idx = torch.arange(self.n_orbitals, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        # For context, use bidirectional (no causal mask) since beta
        # sees the FULL alpha configuration
        x = occ_emb + pos_emb

        # Use alpha blocks but without causal mask
        no_mask = torch.zeros(self.n_orbitals, self.n_orbitals,
                              device=device).bool()
        for block in self.alpha_blocks:
            x = block(x, mask=no_mask)

        return x  # (B, n_orbitals, embed_dim)

    def log_prob(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute log P(config) for a batch of full configurations.

        Args:
            config: (B, 2*n_orbitals) binary occupation string

        Returns:
            log_prob: (B,)
        """
        alpha = config[:, :self.n_orbitals]
        beta = config[:, self.n_orbitals:]

        # Alpha log prob
        alpha_logits = self._alpha_logits(alpha)  # (B, n_orbitals)
        alpha_log_prob = -F.binary_cross_entropy_with_logits(
            alpha_logits, alpha.float(), reduction='none'
        ).sum(dim=-1)  # (B,)

        # Beta log prob (conditioned on alpha)
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
        Autoregressively sample configurations with exact particle number.

        Args:
            n_samples: number of samples
            hard: ignored (always hard for autoregressive)
            temperature: sampling temperature

        Returns:
            configs: (n_samples, 2*n_orbitals) binary configs
            log_probs: (n_samples,) log probabilities
        """
        device = next(self.parameters()).device
        n_orb = self.n_orbitals

        # Sample alpha orbitals
        alpha = torch.zeros(n_samples, n_orb, device=device)
        alpha_log_prob = torch.zeros(n_samples, device=device)

        for i in range(n_orb):
            # Get logit for position i
            logits = self._alpha_logits(alpha[:, :i+1])  # (B, i+1)
            logit_i = logits[:, i] / temperature

            # Constrain: enforce exact n_alpha electrons
            placed = alpha[:, :i].sum(dim=1)  # electrons already placed
            remaining_slots = n_orb - i  # positions remaining (including current)
            needed = self.n_alpha - placed  # electrons still needed

            # Must place: if needed == remaining_slots, force 1
            must_place = (needed >= remaining_slots)
            # Cannot place: if already placed enough
            cannot_place = (placed >= self.n_alpha)

            logit_i = torch.where(must_place, torch.tensor(10.0, device=device), logit_i)
            logit_i = torch.where(cannot_place, torch.tensor(-10.0, device=device), logit_i)

            # Sample
            prob_i = torch.sigmoid(logit_i)
            sample_i = torch.bernoulli(prob_i)
            alpha[:, i] = sample_i

            # Accumulate log prob
            alpha_log_prob += torch.where(
                sample_i == 1,
                F.logsigmoid(logit_i),
                F.logsigmoid(-logit_i),
            )

        # Get alpha context for beta
        alpha_context = self._get_alpha_context(alpha)

        # Sample beta orbitals
        beta = torch.zeros(n_samples, n_orb, device=device)
        beta_log_prob = torch.zeros(n_samples, device=device)

        for i in range(n_orb):
            logits = self._beta_logits(beta[:, :i+1], alpha_context)
            logit_i = logits[:, i] / temperature

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
                sample_i == 1,
                F.logsigmoid(logit_i),
                F.logsigmoid(-logit_i),
            )

        configs = torch.cat([alpha, beta], dim=1)
        log_probs = alpha_log_prob + beta_log_prob

        return configs, log_probs


class TransformerNQS(NeuralQuantumState):
    """
    Transformer-based NQS for log-amplitude estimation.

    Uses bidirectional self-attention (non-causal) to estimate log|ψ(x)|
    for a given configuration. This is used as the "teacher" network
    in physics-guided training.
    """

    def __init__(
        self,
        num_sites: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(num_sites, local_dim=2, complex_output=False)

        if ffn_dim is None:
            ffn_dim = 4 * embed_dim

        self.embed_dim = embed_dim
        self.occ_embedding = nn.Embedding(2, embed_dim)
        self.pos_embedding = nn.Embedding(num_sites, embed_dim)

        # Bidirectional transformer (no causal mask)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)

        # Pool over sites -> scalar log amplitude
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh(),
        )

        self.log_amp_scale = nn.Parameter(torch.tensor(1.0))

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_configuration(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape

        device = next(self.parameters()).device
        # Clamp to valid range [0, 1] then convert to long indices
        x = x.clamp(0, 1).round().long().to(device)

        occ_emb = self.occ_embedding(x)
        pos_idx = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos_idx)

        h = occ_emb + pos_emb

        # Bidirectional: no causal mask
        no_mask = torch.zeros(T, T, device=device).bool()
        for block in self.blocks:
            h = block(h, mask=no_mask)

        h = self.ln(h)

        # Mean pool over sites
        pooled = h.mean(dim=1)  # (B, embed_dim)
        out = self.head(pooled).squeeze(-1)  # (B,)

        return self.log_amp_scale * out

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=next(self.parameters()).device)
