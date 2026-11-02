"""Attention modules for DeepFold3.

Attribute names match the Haiku → PyTorch converter mapping.

GridSelfAttention Haiku → PyTorch:
    act_norm             → norm
    pair_bias_projection → pair_bias  (LinearNoBias, pair_channel → num_heads)
    q_projection         → to_q       (LinearNoBias, transpose_weights=True)
    k_projection         → to_k       (LinearNoBias, transpose_weights=True)
    v_projection         → to_v       (LinearNoBias)
    gating_query         → to_g       (LinearNoBias, transpose_weights=True)
    output_projection    → out_proj   (LinearNoBias)

MSAAttention Haiku → PyTorch:
    act_norm             → norm
    pair_norm            → pair_norm
    pair_logits          → pair_logits (LinearNoBias, pair_channel → num_heads)
    v_projection         → to_v
    gating_query         → to_g
    output_projection    → out_proj

SingleAttention (in PairFormer, single_cond=None):
    single_attention_layer_norm   → norm      (regular LayerNorm, scale+offset)
    single_attention_q_projection → to_q     (Linear w/ bias)
    single_attention_k_projection → to_k     (LinearNoBias)
    single_attention_v_projection → to_v     (LinearNoBias)
    single_attention_gating_query → to_g     (LinearNoBias)
    single_attention_transition2  → out_proj (LinearNoBias)

All modules support both unbatched and batched (leading B dim) inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model import kernel_census
from deepfold3.model.config import OptimConfig
from deepfold3.model.primitives import LayerNorm, LinearNoBias, af3_linear_init_, map_row_chunks


def _load_evo_attn():
    """Load the Triton EvoformerAttention autograd function, or return None."""
    try:
        from deepfold3.model.triton.evo_attention import HAS_TRITON, _EvoformerAttention

        return _EvoformerAttention if HAS_TRITON else None
    except ImportError:
        return None


class GridSelfAttention(nn.Module):
    """Row-wise or column-wise gated self-attention on pair activations.

    Haiku: GridSelfAttention with transpose=False (row) or transpose=True (col).
    Supports both unbatched [N, N, C] and batched [B, N, N, C] inputs.
    """

    def __init__(self, c: int, num_heads: int = 4, transpose: bool = False, optim: OptimConfig | None = None):
        super().__init__()
        assert c % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = max(c // num_heads, 16)
        self.transpose = transpose
        self.scale = self.head_dim**-0.5

        self.norm = LayerNorm(c)
        self.pair_bias = LinearNoBias(c, num_heads)
        self.to_q = LinearNoBias(c, num_heads * self.head_dim)
        self.to_k = LinearNoBias(c, num_heads * self.head_dim)
        self.to_v = LinearNoBias(c, num_heads * self.head_dim)
        # AF3: gating_query uses initializer='zeros' (no bias)
        self.to_g = LinearNoBias(c, num_heads * self.head_dim, initializer="zeros")
        self.out_proj = LinearNoBias(num_heads * self.head_dim, c, initializer="zeros")

        self._evo_attn = None
        self._ln_transpose = None
        self._fused_pair_bias = None
        if optim is not None and optim.use_fused_evo_attention:
            self._evo_attn = _load_evo_attn()
        if optim is not None and optim.use_fused_pair_bias:
            try:
                from deepfold3.model.triton.fused_pair_bias import FusedPairBias

                self._fused_pair_bias = FusedPairBias(c, num_heads)
            except ImportError:
                pass
        if optim is not None and optim.use_fused_ln_transpose and transpose:
            try:
                from deepfold3.model.triton.layernorm_transpose import FusedLayerNormTranspose

                self._ln_transpose = FusedLayerNormTranspose(c, do_transpose=True)
                kernel_census.requested("GridSelfAttention.norm", "fused_ln_transpose")
            except ImportError:
                pass
        if self._evo_attn is None and optim is not None and optim.use_torch_compile:
            # Use SDPA path with torch.compile — cuDNN/FlashAttention backends
            # are faster than our manual matmul+softmax+matmul for large N.
            self._compute_sdpa = torch.compile(self._compute_sdpa)
            self._compute_baseline = self._compute_sdpa

    def _compute_sdpa(
        self, z_flat: torch.Tensor, pb: torch.Tensor, mask: torch.Tensor, B: int, N_rows: int, N_cols: int
    ) -> torch.Tensor:
        """Attention using PyTorch's scaled_dot_product_attention (cuDNN/FlashAttention backend).

        Memory optimization: pair_bias [B, H, N, N] is shared across all N_rows.
        We pack rows into the head dimension: Q/K/V become [B, N_rows*H, N, D] and
        the bias [B, H, N, N] is repeat-interleaved to [B, N_rows*H, N, N] via
        expand (zero-copy virtual broadcast). The mask [B, N_rows, N] is reshaped
        to [B, N_rows*1, 1, N] and broadcast over H heads. The combined bias is
        assembled via addition of expanded views — one N_rows*H*N² allocation
        total, vs the old approach that called .contiguous() creating a full copy.

        dtype fix: cast attn_bias to q.dtype to avoid SDPA bf16/fp32 mismatch.
        """
        H, D = self.num_heads, self.head_dim
        # Q/K/V: [B*N_rows, N_cols, H*D] → [B, N_rows*H, N_cols, D]
        q = self.to_q(z_flat).view(B, N_rows, N_cols, H, D).permute(0, 1, 3, 2, 4).reshape(B, N_rows * H, N_cols, D)
        k = self.to_k(z_flat).view(B, N_rows, N_cols, H, D).permute(0, 1, 3, 2, 4).reshape(B, N_rows * H, N_cols, D)
        v = self.to_v(z_flat).view(B, N_rows, N_cols, H, D).permute(0, 1, 3, 2, 4).reshape(B, N_rows * H, N_cols, D)

        # Apply mask into pb first as [B, H, 1, N] + [B, H, N, N] = [B, H, N, N] (cheap, N²H)
        # Then expand to [B, N_rows*H, N, N] via reshape (one copy, unavoidable for SDPA).
        # This avoids the previous approach of broadcasting a [B, N_rows, 1, 1, N] mask_bias
        # against [B, 1, H, N, N] pb, which materialized a full [B, N_rows, H, N, N] intermediate.
        col_mask, _ = mask.reshape(B, N_rows, N_cols).max(dim=1)  # [B, N] — any-row mask
        pb_masked = pb + 1e9 * (col_mask[:, None, None, :] - 1.0)  # [B, H, N, N] + [B, 1, 1, N]
        attn_bias = pb_masked.unsqueeze(1).expand(B, N_rows, H, N_cols, N_cols)
        attn_bias = attn_bias.reshape(B, N_rows * H, N_cols, N_cols).to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=self.scale)
        # [B, N_rows*H, N_cols, D] → [B, N_rows, H, N_cols, D] → [B*N_rows, N_cols, H*D]
        out = out.view(B, N_rows, H, N_cols, D).permute(0, 1, 3, 2, 4).reshape(B * N_rows, N_cols, H * D)

        g = torch.sigmoid(self.to_g(z_flat))
        return self.out_proj(out * g)

    def _compute_baseline(
        self, z_flat: torch.Tensor, pb: torch.Tensor, mask: torch.Tensor, B: int, N_rows: int, N_cols: int
    ) -> torch.Tensor:
        H, D = self.num_heads, self.head_dim
        q = self.to_q(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)
        k = self.to_k(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)
        v = self.to_v(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        logits = logits.view(B, N_rows, H, N_cols, N_cols)
        logits = logits + pb.unsqueeze(1)
        logits = logits.reshape(B * N_rows, H, N_cols, N_cols)

        mask_flat = mask.reshape(B * N_rows, N_cols)
        logits = logits + 1e9 * (mask_flat.float() - 1.0).view(B * N_rows, 1, 1, N_cols)

        w = F.softmax(logits.float(), dim=-1).to(z_flat.dtype)
        out = torch.matmul(w, v)
        out = out.permute(0, 2, 1, 3).reshape(B * N_rows, N_cols, H * D)

        g = torch.sigmoid(self.to_g(z_flat))
        return self.out_proj(out * g)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:    [B?, N, N, C]  pair activations
            mask: [B?, N, N]     pair mask
        Returns:
            [B?, N, N, C]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        B = z.shape[0]
        N = z.shape[1]  # == z.shape[2] (pair matrix is square)

        # ── Pair bias + LayerNorm ──
        # Compute pair bias and normalize z. For column attention with evo_attn,
        # DEFER the transpose to avoid the N²C .contiguous() copy (see below).
        _defer_transpose = self.transpose and self._evo_attn is not None and z.is_cuda

        if self._fused_pair_bias is not None and z.is_cuda:
            kernel_census.record("GridSelfAttention.norm", "fused_pair_bias")
            pb = self._fused_pair_bias(z, self.norm.weight, self.norm.bias, self.pair_bias.weight)  # [B, H, N, N]
            z = self.norm(z)
            if self.transpose and not _defer_transpose:
                z = z.permute(0, 2, 1, 3)
                mask = mask.permute(0, 2, 1)
        elif self._ln_transpose is not None and z.is_cuda and not _defer_transpose:
            # Not when deferring: that path reads z as untransposed, and this
            # kernel exists only to save a transpose the deferred path never does.
            kernel_census.record("GridSelfAttention.norm", "fused_ln_transpose")
            z_normed = self.norm(z)
            pb = self.pair_bias(z_normed).permute(0, 3, 1, 2)
            z = self._ln_transpose(z, self.norm.weight, self.norm.bias)
            mask = mask.permute(0, 2, 1)
        else:
            kernel_census.record("GridSelfAttention.norm", "stock")
            z = self.norm(z)
            pb = self.pair_bias(z).permute(0, 3, 1, 2)
            if self.transpose and not _defer_transpose:
                z = z.permute(0, 2, 1, 3)
                mask = mask.permute(0, 2, 1)

        _, N_rows, N_cols, C = z.shape
        H = self.num_heads
        D = self.head_dim

        # Fold batch and row dims for per-row attention
        z_flat = z.reshape(B * N_rows, N_cols, C)

        if _defer_transpose:
            kernel_census.record("GridSelfAttention.attention", "evo_attn_col")
            # Column attention fast path: pre-transpose QKV to kernel layout
            # [B, N2, H, N1, D] in one contiguous() call per tensor. The linear
            # output is freed immediately (no view chain keeps it alive), saving
            # ~1.2 GB vs the old approach where caller views + kernel copies coexisted.
            # z_flat is [B*N1, N2, C] from un-transposed z[B, N1, N2, C].
            q = self.to_q(z_flat).view(B, N, N, H, D).permute(0, 2, 3, 1, 4).contiguous()
            k = self.to_k(z_flat).view(B, N, N, H, D).permute(0, 2, 3, 1, 4).contiguous()
            v = self.to_v(z_flat).view(B, N, N, H, D).permute(0, 2, 3, 1, 4).contiguous()
            # Mask: column-wise → permute rows/cols
            col_mask = mask.permute(0, 2, 1)
            res_mask = torch.where(col_mask > 0.5, 0.0, -1e9).view(B, N, 1, 1, N)
            # Pair bias: pb[b,h,n1,n2] from z[B,N1,N2,C]. For col attention attending
            # over N1 (rows) for each N2 (column), query=row_i, key=row_j → bias is
            # pb[b,h,i,j]. No transpose needed — the original pb indexes match.
            pair_bias = pb.unsqueeze(1).float()  # [B, 1, H, N, N]
            out = self._evo_attn.apply(q, k, v, res_mask, pair_bias, True)  # [B, N2, H, N1, D]
            del q, k, v
            # Back to [B, N1, N2, H, D] → [B*N1, N2, H*D]
            out = out.permute(0, 3, 1, 2, 4).contiguous().view(B * N, N, H * D)
            g = torch.sigmoid(self.to_g(z_flat))
            out = self.out_proj(out * g)
        elif self._evo_attn is not None and z.is_cuda:
            kernel_census.record("GridSelfAttention.attention", "evo_attn_row")
            # Row attention: pre-transpose QKV to kernel layout [B, N_rows, H, N_cols, D].
            # Each .contiguous() replaces the linear output immediately, so only the
            # transposed copy survives — avoids 3× duplicate N²C allocations.
            q = self.to_q(z_flat).view(B, N_rows, N_cols, H, D).transpose(-2, -3).contiguous()
            k = self.to_k(z_flat).view(B, N_rows, N_cols, H, D).transpose(-2, -3).contiguous()
            v = self.to_v(z_flat).view(B, N_rows, N_cols, H, D).transpose(-2, -3).contiguous()
            res_mask = torch.where(mask > 0.5, 0.0, -1e9).view(B, N_rows, 1, 1, N_cols)
            pair_bias = pb.unsqueeze(1).float()
            out = self._evo_attn.apply(q, k, v, res_mask, pair_bias, True)
            del q, k, v
            # [B, N_rows, H, N_cols, D] → [B*N_rows, N_cols, H*D]
            out = out.permute(0, 1, 3, 2, 4).contiguous().view(B * N_rows, N_cols, H * D)
            g = torch.sigmoid(self.to_g(z_flat))
            out = self.out_proj(out * g)
        else:
            kernel_census.record("GridSelfAttention.attention", "baseline")
            out = self._compute_baseline(z_flat, pb, mask, B, N_rows, N_cols)

        # Unfold back to [B, N_rows, N_cols, C_out]
        out = out.view(B, N_rows, N_cols, -1)

        if self.transpose and not _defer_transpose:
            out = out.permute(0, 2, 1, 3)

        if not has_batch:
            out = out.squeeze(0)
        return out


class MSAAttention(nn.Module):
    """Row-wise MSA attention with pair bias.

    Haiku module: msa_attention1
    Supports both unbatched [S, N, c] and batched [B, S, N, c] inputs.
    """

    def __init__(self, c_msa: int, c_pair: int, num_heads: int = 8, optim: OptimConfig | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_msa // num_heads
        assert c_msa % num_heads == 0

        self.norm = LayerNorm(c_msa)
        self.pair_norm = LayerNorm(c_pair)
        self.pair_logits = LinearNoBias(c_pair, num_heads)
        self.to_v = LinearNoBias(c_msa, num_heads * self.head_dim)
        # AF3: gating_query uses initializer='zeros' (no bias)
        self.to_g = LinearNoBias(c_msa, num_heads * self.head_dim, initializer="zeros")
        self.out_proj = LinearNoBias(num_heads * self.head_dim, c_msa, initializer="zeros")
        self.row_chunk = optim.msa_row_chunk if optim is not None else 0
        if optim is not None and optim.use_torch_compile:
            self._compute = torch.compile(self._compute)

    def _row_weights(self, msa: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """Pair-derived averaging weights ``[B, H, N, N]``, shared by every row."""
        N = msa.shape[2]
        pair = self.pair_norm(pair)

        logits = self.pair_logits(pair).permute(0, 3, 1, 2)
        col_mask, _ = mask.max(dim=-2)
        logits = logits + 1e9 * (col_mask.float() - 1.0).view(msa.shape[0], 1, 1, N)
        return F.softmax(logits.float(), dim=-1)

    def _row_update(self, msa: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, S, N, C = msa.shape
        H, D = self.num_heads, self.head_dim

        msa = self.norm(msa)
        w = w.to(msa.dtype)
        v = self.to_v(msa).view(B, S, N, H, D)
        v_avg = torch.einsum("bhqk,bskhc->bsqhc", w, v)
        v_avg = v_avg.reshape(B, S, N, H * D)

        g = torch.sigmoid(self.to_g(msa))
        v_avg = v_avg * g
        return self.out_proj(v_avg)

    def _compute(self, msa: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        w = self._row_weights(msa, mask, pair)
        return map_row_chunks(lambda rows: self._row_update(rows, w), msa, self.row_chunk)

    def forward(
        self,
        msa: torch.Tensor,  # [B?, S, N, c_msa]
        mask: torch.Tensor,  # [B?, S, N]
        pair: torch.Tensor,  # [B?, N, N, c_pair]
    ) -> torch.Tensor:
        has_batch = msa.dim() == 4
        if not has_batch:
            msa = msa.unsqueeze(0)
            mask = mask.unsqueeze(0)
            pair = pair.unsqueeze(0)

        result = self._compute(msa, mask, pair)

        if not has_batch:
            result = result.squeeze(0)
        return result


class SingleAttention(nn.Module):
    """Self-attention on single representation with pair-biased logits.

    Used in PairFormerBlock and ConfidencePairFormerBlock for the single track.

    AF3: self_attention(..., single_cond=None) → regular LayerNorm (scale+offset).
    adaptive_zero_init with single_cond=None → plain Linear output (no gate).

    Haiku params stored flat in PairFormer scope (prefix 'single_attention_'):
        layer_norm    → norm      (LayerNorm with scale=weight, offset=bias)
        q_projection  → to_q     (Linear w/ bias)
        k_projection  → to_k     (LinearNoBias)
        v_projection  → to_v     (LinearNoBias)
        gating_query  → to_g     (LinearNoBias)
        transition2   → out_proj (LinearNoBias)

    Supports both unbatched [N, c] and batched [B, N, c] inputs.
    """

    def __init__(
        self,
        c: int,
        num_heads: int = 16,
        key_dim: int = 384,
        value_dim: int = 384,
        optim: OptimConfig | None = None,
        pair_bias_gate: bool = False,
        pair_bias_gate_init_tau: float = 0.25,
    ):
        super().__init__()
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0
        self.num_heads = num_heads
        self.key_dim_per_head = key_dim // num_heads
        self.val_dim_per_head = value_dim // num_heads
        self.scale = self.key_dim_per_head**-0.5

        # Regular LayerNorm (single_cond=None → no AdaLN)
        self.norm = LayerNorm(c)

        # Per-head learnable PairBias gate (log-parameterised). Calibrates
        # the strength of ``pair_logits`` relative to ``q·kᵀ`` inside the
        # attention softmax. Useful when the network needs to re-balance the
        # PairBias contribution per head independently of QK magnitude.
        #     τ_h = exp(a_h),   a_h initialised to log(τ₀)  (default τ₀=0.25)
        # Log-param keeps τ_h > 0 always and grows multiplicatively under
        # additive AdamW updates. Shape ``[num_heads]`` — auto-routed to the
        # no-decay group via the 1-D parameter rule in ``optim.py``.
        if pair_bias_gate:
            if pair_bias_gate_init_tau <= 0.0:
                raise ValueError(f"pair_bias_gate_init_tau must be > 0, got {pair_bias_gate_init_tau}")
            a_init = math.log(float(pair_bias_gate_init_tau))
            self.pair_bias_gate_a = nn.Parameter(torch.full((num_heads,), a_init))
        else:
            self.register_parameter("pair_bias_gate_a", None)

        self.to_q = nn.Linear(c, key_dim)
        af3_linear_init_(self.to_q.weight, "linear")
        nn.init.zeros_(self.to_q.bias)
        self.to_k = LinearNoBias(c, key_dim)
        self.to_v = LinearNoBias(c, value_dim)
        # AF3: gating_query uses initializer='zeros' (no bias)
        self.to_g = LinearNoBias(c, value_dim, initializer="zeros")
        # AF3: adaptive_zero_init with single_cond=None → plain linear output (final_init)
        self.out_proj = LinearNoBias(value_dim, c, initializer="zeros")

        # Triton EvoAttention is NOT beneficial for SingleAttention (N_seq=1 makes kernel
        # overhead dominate — benchmarks show 0.87-1.03x, i.e. slower). Use torch.compile instead.
        if optim is not None and optim.use_torch_compile:
            self._compute_baseline = torch.compile(self._compute_baseline)

    def _compute_baseline(self, xn: torch.Tensor, mask: torch.Tensor, pair_logits: torch.Tensor) -> torch.Tensor:
        B, N, _ = xn.shape
        H, Dk, Dv = self.num_heads, self.key_dim_per_head, self.val_dim_per_head
        q = self.to_q(xn).view(B, N, H, Dk).permute(0, 2, 1, 3)
        k = self.to_k(xn).view(B, N, H, Dk).permute(0, 2, 1, 3)
        v = self.to_v(xn).view(B, N, H, Dv).permute(0, 2, 1, 3)
        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if self.pair_bias_gate_a is not None:
            # τ_h = exp(a_h), shape [H]; broadcast over [B, H, N, N].
            tau = torch.exp(self.pair_bias_gate_a).to(pair_logits.dtype)
            pair_logits = pair_logits * tau.view(1, H, 1, 1)
        logits = logits + pair_logits
        bias_mask = 1e9 * (mask.float() - 1.0)
        logits = logits + bias_mask.view(B, 1, 1, N)
        w = F.softmax(logits.float(), dim=-1).to(xn.dtype)
        out = torch.matmul(w, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * Dv)
        g = torch.sigmoid(self.to_g(xn))
        return self.out_proj(out * g)

    def forward(
        self,
        x: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]
        pair_logits: torch.Tensor,  # [B?, H, N, N]
    ) -> torch.Tensor:
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            pair_logits = pair_logits.unsqueeze(0)

        xn = self.norm(x)
        result = self._compute_baseline(xn, mask, pair_logits)

        if not has_batch:
            result = result.squeeze(0)
        return result
