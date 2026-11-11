"""Atom cross-attention encoder and decoder.

Maps between per-atom layout [N_tok, max_atoms, C] and per-token layout [N_tok, C]
via a mini cross-attention transformer over flat atom subsets.

Flat atom layout with rectangular block attention (matching AF3):
  All real atoms packed into [total_atoms], padded to multiple of Q=32.
  Non-overlapping query subsets of Q=32 atoms (spanning ~6 residues).
  Overlapping key windows of K=128 atoms centered on each query block.

The weights are identical to the per-token sliding window implementation —
only the indexing/layout logic changes.

Two distinct encoders exist in AF3:
  1. EvoformerConditioningEncoder  — produces target_feat contribution (no trunk cond)
  2. AtomEncoder (diffusion)       — conditions on trunk single/pair + noisy positions

Haiku scope prefixes:
    EvoformerConditioningEncoder: 'diffuser/evoformer_conditioning_*'
    AtomEncoder (diffusion):      'diffuser/~/diffusion_head/diffusion_*'
    AtomDecoder (diffusion):      'diffuser/~/diffusion_head/diffusion_*' (decoder part)
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.primitives import LayerNorm, LinearNoBias, af3_linear_init_, layer_norm, map_pair_rows


@dataclasses.dataclass
class AtomSubsets:
    """Precomputed indices for flat atom subset attention.

    Maps between token-atoms layout [N, M] and flat subset layout [S, Q] / [S, K].
    """

    num_subsets: int  # S = num_atoms_padded // Q
    Q: int  # 32 (atoms per query subset)
    K: int  # 128 (atoms per key window)
    N: int  # num_tokens
    M: int  # max_atoms_per_token (24)

    # Gather: token-atoms [N*M] → queries [S, Q]
    ta_to_q_idx: torch.Tensor  # [S, Q] long — indices into flattened [N*M]
    ta_to_q_mask: torch.Tensor  # [S, Q] bool

    # Gather: flat queries [S*Q] → keys [S, K]
    q_to_k_idx: torch.Tensor  # [S, K] long — indices into flattened [S*Q]
    q_to_k_mask: torch.Tensor  # [S, K] bool

    # Token index per query/key atom (for trunk_pair broadcast + offsets_valid)
    token_for_query: torch.Tensor  # [S, Q] long
    token_for_key: torch.Tensor  # [S, K] long

    # Scatter: queries [S*Q] → token-atoms [N, M]
    q_to_ta_idx: torch.Tensor  # [N, M] long — indices into flattened [S*Q]
    q_to_ta_mask: torch.Tensor  # [N, M] bool

    def trim_empty(self) -> AtomSubsets:
        """Drop trailing windows that hold no real atom (batched layout).

        ``S`` follows the ``N*M`` capacity so that training shapes do not depend
        on the data, which leaves most windows empty: 291 of 425 for 566 tokens
        with 4280 atoms. Real atoms are packed first, each window only attends
        within itself, and a key slot past the kept range is padding that
        ``q_to_k_mask`` already masks, so its index is free to be remapped.
        """
        num_real = int(self.q_to_ta_mask.flatten(1).sum(-1).max())
        keep = max(1, math.ceil(num_real / self.Q))
        q_to_k_idx = self.q_to_k_idx[:, :keep]
        return dataclasses.replace(
            self,
            num_subsets=keep,
            ta_to_q_idx=self.ta_to_q_idx[:, :keep],
            ta_to_q_mask=self.ta_to_q_mask[:, :keep],
            q_to_k_idx=q_to_k_idx.masked_fill(q_to_k_idx >= keep * self.Q, 0),
            q_to_k_mask=self.q_to_k_mask[:, :keep],
            token_for_query=self.token_for_query[:, :keep],
            token_for_key=self.token_for_key[:, :keep],
        )


def build_atom_subsets(ref_mask: torch.Tensor, Q: int = 32, K: int = 128) -> AtomSubsets:
    """Build flat atom subset indices from ref_mask.

    Accepts ``[N, M]`` (unbatched) or ``[B, N, M]`` (batched). When batched,
    each sample's index tensors are stacked along a leading dim of size ``B``,
    so the returned ``AtomSubsets`` has fields like ``ta_to_q_idx`` of shape
    ``[B, S, Q]`` instead of ``[S, Q]``.

    1. Flatten ref_mask → find real atom indices in [N*M]
    2. Pad to next multiple of Q based on the FIXED total slot count [N*M]
       (not based on the data-dependent ``num_real``), so ``S`` is constant
       across all samples in a training run. This is required for non-reentrant
       gradient checkpointing under DDP — different samples must produce
       identical saved-tensor shapes inside the encoder.
    3. Build query/key gather indices and scatter-back indices
    """
    if ref_mask.dim() == 3:
        B = ref_mask.shape[0]
        per = [build_atom_subsets(ref_mask[b], Q, K) for b in range(B)]
        first = per[0]
        return AtomSubsets(
            num_subsets=first.num_subsets,
            Q=Q,
            K=K,
            N=first.N,
            M=first.M,
            ta_to_q_idx=torch.stack([s.ta_to_q_idx for s in per]),
            ta_to_q_mask=torch.stack([s.ta_to_q_mask for s in per]),
            q_to_k_idx=torch.stack([s.q_to_k_idx for s in per]),
            q_to_k_mask=torch.stack([s.q_to_k_mask for s in per]),
            token_for_query=torch.stack([s.token_for_query for s in per]),
            token_for_key=torch.stack([s.token_for_key for s in per]),
            q_to_ta_idx=torch.stack([s.q_to_ta_idx for s in per]),
            q_to_ta_mask=torch.stack([s.q_to_ta_mask for s in per]),
        )

    device = ref_mask.device
    N, M = ref_mask.shape

    # Find real atom indices in flattened [N*M] space
    flat_mask = ref_mask.reshape(-1).bool()  # [N*M]
    real_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_real]
    num_real = int(real_indices.shape[0])

    # Pad to next multiple of Q based on the constant [N*M] capacity, not
    # ``num_real``. ``num_padded`` and ``S`` are now constants for the entire
    # training run (function only of the input shape), so subset-layout tensors
    # have data-independent shapes.
    total_slots = N * M
    num_padded = ((total_slots + Q - 1) // Q) * Q
    S = num_padded // Q

    # padded_indices: [num_padded] indices into [N*M]. Real atoms are packed
    # into the first ``num_real`` slots; remaining slots point to index 0 and
    # are masked out via ``padded_mask`` so they contribute zero.
    padded_indices = torch.zeros(num_padded, dtype=torch.long, device=device)
    padded_indices[:num_real] = real_indices
    padded_mask = torch.zeros(num_padded, dtype=torch.bool, device=device)
    padded_mask[:num_real] = True

    ta_to_q_idx = padded_indices.reshape(S, Q)  # [S, Q]
    ta_to_q_mask = padded_mask.reshape(S, Q)  # [S, Q]

    # Key windows: for each subset s, center at (s*Q + Q/2), take K atoms.
    # Key indices are into the flat query space [S*Q]. AF3 uses window
    # shifting (not clamping) at boundaries to avoid duplicate atoms.
    # The shift bound is ``num_padded - 1`` (constant) rather than
    # ``num_real - 1`` (data-dependent) so the resulting indices have a
    # data-independent layout. Padded slots that get attended to are zeroed
    # out via ``q_to_k_mask`` below.
    centers = torch.arange(Q // 2, num_padded, Q, device=device)  # [S]
    key_offsets = torch.arange(-(K // 2), K // 2, device=device)  # [K]
    key_flat_idx = centers.unsqueeze(1) + key_offsets.unsqueeze(0)  # [S, K]

    left_overflow = (-key_flat_idx[:, 0]).clamp(min=0)  # [S]
    right_overflow = (key_flat_idx[:, -1] - (num_padded - 1)).clamp(min=0)  # [S]
    shift = left_overflow - right_overflow
    key_flat_idx = key_flat_idx + shift.unsqueeze(1)
    key_flat_idx = key_flat_idx.clamp(0, num_padded - 1)

    # q_to_k_mask: valid if the key position has a real atom
    q_to_k_mask = padded_mask[key_flat_idx]  # [S, K]
    q_to_k_idx = key_flat_idx  # [S, K] — indices into [S*Q] flat space

    # Token index for each query/key atom (for trunk pair broadcast + offsets_valid)
    flat_token_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, M).reshape(-1)  # [N*M]
    token_for_query = flat_token_idx[ta_to_q_idx]  # [S, Q]
    # Key atoms: first map to [N*M] via ta_to_q_idx, then get token
    key_nm_idx = padded_indices[key_flat_idx]  # [S, K] indices into [N*M]
    token_for_key = flat_token_idx[key_nm_idx]  # [S, K]

    # Scatter back: for each real atom in [N*M], what's its position in [S*Q]?
    q_to_ta_idx = torch.zeros(N * M, dtype=torch.long, device=device)
    q_to_ta_mask = torch.zeros(N * M, dtype=torch.bool, device=device)
    q_positions = torch.arange(num_padded, device=device)  # position in [S*Q]
    q_to_ta_idx[padded_indices[:num_real]] = q_positions[:num_real]
    q_to_ta_mask[padded_indices[:num_real]] = True

    q_to_ta_idx = q_to_ta_idx.reshape(N, M)  # [N, M]
    q_to_ta_mask = q_to_ta_mask.reshape(N, M)  # [N, M]

    return AtomSubsets(
        num_subsets=S,
        Q=Q,
        K=K,
        N=N,
        M=M,
        ta_to_q_idx=ta_to_q_idx,
        ta_to_q_mask=ta_to_q_mask,
        q_to_k_idx=q_to_k_idx,
        q_to_k_mask=q_to_k_mask,
        token_for_query=token_for_query,
        token_for_key=token_for_key,
        q_to_ta_idx=q_to_ta_idx,
        q_to_ta_mask=q_to_ta_mask,
    )


def _batched_take(arr_flat: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-batch advanced index ``arr_flat[b, idx[b, ...]]`` followed by mask.

    Args:
        arr_flat: ``[B, L, *F]`` source flattened along the indexed axis.
        idx:      ``[B, *D]`` long indices into ``L``.
        mask:     ``[B, *D]`` bool/float, same leading shape as ``idx``.

    Returns:
        ``[B, *D, *F]`` with ``mask`` broadcast and applied.
    """
    B = arr_flat.shape[0]
    D = idx.shape[1:]
    b_arange = torch.arange(B, device=arr_flat.device).view(B, *([1] * len(D))).expand(B, *D)
    out = arr_flat[b_arange, idx]  # [B, *D, *F]
    m_shape = mask.shape + (1,) * (out.ndim - mask.ndim)
    return out * mask.view(m_shape).to(out.dtype)


def _flat_take(arr_flat: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Unbatched index ``arr_flat[idx]`` followed by mask.

    Args:
        arr_flat: ``[L, *F]`` source.
        idx:      ``[*D]`` long indices into ``L``.
        mask:     ``[*D]`` bool/float, same shape as ``idx``.
    """
    out = arr_flat[idx]
    m_shape = mask.shape + (1,) * (out.ndim - mask.ndim)
    return out * mask.view(m_shape).to(out.dtype)


def _gather_to_subsets(arr: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather from token-atoms ``[B?, N, M, *F]`` to subset layout
    ``[B?, S, Q_or_K, *F]``.

    Unbatched: ``arr [N, M, *F]`` + ``idx [S, Q_or_K]``.
    Batched:   ``arr [B, N, M, *F]`` + ``idx [B, S, Q_or_K]`` (per-sample subsets).
    """
    if idx.dim() == 3:
        B = idx.shape[0]
        return _batched_take(arr.reshape(B, -1, *arr.shape[3:]), idx, mask)
    return _flat_take(arr.reshape(-1, *arr.shape[2:]), idx, mask)


def _scatter_from_subsets(arr: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """Scatter from subset layout ``[B?, S, Q, *F]`` back to token-atoms
    ``[B?, N, M, *F]``.

    Unbatched: ``arr [S, Q, *F]`` + ``idx [N, M]``.
    Batched:   ``arr [B, S, Q, *F]`` + ``idx [B, N, M]``.
    """
    if idx.dim() == 3:
        B = idx.shape[0]
        return _batched_take(arr.reshape(B, -1, *arr.shape[3:]), idx, mask)
    return _flat_take(arr.reshape(-1, *arr.shape[2:]), idx, mask)


def broadcast_for_samples(t: torch.Tensor, num_samples: int) -> torch.Tensor:
    """``[B, ...]`` state shared by a sample batch: with one molecule it stays
    ``[1, ...]`` and broadcasts, otherwise it is repeated to ``[B*S, ...]``."""
    return t if t.shape[0] == 1 else t.repeat_interleave(num_samples, dim=0)


def expand_for_samples_shared(t: torch.Tensor, num_samples: int) -> torch.Tensor:
    """``[B, ...]`` → ``[B*S, ...]`` read-only: a view with one molecule."""
    return t.expand(num_samples, *t.shape[1:]) if t.shape[0] == 1 else t.repeat_interleave(num_samples, dim=0)


def _gather_q_to_k(arr: torch.Tensor, q_to_k_idx: torch.Tensor, q_to_k_mask: torch.Tensor) -> torch.Tensor:
    """Gather from queries ``[B?, S, Q, *F]`` to keys ``[B?, S, K, *F]``.

    ``q_to_k_idx`` is ``[S, K]`` (unbatched) or ``[B, S, K]`` (batched).
    """
    if q_to_k_idx.dim() == 3:
        B = q_to_k_idx.shape[0]
        return _batched_take(arr.reshape(B, -1, *arr.shape[3:]), q_to_k_idx, q_to_k_mask)
    return _flat_take(arr.reshape(-1, *arr.shape[2:]), q_to_k_idx, q_to_k_mask)


def _offsets_valid(subsets: AtomSubsets, ref_space_uid: torch.Tensor | None = None) -> torch.Tensor:
    """AF3 Algorithm 5 line 3: ``v_lm = (ref_space_uid_l == ref_space_uid_m)``.

    ``ref_space_uid`` numbers each (chain id, residue index) reference conformer,
    so the gate admits every atom pair sharing one conformer. Falls back to token
    identity when the feature is absent (older callers / feature dicts). The two
    agree for multi-atom polymer residue tokens, but NOT for one-atom tokens
    (ligand, modified residue, atomized residue): there token identity collapses
    to the self-pair, dropping the reference-conformer offsets and distances
    entirely — exactly the geometry AF3 feeds the atom encoder.

    Args:
        subsets: atom subset indices from :func:`build_atom_subsets`.
        ref_space_uid: ``[B?, N, M]`` int reference-space ids, or None.

    Returns:
        ``[B?, S, Q, K]`` bool validity mask.
    """
    if ref_space_uid is None:
        return subsets.token_for_query.unsqueeze(-1) == subsets.token_for_key.unsqueeze(-2)
    uid_q = _gather_to_subsets(ref_space_uid.unsqueeze(-1), subsets.ta_to_q_idx, subsets.ta_to_q_mask)
    uid_k = _gather_q_to_k(uid_q, subsets.q_to_k_idx, subsets.q_to_k_mask)
    return uid_q.squeeze(-1).unsqueeze(-1) == uid_k.squeeze(-1).unsqueeze(-2)


@dataclasses.dataclass
class AtomCrossAttEncoderOutput:
    token_act: torch.Tensor  # [N_tok, per_token_channels]
    skip_connection: torch.Tensor  # [S, Q, per_atom_channels]
    queries_mask: torch.Tensor  # [S, Q] float
    queries_single_cond: torch.Tensor  # [S, Q, per_atom_channels]
    keys_mask: torch.Tensor  # [S, K] float
    keys_single_cond: torch.Tensor  # [S, K, per_atom_channels]
    pair_cond: torch.Tensor  # [S, Q, K, pair_ch]
    subsets: AtomSubsets  # precomputed indices for decoder
    decoder_hoist: tuple[AtomBlockHoist, ...] | None = None


@dataclasses.dataclass
class AtomBlockHoist:
    """One :class:`AtomCrossAttTransformer` block's noise-independent terms.

    Everything a block derives from the single / pair conditioning alone, so
    the sampler computes it once instead of once per step and sample.
    """

    logit_bias: torch.Tensor  # [B?, S, H, Q, K] pair logits + key mask
    q_scale: torch.Tensor  # [B?, S, Q, c]
    q_bias: torch.Tensor  # [B?, S, Q, c]
    k_scale: torch.Tensor  # [B?, S, K, c]
    k_bias: torch.Tensor  # [B?, S, K, c]
    attn_gate: torch.Tensor  # [B?, S, Q, c]
    ffn_scale: torch.Tensor  # [B?, S, Q, c]
    ffn_bias: torch.Tensor  # [B?, S, Q, c]
    ffn_gate: torch.Tensor  # [B?, S, Q, c]

    def broadcast_for_samples(self, num_samples: int) -> AtomBlockHoist:
        return AtomBlockHoist(
            *(broadcast_for_samples(getattr(self, f.name), num_samples) for f in dataclasses.fields(self))
        )


@dataclasses.dataclass
class AtomEncoderStatic:
    """Noise-independent encoder state that can be reused across diffusion samples.

    Computed once per (molecule, trunk_embeddings) pair via
    ``AtomEncoder.compute_static`` and reused by ``AtomEncoder.forward`` with
    a ``static=...`` kwarg. Saves the conditioning compute (single embeddings,
    pair MLP, trunk pair gather) on every sample after the first.

    Always stored at batched ``[B, ...]`` layout — :meth:`compute_static`
    auto-promotes unbatched inputs to ``B=1``. Use :meth:`expand_for_samples`
    to materialise the ``[B*S, ...]`` view that aligns with diffusion samples
    collapsed into a single leading batch dim.
    """

    subsets: AtomSubsets
    queries_single: torch.Tensor  # [B, S_sub, Q, c]
    queries_mask: torch.Tensor  # [B, S_sub, Q]
    keys_single: torch.Tensor  # [B, S_sub, K, c]
    keys_mask: torch.Tensor  # [B, S_sub, K]
    atom_pair: torch.Tensor  # [B, S_sub, Q, K, cp]
    ref_mask: torch.Tensor  # [B, N, M] — needed for the aggregation denominator
    encoder_hoist: tuple[AtomBlockHoist, ...] | None = None
    decoder_hoist: tuple[AtomBlockHoist, ...] | None = None

    def trim_empty_subsets(self) -> AtomEncoderStatic:
        """Same state without the windows :meth:`AtomSubsets.trim_empty` drops."""
        subsets = self.subsets.trim_empty()
        keep = subsets.num_subsets
        return dataclasses.replace(
            self,
            subsets=subsets,
            queries_single=self.queries_single[:, :keep],
            queries_mask=self.queries_mask[:, :keep],
            keys_single=self.keys_single[:, :keep],
            keys_mask=self.keys_mask[:, :keep],
            atom_pair=self.atom_pair[:, :keep],
        )

    def share_for_samples(self, num_samples: int) -> AtomEncoderStatic:
        """:meth:`expand_for_samples` for a sampler that only reads this state:
        one molecule yields views, so S samples hold one copy."""

        def exp(t: torch.Tensor) -> torch.Tensor:
            return expand_for_samples_shared(t, num_samples)

        subsets = self.subsets
        return dataclasses.replace(
            self,
            subsets=dataclasses.replace(
                subsets,
                ta_to_q_idx=exp(subsets.ta_to_q_idx),
                ta_to_q_mask=exp(subsets.ta_to_q_mask),
                q_to_k_idx=exp(subsets.q_to_k_idx),
                q_to_k_mask=exp(subsets.q_to_k_mask),
                token_for_query=exp(subsets.token_for_query),
                token_for_key=exp(subsets.token_for_key),
                q_to_ta_idx=exp(subsets.q_to_ta_idx),
                q_to_ta_mask=exp(subsets.q_to_ta_mask),
            ),
            queries_single=exp(self.queries_single),
            queries_mask=exp(self.queries_mask),
            keys_single=exp(self.keys_single),
            keys_mask=exp(self.keys_mask),
            atom_pair=exp(self.atom_pair),
            ref_mask=exp(self.ref_mask),
        )

    def expand_for_samples(self, num_samples: int) -> AtomEncoderStatic:
        """Repeat each field along the leading batch dim by ``num_samples``
        (``[B, ...]`` → ``[B*num_samples, ...]``)."""
        if num_samples == 1:
            return self

        def exp(t: torch.Tensor) -> torch.Tensor:
            return t.repeat_interleave(num_samples, dim=0).contiguous()

        return AtomEncoderStatic(
            subsets=AtomSubsets(
                num_subsets=self.subsets.num_subsets,
                Q=self.subsets.Q,
                K=self.subsets.K,
                N=self.subsets.N,
                M=self.subsets.M,
                ta_to_q_idx=exp(self.subsets.ta_to_q_idx),
                ta_to_q_mask=exp(self.subsets.ta_to_q_mask),
                q_to_k_idx=exp(self.subsets.q_to_k_idx),
                q_to_k_mask=exp(self.subsets.q_to_k_mask),
                token_for_query=exp(self.subsets.token_for_query),
                token_for_key=exp(self.subsets.token_for_key),
                q_to_ta_idx=exp(self.subsets.q_to_ta_idx),
                q_to_ta_mask=exp(self.subsets.q_to_ta_mask),
            ),
            queries_single=exp(self.queries_single),
            queries_mask=exp(self.queries_mask),
            keys_single=exp(self.keys_single),
            keys_mask=exp(self.keys_mask),
            atom_pair=exp(self.atom_pair),
            ref_mask=exp(self.ref_mask),
        )


class AtomPerAtomConditioning(nn.Module):
    """Compute per-atom single + pair conditioning from reference structure.

    Haiku names (prefix = name):
        {name}embed_ref_pos         → embed_ref_pos
        {name}embed_ref_mask        → embed_ref_mask
        {name}embed_ref_element     → embed_ref_element
        {name}embed_ref_charge      → embed_ref_charge
        {name}embed_ref_atom_name   → embed_ref_atom_name
        {name}single_to_pair_cond_row → single_to_pair_row
        {name}single_to_pair_cond_col → single_to_pair_col
        {name}embed_pair_offsets    → embed_pair_offsets
        {name}embed_pair_distances  → embed_pair_distances
        {name}pair_mlp_1            → pair_mlp1
        {name}pair_mlp_2            → pair_mlp2
        {name}pair_mlp_3            → pair_mlp3

    Note: single_to_pair_row/col, embed_pair_offsets/distances, pair_mlp1/2/3
    are exposed as attributes so that encoder modules can reuse them for
    cross-token pair construction (matching AF3's Haiku parameter sharing).
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_atom_pair_channels: int,
        num_element_classes: int = 128,
        num_atom_name_chars: int = 64,
        atom_name_length: int = 4,
        ref_element_zero_index: bool = False,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        self.num_element_classes = num_element_classes
        self.num_atom_name_chars = num_atom_name_chars
        # ``ref_element`` is stored as the raw atomic number Z (one-indexed:
        # H=1, C=6, ...). DeepFold3 one-hots it at index Z by default (index 0
        # is an unused/dead channel, element 128 dropped). This is ALSO the AF3
        # convention: AF3 one-hots at Z over 128 classes ("Element is encoded
        # as atomic number of the periodic table"), so AF3-converted
        # ``embed_ref_element`` weights — copied row-verbatim — load correctly
        # with the default False. Set ``ref_element_zero_index=True`` ONLY for
        # the *Protenix* convention, which one-hots the zero-indexed position
        # (Z-1: H=0, C=5) over 118 classes. Do NOT use True for AF3 — it shifts
        # every element down by one and corrupts the trunk.
        self.ref_element_zero_index = ref_element_zero_index
        self.embed_ref_pos = LinearNoBias(3, c)
        self.embed_ref_mask = LinearNoBias(1, c)
        self.embed_ref_element = LinearNoBias(num_element_classes, c)
        self.embed_ref_charge = LinearNoBias(1, c)
        self.embed_ref_atom_name = LinearNoBias(num_atom_name_chars * atom_name_length, c)
        # pair conditioning from single (accessible from encoder for cross-token pairs)
        self.single_to_pair_row = LinearNoBias(c, cp)
        self.single_to_pair_col = LinearNoBias(c, cp)
        self.embed_pair_offsets = LinearNoBias(3, cp)
        self.embed_pair_distances = LinearNoBias(1, cp)
        # small MLP on pair (accessible from encoder for cross-token pairs)
        # AF3: pair_mlp_1/2 use initializer='relu', pair_mlp_3 uses final_init (zeros)
        self.pair_mlp1 = LinearNoBias(cp, cp, initializer="relu")
        self.pair_mlp2 = LinearNoBias(cp, cp, initializer="relu")
        self.pair_mlp3 = LinearNoBias(cp, cp, initializer="zeros")

    def single_cond(
        self,
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M] float
        ref_element: torch.Tensor,  # [B?, N, M] int64
        ref_charge: torch.Tensor,  # [B?, N, M] float
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4] int64
    ) -> torch.Tensor:
        """Compute per-atom single conditioning ``[B?, N, M, c]``."""
        act = self.embed_ref_pos(ref_pos)
        act = act + self.embed_ref_mask(ref_mask.unsqueeze(-1))
        # One-index Z by default; subtract 1 for the zero-indexed AF3 convention.
        elem_idx = (ref_element - 1).clamp(min=0) if self.ref_element_zero_index else ref_element
        elem_1hot = F.one_hot(elem_idx, self.num_element_classes).float()
        act = act + self.embed_ref_element(elem_1hot)
        act = act + self.embed_ref_charge(torch.arcsinh(ref_charge).unsqueeze(-1))
        name_1hot = F.one_hot(ref_atom_name_chars, self.num_atom_name_chars).float()
        # Flatten the trailing (atom_name_length, num_atom_name_chars) dims
        name_1hot = name_1hot.reshape(*name_1hot.shape[:-2], -1)
        act = act + self.embed_ref_atom_name(name_1hot)
        act = act * ref_mask.unsqueeze(-1)
        return act

    def apply_pair_mlp(self, pair: torch.Tensor) -> torch.Tensor:
        """Apply residual pair MLP to pair activations of any shape [..., cp]."""
        p2 = self.pair_mlp1(F.relu(pair))
        p2 = self.pair_mlp2(F.relu(p2))
        return pair + self.pair_mlp3(F.relu(p2))

    def forward(
        self,
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M] float
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M] float
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (atom_single [N,M,c], intra_pair [N,M,M,cp]).

        The intra_pair output is NOT used by the encoder (cross-token mode).
        It exists so that per_atom's pair parameters are initialised in the
        same Haiku scope as the encoder's pair MLP (matching AF3 weight names).
        """
        act = self.single_cond(ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars)

        # Intra-token pair (same-token atom pairs); negative-axis ops are
        # batch-friendly so the same code path works for [N,M,...] and [B,N,M,...].
        row = self.single_to_pair_row(F.relu(act))  # [..., N, M, cp]
        col = self.single_to_pair_col(F.relu(act))  # [..., N, M, cp]
        pair_act = row.unsqueeze(-2) + col.unsqueeze(-3)  # [..., N, M, M, cp]
        offsets = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)  # [..., N, M, M, 3]
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        pair_act = pair_act + self.embed_pair_offsets(offsets)
        pair_act = pair_act + self.embed_pair_distances(1.0 / (1.0 + sq_dists))
        pair_act = self.apply_pair_mlp(pair_act)

        return act, pair_act


class AtomCrossAttTransformer(nn.Module):
    """Cross-attention transformer for flat atom subsets.

    Queries = Q atoms from a non-overlapping subset;
    Keys = K atoms from an overlapping window centered on that subset.

    At each block, keys_act is re-derived from current queries_act by gathering
    from the key window, matching AF3's CrossAttTransformer behaviour.

    Haiku: CrossAttTransformer
        pair_input_layer_norm → pair_norm  (scale-only, no offset)
        pair_logits_projection → pair_logits_proj
        layers: stacked with __layer_stack_with_per_layer

    Converter attribute names (all per-block, indexed by ModuleList):
        attn_cond_w[i]     ← qsingle_cond_layer_norm  scale
        attn_cond_scale[i] ← qsingle_cond_scale
        attn_cond_bias[i]  ← qsingle_cond_bias
        k_cond_w[i]        ← ksingle_cond_layer_norm  scale
        k_cond_scale[i]    ← ksingle_cond_scale
        k_cond_bias[i]     ← ksingle_cond_bias
        attn_blocks[i].to_q      ← q_projection
        attn_blocks[i].to_k      ← k_projection
        attn_blocks[i].to_v      ← v_projection
        attn_blocks[i].to_g      ← gating_query
        attn_blocks[i].out_proj  ← transition2
        attn_blocks[i].zero_cond ← adaptive_zero_cond
        ffn_cond_w[i]      ← ffw_single_cond_layer_norm scale
        ffn_cond_scale[i]  ← ffw_single_cond_scale
        ffn_cond_bias[i]   ← ffw_single_cond_bias
        ffn_blocks[i].linear1   ← ffw_transition1
        ffn_blocks[i].linear2   ← ffw_transition2
        ffn_blocks[i].zero_cond ← ffw_adaptive_zero_cond
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_atom_pair_channels: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_intermediate_factor: int = 2,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        H = num_heads
        assert c % H == 0
        self.num_heads = H
        self.head_dim = c // H
        self.num_blocks = num_blocks

        # Pair logits (scale-only LN + no-bias projection)
        self.pair_norm_w = nn.Parameter(torch.ones(cp))
        self.pair_logits_proj = LinearNoBias(cp, num_blocks * H)

        num_int = c * num_intermediate_factor
        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            to_q = nn.Linear(c, c)  # q_projection has bias
            af3_linear_init_(to_q.weight, "linear")
            nn.init.zeros_(to_q.bias)
            self.attn_blocks.append(
                nn.ModuleDict(
                    {
                        "to_q": to_q,
                        "to_k": LinearNoBias(c, c),
                        "to_v": LinearNoBias(c, c),
                        # AF3: gating_query uses initializer='zeros' (no bias)
                        "to_g": LinearNoBias(c, c, initializer="zeros"),
                        "out_proj": LinearNoBias(c, c),  # transition2 (default — gate handles zero init)
                        "zero_cond": nn.Linear(c, c),  # adaptive_zero_cond has bias
                    }
                )
            )
            self.ffn_blocks.append(
                nn.ModuleDict(
                    {
                        # AF3: ffw_transition1 uses initializer='relu'
                        "linear1": LinearNoBias(c, num_int * 2, initializer="relu"),
                        "linear2": LinearNoBias(num_int, c),  # transition2 (default — gate handles zero init)
                        "zero_cond": nn.Linear(c, c),
                    }
                )
            )
        # Initialise zero_cond to near-zero gate (sigmoid(-2) ≈ 0.12)
        for b in self.attn_blocks:
            nn.init.zeros_(b["zero_cond"].weight)
            nn.init.constant_(b["zero_cond"].bias, -2.0)
        for b in self.ffn_blocks:
            nn.init.zeros_(b["zero_cond"].weight)
            nn.init.constant_(b["zero_cond"].bias, -2.0)

        # Shared norms (parameter-free, elementwise_affine=False)
        self.attn_norms = nn.ModuleList([LayerNorm(c, elementwise_affine=False) for _ in range(num_blocks)])
        self.ffn_norms = nn.ModuleList([LayerNorm(c, elementwise_affine=False) for _ in range(num_blocks)])

        # q-side AdaLN conditioning
        # AF3: single_cond_scale and single_cond_bias use initializer='zeros'
        self.attn_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.attn_cond_scale = nn.ModuleList()
        self.attn_cond_bias = nn.ModuleList()
        for _ in range(num_blocks):
            s = nn.Linear(c, c)
            af3_linear_init_(s.weight, "zeros")
            nn.init.zeros_(s.bias)
            self.attn_cond_scale.append(s)
            self.attn_cond_bias.append(LinearNoBias(c, c, initializer="zeros"))

        # k-side AdaLN conditioning (keys/values conditioned separately from queries)
        self.k_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.k_cond_scale = nn.ModuleList()
        self.k_cond_bias = nn.ModuleList()
        for _ in range(num_blocks):
            s = nn.Linear(c, c)
            af3_linear_init_(s.weight, "zeros")
            nn.init.zeros_(s.bias)
            self.k_cond_scale.append(s)
            self.k_cond_bias.append(LinearNoBias(c, c, initializer="zeros"))

        # FFN AdaLN conditioning (on queries only)
        self.ffn_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.ffn_cond_scale = nn.ModuleList()
        self.ffn_cond_bias = nn.ModuleList()
        for _ in range(num_blocks):
            s = nn.Linear(c, c)
            af3_linear_init_(s.weight, "zeros")
            nn.init.zeros_(s.bias)
            self.ffn_cond_scale.append(s)
            self.ffn_cond_bias.append(LinearNoBias(c, c, initializer="zeros"))

    def hoist(
        self,
        queries_single_cond: torch.Tensor,  # [B, S, Q, c]
        keys_single_cond: torch.Tensor,  # [B, S, K, c]
        keys_mask: torch.Tensor,  # [B, S, K] float
        pair_cond: torch.Tensor,  # [B, S, Q, K, cp]
    ) -> tuple[AtomBlockHoist, ...]:
        """The per-block terms of :meth:`forward` that ignore ``queries_act``."""
        C = queries_single_cond.shape[-1]
        B, S, Q, K = pair_cond.shape[:4]
        pair_normed = layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w
        pair_logits = self.pair_logits_proj(pair_normed)  # [B, S, Q, K, nb*H]
        pair_logits = pair_logits.reshape(B, S, Q, K, self.num_blocks, self.num_heads).movedim((-2, -1), (0, 3))
        key_bias = 1e9 * (keys_mask.unsqueeze(-2).unsqueeze(-2) - 1.0)  # [B, S, 1, 1, K]
        q_norm = layer_norm(queries_single_cond, [C])
        k_norm = layer_norm(keys_single_cond, [C])

        blocks = []
        for i, (attn, ffn) in enumerate(zip(self.attn_blocks, self.ffn_blocks, strict=True)):
            cn_q = q_norm * self.attn_cond_w[i]
            cn_k = k_norm * self.k_cond_w[i]
            cn_f = q_norm * self.ffn_cond_w[i]
            blocks.append(
                AtomBlockHoist(
                    logit_bias=pair_logits[i] + key_bias,
                    q_scale=torch.sigmoid(self.attn_cond_scale[i](cn_q)),
                    q_bias=self.attn_cond_bias[i](cn_q),
                    k_scale=torch.sigmoid(self.k_cond_scale[i](cn_k)),
                    k_bias=self.k_cond_bias[i](cn_k),
                    attn_gate=torch.sigmoid(attn["zero_cond"](queries_single_cond)),
                    ffn_scale=torch.sigmoid(self.ffn_cond_scale[i](cn_f)),
                    ffn_bias=self.ffn_cond_bias[i](cn_f),
                    ffn_gate=torch.sigmoid(ffn["zero_cond"](queries_single_cond)),
                )
            )
        return tuple(blocks)

    def forward_hoisted(
        self,
        queries_act: torch.Tensor,  # [B, S, Q, c]
        hoist: tuple[AtomBlockHoist, ...],
        q_to_k_idx: torch.Tensor,  # [B, S, K] long
        q_to_k_mask: torch.Tensor,  # [B, S, K] bool
    ) -> torch.Tensor:  # [B, S, Q, c]
        """:meth:`forward` on terms from :meth:`hoist`, the batched path only.

        The key mask sits in ``logit_bias``: ``(qk + pl) + m`` becomes
        ``qk + (pl + m)``, the same bits where ``m`` is 0 and -1e9 either way
        where it is not.
        """
        B, S, Q, C = queries_act.shape
        H = self.num_heads
        D = self.head_dim
        K = q_to_k_idx.shape[-1]
        for i, (attn, ffn, h) in enumerate(zip(self.attn_blocks, self.ffn_blocks, hoist, strict=True)):
            keys_act = _gather_q_to_k(queries_act, q_to_k_idx, q_to_k_mask)
            xn_q = h.q_scale * self.attn_norms[i](queries_act) + h.q_bias
            xn_k = h.k_scale * layer_norm(keys_act, [C]) + h.k_bias

            q = attn["to_q"](xn_q).view(B, S, Q, H, D).permute(0, 1, 3, 2, 4)
            k = attn["to_k"](xn_k).view(B, S, K, H, D).permute(0, 1, 3, 2, 4)
            v = attn["to_v"](xn_k).view(B, S, K, H, D).permute(0, 1, 3, 2, 4)
            logits = torch.matmul(q * (D**-0.5), k.transpose(-2, -1)) + h.logit_bias
            w = F.softmax(logits.float(), dim=-1).to(queries_act.dtype)
            out = torch.matmul(w, v).permute(0, 1, 3, 2, 4).reshape(B, S, Q, C)

            out = out * torch.sigmoid(attn["to_g"](xn_q))
            queries_act = queries_act + h.attn_gate * attn["out_proj"](out)

            xn = h.ffn_scale * self.ffn_norms[i](queries_act) + h.ffn_bias
            a, b = ffn["linear1"](xn).chunk(2, dim=-1)
            queries_act = queries_act + h.ffn_gate * ffn["linear2"](F.silu(a) * b)
        return queries_act

    def forward(
        self,
        queries_act: torch.Tensor,  # [B?, S, Q, c]
        queries_mask: torch.Tensor,  # [B?, S, Q] float
        queries_single_cond: torch.Tensor,  # [B?, S, Q, c]
        keys_single_cond: torch.Tensor,  # [B?, S, K, c]
        keys_mask: torch.Tensor,  # [B?, S, K] float
        pair_cond: torch.Tensor,  # [B?, S, Q, K, cp]
        q_to_k_idx: torch.Tensor,  # [S, K] long — indices into [S*Q] (shared across batch)
        q_to_k_mask: torch.Tensor,  # [S, K] bool
    ) -> torch.Tensor:  # [B?, S, Q, c]
        # Detect leading batch dim (queries_act is 3D unbatched, 4D batched).
        has_batch = queries_act.dim() == 4
        if has_batch:
            B, S, Q, C = queries_act.shape
        else:
            S, Q, C = queries_act.shape
            B = 1
        H = self.num_heads
        D = self.head_dim
        K = keys_mask.shape[-1]

        # Pre-compute pair logits for all blocks (scale-only LN). Permute the
        # ``num_blocks * H`` axis to the front so block ``i`` can be sliced
        # cheaply on the leading dim. ``...`` carries any leading batch dims.
        pair_normed = layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w
        pair_logits = self.pair_logits_proj(pair_normed)  # [..., S, Q, K, nb*H]
        # Move (nb*H) to the front to slice per block. ``movedim`` is
        # batch-friendly (handles any leading dims).
        pair_logits = pair_logits.movedim(-1, 0)  # [nb*H, ..., S, Q, K]
        pair_logits = pair_logits.reshape(self.num_blocks, H, *pair_logits.shape[1:])  # [nb, H, ..., S, Q, K]

        for i, (attn, ffn) in enumerate(zip(self.attn_blocks, self.ffn_blocks, strict=True)):
            pl = pair_logits[i]  # [H, ..., S, Q, K]

            # Re-gather keys from current queries_act (AF3 does this every block).
            keys_act = _gather_q_to_k(queries_act, q_to_k_idx, q_to_k_mask)  # [..., S, K, c]

            # --- q-side AdaLN ---
            xn_q = self.attn_norms[i](queries_act)  # [..., S, Q, c]
            cn_q = layer_norm(queries_single_cond, [C]) * self.attn_cond_w[i]
            xn_q = torch.sigmoid(self.attn_cond_scale[i](cn_q)) * xn_q + self.attn_cond_bias[i](cn_q)

            # --- k-side AdaLN ---
            xn_k = layer_norm(keys_act, [C])  # [..., S, K, c]
            cn_k = layer_norm(keys_single_cond, [C]) * self.k_cond_w[i]
            xn_k = torch.sigmoid(self.k_cond_scale[i](cn_k)) * xn_k + self.k_cond_bias[i](cn_k)

            # --- Cross-attention. Reshape last 3 dims (S, Q, c) to (S, Q, H, D),
            # then permute the head dim in front of (Q or K). With or without
            # leading batch dims, the relative positions are the same.
            if has_batch:
                q = attn["to_q"](xn_q).view(B, S, Q, H, D).permute(0, 1, 3, 2, 4)  # [B,S,H,Q,D]
                k = attn["to_k"](xn_k).view(B, S, K, H, D).permute(0, 1, 3, 2, 4)  # [B,S,H,K,D]
                v = attn["to_v"](xn_k).view(B, S, K, H, D).permute(0, 1, 3, 2, 4)  # [B,S,H,K,D]
                logits = torch.matmul(q * (D**-0.5), k.transpose(-2, -1))  # [B,S,H,Q,K]
                # pl has shape [H, B, S, Q, K]; broadcast to logits' [B,S,H,Q,K]
                logits = logits + pl.permute(1, 2, 0, 3, 4)
                key_mask = keys_mask.unsqueeze(-2).unsqueeze(-2)  # [B,S,1,1,K]
                logits = logits + 1e9 * (key_mask - 1.0)
                w = F.softmax(logits.float(), dim=-1).to(queries_act.dtype)
                out = torch.matmul(w, v).permute(0, 1, 3, 2, 4).reshape(B, S, Q, C)
            else:
                q = attn["to_q"](xn_q).view(S, Q, H, D).permute(0, 2, 1, 3)  # [S,H,Q,D]
                k = attn["to_k"](xn_k).view(S, K, H, D).permute(0, 2, 1, 3)
                v = attn["to_v"](xn_k).view(S, K, H, D).permute(0, 2, 1, 3)
                logits = torch.matmul(q * (D**-0.5), k.transpose(-2, -1))  # [S,H,Q,K]
                logits = logits + pl.permute(1, 0, 2, 3)
                key_mask = keys_mask.unsqueeze(1).unsqueeze(2)  # [S,1,1,K]
                logits = logits + 1e9 * (key_mask - 1.0)
                w = F.softmax(logits.float(), dim=-1).to(queries_act.dtype)
                out = torch.matmul(w, v).permute(0, 2, 1, 3).reshape(S, Q, C)

            g = torch.sigmoid(attn["to_g"](xn_q))  # gate queries
            out = out * g
            out = attn["out_proj"](out)
            gate = torch.sigmoid(attn["zero_cond"](queries_single_cond))
            queries_act = queries_act + gate * out

            # --- FFN on queries only ---
            xn = self.ffn_norms[i](queries_act)
            cn = layer_norm(queries_single_cond, [C]) * self.ffn_cond_w[i]
            xn = torch.sigmoid(self.ffn_cond_scale[i](cn)) * xn + self.ffn_cond_bias[i](cn)
            ff = ffn["linear1"](xn)
            a, b = ff.chunk(2, dim=-1)
            ff = F.silu(a) * b
            ff_out = ffn["linear2"](ff)
            gate = torch.sigmoid(ffn["zero_cond"](queries_single_cond))
            queries_act = queries_act + gate * ff_out

        return queries_act


class EvoformerConditioningEncoder(nn.Module):
    """Evoformer per-atom conditioning encoder.

    Produces token-level activations from reference atom structure features.
    These are concatenated with raw token features to form the full target_feat
    input to the Evoformer (total dim = 63 raw + 384 here = 447).

    No trunk conditioning (pair/single from evoformer are not yet available).
    No noisy atom positions (uses only reference structure).

    Haiku scope prefix: 'diffuser/evoformer_conditioning_'

    Attribute names (for converter):
        per_atom          → AtomPerAtomConditioning (embed_ref_*, single_to_pair_*, ...)
        single_to_pair_row_enc   ← single_to_pair_cond_row_1
        single_to_pair_col_enc   ← single_to_pair_cond_col_1
        enc_embed_pair_offsets   ← embed_pair_offsets_1
        enc_embed_pair_distances ← embed_pair_distances_1
        enc_embed_offsets_valid  ← embed_pair_offsets_valid
        atom_transformer  → AtomCrossAttTransformer (3 blocks)
        proj_for_aggr     ← project_atom_features_for_aggr  (Linear c→384, no bias)
    """

    def __init__(
        self,
        per_atom_channels: int = 128,
        per_token_channels: int = 384,
        per_atom_pair_channels: int = 16,
        num_blocks: int = 3,
        Q: int = 32,
        K: int = 128,
        num_element_classes: int = 128,
        num_atom_name_chars: int = 64,
        atom_name_length: int = 4,
        ref_element_zero_index: bool = False,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        ct = per_token_channels
        self.Q = Q
        self.K = K

        self.per_atom = AtomPerAtomConditioning(
            c, cp, num_element_classes, num_atom_name_chars, atom_name_length, ref_element_zero_index
        )

        # Encoder-specific additional pair conditioning (_1 variants in AF3)
        self.single_to_pair_row_enc = LinearNoBias(c, cp)
        self.single_to_pair_col_enc = LinearNoBias(c, cp)
        self.enc_embed_pair_offsets = LinearNoBias(3, cp)
        self.enc_embed_pair_distances = LinearNoBias(1, cp)
        self.enc_embed_offsets_valid = LinearNoBias(1, cp)

        # Cross-attention transformer
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=num_blocks)

        # Project atom features → token level (masked mean)
        self.proj_for_aggr = LinearNoBias(c, ct)

    def forward(
        self,
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M] float
        ref_element: torch.Tensor,  # [B?, N, M] int64
        ref_charge: torch.Tensor,  # [B?, N, M] float
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4] int64
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
    ) -> torch.Tensor:
        """Returns token_act ``[B?, N, per_token_channels]``."""
        N, M = ref_pos.shape[-3], ref_pos.shape[-2]

        # Build flat atom subsets (batched if ref_mask is 3D)
        subsets = build_atom_subsets(ref_mask.bool(), self.Q, self.K)

        # Single conditioning in token-atoms space [B?, N, M, c]
        atom_single = self.per_atom.single_cond(ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars)

        # Gather to subset layouts (helpers handle batched index tensors)
        queries_single = _gather_to_subsets(atom_single, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [B?, S, Q, c]
        queries_mask = subsets.ta_to_q_mask.to(atom_single.dtype)  # [B?, S, Q]
        keys_single = _gather_q_to_k(queries_single, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [B?, S, K, c]
        keys_mask = subsets.q_to_k_mask.to(atom_single.dtype)  # [B?, S, K]

        # Gather ref_pos to subset layouts
        queries_ref_pos = _gather_to_subsets(ref_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [B?, S, Q, 3]
        keys_ref_pos = _gather_q_to_k(queries_ref_pos, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [B?, S, K, 3]

        # offsets_valid: atom pairs sharing one reference conformer (AF3 Alg. 5 v_lm)
        offsets_valid = _offsets_valid(subsets, ref_space_uid)  # [B?, S, Q, K]

        # Build cross-subset pair_cond [B?, S, Q, K, cp]
        row = self.single_to_pair_row_enc(F.relu(queries_single))  # [B?, S, Q, cp]
        col = self.single_to_pair_col_enc(F.relu(keys_single))  # [B?, S, K, cp]
        atom_pair = row.unsqueeze(-2) + col.unsqueeze(-3)  # [B?, S, Q, K, cp]

        offsets = queries_ref_pos.unsqueeze(-2) - keys_ref_pos.unsqueeze(-3)  # [B?, S, Q, K, 3]
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        ov = offsets_valid.unsqueeze(-1)  # [B?, S, Q, K, 1]
        atom_pair = atom_pair + self.enc_embed_pair_offsets(offsets) * ov
        atom_pair = atom_pair + self.enc_embed_pair_distances(1.0 / (1.0 + sq_dists)) * ov
        atom_pair = atom_pair + self.enc_embed_offsets_valid(ov.to(atom_pair.dtype))

        # Pair MLP applied LAST (after all additive terms, matching AF3)
        atom_pair = self.per_atom.apply_pair_mlp(atom_pair)

        # Cross-attention transformer
        queries_act = self.atom_transformer(
            queries_act=queries_single,
            queries_mask=queries_mask,
            queries_single_cond=queries_single,
            keys_single_cond=keys_single,
            keys_mask=keys_mask,
            pair_cond=atom_pair,
            q_to_k_idx=subsets.q_to_k_idx,
            q_to_k_mask=subsets.q_to_k_mask,
        )
        queries_act = queries_act * queries_mask.unsqueeze(-1)

        # Scatter back to token-atoms and aggregate to token level (masked mean)
        token_atoms_act = _scatter_from_subsets(
            self.proj_for_aggr(queries_act), subsets.q_to_ta_idx, subsets.q_to_ta_mask, N, M
        )  # [N, M, ct]
        denom = ref_mask.sum(-1, keepdim=True).clamp(min=1e-6)
        token_act = (F.relu(token_atoms_act) * ref_mask.unsqueeze(-1)).sum(-2) / denom

        return token_act  # [N, ct=384]


class AtomEncoder(nn.Module):
    """Atom cross-attention encoder: atoms → tokens (for diffusion head).

    Conditions on trunk single/pair representations + noisy atom positions.

    Haiku scope prefix: 'diffusion_*' (within diffuser/~/diffusion_head/)

    Attribute names (for converter):
        per_atom                    → AtomPerAtomConditioning
        lnorm_trunk_single_w        ← lnorm_trunk_single_cond  (scale only)
        embed_trunk_single          ← embed_trunk_single_cond
        lnorm_trunk_pair_w          ← lnorm_trunk_pair_cond    (scale only)
        embed_trunk_pair            ← embed_trunk_pair_cond
        single_to_pair_row_enc      ← single_to_pair_cond_row_1
        single_to_pair_col_enc      ← single_to_pair_cond_col_1
        enc_embed_pair_offsets      ← embed_pair_offsets_1
        enc_embed_pair_distances    ← embed_pair_distances_1
        enc_embed_offsets_valid     ← embed_pair_offsets_valid
        _pos_proj                   ← atom_positions_to_features (Linear(3,c))
        atom_transformer            → AtomCrossAttTransformer
        proj_for_aggr               ← project_atom_features_for_aggr
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_token_channels: int,
        per_atom_pair_channels: int,
        atom_transformer_num_blocks: int = 3,
        Q: int = 32,
        K: int = 128,
        num_element_classes: int = 128,
        num_atom_name_chars: int = 64,
        atom_name_length: int = 4,
        ref_element_zero_index: bool = False,
        c_trunk_single: int = 384,
        c_trunk_pair: int = 128,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        ct = per_token_channels
        self.Q = Q
        self.K = K

        self.per_atom = AtomPerAtomConditioning(
            c, cp, num_element_classes, num_atom_name_chars, atom_name_length, ref_element_zero_index
        )

        # Trunk single conditioning. ``c_trunk_single`` is the width of
        # ``embeddings["single"]`` (evoformer.seq_channel); the AF3 default is 384.
        # AF3: embed_trunk_single_cond uses final_init (zeros)
        self.lnorm_trunk_single_w = nn.Parameter(torch.ones(c_trunk_single))
        self.embed_trunk_single = LinearNoBias(c_trunk_single, c, initializer="zeros")

        # Trunk pair conditioning. NOTE the input here is the DiffusionConditioning
        # output (``pair_cond``), so this is ``diffusion.conditioning_pair_channel``
        # — NOT evoformer.pair_channel. Both were hardcoded to the AF3 widths until
        # a 256-wide pair track tripped the mismatch inside sample()/compute_static.
        # AF3: embed_trunk_pair_cond uses final_init (zeros)
        self.lnorm_trunk_pair_w = nn.Parameter(torch.ones(c_trunk_pair))
        self.embed_trunk_pair = LinearNoBias(c_trunk_pair, cp, initializer="zeros")
        # ``OptimConfig.pair_chunk_budget_bytes``; set by ``DiffusionHead``.
        self.pair_chunk_budget = 0

        # Encoder-specific additional pair conditioning (_1 variants)
        self.single_to_pair_row_enc = LinearNoBias(c, cp)
        self.single_to_pair_col_enc = LinearNoBias(c, cp)
        self.enc_embed_pair_offsets = LinearNoBias(3, cp)
        self.enc_embed_pair_distances = LinearNoBias(1, cp)
        self.enc_embed_offsets_valid = LinearNoBias(1, cp)

        # Project noisy atom positions (3D) → per_atom_channels
        self._pos_proj = LinearNoBias(3, c)

        # Cross-attention transformer
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=atom_transformer_num_blocks)

        # Project atom features → token level
        self.proj_for_aggr = LinearNoBias(c, ct)

    def compute_static(
        self,
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M] float
        ref_element: torch.Tensor,  # [B?, N, M] int64
        ref_charge: torch.Tensor,  # [B?, N, M] float
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4] int64
        trunk_single: torch.Tensor,  # [B?, N, c_single=384]
        trunk_pair: torch.Tensor,  # [B?, N, N, c_pair=128]
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
    ) -> AtomEncoderStatic:
        """Compute the noise-independent encoder state.

        The same static can be reused across many diffusion samples that share
        the same molecule + trunk embeddings (mini-rollout, training chunked
        samples, inference 200-step loop). Pass the result to ``forward`` via
        the ``static`` kwarg, optionally after :meth:`AtomEncoderStatic.expand_for_samples`.
        """
        has_batch = ref_pos.dim() == 4
        N = ref_pos.shape[-3]

        subsets = build_atom_subsets(ref_mask.bool(), self.Q, self.K)

        atom_single = self.per_atom.single_cond(
            ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars
        )  # [B?, N, M, c]

        ts_norm = layer_norm(trunk_single, [trunk_single.shape[-1]]) * self.lnorm_trunk_single_w
        ts_proj = self.embed_trunk_single(ts_norm)  # [B?, N, c]
        atom_single = atom_single + ts_proj.unsqueeze(-2)

        queries_single = _gather_to_subsets(atom_single, subsets.ta_to_q_idx, subsets.ta_to_q_mask)
        queries_mask = subsets.ta_to_q_mask.to(atom_single.dtype)
        keys_single = _gather_q_to_k(queries_single, subsets.q_to_k_idx, subsets.q_to_k_mask)
        keys_mask = subsets.q_to_k_mask.to(atom_single.dtype)

        queries_ref_pos = _gather_to_subsets(ref_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)
        keys_ref_pos = _gather_q_to_k(queries_ref_pos, subsets.q_to_k_idx, subsets.q_to_k_mask)
        offsets_valid = _offsets_valid(subsets, ref_space_uid)

        # --- Trunk pair contribution ---
        tp_proj = map_pair_rows(  # [B?, N, N, cp]
            lambda rows: self.embed_trunk_pair(layer_norm(rows, [rows.shape[-1]]) * self.lnorm_trunk_pair_w),
            trunk_pair,
            self.pair_chunk_budget,
            4 * trunk_pair.shape[-1],
        )
        pair_idx = subsets.token_for_query.unsqueeze(-1) * N + subsets.token_for_key.unsqueeze(-2)
        pair_valid = subsets.ta_to_q_mask.unsqueeze(-1) & subsets.q_to_k_mask.unsqueeze(-2)
        if has_batch:
            B = tp_proj.shape[0]
            trunk_pair_contrib = _batched_take(tp_proj.reshape(B, N * N, -1), pair_idx, pair_valid)
        else:
            trunk_pair_contrib = _flat_take(tp_proj.reshape(N * N, -1), pair_idx, pair_valid)

        # --- Cross-subset pair_cond ---
        row = self.single_to_pair_row_enc(F.relu(queries_single))
        col = self.single_to_pair_col_enc(F.relu(keys_single))
        atom_pair = row.unsqueeze(-2) + col.unsqueeze(-3)

        offsets = queries_ref_pos.unsqueeze(-2) - keys_ref_pos.unsqueeze(-3)
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        ov = offsets_valid.unsqueeze(-1)
        atom_pair = atom_pair + self.enc_embed_pair_offsets(offsets) * ov
        atom_pair = atom_pair + self.enc_embed_pair_distances(1.0 / (1.0 + sq_dists)) * ov
        atom_pair = atom_pair + self.enc_embed_offsets_valid(ov.to(atom_pair.dtype))
        atom_pair = atom_pair + trunk_pair_contrib
        atom_pair = self.per_atom.apply_pair_mlp(atom_pair)

        return AtomEncoderStatic(
            subsets=subsets,
            queries_single=queries_single,
            queries_mask=queries_mask,
            keys_single=keys_single,
            keys_mask=keys_mask,
            atom_pair=atom_pair,
            ref_mask=ref_mask,
        )

    def forward(
        self,
        ref_pos: torch.Tensor | None = None,  # [B?, N, M, 3]
        ref_mask: torch.Tensor | None = None,  # [B?, N, M] float
        ref_element: torch.Tensor | None = None,  # [B?, N, M] int64
        ref_charge: torch.Tensor | None = None,  # [B?, N, M] float
        ref_atom_name_chars: torch.Tensor | None = None,  # [B?, N, M, 4] int64
        ref_space_uid: torch.Tensor | None = None,  # [B?, N, M] int64
        noisy_atom_pos: torch.Tensor | None = None,  # [B?, N, M, 3] or None
        trunk_single: torch.Tensor | None = None,  # [B?, N, c_single=384]
        trunk_pair: torch.Tensor | None = None,  # [B?, N, N, c_pair=128]
        static: AtomEncoderStatic | None = None,
    ) -> AtomCrossAttEncoderOutput:
        """Run the encoder for a single noise sample.

        When ``static`` is provided, the noise-independent state is reused and
        ``ref_*`` / ``trunk_*`` arguments are ignored. Otherwise, the static is
        recomputed inline from those inputs.
        """
        if static is None:
            static = self.compute_static(
                ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars, trunk_single, trunk_pair, ref_space_uid
            )

        subsets = static.subsets
        queries_single = static.queries_single
        queries_mask = static.queries_mask
        keys_single = static.keys_single
        keys_mask = static.keys_mask
        atom_pair = static.atom_pair
        ref_mask_eff = static.ref_mask
        N, M = subsets.N, subsets.M

        # --- Noise-dependent: project noisy positions and add to queries ---
        if noisy_atom_pos is not None:
            noisy_q = _gather_to_subsets(noisy_atom_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)
            queries_act = queries_single + self._pos_proj(noisy_q) * queries_mask.unsqueeze(-1)
        else:
            queries_act = queries_single

        if static.encoder_hoist is not None:
            queries_act = self.atom_transformer.forward_hoisted(
                queries_act, static.encoder_hoist, subsets.q_to_k_idx, subsets.q_to_k_mask
            )
        else:
            queries_act = self.atom_transformer(
                queries_act=queries_act,
                queries_mask=queries_mask,
                queries_single_cond=queries_single,
                keys_single_cond=keys_single,
                keys_mask=keys_mask,
                pair_cond=atom_pair,
                q_to_k_idx=subsets.q_to_k_idx,
                q_to_k_mask=subsets.q_to_k_mask,
            )
        queries_act = queries_act * queries_mask.unsqueeze(-1)

        atom_proj = _scatter_from_subsets(
            self.proj_for_aggr(queries_act), subsets.q_to_ta_idx, subsets.q_to_ta_mask, N, M
        )
        denom = ref_mask_eff.sum(-1, keepdim=True).clamp(min=1e-6)
        token_act = (F.relu(atom_proj) * ref_mask_eff.unsqueeze(-1)).sum(-2) / denom

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=queries_act,
            queries_mask=queries_mask,
            queries_single_cond=queries_single,
            keys_mask=keys_mask,
            keys_single_cond=keys_single,
            pair_cond=atom_pair,
            subsets=subsets,
            decoder_hoist=static.decoder_hoist,
        )


class AtomDecoder(nn.Module):
    """Atom cross-attention decoder: tokens → atom position updates.

    Haiku scope prefix: 'diffusion_*' (within diffuser/~/diffusion_head/)

    Attribute names (for converter):
        proj_tokens    ← project_token_features_for_broadcast
        atom_transformer → AtomCrossAttTransformer
        atom_norm_w    ← atom_features_layer_norm  (scale only, no offset)
        pos_update     ← atom_features_to_position_update
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_token_channels: int,
        per_atom_pair_channels: int,
        atom_transformer_num_blocks: int = 3,
    ):
        super().__init__()
        c = per_atom_channels
        ct = per_token_channels
        cp = per_atom_pair_channels

        self.proj_tokens = LinearNoBias(ct, c)
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=atom_transformer_num_blocks)
        self.atom_norm_w = nn.Parameter(torch.ones(c))
        # AF3: atom_features_to_position_update uses final_init (zeros)
        self.pos_update = LinearNoBias(c, 3, initializer="zeros")

    def forward(
        self,
        token_act: torch.Tensor,  # [B?, N, ct]
        enc: AtomCrossAttEncoderOutput,
    ) -> torch.Tensor:  # [B?, N, M, 3]
        N, M = enc.subsets.N, enc.subsets.M
        c = enc.skip_connection.shape[-1]

        # Project token features → per_atom_channels, broadcast to atoms.
        # Negative-axis ops handle both unbatched [N, ct] and batched [B, N, ct].
        token_proj = self.proj_tokens(token_act)  # [B?, N, c]
        token_atoms = token_proj.unsqueeze(-2).expand(*token_proj.shape[:-1], M, c)
        queries_act = _gather_to_subsets(token_atoms, enc.subsets.ta_to_q_idx, enc.subsets.ta_to_q_mask)
        queries_act = queries_act + enc.skip_connection
        queries_act = queries_act * enc.queries_mask.unsqueeze(-1)

        # Cross-attention transformer (reuses encoder's pair/single conditioning)
        if enc.decoder_hoist is not None:
            queries_act = self.atom_transformer.forward_hoisted(
                queries_act, enc.decoder_hoist, enc.subsets.q_to_k_idx, enc.subsets.q_to_k_mask
            )
        else:
            queries_act = self.atom_transformer(
                queries_act=queries_act,
                queries_mask=enc.queries_mask,
                queries_single_cond=enc.queries_single_cond,
                keys_single_cond=enc.keys_single_cond,
                keys_mask=enc.keys_mask,
                pair_cond=enc.pair_cond,
                q_to_k_idx=enc.subsets.q_to_k_idx,
                q_to_k_mask=enc.subsets.q_to_k_mask,
            )
        queries_act = queries_act * enc.queries_mask.unsqueeze(-1)

        # Normalize and project to 3D position updates
        queries_act = layer_norm(queries_act, [c]) * self.atom_norm_w
        pos_update_q = self.pos_update(queries_act)  # [S, Q, 3]

        # Scatter back to token-atoms [N, M, 3]
        pos_update = _scatter_from_subsets(pos_update_q, enc.subsets.q_to_ta_idx, enc.subsets.q_to_ta_mask, N, M)
        return pos_update
