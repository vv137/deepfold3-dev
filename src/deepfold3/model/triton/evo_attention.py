"""Fused flash-attention with pair bias for Evoformer-style attention.

Provides a Triton kernel that computes multi-head attention with an additive
pair bias term, avoiding the O(N^2) memory of materializing the full attention
matrix. Supports both forward and backward passes.

The kernel expects 5D tensors in [B, N_seq, N_res, H, D] layout (DeepFold3
convention) and internally transposes to [B, N_seq, H, N_res, D] for tiling.
"""

# ``O`` is the Flash-Attention-paper output tensor; the single-letter name is
# domain convention we preserve for readability against the published math.
# ruff: noqa: E741

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _is_hip() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "hip"


if HAS_TRITON:

    @triton.jit
    def _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        res_mask_block_ptr,
        pair_bias_block_ptr,
        block_index_q,
        DIM,
        stride_K_seq,
        stride_V_seq,
        stride_mask_seq,
        stride_pair_bias_seq2,
        softmax_scale,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        offs_q: tl.constexpr,
        offs_kv: tl.constexpr,
        offs_d: tl.constexpr,
        SEQ_LEN: tl.constexpr,
    ):
        lo, hi = 0, SEQ_LEN

        for start_kv in range(lo, hi, BLOCK_SIZE_KV):
            start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

            if EVEN_Q & EVEN_KV:
                pair_bias_block = tl.load(pair_bias_block_ptr)
                res_mask_block = tl.load(res_mask_block_ptr).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_block = tl.load(K_block_ptr)
                    V_block = tl.load(V_block_ptr)
                else:
                    K_block = tl.load(K_block_ptr, mask=offs_d[:, None] < DIM, other=0.0)
                    V_block = tl.load(V_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
            else:
                pair_bias_block = tl.load(
                    pair_bias_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN) & ((start_kv + offs_kv)[None, :] < SEQ_LEN),
                    other=float("-inf"),
                )
                res_mask_block = tl.load(
                    res_mask_block_ptr, mask=(start_kv + offs_kv)[None, :] < SEQ_LEN, other=float("-inf")
                ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_block = tl.load(K_block_ptr, mask=(start_kv + offs_kv)[None, :] < SEQ_LEN, other=0.0)
                    V_block = tl.load(V_block_ptr, mask=(start_kv + offs_kv)[:, None] < SEQ_LEN, other=0.0)
                else:
                    K_block = tl.load(
                        K_block_ptr,
                        mask=((start_kv + offs_kv)[None, :] < SEQ_LEN) & (offs_d[:, None] < DIM),
                        other=0.0,
                    )
                    V_block = tl.load(
                        V_block_ptr,
                        mask=((start_kv + offs_kv)[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                        other=0.0,
                    )

            # ``softmax_scale`` is applied to the fp32 ``tl.dot`` accumulator, NOT
            # pre-multiplied into Q. Pre-scaling re-rounds every Q element to the
            # input dtype: in bf16 that is a ~2^-9 relative perturbation whose
            # effect on a logit grows with the logit's magnitude, so rows with a
            # large norm (|QK^T/sqrt(D)| ~ 1e2) pick up absolute logit errors of
            # ~3e-2, which the softmax exponentiates into a few-percent weight
            # error. Post-scaling in fp32 is exact and costs one fp32 multiply per
            # score tile. See tests/triton/test_evo_attn_robustness.py::
            # test_row_norm_heterogeneity.
            QK_block = tl.dot(Q_block, K_block) * softmax_scale + pair_bias_block + res_mask_block

            if not EVEN_KV:
                QK_block += tl.where((start_kv + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block = QK_block - m_ij[:, None]

            inv_ln2: tl.constexpr = 1.4426950408889634  # 1/ln(2) — exp2 idiom (trifast/protenix)
            P_block = tl.math.exp2(QK_block * inv_ln2)
            l_ij = tl.sum(P_block, 1)

            alpha = tl.math.exp2((m_i - m_ij) * inv_ln2)
            l_i = l_i * alpha + l_ij

            P_block = P_block.to(V_block.dtype)
            O_block = O_block * alpha[:, None]
            O_block = tl.dot(P_block, V_block, O_block)

            m_i = m_ij

            V_block_ptr += BLOCK_SIZE_KV * stride_V_seq
            K_block_ptr += BLOCK_SIZE_KV * stride_K_seq
            pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
            res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

        return O_block, l_i, m_i

    from deepfold3.model.triton.cache import persistent_autotune

    # Autotune over tile sizes for the forward attention kernel.
    # Key insight: optimal BLOCK_SIZE_Q/KV depend on SEQ_LEN (N=128..768).
    # Small N benefits from larger Q blocks; large N needs smaller blocks
    # to avoid register pressure. cuEQ uses fixed tiles per architecture;
    # we autotune at runtime for maximum portability.
    # num_stages=1 included for RTX 3090/4090 with less shared memory.
    _ATTN_FWD_CONFIGS = [
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw)
        for bq in (16, 32, 64)
        for bkv in (32, 64, 128)
        for ns in (1, 2, 3)
        for nw in (4, 8)
    ]

    # Backward kernels: smaller config space (fixed Q/KV roles).
    # num_stages 1-3: 1 for Ampere (48KB shared), 2-3 for Ada/Hopper/Blackwell.
    _ATTN_BWD_DK_DV_CONFIGS = [
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw)
        for bq in (32, 64)
        for bkv in (16, 32)
        for ns in (1, 2, 3)
        for nw in (4, 8)
    ]

    _ATTN_BWD_DQ_CONFIGS = [
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw)
        for bq in (16, 32)
        for bkv in (32, 64)
        for ns in (1, 2, 3)
        for nw in (4, 8)
    ]

    _ATTN_BWD_DB_CONFIGS = [
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_KV": bkv}, num_stages=ns, num_warps=nw)
        for bq in (16, 32, 64)
        for bkv in (16, 32, 64)
        for ns in (1, 2, 3)
        for nw in (4, 8)
    ]

    @persistent_autotune(configs=_ATTN_FWD_CONFIGS, key=["SEQ_LEN", "DIM"])
    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_fwd(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        M,
        O,
        stride_Q_batch,
        stride_Q_msa,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_msa,
        stride_K_head,
        stride_K_seq,
        stride_K_dim,
        stride_V_batch,
        stride_V_msa,
        stride_V_head,
        stride_V_seq,
        stride_V_dim,
        stride_O_batch,
        stride_O_msa,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        BATCH_SIZE,
        HEAD: tl.constexpr,
        N_SEQ: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        block_index_q = tl.program_id(0).to(tl.int64)
        index_batch_msa_head = tl.program_id(1).to(tl.int64)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        qvk_offset = index_batch * stride_Q_batch + index_msa * stride_Q_msa + index_head * stride_Q_head
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        offs_d = tl.arange(0, BLOCK_DIM)

        Q_block_ptr = Q + qvk_offset + (offs_q[:, None] * stride_Q_seq + offs_d[None, :])
        V_block_ptr = V + qvk_offset + (offs_kv[:, None] * stride_V_seq + offs_d[None, :])
        K_block_ptr = K + qvk_offset + (offs_kv[None, :] * stride_K_seq + offs_d[:, None])
        pair_bias_block_ptr = (
            pair_bias
            + index_batch * stride_pair_bias_batch
            + index_head * stride_pair_bias_head
            + (offs_q[:, None] * stride_pair_bias_seq1 + offs_kv[None, :] * stride_pair_bias_seq2)
        )
        O_block_ptr = O + qvk_offset + (offs_q[:, None] * stride_O_seq + offs_d[None, :])
        res_mask_block_ptr = (
            res_mask
            + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
            + (offs_kv[None, :] * stride_mask_seq)
        )

        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
        O_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                Q_block = tl.load(Q_block_ptr)
            else:
                Q_block = tl.load(Q_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
        else:
            if EVEN_DIM:
                Q_block = tl.load(Q_block_ptr, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
            else:
                Q_block = tl.load(Q_block_ptr, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM), other=0.0)

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            res_mask_block_ptr,
            pair_bias_block_ptr,
            block_index_q,
            DIM,
            stride_K_seq,
            stride_V_seq,
            stride_mask_seq,
            stride_pair_bias_seq2,
            softmax_scale,
            EVEN_Q,
            EVEN_KV,
            EVEN_DIM,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            BLOCK_DIM,
            offs_q,
            offs_kv,
            offs_d,
            SEQ_LEN,
        )

        # Fully-masked row handling (sentinel pattern, matches trifast/Tri Dao
        # flash-attn). A row whose attention keys are ALL mask=-1e9 has m_i
        # dominated by that mask value after the loop. In fp32, ULP near -1e9
        # is ~128 >> log(N) (~6), so backward's P = exp2((QK - LSE) * inv_ln2)
        # silently computes exp2(0) = 1 instead of 1/N, giving an exact N×
        # gradient amplification. Fix: store LSE = +inf as sentinel. In backward,
        # QK - (+inf) = -inf, exp2(-inf) = 0 naturally — no explicit sentinel
        # check needed, all gradient contributions collapse to 0.
        row_is_masked = m_i < -1e8  # any valid QK sits far above this
        l_i_safe = tl.where(row_is_masked, 1.0, l_i)
        m_i = tl.where(row_is_masked, float("inf"), m_i + tl.math.log(l_i_safe))
        O_block = tl.where(row_is_masked[:, None], 0.0, O_block / l_i_safe[:, None])
        O_block = O_block.to(O.type.element_ty)
        m_ptrs = M + index_batch_msa_head * SEQ_LEN + offs_q

        if EVEN_Q:
            tl.store(m_ptrs, m_i)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block)
            else:
                tl.store(O_block_ptr, O_block, mask=offs_d[None, :] < DIM)
        else:
            tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block, mask=offs_q[:, None] < SEQ_LEN)
            else:
                tl.store(O_block_ptr, O_block, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM))

    @persistent_autotune(configs=_ATTN_BWD_DQ_CONFIGS, key=["SEQ_LEN", "DIM"])
    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_d(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        M,
        D_ref,
        D_cen,
        D_imax,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        """Build the per-row centring triple ``(dP_imax, imax, C)`` for the backward.

        The dS-forming kernels need ``dS_j = P_j (dP_j - D)``, ``D = sum_i P_i
        dP_i``, and that difference is pure cancellation at the softmax argmax:
        ``dP_imax`` and ``D`` agree to ``O(e^-m)`` in the logit margin ``m``
        (arXiv:2610.31642 §2). This kernel instead supplies the paper's eq. 5
        form, which never subtracts two O(1) quantities: with ``u_j = dP_j -
        dP_imax`` (exactly 0 at the argmax, by index, not by arithmetic),

            dS_j = P_j (u_j - C),   C = sum_i P_i u_i,

        algebraically identical since ``sum_i P_i = 1``, but every term in ``C``
        is ``O(e^-m)`` so nothing is cancelled or absorbed. ``C`` is accumulated
        online over KV tiles with a re-centring step whenever a tile beats the
        running max, the same bookkeeping the forward's online softmax does for
        ``m_i``: ``C += (old_ref - new_ref) * W`` with ``W = sum of P so far``.

        It replaces three separate collapses, all measured at the shape and
        margin of ``tests/triton/test_evo_attn_softmax_collapse.py`` (B1 S2 N256
        H8 D32, margin 14.45; fraction of rows whose ``dS_imax`` is *exactly* 0):

        (a) ``D = rowsum(dO * O)`` from ``_attn_bwd_preprocess``, which routed
            ``D`` through the forward's bf16 ``P`` operand (77% of rows) and its
            bf16 ``O`` store (87%). Cliff at the bf16 ulp, ``m ~ 5.5``.
        (b) ``P = exp2((QK - M) * 1/ln2)`` itself. The kernel does not hold ``P``;
            it rebuilds it from the saved LSE, which the forward rounded to fp32
            at magnitude ``|QK|``. So the rebuilt ``P`` is the true softmax times
            a common ``1 +- ulp(|QK|)``, and ``QK_imax - M`` resolves
            ``1 - P_imax = O(e^-m)`` only down to that same ``ulp(|QK|)``. At this
            test's spike height ``|QK| ~ 20`` it is 2.4e-6 against ``e^-14.45 =
            5.3e-7``: ``P_imax`` rounds to *exactly* 1.0, and ``sum_j P_j``
            misses 1 by ~1e-6. Both kill the uncentred form, where ``dS`` picks up
            ``(1 - sum_j P_j) * D`` with ``D = O(1)``; the centred form couples
            that same drift only to ``(D - dP_imax) = O(e^-m)``, i.e. 1e-6 *of*
            the answer rather than 4x it. An eager fp32 reference escapes (b)
            entirely because it normalises by a sum (``1 + e^-m`` is
            representable) instead of subtracting a log-sum-exp — which is why
            the fp32 reference's cliff is 16.6 while an LSE-based backward's is
            ``log(1/ulp(|QK|))``, ~13 here and a function of the logit magnitude.
            Recomputing ``D`` in fp32 therefore does *not* fix this kernel:
            measured on GPU it still left 40.1% of rows at exactly zero, median
            retention 0.000, and ``d_pair_bias`` row sums at 18.4x their own peak
            element.
        (c) ``D += tl.sum(P * dP, 1)`` per tile, which adds ``O(e^-m)`` tile
            partials onto a running total already holding the winner's ``O(1)``
            term, below its ulp. Kahan compensation fixes (c) but not (b).

        Separate kernel, not an inline first pass in ``_attn_bwd_dq``, because
        ``_attn_bwd_db`` and ``_attn_bwd_dk_dv`` consume the same triple and
        neither can build it: their programs own an (i, j) tile and a KV block
        respectively, so neither ever sees a whole row of j. One program per
        (q block, batch*msa*head); same grid and tiling as ``_attn_bwd_dq``.

        ``imax`` is stored, and the consumers zero ``u`` at that column by index
        rather than relying on ``dP_j - dP_imax`` to cancel, because the three
        consumers form ``dP`` with three different dots (``dO @ V^T``,
        ``V @ dO^T``, ``dO @ trans(V)``) and a per-element disagreement of half an
        ulp of ``|dP| ~ 0.5`` is 3e-8 against a true ``dS_imax`` of ~1e-7. Away
        from the argmax such a disagreement enters only as ``P_j * delta`` and is
        negligible, so ``D_ref`` need only be the value ``C`` was built with.
        """
        index_batch_msa_head = tl.program_id(1).to(tl.int64)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        offset = (index_batch * stride_batch + index_head * stride_head + index_msa * stride_msa).to(tl.int64)
        offset_seq = (index_batch_msa_head * SEQ_LEN).to(tl.int64)

        Q += offset
        K += offset
        V += offset
        dO += offset
        M += offset_seq
        D_ref += offset_seq
        D_cen += offset_seq
        D_imax += offset_seq

        offs_dim = tl.arange(0, BLOCK_DIM)
        block_index_q = tl.program_id(0).to(tl.int64)
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

        if EVEN_Q & EVEN_KV:
            M_block = tl.load(M + offs_q)
            if EVEN_DIM:
                Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :])
                dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :])
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
        else:
            M_block = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            if EVEN_DIM:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )

        # Fully-masked rows: forward stores LSE = +inf → P = exp2(-inf) = 0 → C = 0
        # and no tile ever beats the running max, so dS = P * (u - C) = 0 for them.
        M_block = M_block[:, None]
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        pair_bias_block_ptr = (
            pair_bias
            + (index_batch * stride_pair_bias_batch + index_head * stride_pair_bias_head)
            + offs_q[:, None] * stride_pair_bias_seq1
            + offs_kv[None, :] * stride_pair_bias_seq2
        )
        res_mask_block_ptr = (
            res_mask
            + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
            + (offs_kv[None, :] * stride_mask_seq)
        )

        kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None]
        vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None]

        # Online centred accumulation. ``run_max``/``ref_block``/``imax_block``
        # track the argmax column's logit, its dP and its index; ``C_block`` the
        # centred Jacobian sum; ``W_block`` the softmax mass seen so far, which is
        # what a re-centring has to rescale.
        run_max = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
        ref_block = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)
        imax_block = tl.zeros([BLOCK_SIZE_Q], dtype=tl.int32)
        C_block = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)
        W_block = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)

        num_steps = (SEQ_LEN + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
        for blk_idx in range(num_steps):
            if EVEN_Q & EVEN_KV:
                pb = tl.load(pair_bias_block_ptr)
                rm = tl.load(res_mask_block_ptr).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_T_block = tl.load(kT_ptrs)
                    V_T_block = tl.load(vT_ptrs)
                else:
                    K_T_block = tl.load(kT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
                    V_T_block = tl.load(vT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
            else:
                pb = tl.load(
                    pair_bias_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN) & ((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN),
                    other=float("-inf"),
                )
                rm = tl.load(
                    res_mask_block_ptr, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=float("-inf")
                ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_T_block = tl.load(kT_ptrs, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=0.0)
                    V_T_block = tl.load(vT_ptrs, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=0.0)
                else:
                    K_T_block = tl.load(
                        kT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )
                    V_T_block = tl.load(
                        vT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )

            # Score recompute, op for op as in ``_attn_bwd_dq``: fp32 post-scale,
            # same exp2 idiom, same saved LSE. The consumers multiply ``C`` by
            # their own ``P``, so a different ``P`` here would put the centring
            # constant and the weights it is subtracted from out of step.
            QK_block = tl.dot(Q_block, K_T_block) * softmax_scale + pb + rm
            if not EVEN_KV:
                QK_block += tl.where((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf"))

            inv_ln2: tl.constexpr = 1.4426950408889634  # 1/ln(2)
            P_block = tl.math.exp2((QK_block - M_block) * inv_ln2)
            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)

            # Re-centre on this tile's argmax if it beats the running one. Strict
            # ``>`` matches tl.argmax's first-index tie-break, so the pair
            # (imax, ref) stays the first global argmax.
            tile_max = tl.max(QK_block, 1)
            tile_arg = tl.argmax(QK_block, 1)
            tile_ref = tl.sum(tl.where(offs_kv[None, :] == tile_arg[:, None], dP_block, 0.0), 1)
            better = tile_max > run_max
            new_ref = tl.where(better, tile_ref, ref_block)
            # ``W_block`` is O(e^-m) until the argmax tile is reached, so this
            # correction is never a cancellation; on the first tile W = 0.
            C_block += (ref_block - new_ref) * W_block
            ref_block = new_ref
            imax_block = tl.where(better, (blk_idx * BLOCK_SIZE_KV + tile_arg).to(tl.int32), imax_block)
            run_max = tl.maximum(run_max, tile_max)

            C_block += tl.sum(P_block * (dP_block - ref_block[:, None]), 1)
            W_block += tl.sum(P_block, 1)

            kT_ptrs += BLOCK_SIZE_KV * stride_seq
            vT_ptrs += BLOCK_SIZE_KV * stride_seq
            pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
            res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

        if EVEN_Q:
            tl.store(D_ref + offs_q, ref_block)
            tl.store(D_cen + offs_q, C_block)
            tl.store(D_imax + offs_q, imax_block)
        else:
            tl.store(D_ref + offs_q, ref_block, mask=offs_q < SEQ_LEN)
            tl.store(D_cen + offs_q, C_block, mask=offs_q < SEQ_LEN)
            tl.store(D_imax + offs_q, imax_block, mask=offs_q < SEQ_LEN)

    @persistent_autotune(configs=_ATTN_BWD_DQ_CONFIGS, key=["SEQ_LEN", "DIM"])
    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_dq(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        dQ,
        dK,
        dV,
        M,
        D_ref,
        D_cen,
        D_imax,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        index_batch_msa_head = tl.program_id(1).to(tl.int64)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        offset = (index_batch * stride_batch + index_head * stride_head + index_msa * stride_msa).to(tl.int64)
        offset_seq = (index_batch_msa_head * SEQ_LEN).to(tl.int64)

        Q += offset
        K += offset
        V += offset
        dO += offset
        dQ += offset
        dK += offset
        dV += offset
        M += offset_seq
        D_ref += offset_seq
        D_cen += offset_seq
        D_imax += offset_seq

        offs_dim = tl.arange(0, BLOCK_DIM)
        index_block_kv = tl.program_id(0).to(tl.int64)
        offs_q = index_block_kv * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

        dQ_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

        if EVEN_Q & EVEN_KV:
            M_block = tl.load(M + offs_q)
            ref = tl.load(D_ref + offs_q)
            cen = tl.load(D_cen + offs_q)
            imax = tl.load(D_imax + offs_q)
            if EVEN_DIM:
                Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :])
                dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :])
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
        else:
            M_block = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            ref = tl.load(D_ref + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            cen = tl.load(D_cen + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            imax = tl.load(D_imax + offs_q, mask=offs_q < SEQ_LEN, other=-1)
            if EVEN_DIM:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :], mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )

        # Fully-masked rows: forward stores LSE = +inf sentinel. Natural math
        # handles the rest — QK - (+inf) = -inf → exp2(-inf) = 0 → P = 0 →
        # dQ/dK/dV/d_pair_bias contributions vanish. No explicit check needed.
        M_block = M_block[:, None]
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        pair_bias_block_ptr = (
            pair_bias
            + (index_batch * stride_pair_bias_batch + index_head * stride_pair_bias_head)
            + offs_q[:, None] * stride_pair_bias_seq1
            + offs_kv[None, :] * stride_pair_bias_seq2
        )
        res_mask_block_ptr = (
            res_mask
            + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
            + (offs_kv[None, :] * stride_mask_seq)
        )

        kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None]
        vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None]

        num_steps = (SEQ_LEN + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
        for blk_idx in range(num_steps):
            if EVEN_Q & EVEN_KV:
                pb = tl.load(pair_bias_block_ptr)
                rm = tl.load(res_mask_block_ptr).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_T_block = tl.load(kT_ptrs)
                    V_T_block = tl.load(vT_ptrs)
                else:
                    K_T_block = tl.load(kT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
                    V_T_block = tl.load(vT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
            else:
                pb = tl.load(
                    pair_bias_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN) & ((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN),
                    other=float("-inf"),
                )
                rm = tl.load(
                    res_mask_block_ptr, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=float("-inf")
                ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_T_block = tl.load(kT_ptrs, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=0.0)
                    V_T_block = tl.load(vT_ptrs, mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, other=0.0)
                else:
                    K_T_block = tl.load(
                        kT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )
                    V_T_block = tl.load(
                        vT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )

            # fp32 post-scale, matching the forward kernel (see the note there).
            # The recomputed logits must agree with the forward's to fp32
            # rounding, or P no longer matches the saved LSE in ``M``.
            QK_block = tl.dot(Q_block, K_T_block) * softmax_scale + pb + rm
            if not EVEN_KV:
                QK_block += tl.where((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf"))

            inv_ln2: tl.constexpr = 1.4426950408889634  # 1/ln(2)
            # M_block = +inf for fully-masked rows → P = exp2(-inf) = 0 naturally.
            P_block = tl.math.exp2((QK_block - M_block) * inv_ln2)
            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
            # Centred form, eq. 5 of arXiv:2610.31642: u = dP - dP_imax, forced
            # to exactly 0 at the argmax column by index, then dS = P (u - C).
            # Never forms dP - D, which is pure cancellation there. See
            # ``_attn_bwd_d`` for why the uncentred form cannot be rescued.
            u_block = tl.where(
                (blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] == imax[:, None], 0.0, dP_block - ref[:, None]
            )
            dS_block = P_block * (u_block - cen[:, None])
            dS_block = dS_block.to(K_T_block.dtype)

            dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

            kT_ptrs += BLOCK_SIZE_KV * stride_seq
            vT_ptrs += BLOCK_SIZE_KV * stride_seq
            pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
            res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

        dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :]
        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                tl.store(dQ_block_ptrs, dQ_block)
            else:
                tl.store(dQ_block_ptrs, dQ_block, mask=offs_dim[None, :] < DIM)
        else:
            if EVEN_DIM:
                tl.store(dQ_block_ptrs, dQ_block, mask=offs_q[:, None] < SEQ_LEN)
            else:
                tl.store(dQ_block_ptrs, dQ_block, mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM))

    @persistent_autotune(configs=_ATTN_BWD_DB_CONFIGS, key=["SEQ_LEN", "DIM"])
    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_db(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        d_pair_bias,
        M,
        D_ref,
        D_cen,
        D_imax,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        stride_d_pair_bias_batch,
        stride_d_pair_bias_head,
        stride_d_pair_bias_seq1,
        stride_d_pair_bias_seq2,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        """d_pair_bias: one program per (i_block, j_block, batch*head) tile, loops
        N_SEQ internally, accumulates fp32 locally, single non-atomic store. Bitwise
        deterministic by construction.
        """
        index_i = tl.program_id(0).to(tl.int64)
        index_j = tl.program_id(1).to(tl.int64)
        index_batch_head = tl.program_id(2).to(tl.int64)
        index_batch = index_batch_head // HEAD
        index_head = index_batch_head % HEAD

        offs_q = index_i * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = index_j * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        offs_dim = tl.arange(0, BLOCK_DIM)

        pair_bias_block_ptr = (
            pair_bias
            + (index_batch * stride_pair_bias_batch + index_head * stride_pair_bias_head)
            + offs_q[:, None] * stride_pair_bias_seq1
            + offs_kv[None, :] * stride_pair_bias_seq2
        )
        d_pair_bias_block_ptr = (
            d_pair_bias
            + (index_batch * stride_d_pair_bias_batch + index_head * stride_d_pair_bias_head)
            + offs_q[:, None] * stride_d_pair_bias_seq1
            + offs_kv[None, :] * stride_d_pair_bias_seq2
        )

        mask_q = offs_q[:, None] < SEQ_LEN
        mask_kv = offs_kv[None, :] < SEQ_LEN
        tile_mask = mask_q & mask_kv

        # pair_bias tile is shared across all N_SEQ MSA rows — load once.
        if EVEN_Q & EVEN_KV:
            pb = tl.load(pair_bias_block_ptr)
        else:
            pb = tl.load(pair_bias_block_ptr, mask=tile_mask, other=float("-inf"))

        db_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_SIZE_KV], dtype=tl.float32)

        inv_ln2: tl.constexpr = 1.4426950408889634  # 1/ln(2)

        for index_msa in range(0, N_SEQ):
            base_qkv = (index_batch * stride_batch + index_msa * stride_msa + index_head * stride_head).to(tl.int64)
            # M/D packed as [batch*N_SEQ*H + msa*H + head] * SEQ_LEN (matches dq/dk_dv).
            base_seq = ((index_batch * N_SEQ * HEAD + index_msa * HEAD + index_head) * SEQ_LEN).to(tl.int64)

            Q_ptrs = Q + base_qkv + offs_q[:, None] * stride_seq + offs_dim[None, :]
            K_ptrs = K + base_qkv + offs_kv[:, None] * stride_seq + offs_dim[None, :]
            V_ptrs = V + base_qkv + offs_kv[:, None] * stride_seq + offs_dim[None, :]
            dO_ptrs = dO + base_qkv + offs_q[:, None] * stride_seq + offs_dim[None, :]

            res_mask_ptr = (
                res_mask
                + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
                + offs_kv[None, :] * stride_mask_seq
            )

            if EVEN_Q & EVEN_KV:
                m_block = tl.load(M + base_seq + offs_q)
                ref = tl.load(D_ref + base_seq + offs_q)
                cen = tl.load(D_cen + base_seq + offs_q)
                imax = tl.load(D_imax + base_seq + offs_q)
                rm = tl.load(res_mask_ptr).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    Q_block = tl.load(Q_ptrs)
                    K_block = tl.load(K_ptrs)
                    V_block = tl.load(V_ptrs)
                    dO_block = tl.load(dO_ptrs)
                else:
                    Q_block = tl.load(Q_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
                    K_block = tl.load(K_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
                    V_block = tl.load(V_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
            else:
                # `other=0.0` for M: combined with the explicit `-inf` OOB mask
                # applied to QK below, this gives QK - m = -inf for OOB rows →
                # P = exp2(-inf) = 0 → dS = 0. (Using other=inf here would
                # propagate inf - inf = NaN.)
                m_block = tl.load(M + base_seq + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                ref = tl.load(D_ref + base_seq + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                cen = tl.load(D_cen + base_seq + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                imax = tl.load(D_imax + base_seq + offs_q, mask=offs_q < SEQ_LEN, other=-1)
                rm = tl.load(res_mask_ptr, mask=mask_kv, other=float("-inf")).broadcast_to(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
                )
                if EVEN_DIM:
                    Q_block = tl.load(Q_ptrs, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
                    K_block = tl.load(K_ptrs, mask=offs_kv[:, None] < SEQ_LEN, other=0.0)
                    V_block = tl.load(V_ptrs, mask=offs_kv[:, None] < SEQ_LEN, other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
                else:
                    Q_block = tl.load(
                        Q_ptrs,
                        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                        other=0.0,
                    )
                    K_block = tl.load(
                        K_ptrs,
                        mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                        other=0.0,
                    )
                    V_block = tl.load(
                        V_ptrs,
                        mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                        other=0.0,
                    )
                    dO_block = tl.load(
                        dO_ptrs,
                        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                        other=0.0,
                    )

            # fp32 post-scale, matching the forward kernel (see the note there).
            QK_block = tl.dot(Q_block, tl.trans(K_block)) * softmax_scale + pb + rm
            if not EVEN_KV:
                QK_block += tl.where(mask_kv, 0, float("-inf"))
            if not EVEN_Q:
                QK_block += tl.where(mask_q, 0, float("-inf"))

            # m_block = +inf for fully-masked rows → P = exp2(-inf) = 0 → dS = 0.
            P_block = tl.math.exp2((QK_block - m_block[:, None]) * inv_ln2)
            dP_block = tl.dot(dO_block, tl.trans(V_block)).to(tl.float32)
            # Centred form — see ``_attn_bwd_d``. ``offs_kv`` is already the
            # global column index in this kernel.
            u_block = tl.where(offs_kv[None, :] == imax[:, None].to(offs_kv.dtype), 0.0, dP_block - ref[:, None])
            dS_block = P_block * (u_block - cen[:, None])

            db_block += dS_block

        if EVEN_Q & EVEN_KV:
            tl.store(d_pair_bias_block_ptr, db_block)
        else:
            tl.store(d_pair_bias_block_ptr, db_block, mask=tile_mask)

    @persistent_autotune(configs=_ATTN_BWD_DK_DV_CONFIGS, key=["SEQ_LEN", "DIM"])
    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_dk_dv(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        dQ,
        dK,
        dV,
        M,
        D_ref,
        D_cen,
        D_imax,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        index_batch_msa_head = tl.program_id(1).to(tl.int64)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        offset = (index_batch * stride_batch + index_msa * stride_msa + index_head * stride_head).to(tl.int64)
        offset_seq = (index_batch_msa_head * SEQ_LEN).to(tl.int64)

        Q += offset
        K += offset
        V += offset
        dO += offset
        dQ += offset
        dK += offset
        dV += offset
        M += offset_seq
        D_ref += offset_seq
        D_cen += offset_seq
        D_imax += offset_seq

        offs_dim = tl.arange(0, BLOCK_DIM)
        index_block_kv = tl.program_id(0).to(tl.int64)
        offs_kv = index_block_kv * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        offs_q = tl.arange(0, BLOCK_SIZE_Q)

        dK_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)
        dV_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)

        res_mask_block_ptr = (
            res_mask
            + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
            + offs_kv[None, :] * stride_mask_seq
        )

        if EVEN_Q & EVEN_KV:
            res_mask_T_block = tl.trans(tl.load(res_mask_block_ptr)).broadcast_to((BLOCK_SIZE_KV, BLOCK_SIZE_Q))
            if EVEN_DIM:
                K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :])
                V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :])
            else:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :], mask=offs_dim[None, :] < DIM, other=0.0
                )
        else:
            res_mask_T_block = tl.trans(
                tl.load(res_mask_block_ptr, mask=offs_kv[None, :] < SEQ_LEN, other=float("-inf"))
            ).broadcast_to((BLOCK_SIZE_KV, BLOCK_SIZE_Q))
            if EVEN_DIM:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :], mask=offs_kv[:, None] < SEQ_LEN, other=0.0
                )
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :], mask=offs_kv[:, None] < SEQ_LEN, other=0.0
                )
            else:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )

        pair_bias_T_ptr = (
            pair_bias
            + (index_batch.to(tl.int64) * stride_pair_bias_batch + index_head.to(tl.int64) * stride_pair_bias_head)
            + offs_q[None, :] * stride_pair_bias_seq1
            + offs_kv[:, None] * stride_pair_bias_seq2
        )
        qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None]
        dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :]

        num_steps = (SEQ_LEN + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
        curr_q = 0
        for _blk_idx in range(num_steps):
            offs_q = curr_q + tl.arange(0, BLOCK_SIZE_Q)

            if EVEN_Q & EVEN_KV:
                m = tl.load(M + offs_q)
                pb_T = tl.load(pair_bias_T_ptr)
                ref = tl.load(D_ref + offs_q)
                cen = tl.load(D_cen + offs_q)
                imax = tl.load(D_imax + offs_q)
                if EVEN_DIM:
                    qT_block = tl.load(qT_ptrs)
                    dO_block = tl.load(dO_ptrs)
                else:
                    qT_block = tl.load(qT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
            else:
                m = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                pb_T = tl.load(
                    pair_bias_T_ptr,
                    mask=(offs_q[None, :] < SEQ_LEN) & (offs_kv[:, None] < SEQ_LEN),
                    other=float("-inf"),
                )
                ref = tl.load(D_ref + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                cen = tl.load(D_cen + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                imax = tl.load(D_imax + offs_q, mask=offs_q < SEQ_LEN, other=-1)
                if EVEN_DIM:
                    qT_block = tl.load(qT_ptrs, mask=offs_q[None, :] < SEQ_LEN, other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
                else:
                    qT_block = tl.load(qT_ptrs, mask=(offs_q[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM), other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM), other=0.0)

            # fp32 post-scale. This kernel used to pre-scale K in the input dtype
            # while the forward pre-scaled Q, so the recomputed logits disagreed
            # with the forward's by ~3e-2 in bf16 for high-norm rows and P no
            # longer matched the saved LSE — the dominant dK / dV error.
            QK_T_block = tl.dot(K_block, qT_block) * softmax_scale + pb_T + res_mask_T_block
            if not (EVEN_Q & EVEN_KV):
                QK_T_block += tl.where((offs_kv[:, None] < SEQ_LEN) & (offs_q[None, :] < SEQ_LEN), 0, float("-inf"))

            inv_ln2: tl.constexpr = 1.4426950408889634  # 1/ln(2)
            # m = +inf for fully-masked rows → P_T = exp2(-inf) = 0 naturally,
            # so dV/dK contributions from masked rows are 0.
            P_T_block = tl.math.exp2((QK_T_block - m[None, :]) * inv_ln2)
            dV_block += tl.dot(P_T_block.to(K_block.dtype), dO_block)

            dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
            # Centred form, transposed — see ``_attn_bwd_d``. Here the KV axis is
            # the row axis, so the argmax column of query row ``i`` is masked at
            # ``offs_kv == imax[i]``. ``offs_kv`` is the global column index.
            u_T_block = tl.where(offs_kv[:, None] == imax[None, :].to(offs_kv.dtype), 0.0, dpT_block - ref[None, :])
            dS_T_block = P_T_block * (u_T_block - cen[None, :])
            dS_T_block = dS_T_block.to(K_block.dtype)

            dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

            curr_q += BLOCK_SIZE_Q
            qT_ptrs += BLOCK_SIZE_Q * stride_seq
            dO_ptrs += BLOCK_SIZE_Q * stride_seq
            pair_bias_T_ptr += BLOCK_SIZE_Q * stride_pair_bias_seq1

        dV_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :]
        dK_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :]

        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                tl.store(dV_ptrs, dV_block)
                tl.store(dK_ptrs, dK_block)
            else:
                tl.store(dV_ptrs, dV_block, mask=offs_dim[None, :] < DIM)
                tl.store(dK_ptrs, dK_block, mask=offs_dim[None, :] < DIM)
        else:
            if EVEN_DIM:
                tl.store(dV_ptrs, dV_block, mask=offs_kv[:, None] < SEQ_LEN)
                tl.store(dK_ptrs, dK_block, mask=offs_kv[:, None] < SEQ_LEN)
            else:
                tl.store(dV_ptrs, dV_block, mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM))
                tl.store(dK_ptrs, dK_block, mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM))


class _EvoformerAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, res_mask, pair_bias, pre_transposed=False):
        """Forward pass.

        Args:
            Q, K, V: [B, N_seq, N_res, H, D] (default) or
                      [B, N_seq, H, N_res, D] (pre_transposed=True, kernel layout).
            res_mask: [B, N_seq, 1, 1, N_res]
            pair_bias: [B, 1, H, N_res, N_res]
            pre_transposed: If True, inputs are already in kernel layout and
                contiguous. Skips the internal transpose+contiguous copies,
                saving ~1.2 GB at N=1280 (3 fewer N²·C allocations).
        """
        DIM_Q, DIM_K, DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        assert DIM_Q == DIM_K and DIM_K == DIM_V

        if not pre_transposed:
            Q = Q.transpose(-2, -3).contiguous()
            K = K.transpose(-2, -3).contiguous()
            V = V.transpose(-2, -3).contiguous()

        BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = Q.shape
        softmax_scale = DIM**-0.5
        BLOCK_DIM = max(triton.next_power_of_2(DIM), 16)

        O = torch.empty_like(Q)

        extra_kern_args = {}
        if _is_hip():
            waves_per_eu = 3 if DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        def grid(args):
            return (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * N_SEQ * HEAD, 1)

        M = torch.empty((BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_msa=Q.stride(1),
            stride_Q_head=Q.stride(2),
            stride_Q_seq=Q.stride(3),
            stride_Q_dim=Q.stride(4),
            stride_K_batch=K.stride(0),
            stride_K_msa=K.stride(1),
            stride_K_head=K.stride(2),
            stride_K_seq=K.stride(3),
            stride_K_dim=K.stride(4),
            stride_V_batch=V.stride(0),
            stride_V_msa=V.stride(1),
            stride_V_head=V.stride(2),
            stride_V_seq=V.stride(3),
            stride_V_dim=V.stride(4),
            stride_O_batch=O.stride(0),
            stride_O_msa=O.stride(1),
            stride_O_head=O.stride(2),
            stride_O_seq=O.stride(3),
            stride_O_dim=O.stride(4),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(2),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(1),
            stride_mask_seq=res_mask.stride(4),
            BATCH_SIZE=BATCH_SIZE,
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            DIM=DIM,
            BLOCK_DIM=BLOCK_DIM,
            **extra_kern_args,
        )

        # ``O`` is deliberately not saved: the backward's only consumer of it was
        # the D = rowsum(dO * O) preprocess pass, which is gone (see
        # ``_attn_bwd_d``). Dropping it frees one [B, N_seq, H, N_res, D]
        # activation per attention call.
        ctx.save_for_backward(Q, K, V, res_mask, pair_bias, M)
        ctx.softmax_scale = softmax_scale
        ctx.DIM = DIM
        ctx.pre_transposed = pre_transposed

        if not pre_transposed:
            O = O.transpose(-2, -3).contiguous()
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, res_mask, pair_bias, M = ctx.saved_tensors
        if not ctx.pre_transposed:
            dO = dO.transpose(-2, -3).contiguous()
        elif not dO.is_contiguous():
            dO = dO.contiguous()

        assert Q.stride() == K.stride() == V.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = dQ.shape

        d_pair_bias = torch.zeros((BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN), device=pair_bias.device, dtype=torch.float32)

        BLOCK_DIM = max(triton.next_power_of_2(DIM), 16)
        # Per-row centring triple for dS = P * ((dP - dP_imax) - C), eq. 5 of
        # arXiv:2610.31642. Replaces the single D = rowsum(dO * O) that
        # _attn_bwd_preprocess used to produce, which collapsed dS at the softmax
        # argmax — see _attn_bwd_d's docstring for the three mechanisms.
        D_ref = torch.empty_like(M)
        D_cen = torch.empty_like(M)
        D_imax = torch.empty(M.shape, device=M.device, dtype=torch.int32)

        def bwd_d_grid(args):
            return (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * N_SEQ * HEAD, 1)

        _attn_bwd_d[bwd_d_grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            M=M,
            D_ref=D_ref,
            D_cen=D_cen,
            D_imax=D_imax,
            stride_batch=Q.stride(0),
            stride_msa=Q.stride(1),
            stride_head=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(2),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(1),
            stride_mask_seq=res_mask.stride(4),
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_DIM=BLOCK_DIM,
            DIM=ctx.DIM,
        )

        def bwd_dk_dv_grid(args):
            return (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]), BATCH_SIZE * N_SEQ * HEAD, 1)

        _attn_bwd_dk_dv[bwd_dk_dv_grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D_ref=D_ref,
            D_cen=D_cen,
            D_imax=D_imax,
            stride_batch=Q.stride(0),
            stride_msa=Q.stride(1),
            stride_head=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(2),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(1),
            stride_mask_seq=res_mask.stride(4),
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_DIM=BLOCK_DIM,
            DIM=ctx.DIM,
        )

        def bwd_dq_grid(args):
            return (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * N_SEQ * HEAD, 1)

        _attn_bwd_dq[bwd_dq_grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D_ref=D_ref,
            D_cen=D_cen,
            D_imax=D_imax,
            stride_batch=Q.stride(0),
            stride_msa=Q.stride(1),
            stride_head=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(2),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(1),
            stride_mask_seq=res_mask.stride(4),
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_DIM=BLOCK_DIM,
            DIM=ctx.DIM,
        )

        def bwd_db_grid(args):
            return (
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
                BATCH_SIZE * HEAD,
            )

        _attn_bwd_db[bwd_db_grid](
            Q=Q,
            K=K,
            V=V,
            res_mask=res_mask,
            pair_bias=pair_bias,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            d_pair_bias=d_pair_bias,
            M=M,
            D_ref=D_ref,
            D_cen=D_cen,
            D_imax=D_imax,
            stride_batch=Q.stride(0),
            stride_msa=Q.stride(1),
            stride_head=Q.stride(2),
            stride_seq=Q.stride(3),
            stride_pair_bias_batch=pair_bias.stride(0),
            stride_pair_bias_head=pair_bias.stride(2),
            stride_pair_bias_seq1=pair_bias.stride(3),
            stride_pair_bias_seq2=pair_bias.stride(4),
            stride_mask_batch=res_mask.stride(0),
            stride_mask_msa=res_mask.stride(1),
            stride_mask_seq=res_mask.stride(4),
            stride_d_pair_bias_batch=d_pair_bias.stride(0),
            stride_d_pair_bias_head=d_pair_bias.stride(2),
            stride_d_pair_bias_seq1=d_pair_bias.stride(3),
            stride_d_pair_bias_seq2=d_pair_bias.stride(4),
            HEAD=HEAD,
            N_SEQ=N_SEQ,
            SEQ_LEN=SEQ_LEN,
            BLOCK_DIM=BLOCK_DIM,
            DIM=ctx.DIM,
        )

        if not ctx.pre_transposed:
            dQ = dQ.transpose(-2, -3).contiguous()
            dK = dK.transpose(-2, -3).contiguous()
            dV = dV.transpose(-2, -3).contiguous()

        return dQ, dK, dV, None, d_pair_bias.to(dO.dtype), None


def fused_evo_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fused attention with pair bias.

    Args:
        q, k, v: [B*N_rows, N_cols, H, D] — DeepFold3 GridSelfAttention layout.
        pair_bias: [B, H, N, N] or [1, H, N, N].
        mask: [B*N_rows, N_cols] — per-row mask.

    Returns:
        [B*N_rows, N_cols, H, D] — attention output.
    """
    BN, N_cols, H, D = q.shape
    # Reshape to [B, N_rows, N_cols, H, D] — need B from pair_bias
    B = pair_bias.shape[0]
    N_rows = BN // B

    q5 = q.view(B, N_rows, N_cols, H, D)
    k5 = k.view(B, N_rows, N_cols, H, D)
    v5 = v.view(B, N_rows, N_cols, H, D)

    # res_mask: [B, N_rows, 1, 1, N_cols] from mask [B*N_rows, N_cols]
    res_mask = mask.view(B, N_rows, 1, 1, N_cols).float()
    res_mask = torch.where(res_mask > 0.5, 0.0, float("-inf"))

    # pair_bias: [B, 1, H, N, N]
    if pair_bias.dim() == 4:
        pair_bias = pair_bias.unsqueeze(1)

    o5 = _EvoformerAttention.apply(q5, k5, v5, res_mask, pair_bias.float())
    return o5.view(BN, N_cols, H, D)
