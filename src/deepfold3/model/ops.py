"""Triangle multiplication and outer-product mean ops.

Attribute names map to Haiku parameter scopes in the AF3 converter.

TriangleMultiplication Haiku → PyTorch mapping:
    left_norm_input       → norm_left    (LayerNorm)
    right_norm_input      → norm_right   (LayerNorm)   [unused in AF3 — same input]
    projection            → proj         (LinearNoBias, c_z → c_z*2)
    gate                  → gate         (LinearNoBias, c_z → c_z*2)
    center_norm           → norm_center  (LayerNorm, axis=0 in Haiku ≡ on channel axis)
    output_projection     → out_proj     (LinearNoBias, c_z → c_z)
    gating_linear         → out_gate     (LinearNoBias, c_z → c_z)

OuterProductMean Haiku → PyTorch mapping:
    layer_norm_input  → norm      (LayerNorm, c_msa)
    left_projection   → proj_a    (LinearNoBias, c_msa → c_hidden)
    right_projection  → proj_b    (LinearNoBias, c_msa → c_hidden)
    output_w          → out_w     (parameter [c_hidden, c_hidden, c_z])
    output_b          → out_b     (parameter [c_z])

All modules support both unbatched and batched (leading B dim) inputs.

Triangle multiply optimization design (cf. NVIDIA cuEquivariance):
─────────────────────────────────────────────────────────────────
cuEQ fuses the entire triangle multiply into a pipeline of 3 fused ops:

  1. layer_norm_transpose(x, layout="bijd->bijd")        # fused LN
  2. fused_sigmoid_gated_dual_gemm(x, W_gate, W_proj,    # two GEMMs + sigmoid
         mask, transpose_out=True)                         # + layout transpose
     → a, b = chunk(ab, 2, dim=0)                        # split in D-first layout
  3. einsum("dbik,dbjk->dbij", a, b)                     # contraction
  4. layer_norm_transpose(x, layout="dbij->bijd")        # fused LN + transpose
  5. fused_sigmoid_gated_dual_gemm_dual_x(x_in, x_out,   # output gating
         W_g_out, W_p_out)

Our current optimization stack:
  Level 1: FusedTriangleMulGate — LN + packed GEMM (3 projections) + sigmoid postprocess
  Level 2: FusedLayerNormLinear for norm_center + out_proj (post-einsum)
  Level 3: FusedGatedDualGEMM — two GEMMs + sigmoid gate in one kernel (cuEQ pattern)

Future: D-first layout for einsum (transpose_out=True → "dbik,dbjk->dbij"),
fused LN+transpose on output, output gating via second dual GEMM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepfold3.model import kernel_census
from deepfold3.model.config import OptimConfig
from deepfold3.model.primitives import LayerNorm, LinearNoBias, pad_row_chunk


def _triangle_matmul_outgoing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``bikc,bjkc->bijc`` via matmul on permuted views (zero-copy).

    Equivalent to ``torch.einsum('bikc,bjkc->bijc', a, b)`` but uses cuBLAS
    strided batch GEMM on non-contiguous views, avoiding the internal
    ``.contiguous()`` copies that ``torch.einsum`` creates.

    Args:
        a: Tensor ``[B, N, N, C]`` (interpreted as ``[B, i, k, C]``).
        b: Tensor ``[B, N, N, C]`` (interpreted as ``[B, j, k, C]``).

    Returns:
        Tensor ``[B, N, N, C]`` — contraction over k.
    """
    return torch.matmul(a.permute(0, 3, 1, 2), b.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)


def _triangle_matmul_incoming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``bkjc,bkic->bijc`` via matmul on permuted views (zero-copy).

    Equivalent to ``torch.einsum('bkjc,bkic->bijc', a, b)`` but uses cuBLAS
    strided batch GEMM on non-contiguous views. Critical for incoming
    triangle multiply — ``torch.einsum`` for this contraction pattern creates
    ~10 extra ``[N², C]`` contiguous copies (1.5x memory overhead), while
    ``torch.matmul`` on views creates zero.

    Args:
        a: Tensor ``[B, N, N, C]`` (interpreted as ``[B, k, j, C]``).
        b: Tensor ``[B, N, N, C]`` (interpreted as ``[B, k, i, C]``).

    Returns:
        Tensor ``[B, N, N, C]`` — contraction over k.
    """
    return torch.matmul(a.permute(0, 3, 2, 1), b.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)


_TRIANGLE_MATMUL = {
    "bikc,bjkc->bijc": _triangle_matmul_outgoing,
    "bkjc,bkic->bijc": _triangle_matmul_incoming,
}


def set_router_stats_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle MHR router read-out on every triangle-multiply in ``model``."""
    for mod in model.modules():
        if isinstance(mod, _TriangleMulBase) and mod.use_router:
            mod.router_stats_enabled = bool(enabled)


def collect_router_stats(model: nn.Module, prefix: str = "router", reset: bool = True) -> dict[str, float]:
    """Router read-out, averaged over the logging window and over depth.

    Read the two statistics together (see ``_record_router_stats``):
    ``entropy`` at 1.0 means the router never left uniform, and ``g_std`` at ~0
    means whatever it settled on is the same for every pair — a constant
    per-head rescale, which is a reparameterization of ``W_o`` and therefore
    inside AF3's function class. Only ``g_std > 0`` is evidence of routing AF3
    could not already do.

    Values are accumulated across every forward in the window (recycling means
    several per step) and reset here, matching the activation meter rather than
    reporting whichever forward happened to be last.

    Rank-local (the caller logs from rank 0); no collective, so it is safe to
    call on any subset of ranks.
    """
    rows = [
        (name, mod._flush_router_stats(reset))
        for name, mod in model.named_modules()
        if isinstance(mod, _TriangleMulBase) and mod.use_router
    ]
    rows = [(n, r) for n, r in rows if r is not None]
    if not rows:
        return {}
    ent = [r["entropy"] for _, r in rows]
    std = [r["g_std"] for _, r in rows]
    out = {
        f"{prefix}/entropy": sum(ent) / len(ent),
        f"{prefix}/entropy_min": min(ent),
        # marginal - entropy = I(pair; head) / log H. Near zero ⇒ head death.
        f"{prefix}/marginal_entropy": sum(r["marginal_entropy"] for _, r in rows) / len(rows),
        f"{prefix}/specialization_gap": sum(r["marginal_entropy"] - r["entropy"] for _, r in rows) / len(rows),
        # The one that decides whether the router is redundant with W_o.
        f"{prefix}/g_std": sum(std) / len(std),
        f"{prefix}/g_std_max": max(std),
        # 1.0 by construction for softmax; drift for sigmoid.
        f"{prefix}/g_mean": sum(r["g_mean"] for _, r in rows) / len(rows),
        f"{prefix}/occupancy_max": max(r["occupancy_max"] for _, r in rows),
        f"{prefix}/occupancy_min": min(r["occupancy_min"] for _, r in rows),
        # sqrt(mean_h g^2): what the branch magnitude actually does. The softmax
        # constraint pins mean_h g = 1, not the norm, so this rises from 1
        # toward sqrt(H) as routing sharpens and out_gate has to absorb it.
        f"{prefix}/g_rms": sum(r["g_rms"] for _, r in rows) / len(rows),
        f"{prefix}/g_rms_max": max(r["g_rms"] for _, r in rows),
    }
    for name, r in rows:
        out[f"{prefix}/entropy_by_module/{name}"] = r["entropy"]
        out[f"{prefix}/marginal_entropy_by_module/{name}"] = r["marginal_entropy"]
        out[f"{prefix}/g_rms_by_module/{name}"] = r["g_rms"]
        out[f"{prefix}/g_std_by_module/{name}"] = r["g_std"]
    return out


class _TriangleMulBase(nn.Module):
    """Shared base for outgoing/incoming triangle multiplication.

    Subclasses only differ in the einsum contraction equation.
    """

    _equation: str  # set by subclasses

    def __init__(
        self,
        c_z: int,
        optim: OptimConfig | None = None,
        num_heads: int = 0,
        head_norm: str = "channel",
        router: bool = False,
        router_kind: str = "softmax",
    ):
        """
        Args:
            c_z: Pair channel width.
            optim: Fused-kernel configuration.
            num_heads: MHR head count ``H`` (``0`` ≡ off ≡ AF3). ``d = c_z / H``.
            head_norm: Post-contraction normalization —
                ``"channel"`` = AF3's LayerNorm over all ``c_z``;
                ``"head_rms"`` = RMSNorm over ``d`` within each head;
                ``"head_ln"`` = LayerNorm over ``d`` within each head;
                ``"post_channel"`` = LayerNorm over all ``c_z``, applied
                *after* the router instead of before it. The other three
                normalize first, which leaves nothing bounding what the gate
                then multiplies in: ``sum_h g = H`` pins the mean of ``g`` but
                not its norm, so what reaches ``out_proj`` grows toward
                ``sqrt(H)`` as the routing sharpens (measured: 2.26x in r17's
                deep trunk by step 8000). Normalizing afterwards lets the gate
                set the *direction* of the ``c_z`` vector while the norm fixes
                its length. There is deliberately no per-head equivalent: a
                per-head norm applied after gating divides out ``g^(h)``
                exactly, since ``g`` is constant within a head, and the router
                becomes a no-op.
            router: Target-pair-conditioned router over heads.
            router_kind: ``"softmax"`` — ``g = H·softmax_h(R(x))``, so
                ``mean_h g = 1`` and the router can only *reallocate*; heads
                compete for a fixed budget. ``"sigmoid"`` — ``g = 2σ(R(x))``,
                per-head independent and bounded in (0, 2); heads can
                co-activate, at the cost of letting the branch magnitude drift
                (which ``out_gate`` then has to absorb). Both are exactly
                ``g ≡ 1`` at zero-init. Default ``"softmax"``.
        """
        super().__init__()
        self.norm_left = LayerNorm(c_z)
        self.proj = LinearNoBias(c_z, c_z * 2)
        # AF3: gate uses final_init (zeros)
        self.gate = LinearNoBias(c_z, c_z * 2, initializer="zeros")
        # ``head_rms`` is RMSNorm proper — gain, no shift. Dropping the shift
        # also removes a parameter that would otherwise have no consumer, which
        # DDP reports as unused; and the trunk's LN/linear biases are already
        # measured to wander with no compensating benefit.
        self.norm_center = LayerNorm(c_z, bias=head_norm != "head_rms")
        # AF3: output_projection uses final_init (zeros)
        self.out_proj = LinearNoBias(c_z, c_z, initializer="zeros")
        # AF3: gating_linear uses final_init (zeros)
        self.out_gate = LinearNoBias(c_z, c_z, initializer="zeros")

        # ── MHR-TriMul: Multi-Head **Routed** triangle multiplication ─────
        # (see export/mhr-trimul/README.md)
        #
        # "Routed", not "gated": every gate in this module — sigma(G_a),
        # sigma(G_b), sigma(G_o) — is AF3's, unchanged. What is new is the
        # router, and a softmax router is not a gate: sum_h g^(h) = H binds the
        # heads together, so it *reallocates* rather than gates. Whether the
        # module fires at all stays with out_gate; the router only chooses which
        # head explains the pair.
        #
        # The contraction is untouched: einsum("ikhd,jkhd->ijhd") is bitwise
        # equal to einsum("ikc,jkc->ijc") under c = (h, d), because only k is
        # contracted. The head axis earns its keep in the *normalization* and
        # the *router*, both of which sit between the contraction and out_proj.
        if head_norm not in ("channel", "head_rms", "head_ln", "post_channel"):
            raise ValueError(f"head_norm must be channel|head_rms|head_ln|post_channel, got {head_norm!r}")
        if head_norm == "post_channel" and not router:
            # Without a gate there is nothing to normalize after, and the mode
            # would silently equal "channel" — a second spelling for one
            # behaviour is exactly what section 12's incident was made of.
            raise ValueError("head_norm='post_channel' is meaningless without router=True")
        if router_kind not in ("softmax", "sigmoid"):
            raise ValueError(f"router_kind must be softmax|sigmoid, got {router_kind!r}")
        if head_norm != "channel" or router:
            if num_heads <= 0:
                raise ValueError("num_heads > 0 is required for per-head norm or routing")
            if c_z % num_heads:
                raise ValueError(f"c_z={c_z} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = c_z // num_heads if num_heads else 0
        self.head_norm = head_norm
        self.use_router = bool(router)
        self.router_kind = router_kind
        self.is_mhr = head_norm != "channel" or self.use_router
        # Router logits from the *target* pair x_ij. Zero-init ⇒ g ≡ 1, so the
        # module starts function-identical to AF3 TriMul, while the softmax
        # Jacobian at p = 1/H is (δ_hh' − 1/H), i.e. O(1): the router receives a
        # real gradient from step 0 rather than waking up through a sigmoid.
        self.router = LinearNoBias(c_z, num_heads, initializer="zeros") if self.use_router else None
        # Per-head affine reuses norm_center's gain viewed as [H, d], so the
        # state_dict keys line up across the ablation grid; ``head_rms`` is the
        # one cell that drops ``norm_center.bias``, so warm-starting into or out
        # of it needs ``load_strict=false``.
        self._router_stats: dict[str, float] | None = None
        self.router_stats_enabled = False

        self._fused_gate = None
        self._fused_dual_gemm = None
        self._fused_center_proj = None
        if optim is not None and optim.use_fused_triangle_mul:
            # Level 3: Fused dual GEMM for proj+gate (cuEQ pattern — two GEMMs
            # share input tile loads, sigmoid+multiply fused in epilogue).
            # Falls back to Level 1 (FusedTriangleMulGate) if unavailable.
            try:
                from deepfold3.model.triton.fused_gated_dual_gemm import FusedGatedDualGEMM

                self._fused_dual_gemm = FusedGatedDualGEMM(c_z, c_z * 2)
            except ImportError:
                pass
            if self._fused_dual_gemm is None:
                try:
                    from deepfold3.model.triton.triangle_mul_gate import FusedTriangleMulGate

                    self._fused_gate = FusedTriangleMulGate(c_z)
                except ImportError:
                    pass
            if self.use_router and self._fused_dual_gemm is None:
                # Level 1 never materializes LN(z), which the router reads.
                self._fused_gate = None
            # Level 2: Fused LN+Linear for post-einsum norm_center + out_proj.
            # Incompatible with MHR — per-head norm and routing sit between the
            # two fused halves.
            if not self.is_mhr:
                try:
                    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear

                    self._fused_center_proj = FusedLayerNormLinear(c_z, c_z, has_linear_bias=False)
                except ImportError:
                    pass

        _has_triton_fused = (
            self._fused_gate is not None or self._fused_dual_gemm is not None or self._fused_center_proj is not None
        )
        if optim is not None and optim.use_torch_compile and not _has_triton_fused:
            # Don't combine torch.compile with fused Triton kernels — the compiled
            # graph mishandles nested Triton kernel launches (SymNodeVariable error).
            self._compute = torch.compile(self._compute)

    def _compute(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._fused_dual_gemm is not None and z.is_cuda:
            # Level 3: dual GEMM (cuEQ) — sigmoid + multiply fused in GEMM epilogue.
            z0 = self.norm_left(z)
            gated_proj = self._fused_dual_gemm(
                z0, w_gate=self.gate.weight, w_proj=self.proj.weight
            )  # sigmoid(z0@Wg.T) * (z0@Wp.T)
            out_gate = torch.sigmoid(self.out_gate(z0))
            router_logits = self.router(z0) if self.use_router else None
            del z0  # free normalized input (400 MB at N=1280)
            a = gated_proj[..., 0::2]
            b = gated_proj[..., 1::2]
            del gated_proj  # free dual GEMM output (800 MB) — a, b are stride-2 views
        elif self._fused_gate is not None and z.is_cuda:
            # Level 1: LN + packed GEMM + sigmoid postprocess.
            a, b, out_gate = self._fused_gate(
                z,
                self.norm_left.weight,
                self.norm_left.bias,
                self.proj.weight,
                self.gate.weight,
                self.out_gate.weight,
            )
            router_logits = None  # ``use_router`` disables this path in __init__
        else:
            z0 = self.norm_left(z)
            proj = self.proj(z0)
            gate = torch.sigmoid(self.gate(z0))
            out_gate = torch.sigmoid(self.out_gate(z0))
            router_logits = self.router(z0) if self.use_router else None
            del z0
            proj = proj * gate
            a = proj[..., 0::2]
            b = proj[..., 1::2]
        m = mask.unsqueeze(-1)
        # Materialize masked a, b as contiguous tensors and free the gated_proj
        # backing store (800 MB at N=1280). Without this, gated_proj stays alive
        # through the a/b stride-2 views until the function returns.
        a = (a * m).contiguous()
        b = (b * m).contiguous()
        del m
        act = _TRIANGLE_MATMUL[self._equation](a, b)
        del a, b  # free masked projections before output gating
        if self.is_mhr:
            act = self._mhr_head_mix(act, router_logits)
            act = self.out_proj(act)
        elif self._fused_center_proj is not None and act.is_cuda:
            act = self._fused_center_proj(
                act,
                self.norm_center.weight,
                self.norm_center.bias,
                self.out_proj.weight.T,
            )
        else:
            act = self.norm_center(act)
            act = self.out_proj(act)
        return act * out_gate

    def _mhr_head_mix(self, act: torch.Tensor, router_logits: torch.Tensor | None) -> torch.Tensor:
        """Per-head normalization then target-pair-conditioned routing.

        ``act`` is the raw contraction ``T_ij`` of shape ``[B, N, N, c_z]``;
        the return value is ``[g^(1) T̂^(1) ‖ … ‖ g^(H) T̂^(H)]``, still
        ``[B, N, N, c_z]``, ready for ``out_proj``.

        The affine parameters are ``norm_center.weight/bias`` viewed as
        ``[H, d]`` — one gain (and shift) per head channel, same parameter
        count and same state_dict keys as the AF3 ``LayerNorm(c_z)`` it
        replaces.
        """
        h, d = self.num_heads, self.head_dim
        if self.head_norm == "post_channel":
            # Gate first, normalize after: the router chooses the direction and
            # LayerNorm(c_z) fixes the length, so sharpening cannot inflate the
            # branch. It also stops the softmax sum constraint from being
            # load-bearing, which is what makes ``router_kind="sigmoid"``
            # usable here — the anchor moves from the constraint to the norm.
            t = act.unflatten(-1, (h, d))
            if router_logits is not None:
                r = router_logits.float()
                g = float(h) * torch.softmax(r, dim=-1) if self.router_kind == "softmax" else 2.0 * torch.sigmoid(r)
                if self.router_stats_enabled:
                    self._record_router_stats(g)
                t = t * g.to(t.dtype).unsqueeze(-1)
            return self.norm_center(t.flatten(-2))
        if self.head_norm == "channel":
            # AF3's LayerNorm over the whole of c_z. This branch was missing —
            # "channel" fell through to the head_ln case, so a run configured
            # for AF3 normalization silently got per-head LayerNorm instead.
            # ``test_zero_init_router_is_exactly_af3`` did not catch it because
            # out_proj and out_gate are zero-initialized, so both sides of the
            # comparison were exactly zero. See the regression test that now
            # perturbs those weights before comparing.
            t = self.norm_center(act).unflatten(-1, (h, d))
        else:
            t = act.unflatten(-1, (h, d))
            w = self.norm_center.weight.view(h, d)
            eps = self.norm_center.eps
            if self.head_norm == "head_rms":
                # RMSNorm, not LayerNorm: at d = 32 the LN mean carries
                # σ/sqrt(32) ≈ 0.18σ of estimation noise and subtracts it from
                # the signal. Dropping the mean removes that noise source
                # outright. No shift — ``norm_center`` has no bias in this mode.
                t32 = t.float()
                t = (t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + eps)).to(act.dtype) * w
            else:  # head_ln
                t32 = t.float()
                mu = t32.mean(-1, keepdim=True)
                t32 = (t32 - mu) * torch.rsqrt(t32.var(-1, keepdim=True, unbiased=False) + eps)
                t = t32.to(act.dtype) * w + self.norm_center.bias.view(h, d)

        if router_logits is not None:
            # Both forms are exactly g ≡ 1 at zero-init; they differ in whether
            # the heads compete.
            #
            #   softmax: g = H·softmax_h(R) ⇒ mean_h g = 1. The router can only
            #     reallocate, never change the branch's global scale — that
            #     stays with ``out_gate``. But raising one head necessarily
            #     lowers the others, which is an architectural claim that heads
            #     share a fixed budget. Its fixed point under an unregularized,
            #     growing W_r is a vertex, i.e. collapse to one head.
            #
            #   sigmoid: g = 2σ(R), independent per head, bounded in (0, 2).
            #     Complementary relation subspaces can co-activate, and there is
            #     no winner-take-all attractor to balance away. The cost is that
            #     mean_h g may drift from 1, so the branch magnitude drifts and
            #     competes with ``out_gate``.
            r = router_logits.float()
            g = float(h) * torch.softmax(r, dim=-1) if self.router_kind == "softmax" else 2.0 * torch.sigmoid(r)
            if self.router_stats_enabled:
                self._record_router_stats(g)
            t = t * g.to(t.dtype).unsqueeze(-1)
        return t.flatten(-2)

    @torch.no_grad()
    def _record_router_stats(self, g: torch.Tensor) -> None:
        """Accumulate a detached read-out of whether the router is doing anything.

        Two statistics, because entropy alone cannot answer the question.

        ``entropy`` is ``H(p)/log H`` averaged over pairs. 1.0 means the router
        never left uniform — inert, functionally AF3 TriMul.

        ``g_std`` is the **across-pair** standard deviation of ``g`` itself,
        averaged over heads, and it is what separates useful routing from a
        redundant one. A router that is sharp but the *same* for every pair
        (low entropy, ``g_std ≈ 0``) merely scales each column block of
        ``W_o`` by a constant — a reparameterization of ``W_o``, inside AF3's
        function class, contributing nothing. Only the pair-conditional part,
        which ``g_std`` measures, is outside it. ``g`` sits at 1 per head when
        the router is inert, so 0.1 is a modest effect.

        ``entropy`` and ``occupancy`` are computed from the *normalized* shares
        ``p = g / Σ_h g``, which makes them identical to the softmax
        probabilities under ``router_kind="softmax"`` (where ``Σ_h g = H``) and
        keeps the two parameterizations directly comparable.

        ``g_mean`` is exactly 1.0 for the softmax router — a free sanity check —
        and free to drift for the sigmoid one, where it measures the branch
        rescaling that ``out_gate`` then has to absorb.

        ``marginal_entropy`` is the same functional with the averaging order
        swapped: the entropy of the pair-averaged shares ``H(E_ij[p])`` rather
        than the average of the per-pair entropies ``E_ij[H(p)]``. By Jensen the
        former is never smaller, and their difference is exactly the normalized
        mutual information between which pair it is and which head fires,
        ``I(pair; head) / log H``. That separates two states ``entropy`` alone
        cannot tell apart, because both drive it down:

        * **specialization** — each pair commits to one head, but *which* head
          varies across pairs. Low ``entropy``, high ``marginal_entropy``.
        * **head death** — the same few heads win for every pair and the rest
          are permanently off. Both are low.

        A gap near zero means the head choice carries no information about the
        pair: the module has simply stopped using most of its heads.

        So: ``entropy ≈ 1`` ⇒ inert. ``entropy < 1`` with ``g_std ≈ 0`` ⇒
        active but redundant. ``g_std > 0`` ⇒ genuinely conditional routing.

        One forward is already a precise estimate — there are ``N²`` pairs in
        it — but it measures only *within-example* variation. A router that
        varied across examples while staying constant within one would also be
        outside the function class and would read here as 0, so this is a
        conservative lower bound on how much conditioning the router carries.
        """
        g = g.reshape(-1, self.num_heads)
        p = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
        ent = -(p * (p + 1e-9).log()).sum(-1).mean() / math.log(self.num_heads)
        share = p.mean(0)
        marg = -(share * (share + 1e-9).log()).sum() / math.log(self.num_heads)
        acc = self._router_stats
        if acc is None:
            acc = self._router_stats = dict(
                entropy=0.0,
                marginal_entropy=0.0,
                g_rms=0.0,
                g_std=0.0,
                g_mean=0.0,
                occupancy_max=0.0,
                occupancy_min=0.0,
                n=0.0,
            )
        acc["entropy"] += float(ent)
        acc["marginal_entropy"] += float(marg)
        # ``mean_h g = 1`` under softmax pins the arithmetic mean, not the norm.
        # What out_proj actually receives scales with sqrt(mean_h g^2), which by
        # Jensen is >= 1 and grows to sqrt(H) as the routing sharpens. So a
        # router that is doing its job inflates the branch, and out_gate has to
        # absorb it. This records the inflation directly.
        acc["g_rms"] += float(g.pow(2).mean(-1).sqrt().mean())
        acc["g_std"] += float(g.std(0).mean())
        acc["g_mean"] += float(g.mean())
        acc["occupancy_max"] += float(share.max())
        acc["occupancy_min"] += float(share.min())
        acc["n"] += 1.0

    def _flush_router_stats(self, reset: bool = True) -> dict[str, float] | None:
        """Window means of the accumulated read-out, or ``None`` if nothing ran."""
        acc = self._router_stats
        if acc is None or acc["n"] == 0:
            return None
        n = acc["n"]
        out = {k: v / n for k, v in acc.items() if k != "n"}
        if reset:
            self._router_stats = None
        return out

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B?, N, N, c_z]
            mask: [B?, N, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        result = self._compute(z, mask)

        if not has_batch:
            result = result.squeeze(0)
        return result


class TriangleMulOutgoing(_TriangleMulBase):
    """Triangle Multiplication — outgoing (equation: ikc,jkc->ijc).

    Haiku module name: triangle_multiplication_outgoing
    """

    _equation = "bikc,bjkc->bijc"


class TriangleMulIncoming(_TriangleMulBase):
    """Triangle Multiplication — incoming (equation: kjc,kic->ijc).

    Haiku module name: triangle_multiplication_incoming
    """

    _equation = "bkjc,bkic->bijc"


class OuterProductMean(nn.Module):
    """Outer product mean for MSA → pair update.

    Haiku → PyTorch:
        layer_norm_input → norm
        left_projection  → proj_a
        right_projection → proj_b
        output_w         → out_w  (register_parameter)
        output_b         → out_b  (register_parameter)
    """

    def __init__(self, c_msa: int, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.norm = LayerNorm(c_msa)
        self.proj_a = LinearNoBias(c_msa, c_hidden)
        self.proj_b = LinearNoBias(c_msa, c_hidden)
        # AF3: output_w uses Constant(0.0) when final_init='zeros'
        self.out_w = nn.Parameter(torch.zeros(c_hidden, c_hidden, c_z))
        self.out_b = nn.Parameter(torch.zeros(c_z))
        self._chunk_size: int = 0  # 0 = disabled, >0 = tiled OPM
        self._row_chunk: int = 0  # 0 = disabled, >0 = OptimConfig.msa_row_chunk

    def _outer_product_contract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute outer product mean contraction: (a^T b) @ w + bias.

        Args:
            a: [B, S, N, c_hidden] left projection.
            b: [B, S, N, c_hidden] right projection.
        Returns:
            [B, N, N, c_z] contracted result.
        """
        ab = torch.einsum("bsie,bsjf->bijef", a, b)
        return torch.einsum("bijef,efc->bijc", ab, self.out_w) + self.out_b

    def _project_rows(self, msa: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The left and right projections, with the LayerNorm output alive one chunk at a time."""
        a = b = None
        for s0 in range(0, msa.shape[1], self._row_chunk):
            s1 = s0 + self._row_chunk
            rows = self.norm(pad_row_chunk(msa[:, s0:s1], self._row_chunk))
            n = min(self._row_chunk, msa.shape[1] - s0)
            a_r, b_r = self.proj_a(rows)[:, :n] * m[:, s0:s1], self.proj_b(rows)[:, :n] * m[:, s0:s1]
            if a is None:
                shape = (*msa.shape[:3], a_r.shape[-1])
                a, b = a_r.new_empty(shape), b_r.new_empty(shape)
            a[:, s0:s1], b[:, s0:s1] = a_r, b_r
        return a, b

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa:  [B?, num_seq, N, c_msa]
            mask: [B?, num_seq, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = msa.dim() == 4
        if not has_batch:
            msa = msa.unsqueeze(0)
            mask = mask.unsqueeze(0)

        m = mask.unsqueeze(-1)  # [B, S, N, 1]
        if 0 < self._row_chunk < msa.shape[1] and not torch.is_grad_enabled():
            kernel_census.record("OuterProductMean.project", "row_chunked")
            a, b = self._project_rows(msa, m)
        else:
            kernel_census.record("OuterProductMean.project", "whole")
            msa = self.norm(msa)
            a = self.proj_a(msa) * m  # [B, S, N, c_hidden]
            b = self.proj_b(msa) * m  # [B, S, N, c_hidden]

        chunk = self._chunk_size
        N = a.shape[2]
        kernel_census.record("OuterProductMean.contract", "tiled" if 0 < chunk < N else "whole")
        if chunk > 0 and chunk < N:
            from deepfold3.model.tiled_linear import tiled_opm_contract

            num_shards = max(1, math.ceil(N / chunk))
            if self.training and a.requires_grad:
                act = tiled_opm_contract(a, b, self._outer_product_contract, num_shards)
            else:
                # Inference: 2D chunked forward — chunk both i and j to keep
                # intermediate at [B, chunk, chunk, c_h, c_h] (constant in N).
                c_z = self.out_w.shape[2]
                act = a.new_empty(a.shape[0], N, N, c_z)
                for i0 in range(0, N, chunk):
                    i1 = min(i0 + chunk, N)
                    a_i = a[:, :, i0:i1, :]
                    for j0 in range(0, N, chunk):
                        j1 = min(j0 + chunk, N)
                        act[:, i0:i1, j0:j1] = self._outer_product_contract(a_i, b[:, :, j0:j1, :])
        else:
            act = self._outer_product_contract(a, b)

        # normalize by number of valid sequence positions
        eps = 1e-3
        mask_f = mask.float()
        norm = torch.einsum("bsi,bsj->bij", mask_f, mask_f)
        result = act / (eps + norm.unsqueeze(-1))

        if not has_batch:
            result = result.squeeze(0)
        return result
