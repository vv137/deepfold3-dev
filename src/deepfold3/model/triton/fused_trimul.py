"""Forward-only triangle multiply in three launches, on channel-major planes.

    x = LN_in(z);  a, b = sigmoid(x Wg^T) * (x Wp^T) * mask   [K1, writes [2C, B, Np, Np] bf16]
    X = a @ b^T (outgoing) | a^T @ b (incoming)              [one strided-batched cuBLAS GEMM]
    out = sigmoid(x Wog^T) * (LN_out(X) Wo^T)                [K3, recomputes LN_in on z tiles]

The GEMM operands are contiguous and their extents are multiples of 16, which is
what makes the contraction fast: on [1, 640, 640, 256] bf16 pairs it takes 1.7 ms
here and 5.4 ms on the permuted views of ``ops._triangle_matmul_*`` (RTX 6000 Ada).

Rounding points: fp32 LayerNorm statistics, bf16 LN output, gating on the fp32
accumulators, bf16 planes, bf16 LN_out output, fp32 output gating.

``fused_trimul`` is forward-only and refuses grad. ``FusedTriMulFunction`` is the
training entry: the input LayerNorm stays in autograd, the forward keeps the planes
and the contraction, and the backward walks the same stages with two tile kernels
and cuBLAS GEMMs. Elementwise work is what costs here, not the GEMMs: on the same
pairs a GEMM takes 0.8 ms, the output LayerNorm and its autograd backward 36 ms.
Saving activations instead of recomputing them follows the cuEquivariance training
design, so activation checkpointing drops and rebuilds them with ``save_for_backward``.

Ported from ``fpf_trimul_v4`` in anthropics/uplifting-biomolecular-modeling; see
``fused_trimul.NOTICE``. Modified: pointer kernels only, without the batch-limit,
bias, residual and descriptor paths, and with our autotune in place of its cell
table. The two VJP kernels are ours.
"""

from __future__ import annotations

import torch

from deepfold3.model.inference_guard import inference_only

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

SUPPORTED_C = (128, 256)
MIN_TOKENS = 101  # Below this the launches cost more than the torch formula.
_PAD = 16

if HAS_TRITON:
    from deepfold3.model.triton.cache import persistent_autotune

    def _configs(bm, bn, warps, stages):
        return [
            triton.Config({"BM": m, "BN": n}, num_warps=w, num_stages=s)
            for m in bm
            for n in bn
            for w in warps
            for s in stages
        ]

    # The first config of each list is what ``--det`` serves, so it must build on
    # every architecture: SAFE cells upstream, or the smallest tiles for the VJPs.
    # Tiles change only the order of floating-point sums, and BN must divide C.
    _PROJECT_CONFIGS = _configs((64, 128), (32, 64, 128), (4, 8), (1, 2))
    _OUTPUT_CONFIGS = _configs((64, 32), (64, 32), (4, 8), (1,))
    _PROJECT_VJP_CONFIGS = _configs((32, 16), (32, 64), (4, 8), (1, 2))
    _OUTPUT_VJP_CONFIGS = _configs((32, 16), (32, 16), (4, 8), (1,))

    @persistent_autotune(configs=_PROJECT_CONFIGS, key=["N", "C"], kernel_name="fused_trimul_project")
    @triton.jit
    def _project_kernel(
        z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, mask_ptr, ab_ptr,
        N, Np, plane_stride, eps, NN, B,
        C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NORM: tl.constexpr,
    ):  # fmt: skip
        """LN_in and both gated projections for one (token tile, row, batch element)."""
        pid_j = tl.program_id(0)
        i = tl.program_id(1).to(tl.int64)
        bidx = tl.program_id(2).to(tl.int64)
        z_ptr += bidx * NN * C
        mask_ptr += bidx * NN
        rows = pid_j * BM + tl.arange(0, BM)
        rows64 = rows.to(tl.int64)
        lmask = (rows < N) & (i < N)
        cols = tl.arange(0, C)
        cols64 = cols.to(tl.int64)
        zf = tl.load(z_ptr + ((i * N + rows64) * C)[:, None] + cols64[None, :], mask=lmask[:, None], other=0.0)
        if NORM:
            zf = zf.to(tl.float32)
            mean = tl.sum(zf, axis=1) / C
            zc = zf - mean[:, None]
            rstd = tl.rsqrt(tl.sum(zc * zc, axis=1) / C + eps)
            x = (zc * rstd[:, None] * tl.load(lnw_ptr + cols)[None, :] + tl.load(lnb_ptr + cols)[None, :]).to(tl.bfloat16)
        else:
            x = zf.to(tl.bfloat16)
        mv = tl.load(mask_ptr + i * N + rows64, mask=lmask, other=0.0).to(tl.float32)
        obase = i * Np + rows64
        smask = rows < Np
        ncols = tl.arange(0, BN)
        for nb in range(0, 2 * C // BN):
            n0 = nb * BN
            woff = cols64[:, None] * (2 * C) + (n0 + ncols)[None, :]
            acc_g = tl.dot(x, tl.load(wgT_ptr + woff))
            acc_p = tl.dot(x, tl.load(wpT_ptr + woff))
            val = tl.where(lmask[:, None], tl.sigmoid(acc_g) * acc_p * mv[:, None], 0.0)
            nch = n0 + ncols
            # BN divides C, so a block never straddles the a|b boundary. Planes: [a, b] of [C, B].
            plane = nch.to(tl.int64) + bidx * C + (nch >= C).to(tl.int64) * (B - 1) * C
            tl.store(ab_ptr + obase[:, None] + (plane * plane_stride)[None, :], val.to(tl.bfloat16), mask=smask[:, None])

    @persistent_autotune(configs=_OUTPUT_CONFIGS, key=["N", "C"], kernel_name="fused_trimul_output")
    @triton.jit
    def _output_kernel(
        x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wz_ptr, wg_ptr, out_ptr,
        N, Np, plane_stride, eps, NN,
        C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, IN_F32: tl.constexpr, NORM: tl.constexpr,
    ):  # fmt: skip
        """LN_out, output projection and output gate for one (token tile, row, batch element)."""
        pid_j = tl.program_id(0)
        i = tl.program_id(1).to(tl.int64)
        bidx = tl.program_id(2).to(tl.int64)
        x_ptr += bidx * C * plane_stride
        z_ptr += bidx * NN * C
        out_ptr += bidx * NN * C
        rows = pid_j * BM + tl.arange(0, BM)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        ch = tl.arange(0, C)
        ch64 = ch.to(tl.int64)
        xf = tl.load(x_ptr + (ch64 * plane_stride)[:, None] + (i * Np + rows64)[None, :], mask=rmask[None, :], other=0.0)
        xf = xf.to(tl.float32)
        xc = xf - (tl.sum(xf, axis=0) / C)[None, :]
        rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / C + eps)
        xln = (xc * rstd[None, :] * tl.load(lnow_ptr + ch)[:, None] + tl.load(lnob_ptr + ch)[:, None]).to(tl.bfloat16)
        ztok = (i * N + rows64) * C
        zf = tl.load(z_ptr + ch64[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0)
        if NORM:
            zf = zf.to(tl.float32)
            zc = zf - (tl.sum(zf, axis=0) / C)[None, :]
            rstd2 = tl.rsqrt(tl.sum(zc * zc, axis=0) / C + eps)
            zln = (zc * rstd2[None, :] * tl.load(lniw_ptr + ch)[:, None] + tl.load(lnib_ptr + ch)[:, None]).to(tl.bfloat16)
        else:
            zln = zf.to(tl.bfloat16)
        nr = tl.arange(0, BN)
        for nb in range(0, C // BN):
            n64 = (nb * BN + nr).to(tl.int64)
            acc_p = tl.dot(tl.load(wz_ptr + n64[:, None] * C + ch[None, :]), xln)
            acc_g = tl.dot(tl.load(wg_ptr + n64[:, None] * C + ch[None, :]), zln)
            o = tl.sigmoid(acc_g) * acc_p
            optrs = out_ptr + n64[:, None] + ztok[None, :]
            if IN_F32:
                tl.store(optrs, o.to(tl.bfloat16).to(tl.float32), mask=rmask[None, :])
            else:
                tl.store(optrs, o.to(tl.bfloat16), mask=rmask[None, :])

    @persistent_autotune(
        configs=_OUTPUT_VJP_CONFIGS,
        key=["N", "C"],
        kernel_name="fused_trimul_output_vjp",
        reset_to_zero=["glnw_ptr", "glnb_ptr"],
    )
    @triton.jit
    def _output_vjp_kernel(
        g_ptr, plane_ptr, x_ptr, lnow_ptr, lnob_ptr, wz_ptr, wg_ptr,
        gp_ptr, gg_ptr, xln_ptr, gplane_ptr, gx_ptr, glnw_ptr, glnb_ptr,
        N, Np, plane_stride, eps, NN, M,
        C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):  # fmt: skip
        """Cotangent of the output stage for one (token tile, row, batch element).

        With g the output cotangent, s = sigmoid(Wog x) and p = Wo LN_out(X):
        gp = g s and gg = g p s (1 - s) feed the weight GEMMs, ``Wog^T gg`` goes to x,
        and ``Wo^T gp`` is pulled through LN_out to the contraction. With xn the
        standardized X and u the cotangent of xn, the pullback is
        rstd (u - mean(u) - xn mean(u xn)), means over channels.
        """
        pid_j = tl.program_id(0)
        i = tl.program_id(1).to(tl.int64)
        bidx = tl.program_id(2).to(tl.int64)
        plane_ptr += bidx * C * plane_stride
        gplane_ptr += bidx * C * plane_stride
        rows = pid_j * BM + tl.arange(0, BM)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        tok = bidx * NN + i * N + rows64
        ch = tl.arange(0, C)
        ch64 = ch.to(tl.int64)
        pos = (i * Np + rows64)[None, :]
        xf = tl.load(plane_ptr + (ch64 * plane_stride)[:, None] + pos, mask=rmask[None, :], other=0.0).to(tl.float32)
        xc = xf - (tl.sum(xf, axis=0) / C)[None, :]
        rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / C + eps)
        xn = xc * rstd[None, :]
        low = tl.load(lnow_ptr + ch)[:, None]
        xln = (xn * low + tl.load(lnob_ptr + ch)[:, None]).to(tl.bfloat16)
        zln = tl.load(x_ptr + ch64[:, None] + (tok * C)[None, :], mask=rmask[None, :], other=0.0).to(tl.bfloat16)
        acc_o = tl.zeros((C, BM), dtype=tl.float32)
        acc_x = tl.zeros((C, BM), dtype=tl.float32)
        nr = tl.arange(0, BN)
        for nb in range(0, C // BN):
            n64 = (nb * BN + nr).to(tl.int64)
            wo = tl.load(wz_ptr + n64[:, None] * C + ch[None, :])
            wg = tl.load(wg_ptr + n64[:, None] * C + ch[None, :])
            acc_p = tl.dot(wo, xln)
            s = tl.sigmoid(tl.dot(wg, zln))
            g = tl.load(g_ptr + n64[:, None] + (tok * C)[None, :], mask=rmask[None, :], other=0.0).to(tl.float32)
            gp = (g * s).to(tl.bfloat16)
            gg = (g * acc_p * s * (1.0 - s)).to(tl.bfloat16)
            acc_o = acc_o + tl.dot(tl.trans(wo), gp)
            acc_x = acc_x + tl.dot(tl.trans(wg), gg)
            tl.store(gp_ptr + (n64 * M)[:, None] + tok[None, :], gp, mask=rmask[None, :])
            tl.store(gg_ptr + (n64 * M)[:, None] + tok[None, :], gg, mask=rmask[None, :])
        u = acc_o * low
        gx = rstd[None, :] * (u - (tl.sum(u, axis=0) / C)[None, :] - xn * (tl.sum(u * xn, axis=0) / C)[None, :])
        tl.store(gplane_ptr + (ch64 * plane_stride)[:, None] + pos, gx.to(tl.bfloat16), mask=rmask[None, :])
        tl.store(xln_ptr + (ch64 * M)[:, None] + tok[None, :], xln, mask=rmask[None, :])
        tl.store(gx_ptr + ch64[:, None] + (tok * C)[None, :], acc_x, mask=rmask[None, :])
        tl.atomic_add(glnw_ptr + ch, tl.sum(acc_o * xn, axis=1))
        tl.atomic_add(glnb_ptr + ch, tl.sum(acc_o, axis=1))

    @persistent_autotune(configs=_PROJECT_VJP_CONFIGS, key=["N", "C"], kernel_name="fused_trimul_project_vjp")
    @triton.jit
    def _project_vjp_kernel(
        gab_ptr, x_ptr, wgT_ptr, wpT_ptr, mask_ptr, gp_ptr, gg_ptr, gx_ptr, gx_out_ptr,
        N, Np, plane_stride, NN, M, B,
        C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):  # fmt: skip
        """Cotangent of the gated projections for one (token tile, row, batch element).

        Adds ``Wp^T gp + Wg^T gg`` to the output stage's x cotangent. The sum goes to a
        separate buffer: autotuning reruns the kernel on the same arguments.
        """
        pid_j = tl.program_id(0)
        i = tl.program_id(1).to(tl.int64)
        bidx = tl.program_id(2).to(tl.int64)
        rows = pid_j * BM + tl.arange(0, BM)
        lmask = rows < N
        rows64 = rows.to(tl.int64)
        tok = bidx * NN + i * N + rows64
        cols = tl.arange(0, C)
        cols64 = cols.to(tl.int64)
        toff = (tok * C)[:, None] + cols64[None, :]
        x = tl.load(x_ptr + toff, mask=lmask[:, None], other=0.0).to(tl.bfloat16)
        acc = tl.load(gx_ptr + toff, mask=lmask[:, None], other=0.0)
        mv = tl.load(mask_ptr + tok, mask=lmask, other=0.0).to(tl.float32)
        obase = i * Np + rows64
        ncols = tl.arange(0, BN)
        for nb in range(0, 2 * C // BN):
            n0 = nb * BN
            nch = n0 + ncols
            nch64 = nch.to(tl.int64)
            woff = cols64[:, None] * (2 * C) + nch[None, :]
            wg = tl.load(wgT_ptr + woff)
            wp = tl.load(wpT_ptr + woff)
            acc_p = tl.dot(x, wp)
            s = tl.sigmoid(tl.dot(x, wg))
            plane = nch64 + bidx * C + (nch >= C).to(tl.int64) * (B - 1) * C
            g = tl.load(gab_ptr + obase[:, None] + (plane * plane_stride)[None, :], mask=lmask[:, None], other=0.0)
            g = g.to(tl.float32) * mv[:, None]
            gp = (g * s).to(tl.bfloat16)
            gg = (g * acc_p * s * (1.0 - s)).to(tl.bfloat16)
            acc = acc + tl.dot(gp, tl.trans(wp)) + tl.dot(gg, tl.trans(wg))
            tl.store(gp_ptr + tok[:, None] + (nch64 * M)[None, :], gp, mask=lmask[:, None])
            tl.store(gg_ptr + tok[:, None] + (nch64 * M)[None, :], gg, mask=lmask[:, None])
        tl.store(gx_out_ptr + toff, acc, mask=lmask[:, None])


def _forward(z, m, outgoing, lniw, lnib, wgT, wpT, lnow, lnob, wz, wg, eps_in, eps_out, norm):
    """Launch K1, the contraction and K3. Returns the planes, the contraction and the update."""
    B, N, _, C = z.shape
    Np = triton.cdiv(N, _PAD) * _PAD
    ab = torch.empty((2, B, C, Np, Np), dtype=torch.bfloat16, device=z.device)
    _project_kernel[lambda meta: (triton.cdiv(Np, meta["BM"]), Np, B)](
        z, lniw, lnib, wgT, wpT, m, ab, N, Np, Np * Np, eps_in, N * N, B, C=C, NORM=norm,
    )  # fmt: skip
    a, b = ab[0].view(B * C, Np, Np), ab[1].view(B * C, Np, Np)
    act = torch.bmm(a, b.transpose(1, 2)) if outgoing else torch.bmm(a.transpose(1, 2), b)
    out = torch.empty_like(z)
    _output_kernel[lambda meta: (triton.cdiv(N, meta["BM"]), N, B)](
        act, z, lnow, lnob, lniw, lnib, wz, wg, out, N, Np, Np * Np, eps_out, N * N,
        C=C, IN_F32=z.dtype == torch.float32, NORM=norm,
    )  # fmt: skip
    return ab, act, out


class FusedTriMulFunction(torch.autograd.Function):
    """Trainable triangle multiply after the input LayerNorm, with a fused backward."""

    @staticmethod
    def forward(ctx, x, mask, w_gate, w_proj, lnow, lnob, w_out, w_out_gate, eps, outgoing):
        """Args as in :func:`fused_trimul`, with ``x`` already normalized, [B, N, N, C] contiguous."""
        wgT, wpT = w_gate.to(torch.bfloat16).T.contiguous(), w_proj.to(torch.bfloat16).T.contiguous()
        wz, wg = w_out.to(torch.bfloat16).contiguous(), w_out_gate.to(torch.bfloat16).contiguous()
        lnow32, lnob32 = lnow.float().contiguous(), lnob.float().contiguous()
        m = mask.to(torch.float32).contiguous()
        ab, act, out = _forward(x, m, outgoing, lnow32, lnob32, wgT, wpT, lnow32, lnob32, wz, wg, eps, eps, False)
        ctx.save_for_backward(x, m, ab, act, wgT, wpT, wz, wg, lnow32, lnob32)
        ctx.eps, ctx.outgoing = eps, outgoing
        return out

    @staticmethod
    def backward(ctx, g):
        x, m, ab, act, wgT, wpT, wz, wg, lnow, lnob = ctx.saved_tensors
        B, N, _, C = x.shape
        Np, M = act.shape[-1], B * N * N
        g = g.float().contiguous()
        bf = dict(dtype=torch.bfloat16, device=x.device)
        gp_o, gg_o, xln = torch.empty((C, M), **bf), torch.empty((C, M), **bf), torch.empty((C, M), **bf)
        gact = torch.zeros((B, C, Np, Np), **bf)
        gx = torch.empty_like(g)
        glnw, glnb = torch.zeros_like(lnow), torch.zeros_like(lnob)
        _output_vjp_kernel[lambda meta: (triton.cdiv(N, meta["BM"]), N, B)](
            g, act, x, lnow, lnob, wz, wg, gp_o, gg_o, xln, gact, gx, glnw, glnb,
            N, Np, Np * Np, ctx.eps, N * N, M, C=C,
        )  # fmt: skip
        xb = x.reshape(M, C).to(torch.bfloat16)
        g_out, g_out_gate = (gp_o @ xln.T).float(), (gg_o @ xb).float()
        del g, gp_o, gg_o, xln
        ga = gact.view(B * C, Np, Np)
        a, b = ab[0].view(B * C, Np, Np), ab[1].view(B * C, Np, Np)
        if ctx.outgoing:
            gab = torch.stack((torch.bmm(ga, b), torch.bmm(ga.transpose(1, 2), a)))
        else:
            gab = torch.stack((torch.bmm(b, ga.transpose(1, 2)), torch.bmm(a, ga)))
        del gact, ga, a, b
        gp_i, gg_i = torch.empty((2 * C, M), **bf), torch.empty((2 * C, M), **bf)
        gx_out = torch.empty_like(gx)
        _project_vjp_kernel[lambda meta: (triton.cdiv(N, meta["BM"]), N, B)](
            gab, x, wgT, wpT, m, gp_i, gg_i, gx, gx_out, N, Np, Np * Np, N * N, M, B, C=C,
        )  # fmt: skip
        del gab, gx
        return gx_out.to(x.dtype), None, (gg_i @ xb).float(), (gp_i @ xb).float(), glnw, glnb, g_out, g_out_gate, None, None


@inference_only("fused_trimul")
def fused_trimul(
    z: torch.Tensor,
    mask: torch.Tensor,
    outgoing: bool,
    norm_in: torch.nn.Module,
    w_gate: torch.Tensor,
    w_proj: torch.Tensor,
    norm_out: torch.nn.Module,
    w_out: torch.Tensor,
    w_out_gate: torch.Tensor,
) -> torch.Tensor:
    """Run the triangle multiply in three launches.

    Args:
        z: [B, N, N, C] pair activations, bf16 or fp32.
        mask: [B, N, N] pair mask.
        outgoing: Contract ``a @ b^T``; otherwise ``a^T @ b``.
        norm_in: Input LayerNorm, with fp32 affine parameters.
        w_gate: [2C, C] gate weights, rows ``[a; b]``.
        w_proj: [2C, C] projection weights, rows ``[a; b]``.
        norm_out: Output LayerNorm, with fp32 affine parameters.
        w_out: [C, C] output projection.
        w_out_gate: [C, C] output gate.

    Returns:
        [B, N, N, C] update in the dtype of ``z``.
    """
    lniw, lnib = norm_in.weight.float().contiguous(), norm_in.bias.float().contiguous()
    lnow, lnob = norm_out.weight.float().contiguous(), norm_out.bias.float().contiguous()
    return _forward(
        z.contiguous(), mask.to(torch.float32).contiguous(), outgoing, lniw, lnib,
        w_gate.to(torch.bfloat16).T.contiguous(), w_proj.to(torch.bfloat16).T.contiguous(), lnow, lnob,
        w_out.to(torch.bfloat16).contiguous(), w_out_gate.to(torch.bfloat16).contiguous(),
        norm_in.eps, norm_out.eps, True,
    )[2]  # fmt: skip
