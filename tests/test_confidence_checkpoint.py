"""Per-layer checkpointing in the confidence head changes memory, not numbers."""

from __future__ import annotations

import copy

import torch

from deepfold3.model.confidence import ConfidenceHead
from deepfold3.model.config import ConfidenceConfig, OptimConfig

N, S, F = 6, 2, 11


def _head(ckpt: bool) -> ConfidenceHead:
    cfg = ConfidenceConfig(
        num_layer=2, pair_channel=16, seq_channel=32, pair_attention_num_head=2, single_attention_num_head=2
    )
    torch.manual_seed(0)
    head = ConfidenceHead(cfg, target_feat_dim=F, optim=OptimConfig(use_gradient_checkpointing=ckpt))
    for p in head.parameters():  # the logit projections are zero-initialised
        torch.nn.init.normal_(p, std=0.1)
    return head


def _inputs():
    g = torch.Generator().manual_seed(1)
    pair = torch.randn(N, N, 16, generator=g, requires_grad=True)
    single = torch.randn(N, 32, generator=g, requires_grad=True)
    return dict(
        pair=pair,
        single=single,
        pair_mask=torch.ones(N, N),
        seq_mask=torch.ones(N),
        target_feat=torch.randn(N, F, generator=g),
        atom_positions=torch.randn(S, N, 24, 3, generator=g) * 5,
    )


def _run(head):
    x = _inputs()
    torch.manual_seed(2)  # same dropout masks in both runs; checkpoint replays them on recompute
    # Count layer forward calls by wrapping forward: module hooks do not fire
    # on a checkpoint's recompute of a nested checkpoint region.
    calls = []
    for layer in head.layers:
        fwd = layer.forward

        def counted(*a, _fwd=fwd, **k):
            calls.append(1)
            return _fwd(*a, **k)

        layer.forward = counted
    out = head(**x)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()
    grads = [p.grad.clone() for p in head.parameters()] + [x["pair"].grad, x["single"].grad]
    return out, grads, len(calls)


def test_layer_checkpoint_matches_and_recomputes_per_layer():
    plain = _head(ckpt=False)
    ckpt = copy.deepcopy(plain)
    ckpt._grad_checkpoint = True
    out_a, grads_a, calls_a = _run(plain)
    out_b, grads_b, calls_b = _run(ckpt)
    for k in out_a:
        torch.testing.assert_close(out_b[k], out_a[k])
    for a, b in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(b, a)
    layers = len(plain.layers) * S
    # Sample checkpoint alone: forward + one recompute. Per layer: one more.
    assert (calls_a, calls_b) == (2 * layers, 3 * layers)


def test_flag_follows_optim_config():
    assert _head(ckpt=True)._grad_checkpoint
    assert not _head(ckpt=False)._grad_checkpoint


def test_inference_preallocated_outputs_match_the_stacked_ones():
    """No grad: outputs are written into one preallocated tensor, same values and shapes."""
    head = _head(ckpt=False).eval()
    x = _inputs()
    with torch.no_grad():
        pre = head(**x)
    with torch.enable_grad():
        stacked = head(**x)
    assert set(pre) == set(stacked)
    for k in pre:
        assert pre[k].shape == stacked[k].shape and pre[k].shape[0] == S
        torch.testing.assert_close(pre[k], stacked[k].detach())
