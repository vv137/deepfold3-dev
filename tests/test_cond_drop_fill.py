"""``cond_drop_fill: learned``: a trainable vector stands in for dropped trunk-pair entries."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import DiffusionConfig
from deepfold3.model.diffusion import interface_pair_keep, mask_trunk_pair
from tests.test_diffusion_guidance import _asym, _head, _inputs, _static, _tiny_batch, _tiny_model


def test_default_fill_is_zero_and_has_no_parameter():
    assert DiffusionConfig().cond_drop_fill == "zero"
    assert _head().null_pair is None
    assert "null_pair" not in dict(_head().named_parameters())


def test_learned_fill_at_init_equals_zero_fill():
    zero, learned = _head(), _head(cond_drop_fill="learned")
    learned.load_state_dict({**zero.state_dict(), "null_pair": torch.zeros(128)})
    x = _inputs()
    with torch.no_grad():
        a, b = _static(zero, x, False), _static(learned, x, False)
    assert torch.allclose(a.pair_cond, b.pair_cond)


def test_full_drop_uses_the_vector_everywhere():
    head, x = _head(cond_drop_fill="learned"), _inputs()
    v = head.null_pair.detach()
    const = {**x["embeddings"], "pair": v.expand_as(x["embeddings"]["pair"]).clone()}
    with torch.no_grad():
        dropped = _static(head, x, False)
        explicit = head.compute_static(**{**x, "embeddings": const})
    assert torch.allclose(dropped.pair_cond, explicit.pair_cond, atol=1e-6)


def test_masked_pair_takes_the_vector_where_dropped():
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    pair = torch.randn(1, 3, 3, 128)
    fill = torch.randn(128)
    out = mask_trunk_pair({"pair": pair}, keep, fill)["pair"]
    assert torch.equal(out[keep], pair[keep])
    assert torch.equal(out[~keep], fill.expand(int((~keep).sum()), 128))
    assert torch.equal(mask_trunk_pair({"pair": pair}, keep)["pair"][~keep], torch.zeros_like(pair[~keep]))


def test_learned_masked_pair_matches_the_full_drop_where_dropped():
    head, x = _head(cond_drop_fill="learned"), _inputs()
    keep = interface_pair_keep(_asym(2, 1)[0])
    with torch.no_grad():
        c, u = _static(head, x, True), _static(head, x, False)
        part = head.compute_static(**{**x, "embeddings": mask_trunk_pair(x["embeddings"], keep, head.null_pair)})
    k = keep[0]
    assert torch.allclose(part.pair_cond[0][k], c.pair_cond[0][k], atol=1e-6)
    assert torch.allclose(part.pair_cond[0][~k], u.pair_cond[0][~k], atol=1e-6)


@pytest.mark.parametrize("use_conditioning", [False, True])
def test_null_pair_is_in_every_training_graph(use_conditioning):
    """Gradient on a drop; an all-zero (not missing) gradient otherwise, so DDP
    sees the same parameter set on every rank."""
    head, x = _head(cond_drop_fill="learned").train(), _inputs()
    st = head.compute_static(**x, use_conditioning=use_conditioning)
    (st.pair_cond.sum() + st.single_cond_pre_noise.sum()).backward()
    g = head.null_pair.grad
    assert g is not None
    assert bool(g.abs().sum() > 0) is (not use_conditioning)


def test_training_drop_through_forward_training_reaches_null_pair():
    torch.manual_seed(0)
    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0, cond_drop_fill="learned").train()
    with torch.no_grad():
        model.diffusion_head.null_pair.normal_()
    batch = _tiny_batch()
    seen = {}
    orig = model.diffusion_head.sample_training

    def spy(*a, **kw):
        seen["pair"] = kw["embeddings"]["pair"][0].detach()
        return orig(*a, **kw)

    model.diffusion_head.sample_training = spy
    model.forward_training(
        batch, num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
    )
    intra = interface_pair_keep(batch["asym_id"])[0]
    v = model.diffusion_head.null_pair.detach()
    dropped = (seen["pair"] == v).all(-1)
    assert not bool(dropped[intra].any())
    assert bool(dropped[~intra].any())


def test_strict_load_tolerates_only_a_missing_null_pair(tmp_path):
    from deepfold3.training.checkpointing import load_pretrained_weights

    old = _tiny_model()  # zero fill: no null_pair
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict()}, path)
    new = _tiny_model(cond_drop_fill="learned")
    load_pretrained_weights(new, path, strict=True, device=torch.device("cpu"))
    assert torch.equal(new.diffusion_head.null_pair, torch.zeros_like(new.diffusion_head.null_pair))
    k = next(iter(old.state_dict()))
    torch.save({"model": {n: v for n, v in old.state_dict().items() if n != k}}, path)
    with pytest.raises(RuntimeError, match="missing keys"):
        load_pretrained_weights(new, path, strict=True, device=torch.device("cpu"))


def test_null_pair_stats():
    from deepfold3.model.diffusion import null_pair_stats

    asym = torch.tensor([[0, 0, 1]])
    pair = torch.zeros(1, 3, 3, 4)
    pair[0, :2, :2] = torch.tensor([1.0, 0, 0, 0])  # intra (chain 0)
    pair[0, 2, 2] = torch.tensor([1.0, 0, 0, 0])  # intra (chain 1)
    pair[0, :2, 2] = pair[0, 2, :2] = torch.tensor([0, 2.0, 0, 0])  # inter
    v = torch.tensor([0, 3.0, 0, 0])
    s = null_pair_stats(v, pair, asym, torch.ones(1, 3))
    assert torch.isclose(s["norm"], torch.tensor(3.0))
    assert torch.isclose(s["cos_inter"], torch.tensor(1.0))
    assert torch.isclose(s["cos_intra"], torch.tensor(0.0))
    assert torch.isclose(s["norm_ratio"], torch.tensor(3.0 / ((5 * 1.0 + 4 * 2.0) / 9)))
    assert "cos_inter" not in null_pair_stats(v, pair, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3))


@pytest.mark.parametrize("fill", ["learned", "zero"])
def test_forward_training_emits_null_pair_stats_only_when_learned(fill, monkeypatch):
    import deepfold3.model.diffusion as diffusion

    model = _tiny_model(use_conditioning_prob=0.8, cond_drop_fill=fill).train()
    seen = {}
    real = diffusion.null_pair_stats

    def spy(*a, **kw):
        seen["stats"] = real(*a, **kw)
        return seen["stats"]

    monkeypatch.setattr(diffusion, "null_pair_stats", spy)
    torch.manual_seed(0)
    model.forward_training(
        _tiny_batch(), num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
    )
    if fill == "zero":
        assert "stats" not in seen
    else:
        assert {"norm", "norm_ratio", "cos_intra", "cos_inter"} <= set(seen["stats"])


def test_wrapped_optimizer_records_null_pair_grad_norm():
    from unittest.mock import MagicMock

    from torch import nn
    from torch.amp import GradScaler

    from deepfold3.model.config import TrainingConfig
    from deepfold3.training.distributed import DistInfo
    from deepfold3.training.optim import WrappedOptimizer
    from deepfold3.training.state import TrainState

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_head = nn.Module()
            self.diffusion_head.null_pair = nn.Parameter(torch.zeros(4))

    model = M()
    opt = WrappedOptimizer(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=MagicMock(),
        scaler=GradScaler("cpu", enabled=False),
        model=model,
        dist_info=DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
        cfg=TrainingConfig(lr=1e-3, accumulate_grad_batches=1, grad_norm_fp64=True),
    )
    (model.diffusion_head.null_pair * torch.tensor([3.0, 4.0, 0.0, 0.0])).sum().backward()
    state = TrainState()
    opt._unscale_clip_and_measure(state)
    assert torch.isclose(state.last_null_pair_grad_norm, torch.tensor(5.0))
