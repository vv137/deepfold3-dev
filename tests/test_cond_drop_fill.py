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
    assert not any("null_pair" in n for n, _ in _head().named_parameters())


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
    g = head.null_pair_dir.grad
    assert g is not None
    assert bool(g.abs().sum() > 0) is (not use_conditioning)


def test_training_drop_through_forward_training_reaches_null_pair():
    torch.manual_seed(0)
    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0, cond_drop_fill="learned").train()
    with torch.no_grad():
        model.diffusion_head.null_pair_dir.normal_()
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
            self.diffusion_head.null_pair_dir = nn.Parameter(torch.zeros(4))

    model = M()
    opt = WrappedOptimizer(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=MagicMock(),
        scaler=GradScaler("cpu", enabled=False),
        model=model,
        dist_info=DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
        cfg=TrainingConfig(lr=1e-3, accumulate_grad_batches=1, grad_norm_fp64=True),
    )
    (model.diffusion_head.null_pair_dir * torch.tensor([3.0, 4.0, 0.0, 0.0])).sum().backward()
    state = TrainState()
    opt._unscale_clip_and_measure(state)
    assert torch.isclose(state.last_null_pair_grad_norm, torch.tensor(5.0))


def test_init_null_pair_sets_the_fill_as_scale_times_unit_rms_direction():
    head = _head(cond_drop_fill="learned")
    vec = torch.randn(128) * 400.0 + 150.0
    head.init_null_pair(vec)
    assert torch.allclose(head.null_pair, vec, rtol=1e-5)
    assert torch.isclose(head.null_pair_dir.pow(2).mean().sqrt(), torch.tensor(1.0), rtol=1e-5)
    assert torch.isclose(head.null_pair_log_scale.exp(), vec.pow(2).mean().sqrt(), rtol=1e-5)
    with pytest.raises(ValueError, match="shape"):
        head.init_null_pair(torch.zeros(7))
    with pytest.raises(ValueError, match="learned"):
        _head().init_null_pair(vec)


def test_a_single_vector_checkpoint_loads_into_direction_and_scale():
    """Checkpoints written before the split store ``null_pair`` as one vector."""
    head = _head(cond_drop_fill="learned")
    vec = torch.linspace(-0.3, 0.2, 128)
    sd = {k: v for k, v in head.state_dict().items() if "null_pair" not in k}
    head.load_state_dict({**sd, "null_pair": vec})
    assert torch.allclose(head.null_pair, vec, atol=1e-6)


def test_the_log_scale_moves_the_fill_by_a_factor_per_step():
    """Adam steps the log scale by ~lr, so the norm changes by a ratio, not by lr in absolute units."""
    head = _head(cond_drop_fill="learned")
    head.init_null_pair(torch.full((128,), 2000.0))
    opt = torch.optim.Adam([head.null_pair_log_scale], lr=0.01)
    before = float(head.null_pair.norm())
    for _ in range(10):
        opt.zero_grad()
        head.null_pair.norm().backward()  # push the scale down
        opt.step()
    assert float(head.null_pair.norm()) == pytest.approx(before * float(torch.tensor(-0.1).exp()), rel=1e-3)


@pytest.mark.parametrize(
    ("train", "dir_grad", "scale_grad"), [("all", True, True), ("direction", True, False), ("none", False, False)]
)
def test_trainer_inits_from_file_and_freezes_per_null_pair_train(tmp_path, train, dir_grad, scale_grad):
    from types import SimpleNamespace

    from deepfold3.training.trainer import Trainer

    model = _tiny_model(cond_drop_fill="learned")
    c = model.diffusion_head.null_pair_dir.shape[0]
    vec = torch.arange(c, dtype=torch.float32) + 1.0
    path = tmp_path / "stats.pt"
    torch.save({"mean_all": vec, "mean_inter": -vec}, path)
    dcfg = SimpleNamespace(null_pair_init=str(path), null_pair_init_key="mean_inter", null_pair_train=train)
    Trainer._init_null_pair(SimpleNamespace(model=model, model_cfg=SimpleNamespace(diffusion=dcfg)))
    head = model.diffusion_head
    assert torch.allclose(head.null_pair, -vec, rtol=1e-5)
    assert head.null_pair_dir.requires_grad is dir_grad
    assert head.null_pair_log_scale.requires_grad is scale_grad
