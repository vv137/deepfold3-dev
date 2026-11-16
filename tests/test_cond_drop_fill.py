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
