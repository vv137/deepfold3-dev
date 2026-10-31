"""Classifier-free guidance in ``DiffusionHead.sample`` and the ``cond_drop`` modes."""

from __future__ import annotations

import warnings

import pytest
import torch

from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.diffusion import (
    DiffusionHead,
    GuidanceTerm,
    cond_drop_stats,
    draw_interface_pair_keep,
    interface_pair_keep,
    mask_trunk_pair,
)

N, M = 3, 2


def _head(**kw) -> DiffusionHead:
    cfg = DiffusionConfig(steps=4, num_samples=2, **kw)
    head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig()).eval()
    # AF3 zero-inits the output projections, which would make every denoiser
    # output identical and every comparison below vacuous.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.05)
    return head


def _inputs(pair: torch.Tensor | None = None) -> dict:
    g = torch.Generator().manual_seed(1)
    single = torch.randn(1, N, 384, generator=g)
    drawn = torch.randn(1, N, N, 128, generator=g)  # always drawn: keeps the stream aligned
    return {
        "embeddings": {
            "single": single,
            "pair": drawn if pair is None else pair,
            "target_feat": torch.randn(1, N, 447, generator=g),
        },
        "ref_pos": torch.randn(1, N, M, 3, generator=g),
        "ref_mask": torch.ones(1, N, M),
        "ref_element": torch.zeros(1, N, M, dtype=torch.int64),
        "ref_charge": torch.zeros(1, N, M),
        "ref_atom_name_chars": torch.zeros(1, N, M, 4, dtype=torch.int64),
        "seq_mask": torch.ones(1, N),
        "rel_feat": torch.randn(1, N, N, 8, generator=g),
    }


def _sample(head: DiffusionHead, inputs: dict) -> torch.Tensor:
    return head.sample(**inputs, generator=torch.Generator().manual_seed(2))["atom_positions"]


def _static(head: DiffusionHead, inputs: dict, use_conditioning: bool):
    return head.compute_static(**inputs, use_conditioning=use_conditioning)


@pytest.mark.parametrize("mode", ["pair", "both"])
def test_cond_drop_zeroes_only_what_it_names(mode):
    head, x = _head(cond_drop=mode), _inputs()
    with torch.no_grad():
        c, u = _static(head, x, True), _static(head, x, False)
        zero_pair = _inputs(pair=torch.zeros(1, N, N, 128))
        ref = _static(head, zero_pair, True)
    assert torch.allclose(u.pair_cond, ref.pair_cond)
    assert not torch.allclose(u.pair_cond, c.pair_cond)
    same_single = torch.allclose(u.single_cond_pre_noise, c.single_cond_pre_noise)
    assert same_single == (mode == "pair")


def test_default_drop_is_pair():
    assert DiffusionConfig().cond_drop == "pair"


def test_scale_one_is_the_unguided_path(monkeypatch):
    head, x = _head(), _inputs()
    calls = []
    orig = head.compute_static

    def spy(*a, **kw):
        calls.append(kw.get("use_conditioning", True))
        return orig(*a, **kw)

    monkeypatch.setattr(head, "compute_static", spy)
    base = _sample(head, x)
    assert calls == [True]  # no unconditional static, no second forward

    head.cfg.guidance_scale = 2.0
    calls.clear()
    guided = _sample(head, x)
    assert calls == [True, False]
    assert not torch.allclose(base, guided)


def test_scale_zero_is_the_unconditional_model():
    head = _head(guidance_scale=0.0)
    guided = _sample(head, _inputs())
    head.cfg.guidance_scale = 1.0
    uncond = _sample(head, _inputs(pair=torch.zeros(1, N, N, 128)))
    assert torch.allclose(guided, uncond, atol=1e-5)


def test_sigma_range_outside_schedule_is_unguided():
    x = _inputs()
    base = _sample(_head(), x)
    off = _sample(_head(guidance_scale=3.0, guidance_sigma_range=(1e6, 1e7)), x)
    assert torch.equal(base, off)


def test_config_validation():
    with pytest.raises(ValueError):
        DiffusionConfig(cond_drop="single")
    with pytest.raises(ValueError):
        DiffusionConfig(guidance_sigma_range=(5.0, 1.0))
    assert DiffusionConfig(guidance_sigma_range=[1.0, 5.0]).guidance_sigma_range == (1.0, 5.0)


def _asym(*sizes: int, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [c for c, n in enumerate(sizes) for _ in range(n)] + [0] * pad
    mask = [1] * sum(sizes) + [0] * pad
    return torch.tensor([ids]), torch.tensor([mask])


def test_interface_keep_drops_every_inter_chain_pair():
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)[0]
    assert keep.tolist() == [[True, True, False], [True, True, False], [False, False, True]]


def test_drawn_interface_keep_is_a_proper_chain_pair_subset():
    asym, seq_mask = _asym(2, 2, 2, 1, pad=2)
    g = torch.Generator().manual_seed(0)
    n_dropped = set()
    for _ in range(200):
        keep = draw_interface_pair_keep(asym, seq_mask, generator=g)[0]
        assert torch.equal(keep, keep.T)
        assert bool(keep[interface_pair_keep(asym)[0] & seq_mask[0].bool()[:, None]].all())  # intra kept
        assert bool(keep[-2:].all() and keep[:, -2:].all())  # padding untouched
        # a whole chain-pair block at a time
        blocks = {(int(asym[0, i]), int(asym[0, j])): set() for i in range(7) for j in range(7)}
        for i in range(7):
            for j in range(7):
                blocks[(int(asym[0, i]), int(asym[0, j]))].add(bool(keep[i, j]))
        assert all(len(v) == 1 for v in blocks.values())
        dropped = sum(1 for (a, b), v in blocks.items() if a < b and v == {False})
        assert dropped >= 1
        n_dropped.add(dropped)
    assert 1 in n_dropped and 6 in n_dropped  # single interfaces through all of them


def test_drawn_interface_keep_needs_two_chains():
    asym, seq_mask = _asym(4, pad=2)
    assert draw_interface_pair_keep(asym, seq_mask) is None


def test_masked_pair_matches_the_full_drop_where_dropped():
    head, x = _head(), _inputs()
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    with torch.no_grad():
        c, u = _static(head, x, True), _static(head, x, False)
        part = head.compute_static(**{**x, "embeddings": mask_trunk_pair(x["embeddings"], keep)})
    k = keep[0]
    assert torch.allclose(part.pair_cond[0][k], c.pair_cond[0][k])
    assert torch.allclose(part.pair_cond[0][~k], u.pair_cond[0][~k])
    assert torch.allclose(part.single_cond_pre_noise, c.single_cond_pre_noise)


def test_guidance_pair_keep_is_the_unconditional_branch():
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    x = _inputs()
    head = _head(guidance_scale=0.0, guidance_drop="interface")
    guided = head.sample(**x, generator=torch.Generator().manual_seed(2), guidance_pair_keep=keep)["atom_positions"]
    head.cfg.guidance_scale = 1.0
    ref = _sample(head, {**x, "embeddings": mask_trunk_pair(x["embeddings"], keep)})
    assert torch.allclose(guided, ref, atol=1e-5)


def test_interface_guidance_needs_a_keep_mask():
    with pytest.raises(ValueError):
        _sample(_head(guidance_scale=2.0, guidance_drop="interface"), _inputs())


def test_guidance_false_is_unguided():
    x = _inputs()
    base = _sample(_head(), x)
    off = _head(guidance_scale=3.0).sample(**x, generator=torch.Generator().manual_seed(2), guidance=False)
    assert torch.equal(base, off["atom_positions"])


def test_interface_config_validation():
    with pytest.raises(ValueError):
        DiffusionConfig(guidance_drop="chain")
    with pytest.raises(ValueError):
        DiffusionConfig(cond_drop_interface_frac=1.5)


class _Stop(Exception):
    pass


def _tiny_model(**diff_kw):
    from deepfold3.model.model import DeepFold3
    from tests.test_batch import _tiny_config

    cfg = _tiny_config()
    for k, v in diff_kw.items():
        setattr(cfg.diffusion, k, v)
    return DeepFold3(cfg)


def _tiny_batch():
    from deepfold3.data.module.collate import collate_fn
    from tests.test_batch import _make_features

    f = _make_features(6)
    f["asym_id"] = torch.tensor([0, 0, 0, 1, 1, 2])
    f["atom_gt_positions"] = torch.randn(6, 24, 3)
    f["atom_gt_mask"] = f["ref_mask"].clone()
    return collate_fn([f])


def test_training_interface_drop_masks_only_inter_chain_pair():
    """End to end through ``forward_training``: an always-firing interface drop
    reaches the diffusion head as a masked trunk pair with conditioning on,
    and the mini-rollout runs unguided even with ``guidance_scale`` set."""
    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0, guidance_scale=2.0).train()
    batch = _tiny_batch()
    seen = {}
    sample = model.diffusion_head.sample

    def spy_sample(*a, **kw):
        seen.setdefault("rollout_guidance", kw.get("guidance", True))
        return sample(*a, **kw)

    def spy_training(*a, **kw):
        seen["pair"], seen["use_conditioning"] = kw["embeddings"]["pair"][0], kw["use_conditioning"]
        raise _Stop

    model.diffusion_head.sample = spy_sample
    model.diffusion_head.sample_training = spy_training
    torch.manual_seed(0)
    with pytest.raises(_Stop):
        model.forward_training(
            batch, num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
        )
    assert seen["rollout_guidance"] is False
    assert seen["use_conditioning"] is True
    zero = seen["pair"].abs().sum(-1) == 0
    intra = interface_pair_keep(batch["asym_id"])[0]
    assert not bool(zero[intra].any())  # intra-chain pair untouched
    assert bool(zero[~intra].any())  # at least one interface dropped
    assert torch.equal(zero, zero.T)


def test_chunked_training_backward_with_interface_drop():
    """Chunked diffusion back-props once per chunk through the detached trunk
    leaves. The interface mask must be applied per chunk: masked once outside
    the loop, the second chunk's backward re-walks the freed mask node
    ("Trying to backward through the graph a second time"), the crash of the
    first 2-node stage2 smoke run."""
    from deepfold3.model.config import TrainingConfig

    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0).train()
    batch = _tiny_batch()
    torch.manual_seed(0)
    model.forward_training(
        batch,
        num_recycles=0,
        diffusion_batch_size=2,
        diffusion_chunk_size=1,
        mini_rollout_steps=2,
        compute_confidence=False,
        backward_fn=lambda loss: loss.backward(),
        loss_cfg=TrainingConfig(),
    )
    assert next(model.diffusion_head.parameters()).grad is not None


def test_inference_interface_guidance_passes_the_interface_mask():
    model = _tiny_model(guidance_scale=2.0, guidance_drop="interface").eval()
    batch = _tiny_batch()
    seen = {}
    sample = model.diffusion_head.sample

    def spy(*a, **kw):
        seen["keep"] = kw.get("guidance_pair_keep")
        return sample(*a, **kw)

    model.diffusion_head.sample = spy
    with torch.no_grad():
        model(batch, num_recycles=0)
    assert torch.equal(seen["keep"], interface_pair_keep(batch["asym_id"]))


def test_stage2_yaml_enables_the_drop_on_top_of_stage1():
    from deepfold3.config_loader import load_config
    from deepfold3.model.config import ModelConfig
    from deepfold3.scripts.train import _apply_overrides

    s1, s2 = ModelConfig(), ModelConfig()
    _apply_overrides(s1, load_config("configs/stage1.yaml")["model"])
    _apply_overrides(s2, load_config("configs/stage2.yaml")["model"])
    d = s2.diffusion
    assert (d.use_conditioning_prob, d.cond_drop, d.cond_drop_interface_frac) == (0.8, "pair", 0.5)
    assert d.guidance_scale == 1.0  # training config: eval rollouts stay unguided
    for k in ("use_conditioning_prob", "cond_drop", "cond_drop_interface_frac"):
        setattr(d, k, getattr(s1.diffusion, k))
    assert s2 == s1  # nothing else differs from stage1


def test_yaml_overrides_are_validated():
    from deepfold3.model.config import ModelConfig
    from deepfold3.scripts.train import _apply_overrides

    with pytest.raises(ValueError):
        _apply_overrides(ModelConfig(), {"diffusion": {"guidance_drop": "interfce"}})
    cfg = ModelConfig()
    _apply_overrides(cfg, {"diffusion": {"guidance_sigma_range": [1.0, 160.0]}})
    assert cfg.diffusion.guidance_sigma_range == (1.0, 160.0)


def test_cond_drop_stats_one_hot_and_pair_frac():
    asym, seq_mask = _asym(2, 1, 1, pad=1)
    st = cond_drop_stats(asym, seq_mask, use_conditioning=True, pair_keep=None)
    assert {k: float(v) for k, v in st.items()} == {"none": 1.0, "full": 0.0, "interface": 0.0, "multichain": 1.0}
    assert float(cond_drop_stats(asym, seq_mask, use_conditioning=False, pair_keep=None)["full"]) == 1.0
    keep = interface_pair_keep(asym).clone()
    keep[0, 2, 3] = keep[0, 3, 2] = True  # leave the chain-1/chain-2 interface conditioned
    st = cond_drop_stats(asym, seq_mask, use_conditioning=True, pair_keep=keep)
    assert float(st["interface"]) == 1.0 and float(st["none"]) == 0.0
    # inter-chain valid pairs: 2*(2*1 + 2*1 + 1*1) = 10; 2 kept -> 8/10 zeroed
    assert float(st["interface_pair_frac"]) == pytest.approx(0.8)
    single, sm = _asym(3, pad=1)
    assert float(cond_drop_stats(single, sm, use_conditioning=False, pair_keep=None)["multichain"]) == 0.0


def test_training_forward_reports_cond_drop_stats():
    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0).train()
    seen = {}

    def stop(*a, **kw):
        raise _Stop

    model.diffusion_head.sample_training = stop
    orig = model.forward_training

    # ``__cond_drop__`` is written into ``pred`` before the diffusion step;
    # read it off the dict the forward builds.
    import deepfold3.model.diffusion as diffusion

    real = diffusion.cond_drop_stats

    def spy(*a, **kw):
        seen["stats"] = real(*a, **kw)
        return seen["stats"]

    diffusion.cond_drop_stats = spy
    try:
        with pytest.raises(_Stop):
            orig(_tiny_batch(), num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False)
    finally:
        diffusion.cond_drop_stats = real
    st = {k: float(v) for k, v in seen["stats"].items()}
    assert st["interface"] == 1.0 and st["multichain"] == 1.0 and 0.0 < st["interface_pair_frac"] <= 1.0


def test_no_cond_drop_stats_when_drop_is_off():
    import deepfold3.model.diffusion as diffusion

    model = _tiny_model().train()
    called = []
    real = diffusion.cond_drop_stats
    diffusion.cond_drop_stats = lambda *a, **kw: called.append(1) or real(*a, **kw)

    def stop(*a, **kw):
        raise _Stop

    model.diffusion_head.sample_training = stop
    try:
        with pytest.raises(_Stop):
            model.forward_training(
                _tiny_batch(), num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
            )
    finally:
        diffusion.cond_drop_stats = real
    assert called == []  # default config logs nothing new


def test_training_step_logs_cond_drop_under_its_own_prefix(monkeypatch):
    """``pred["__cond_drop__"]`` reaches ``loss_dict`` as ``cond_drop/*`` with
    count 1 each (one draw per forward), which the trainer logs as
    ``train/cond_drop/*``."""
    from contextlib import nullcontext

    import deepfold3.training.step as step
    from deepfold3.model.config import ModelConfig, TrainingConfig

    stats = {"none": torch.tensor(0.0), "full": torch.tensor(0.0), "interface": torch.tensor(1.0)}

    def fake_model(features, **kw):
        return {"__cond_drop__": stats}

    monkeypatch.setattr(
        step, "training_loss", lambda pred, feats, cfg: (torch.tensor(1.0), {"loss": torch.tensor(1.0)})
    )
    monkeypatch.setattr(step, "prepare_features", lambda f, d: f)

    class _Acc:
        def chunk_context(self, **kw):
            return nullcontext()

    _, loss_dict, counts, *_ = step.training_step(
        fake_model,
        {"seq_mask": torch.ones(1, 3)},
        cfg=TrainingConfig(batch_size_per_rank=1),
        model_cfg=ModelConfig(),
        global_step=0,
        accumulator=_Acc(),
        scaler=None,
        device=torch.device("cpu"),
    )
    assert {k: float(v) for k, v in loss_dict.items() if k.startswith("cond_drop/")} == {
        "cond_drop/none": 0.0,
        "cond_drop/full": 0.0,
        "cond_drop/interface": 1.0,
    }
    assert all(counts[k] == 1 for k in loss_dict if k.startswith("cond_drop/"))


def _parse(text: str):
    from deepfold3.model.guidance import parse_guidance_spec

    return parse_guidance_spec(text)


def test_guidance_spec_syntax():
    from deepfold3.model.guidance import ChainSel

    assert _parse("2").target == "full"
    assert _parse("1.5@all").target == "all"
    pairs = _parse("2@A:B,AA:B")
    assert (pairs.target, pairs.pairs) == ("pairs", ((ChainSel("A"), ChainSel("B")), (ChainSel("AA"), ChainSel("B"))))
    assert _parse("2@full@1e-4..160").sigma_range == (1e-4, 160.0)


def test_residue_selector_syntax():
    from deepfold3.model.guidance import ChainSel, has_residue_ranges

    spec = _parse("2@A/25-40+88:B")
    assert spec.pairs == ((ChainSel("A", ((25, 40), (88, 88))), ChainSel("B")),)
    assert str(spec.pairs[0][0]) == "A/25-40+88"  # round-trips for the run manifest / echo
    assert has_residue_ranges([spec])
    assert not has_residue_ranges([_parse("2@A:B"), _parse("2@all"), _parse("2")])
    assert not has_residue_ranges([_parse("1@A/25-40:B")])  # w = 1 is inert


@pytest.mark.parametrize(
    "text",
    [
        "",
        "x",
        "nan",
        "2@A",
        "2@A:A",
        "2@A:B:C",
        "2@A:",
        "2@A:B,",
        "2@all@5..1",
        "2@all@1",
        "2@all@1..2@x",
        "2@A/:B",  # '/' with no residues
        "2@A/x:B",  # not a number
        "2@A/40-25:B",  # LO > HI
        "2@A/25-:B",
        "2@A/25-40-60:B",
        "2@A/-5:B",  # no negative residue numbers
        "2@A/25+:B",
        "2@A/25-40:A/60-80",  # still the same chain
    ],
)
def test_guidance_spec_rejects_malformed(text):
    with pytest.raises(ValueError):
        _parse(text)


def test_pair_term_drops_exactly_the_named_chain_pairs():
    from deepfold3.model.guidance import build_guidance_terms

    asym, _ = _asym(2, 2, 1)
    index = {"A": 0, "B": 1, "AA": 2}
    keep = build_guidance_terms([_parse("2@AA:A")], asym[0], index)[0].pair_keep
    dropped = {(int(asym[0, i]), int(asym[0, j])) for i, j in (~keep).nonzero().tolist()}
    assert dropped == {(0, 2), (2, 0)}
    assert torch.equal(keep, keep.T)


def test_residue_selector_drops_only_the_named_blocks():
    from deepfold3.model.guidance import build_guidance_terms

    asym = torch.tensor([0, 0, 0, 1, 1])
    res = torch.tensor([1, 2, 3, 1, 2])
    keep = build_guidance_terms([_parse("2@A/2-3:B")], asym, {"A": 0, "B": 1}, residue_index=res)[0].pair_keep
    dropped = {tuple(ij) for ij in (~keep).nonzero().tolist()}
    # tokens 1,2 (chain A, residues 2-3) against tokens 3,4 (chain B), both blocks
    assert dropped == {(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (4, 1), (3, 2), (4, 2)}
    assert bool(keep[0].all()) and bool(keep[:, 0].all())  # residue 1 keeps its conditioning


def test_pairs_of_one_term_are_a_union_not_two_branches():
    """Commas inside one term union into ONE mask, so one more denoiser pass.

    Two ``--guidance`` values of the same weight are a different thing: two
    unconditional branches, two differences, two passes.
    """
    from deepfold3.model.guidance import build_guidance_terms

    asym = torch.tensor([0, 0, 1, 2])
    res = torch.tensor([1, 2, 1, 1])
    args = (asym, {"A": 0, "B": 1, "C": 2})
    both = build_guidance_terms([_parse("2@A/1:B,A/2:C")], *args, residue_index=res)
    assert len(both) == 1
    one = build_guidance_terms([_parse("2@A/1:B")], *args, residue_index=res)[0].pair_keep
    two = build_guidance_terms([_parse("2@A/2:C")], *args, residue_index=res)[0].pair_keep
    assert torch.equal(both[0].pair_keep, one & two)


def test_overlapping_pairs_in_one_term_are_an_error():
    from deepfold3.model.guidance import build_guidance_terms

    asym, res = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 1])
    with pytest.raises(ValueError, match="already dropped by"):
        build_guidance_terms([_parse("2@A/1-2:B,A/2-3:B")], asym, {"A": 0, "B": 1}, residue_index=res)
    # disjoint residues of the same chain pair are fine
    build_guidance_terms([_parse("2@A/1:B,A/3:B")], asym, {"A": 0, "B": 1}, residue_index=res)


def test_overlapping_terms_warn_about_the_summed_weight():
    from deepfold3.model.guidance import build_guidance_terms

    asym, res = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 1])
    args = (asym, {"A": 0, "B": 1})
    with pytest.warns(UserWarning, match="not either weight"):
        build_guidance_terms([_parse("2@A/1-2:B"), _parse("-1@A/2-3:B")], *args, residue_index=res)
    with warnings.catch_warnings():  # disjoint blocks, and disjoint σ̂ ranges, stay quiet
        # (σ̂ bounds are inclusive on both sides: 0..100 and 100..4608 would share σ̂ = 100)
        warnings.simplefilter("error")
        build_guidance_terms([_parse("2@A/1:B"), _parse("-1@A/3:B")], *args, residue_index=res)
        build_guidance_terms([_parse("2@A/1-2:B@100..4608"), _parse("-1@A/2-3:B@0..99.9")], *args, residue_index=res)
        build_guidance_terms([_parse("2"), _parse("2@all")], *args, residue_index=res)  # full is exempt


def test_residue_selector_needs_residue_index():
    from deepfold3.model.guidance import build_guidance_terms

    with pytest.raises(ValueError, match="residue_index"):
        build_guidance_terms([_parse("2@A/2-3:B")], torch.tensor([0, 1]), {"A": 0, "B": 1})


def test_residue_selector_that_selects_nothing_is_an_error():
    from deepfold3.model.guidance import build_guidance_terms

    asym, res = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 1])
    with pytest.raises(ValueError, match="A/9-20 selects no token"):
        build_guidance_terms([_parse("2@A/9-20:B")], asym, {"A": 0, "B": 1}, residue_index=res)


def test_residue_selector_never_drops_bucket_padding():
    asym = torch.tensor([0, 0, 1, 0, 0])  # the last two are padding, asym_id 0
    res = torch.tensor([1, 2, 1, 1, 2])  # padding repeats the residue numbers
    from deepfold3.model.guidance import build_guidance_terms

    keep = build_guidance_terms([_parse("2@A/1-2:B")], asym, {"A": 0, "B": 1}, n_tokens=3, residue_index=res)[
        0
    ].pair_keep
    assert (~keep).sum() == 4  # (0,2), (1,2) and their mirrors
    assert bool(keep[3:].all()) and bool(keep[:, 3:].all())


def test_pair_term_never_drops_bucket_padding():
    from deepfold3.model.guidance import build_guidance_terms

    asym = torch.tensor([0, 0, 1, 0, 0])  # the last two are padding, asym_id 0
    keep = build_guidance_terms([_parse("2@A:B")], asym, {"A": 0, "B": 1}, n_tokens=3)[0].pair_keep
    assert (~keep).sum() == 4  # (0,2), (1,2) and their mirrors
    assert bool(keep[2, 3:].all())


@pytest.mark.parametrize("text", ["2@A:Z", "2@Z:A"])
def test_unknown_chain_is_an_error(text):
    from deepfold3.model.guidance import build_guidance_terms

    with pytest.raises(ValueError, match="unknown chain"):
        build_guidance_terms([_parse(text)], torch.tensor([0, 1]), {"A": 0, "B": 1})


@pytest.mark.parametrize("text", ["2@all", "2@A:B"])
def test_interface_term_needs_two_chains(text):
    from deepfold3.model.guidance import build_guidance_terms

    with pytest.raises(ValueError, match="two chains"):
        build_guidance_terms([_parse(text)], torch.zeros(3, dtype=torch.int64), {"A": 0})


def test_checkpoint_must_have_trained_the_drop():
    from deepfold3.model.guidance import check_guidance_training

    with pytest.raises(ValueError, match="use_conditioning_prob"):
        check_guidance_training([_parse("2")], DiffusionConfig())
    with pytest.raises(ValueError, match="cond_drop_interface_frac"):
        check_guidance_training([_parse("2@all")], DiffusionConfig(use_conditioning_prob=0.8))
    check_guidance_training([_parse("1")], DiffusionConfig())  # w = 1 is inert
    check_guidance_training([_parse("2")], DiffusionConfig(use_conditioning_prob=0.8))
    check_guidance_training([_parse("2@all")], DiffusionConfig(use_conditioning_prob=0.8, cond_drop_interface_frac=0.5))


def _sample_terms(head: DiffusionHead, inputs: dict, terms) -> torch.Tensor:
    return head.sample(**inputs, generator=torch.Generator().manual_seed(2), guidance_terms=terms)["atom_positions"]


def test_one_term_equals_the_config_path():
    x = _inputs()
    by_cfg = _sample(_head(guidance_scale=2.0, guidance_sigma_range=(1.0, 160.0)), x)
    by_term = _sample_terms(_head(), x, [GuidanceTerm(2.0, sigma_range=(1.0, 160.0))])
    assert torch.allclose(by_cfg, by_term, atol=1e-5)


def test_terms_override_the_config_and_scale_one_is_inert():
    x = _inputs()
    base = _sample(_head(), x)
    assert torch.equal(base, _sample_terms(_head(guidance_scale=3.0), x, []))
    assert torch.equal(base, _sample_terms(_head(), x, [GuidanceTerm(1.0)]))


def test_two_terms_are_the_linear_combination(monkeypatch):
    """Every step adds both terms' ``(w − 1)·(D_c − D_u)`` to the same ``D_c``."""
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    head, x = _head(), _inputs()
    head.cfg.steps = 1
    orig = head._denoise_samples
    denoised: list[torch.Tensor] = []

    def spy(*a, **kw):
        denoised.append(orig(*a, **kw))
        return denoised[-1]

    monkeypatch.setattr(head, "_denoise_samples", spy)
    captured = []
    step = head._sample_impl

    def spy_impl(*a, **kw):
        captured.append(kw["guidance_terms"])
        return step(*a, **kw)

    monkeypatch.setattr(head, "_sample_impl", spy_impl)
    guided = _sample_terms(head, x, [GuidanceTerm(1.5), GuidanceTerm(3.0, keep)])
    assert len(denoised) == 3 and len(captured[0]) == 2  # conditional + one per term

    den_c, den_u1, den_u2 = denoised
    combined = den_c + 0.5 * (den_c - den_u1) + 2.0 * (den_c - den_u2)
    denoised.clear()
    head._denoise_samples = lambda *a, **kw: combined
    ref = _sample_terms(head, x, [])
    assert torch.allclose(guided, ref, atol=1e-4)


def test_sigma_range_limits_only_its_own_term(monkeypatch):
    head, x = _head(), _inputs()
    calls = []
    orig = head._denoise_samples

    def spy(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(head, "_denoise_samples", spy)
    _sample_terms(head, x, [GuidanceTerm(2.0, sigma_range=(1e6, 1e7)), GuidanceTerm(2.0)])
    assert len(calls) == 2 * head.cfg.steps


def test_forward_passes_terms_to_the_sampler():
    model = _tiny_model().eval()
    batch = _tiny_batch()
    terms = (GuidanceTerm(2.0),)
    seen = {}
    sample = model.diffusion_head.sample

    def spy(*a, **kw):
        seen["terms"] = kw["guidance_terms"]
        return sample(*a, **kw)

    model.diffusion_head.sample = spy
    model(batch, num_recycles=0, num_samples=1, guidance_terms=terms)
    assert seen["terms"] is terms


def test_allow_overlap_takes_the_union_quietly():
    from deepfold3.model.guidance import build_guidance_terms

    asym, res = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 1])
    args = (asym, {"A": 0, "B": 1})
    union = build_guidance_terms([_parse("2@A/1-2:B,A/2-3:B")], *args, residue_index=res, allow_overlap=True)[
        0
    ].pair_keep
    expected = build_guidance_terms([_parse("2@A/1-3:B")], *args, residue_index=res)[0].pair_keep
    assert torch.equal(union, expected)
    with warnings.catch_warnings():  # and it silences the cross-term warning too
        warnings.simplefilter("error")
        build_guidance_terms([_parse("2@A/1-2:B"), _parse("-1@A/2-3:B")], *args, residue_index=res, allow_overlap=True)


def test_residue_numbers_are_the_input_sequence_positions(tmp_path):
    """End to end: the selector's numbering is the job JSON's own, 1-based.

    The guidance mask is built from ``feats["residue_index"]``, which carries
    res_id. For a sequence given in the JSON that is its 1-based position, so
    ``B/2-3`` must be the second and third residue of the chain the input calls
    B — not an internal index, and not 0-based.
    """
    import json

    from deepfold3.data.module.dataset import InferenceInput, job_chain_ids
    from deepfold3.data.parse.json_input import parse_json_input
    from deepfold3.io.mmcif import resolve_chain_name
    from deepfold3.model.guidance import build_guidance_terms

    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "name": "t",
                "dialect": "deepfold3",
                "version": 3,
                "modelSeeds": [1],
                "sequences": [
                    {"protein": {"id": "A", "sequence": "GGGG"}},
                    {"protein": {"id": "B", "sequence": "AAWAA"}},
                ],
            }
        )
    )
    job = parse_json_input(path)[0]
    feats = InferenceInput.from_job(job)
    names = job_chain_ids(job)  # exactly what ``inference`` passes
    chain_index = {resolve_chain_name(a, names): a for a in feats["asym_id"].unique().tolist()}

    keep = build_guidance_terms(
        [_parse("2@B/2-3:A")],
        feats["asym_id"],
        chain_index,
        residue_index=feats["residue_index"],
    )[0].pair_keep
    rows = sorted({int(i) for i, _ in (~keep).nonzero().tolist()} & set(range(4, 9)))
    assert rows == [5, 6]  # chain A is tokens 0..3, so B's residues 2,3 are tokens 5,6
    assert feats["residue_index"][5:7].tolist() == [2, 3]


# --- distance restraints (model/restraints.py) -------------------------------


def _restraint(text: str):
    from deepfold3.model.restraints import parse_restraint_spec

    return parse_restraint_spec(text)


def test_restraint_syntax():
    from deepfold3.model.guidance import ChainSel
    from deepfold3.model.restraints import DEFAULT_DISTANCE, describe_restraint

    r = _restraint("A/25-40+88:B")
    assert r.groups == (ChainSel("A", ((25, 40), (88, 88))), ChainSel("B"))
    assert (r.distance, r.sigma_range) == (DEFAULT_DISTANCE, None)
    r = _restraint("A:B@8@16..4608")
    assert (r.distance, r.sigma_range) == (8.0, (16.0, 4608.0))
    assert describe_restraint(_restraint("A/2:B@4")) == "A/2:B ≤ 4 Å"


@pytest.mark.parametrize(
    "text", ["", "A", "A:A", "A:B:C", "A:B@1", "A:B@50", "A:B@x", "A/:B", "A:B@6@5..1", "A:B@6@1..2@x"]
)
def test_restraint_rejects_malformed(text):
    with pytest.raises(ValueError):
        _restraint(text)


def _two_chains(sep: float):
    """5 tokens x 2 atoms: chain A at x = 0,1,2 and chain B `sep` away."""
    pos = torch.zeros(1, 5, 2, 3)
    pos[:, :3, :, 0] = torch.arange(3.0).view(1, 3, 1)
    pos[:, 3:, :, 0] = sep
    return pos, torch.ones(5, 2, dtype=torch.bool), torch.tensor([0, 0, 0, 1, 1]), torch.tensor([1, 2, 3, 1, 2])


def test_restraint_pulls_only_the_named_residues():
    from deepfold3.model.restraints import build_restraints, restraint_gradient

    pos, mask, asym, res = _two_chains(30.0)
    r = build_restraints([_restraint("A/2-3:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
    g = restraint_gradient(pos, mask, r, sharpness=1.0)
    per_token = g.norm(dim=-1).sum(-1).squeeze(0)
    assert float(per_token[0]) == 0.0  # residue 1 is not named
    assert all(float(per_token[i]) > 0 for i in (1, 2, 3, 4))
    moved = pos - 0.5 * 0.2 * g
    assert moved[0, 2, 0, 0] > pos[0, 2, 0, 0]  # the two groups approach each other
    assert moved[0, 3, 0, 0] < pos[0, 3, 0, 0]


def test_restraint_already_satisfied_is_a_no_op():
    from deepfold3.model.restraints import build_restraints, restraint_gradient

    pos, mask, asym, res = _two_chains(4.0)  # inside the 6 Å bottom
    r = build_restraints([_restraint("A/2-3:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
    assert float(restraint_gradient(pos, mask, r, sharpness=1.0).abs().max()) == 0.0


def test_restraint_force_falls_off_as_it_is_met():
    """Quadratic, not Boltz's linear flat bottom: the pull shrinks on approach."""
    from deepfold3.model.restraints import build_restraints, restraint_gradient

    def pull(sep: float) -> float:
        pos, mask, asym, res = _two_chains(sep)
        r = build_restraints([_restraint("A/2-3:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
        return float(restraint_gradient(pos, mask, r, sharpness=1.0).norm())

    assert pull(30.0) > pull(15.0) > pull(8.0) > 0.0


def test_restraint_sigma_range_selects_the_steps():
    from deepfold3.model.restraints import active_restraints, build_restraints

    _, _, asym, res = _two_chains(30.0)
    rs = build_restraints(
        [_restraint("A/2:B@6@100..4608"), _restraint("A/3:B@6")], asym, {"A": 0, "B": 1}, residue_index=res
    )
    assert len(active_restraints(rs, 500.0)) == 2
    assert len(active_restraints(rs, 10.0)) == 1  # only the unrestricted one


def test_restraint_never_pulls_bucket_padding():
    from deepfold3.model.restraints import build_restraints

    asym = torch.tensor([0, 0, 1, 0, 0])  # the last two are padding, asym_id 0
    res = torch.tensor([1, 2, 1, 1, 2])
    r = build_restraints([_restraint("A/1-2:B@6")], asym, {"A": 0, "B": 1}, n_tokens=3, residue_index=res)[0]
    assert r.tokens_a.tolist() == [True, True, False, False, False]


def test_restraint_gradient_survives_inference_mode():
    """The sampler runs under inference_mode, where enable_grad is not enough.

    A tensor created inside inference mode records no graph even within
    enable_grad, so the energy comes back detached and the restraint silently
    does nothing. This pins the inference_mode(False) copy that fixes it.
    """
    from deepfold3.model.restraints import build_restraints, restraint_gradient

    pos, mask, asym, res = _two_chains(30.0)
    r = build_restraints([_restraint("A/2-3:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
    with torch.inference_mode():
        g = restraint_gradient(pos, mask, r, sharpness=1.0)
    assert float(g.norm()) > 0.0


def _two_bodies():
    """Chain A (8 residues) and chain B (6), 30 Å apart and offset, 3 atoms each."""
    n_a, n_b, m = 8, 6, 3
    pos = torch.zeros(1, n_a + n_b, m, 3)
    for i in range(n_a):
        pos[0, i, :, 0] = i * 3.8
        pos[0, i, :, 1] = torch.tensor([0.0, 1.0, 2.0])
    for j in range(n_b):
        pos[0, n_a + j, :, 0] = 30.0 + j * 3.8
        pos[0, n_a + j, :, 1] = torch.tensor([20.0, 21.0, 22.0])
    mask = torch.ones(n_a + n_b, m, dtype=torch.bool)
    asym = torch.tensor([0] * n_a + [1] * n_b)
    res = torch.tensor(list(range(1, n_a + 1)) + list(range(1, n_b + 1)))
    return pos, mask, asym, res


def test_rigid_mode_keeps_every_internal_distance():
    """The whole point of the rigid projection: no stretched bonds."""
    from deepfold3.model.restraints import build_restraints, rigid_displacement

    pos, mask, asym, res = _two_bodies()
    r = build_restraints([_restraint("A/1-2:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)

    def inner(p, s):
        return torch.cdist(p[0, s].reshape(-1, 3), p[0, s].reshape(-1, 3))

    before = (inner(pos, slice(0, 8)), inner(pos, slice(8, None)))
    p = pos.clone()
    for _ in range(40):
        p = p + rigid_displacement(p, mask, r, weight=0.4, sharpness=1.0, max_step=2.0)
    after = (inner(p, slice(0, 8)), inner(p, slice(8, None)))
    assert all(float((x - y).abs().max()) < 1e-3 for x, y in zip(before, after, strict=True))
    far = torch.cdist(pos[0, :2].reshape(-1, 3), pos[0, 8:].reshape(-1, 3)).min()
    near = torch.cdist(p[0, :2].reshape(-1, 3), p[0, 8:].reshape(-1, 3)).min()
    assert float(far) > 25.0 and float(near) < 7.0  # and it still closes the restraint


def test_rigid_mode_rotates_not_only_translates():
    """A patch facing away has to be turned, not just moved in."""
    from deepfold3.model.restraints import build_restraints, rigid_displacement

    pos, mask, asym, res = _two_bodies()
    r = build_restraints([_restraint("A/1-2:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
    p = pos.clone()
    for _ in range(40):
        p = p + rigid_displacement(p, mask, r, weight=0.4, sharpness=1.0, max_step=2.0)
    # the chain's own axis has turned: compare the first-to-last residue vector
    v0 = (pos[0, 7, 0] - pos[0, 0, 0]) / (pos[0, 7, 0] - pos[0, 0, 0]).norm()
    v1 = (p[0, 7, 0] - p[0, 0, 0]) / (p[0, 7, 0] - p[0, 0, 0]).norm()
    assert float(torch.dot(v0, v1)) < 0.99


def test_rigid_step_is_capped():
    from deepfold3.model.restraints import build_restraints, rigid_displacement

    pos, mask, asym, res = _two_bodies()
    r = build_restraints([_restraint("A/1-2:B@6")], asym, {"A": 0, "B": 1}, residue_index=res)
    for cap in (0.5, 2.0):
        d = rigid_displacement(pos, mask, r, weight=10.0, sharpness=1.0, max_step=cap)
        assert float(d.norm(dim=-1).max()) <= cap * 2 + 1e-4  # translation + rotation arc


def _pair(text: str):
    from deepfold3.model.restraints import parse_restraint_spec

    return parse_restraint_spec(text, paired=True)


def test_paired_restraint_syntax():
    from deepfold3.model.restraints import describe_restraint

    spec = _pair("C/118+120+145:A/31+52+101@5")
    assert spec.paired and spec.distance == 5.0
    assert describe_restraint(spec) == "C/118+120+145:A/31+52+101 ≤ 5 Å (3 pairs)"


@pytest.mark.parametrize("text", ["A:B", "A/2-3:B", "A:B/1", "A/2-3:B/1", "A/1+2:B/1+2+3"])
def test_paired_restraint_needs_matching_residues(text):
    with pytest.raises(ValueError):
        _pair(text)


def test_paired_restraint_pulls_the_named_partners():
    """A group restraint cannot say which residue meets which; this can."""
    from deepfold3.model.restraints import build_restraints, restraint_gradient

    pos, mask, asym, res = _two_chains(30.0)
    r = build_restraints([_pair("A/2+3:B/1+2@5")], asym, {"A": 0, "B": 1}, residue_index=res)
    assert r[0].pair_index == ((1, 3), (2, 4))
    g = restraint_gradient(pos, mask, r, sharpness=1.0)
    per_token = g.norm(dim=-1).sum(-1).squeeze(0)
    assert float(per_token[0]) == 0.0  # residue 1 of A is in neither pair
    assert all(float(per_token[i]) > 0 for i in (1, 2, 3, 4))


def test_paired_restraint_rejects_a_missing_residue():
    from deepfold3.model.restraints import build_restraints

    _, _, asym, res = _two_chains(30.0)
    with pytest.raises(ValueError, match="not a residue"):
        build_restraints([_pair("A/2+99:B/1+2@5")], asym, {"A": 0, "B": 1}, residue_index=res)


def test_repel_pushes_apart_below_the_bound_only():
    from deepfold3.model.restraints import build_restraints, parse_restraint_spec, restraint_gradient

    def grad(sep: float, bound: float):
        pos, mask, asym, res = _two_chains(sep)
        spec = parse_restraint_spec(f"A/1:B@{bound:g}", repel=True)
        r = build_restraints([spec], asym, {"A": 0, "B": 1}, residue_index=res)
        return restraint_gradient(pos, mask, r, sharpness=1.0)

    assert float(grad(30.0, 20.0).abs().max()) == 0.0  # already further than the bound
    g = grad(10.0, 20.0)
    pos, mask, asym, res = _two_chains(10.0)
    moved = pos - 0.05 * g
    before = float(pos[0, 3, 0, 0] - pos[0, 0, 0, 0])
    after = float(moved[0, 3, 0, 0] - moved[0, 0, 0, 0])
    assert after > before  # pushed apart


def test_repel_reads_in_the_description():
    from deepfold3.model.restraints import describe_restraint, parse_restraint_spec

    assert describe_restraint(parse_restraint_spec("A/5-9:B@12", repel=True)) == "A/5-9:B ≥ 12 Å"
