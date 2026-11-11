"""``OptimConfig.pair_chunk_budget_bytes``: exact, inference only, and the whole path when it fits."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import OptimConfig
from deepfold3.model.primitives import budget_rows, map_pair_rows, map_row_chunks

N = 10


def _randomize(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    # Most output projections are zero-initialized; perturb them so a chunking
    # mistake cannot hide behind an all-zero branch.
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return module.eval()


def _row_counts(module: torch.nn.Module) -> list[int]:
    """Rows (dim -3 of the first input) of every call ``module`` receives."""
    seen: list[int] = []
    module.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[-3]))
    return seen


def test_budget_rows():
    assert budget_rows(0, 2560, 4096) == 0  # off
    assert budget_rows(4 << 30, 1024, 4096) == 0  # the whole tensor fits
    assert budget_rows(4 << 30, 2560, 4096) == 256  # 409 rows, down to a power of two
    assert budget_rows(4 << 30, 2560, 8192) == 128
    assert budget_rows(1, 2560, 4096) == 1  # never 0 once chunking is needed


def test_map_row_chunks_slices_every_input():
    x, y = torch.randn(2, 7, 5, 3), torch.randn(2, 7, 5, 4)
    with torch.no_grad():
        out = map_row_chunks(lambda a, b: torch.cat([a, b], -1), x, 3, y)
    assert torch.equal(out, torch.cat([x, y], -1))


def test_map_pair_rows_runs_whole_with_grad_or_when_it_fits():
    x = torch.randn(N, N, 4)
    calls: list[int] = []

    def fn(r):
        calls.append(r.shape[-3])
        return r * 2

    map_pair_rows(fn, x, 1, 16)  # grad enabled
    with torch.no_grad():
        map_pair_rows(fn, x, N * N * 16, 16)  # fits
        map_pair_rows(fn, x, 0, 16)  # off
        map_pair_rows(fn, x, 4 * N * 16, 16)  # 4 rows
    assert calls == [N, N, N, 4, 4, 4]


def test_presets():
    assert OptimConfig().pair_chunk_budget_bytes == 0
    assert OptimConfig.none().pair_chunk_budget_bytes == 0
    assert OptimConfig.inference().pair_chunk_budget_bytes > 0
    assert OptimConfig.inference_cueq().pair_chunk_budget_bytes == OptimConfig.inference().pair_chunk_budget_bytes


@pytest.mark.parametrize("budget", [1, 3 * N * 4 * 32 * 4])  # 1 row, and 2 rows with a padded tail
def test_pairformer_block_chunked_equals_whole(budget):
    from deepfold3.model.evoformer import PairFormerBlock

    whole = _randomize(PairFormerBlock(c_pair=32, c_single=32, optim=OptimConfig()))
    rows = PairFormerBlock(c_pair=32, c_single=32, optim=OptimConfig(pair_chunk_budget_bytes=budget)).eval()
    rows.load_state_dict(whole.state_dict())
    seen = _row_counts(rows.pair_transition)
    torch.manual_seed(2)
    args = (torch.randn(1, N, N, 32), torch.randn(1, N, 32), torch.ones(1, N, N), torch.ones(1, N))
    with torch.no_grad():
        a, b = whole(*args), rows(*args)
    assert max(seen) < N
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y, atol=1e-5)


def test_grad_enabled_is_not_chunked():
    from deepfold3.model.evoformer import PairFormerBlock

    block = _randomize(PairFormerBlock(c_pair=32, c_single=32, optim=OptimConfig(pair_chunk_budget_bytes=1)))
    seen = _row_counts(block.pair_transition)
    args = (torch.randn(1, N, N, 32), torch.randn(1, N, 32), torch.ones(1, N, N), torch.ones(1, N))
    pair, _ = block(*args)
    pair.sum().backward()
    assert seen == [N]


def test_diffusion_conditioning_chunked_equals_whole():
    from deepfold3.model.diffusion import DiffusionConditioningBlock

    def make(budget):
        return DiffusionConditioningBlock(
            c_pair_in=32, c_single_in=16, rel_feat_dim=7, c_pair_out=32, c_single_out=16, target_feat_dim=8,
            optim=OptimConfig(pair_chunk_budget_bytes=budget),
        )  # fmt: skip

    whole = _randomize(make(0))
    rows = make(1).eval()
    rows.load_state_dict(whole.state_dict())
    seen = _row_counts(rows.pair_trans0)
    torch.manual_seed(3)
    args = (torch.randn(1, N, 16), torch.randn(1, N, N, 32), torch.randn(1, N, 8), torch.randn(1, N, N, 7))
    with torch.no_grad():
        a, b = whole.compute_static(*args), rows.compute_static(*args)
    assert seen == [1] * N
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y, atol=1e-5)


def _model(budget: int):
    from deepfold3.model.model import DeepFold3
    from tests.test_batch import _tiny_config

    cfg = _tiny_config()
    cfg.diffusion.num_samples = 2
    cfg.diffusion.hoist_sampler = True
    cfg.optim = OptimConfig(pair_chunk_budget_bytes=budget)
    return DeepFold3(cfg).eval()


def _features():
    from tests.test_batch import _make_features

    f = _make_features(N)
    T = 2
    torch.manual_seed(5)
    f.update(
        asym_id=(torch.arange(N) >= N // 2).long(),
        template_aatype=torch.randint(0, 20, (T, N)),
        template_pseudo_beta=10 * torch.randn(T, N, 3),
        template_pseudo_beta_mask=torch.ones(T, N),
        template_backbone_frame=torch.eye(3).expand(T, N, 3, 3).clone(),
        template_backbone_mask=torch.ones(T, N),
        template_ca=10 * torch.randn(T, N, 3),
        template_mask=torch.ones(T),
    )
    return f


def _budgeted_sites(model) -> dict[str, torch.nn.Module]:
    evo, diff = model.evoformer, model.diffusion_head
    return {
        "prev_embedding": evo.prev_embedding,
        "template embed_query": evo.template_embedding.single_template_emb.embed_query,
        "template pair_transition": evo.template_embedding.single_template_emb.pairformer[0].pair_transition,
        "msa pair_logits": evo.msa_stack[0].row_attn.pair_logits,
        "msa pair_transition": evo.msa_stack[0].pair_transition,
        "trunk pair_transition": evo.trunk_pairformer[0].pair_transition,
        "trunk single_to_pair": evo.trunk_pairformer[0].single_to_pair,
        "diffusion pair_cond_proj": diff.conditioning.pair_cond_proj,
        "diffusion pair_trans1": diff.conditioning.pair_trans1,
        "diffusion pair_logits_proj": diff.transformer.pair_logits_proj[0],
        "atom encoder embed_trunk_pair": diff.atom_encoder.embed_trunk_pair,
        "confidence pair_transition": model.confidence_head.layers[0].pair_transition,
        "confidence pae_logits": model.confidence_head.pae_logits,
        "confidence pde_half_logits": model.confidence_head.pde_half_logits,
        "distogram half_logits": model.distogram_head.half_logits,
    }


def test_model_chunked_equals_whole():
    """Every budgeted site chunks under a tiny budget and none does under a large one; outputs agree."""
    whole = _randomize(_model(1 << 40))
    rows = _model(1)
    rows.load_state_dict(whole.state_dict())
    seen_whole = {k: _row_counts(m) for k, m in _budgeted_sites(whole).items()}
    seen_rows = {k: _row_counts(m) for k, m in _budgeted_sites(rows).items()}
    feats = _features()
    with torch.no_grad():
        a = whole(feats, seed=7, num_recycles=1)
        b = rows(feats, seed=7, num_recycles=1)
    for site in seen_whole:
        # Whole: one full-size call per forward of the site. ``nn.Linear`` sees the
        # [N, N, C] tensor too, so its dim -3 is N.
        assert seen_whole[site] and set(seen_whole[site]) == {N}, site
        assert seen_rows[site] and set(seen_rows[site]) == {1}, site
    for name in ("atom_positions", "plddt", "pae", "pde", "distogram"):
        x, y = getattr(a, name), getattr(b, name)
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-4), (name, (x - y).abs().max())


def test_large_budget_is_bitwise_the_unbudgeted_model():
    off = _randomize(_model(0))
    big = _model(1 << 40)
    big.load_state_dict(off.state_dict())
    feats = _features()
    with torch.no_grad():
        a = off(feats, seed=7, num_recycles=1)
        b = big(feats, seed=7, num_recycles=1)
    for name in ("atom_positions", "plddt", "pae", "pde", "distogram"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
