"""Template feature construction — placement for both builders.

Regression guard for the ``_fill_template_slot(start, length)`` ->
``(dst_pos, src_idx)`` signature change (ab9369f): the single-chain (flat-list)
path was left passing ``(0, L)``, so ``src_idx=L`` read one past the template
array end -> ``IndexError: index L is out of bounds for axis 0 with size L``,
crashing all inference that uses flat-list templates. Training uses the
per-chain builder (also covered here) and was unaffected.
"""

import numpy as np
import pytest

from deepfold3.data import const
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.parse.templates import TemplateFeatures
from deepfold3.data.types import TOKEN_DTYPE
from tests._util import wake_zero_init


def _make_template(length: int) -> TemplateFeatures:
    # aatype[k] = k so placement is directly checkable.
    return TemplateFeatures(
        aatype=np.arange(length, dtype=np.int64),
        atom_positions=np.zeros((length, const.MAX_ATOMS_PER_TOKEN, 3), np.float32),
        atom_mask=np.ones((length, const.MAX_ATOMS_PER_TOKEN), bool),
    )


def _make_tokens(residues: list[int], asym: int = 0) -> np.ndarray:
    toks = np.zeros(len(residues), dtype=TOKEN_DTYPE)
    for i, r in enumerate(residues):
        toks[i]["token_idx"] = i
        toks[i]["asym_id"] = asym
        toks[i]["residue_index"] = r + 1  # residue_index is 1-based
        toks[i]["mol_type"] = const.MOL_PROTEIN
    return toks


# --- single-chain (flat-list) path — the inference path that regressed ---


@pytest.mark.parametrize("n", [1, 12, 146])
def test_single_chain_no_index_error_and_placement(n):
    """Row k lands at query position k for a full-length (L == N) template."""
    out = Featurizer()._build_single_chain_template_features([_make_template(n)], n)
    aatype = out["template_aatype"]
    assert aatype.shape == (1, n)
    assert aatype[0].tolist() == list(range(n))


def test_single_chain_truncates_to_n():
    """A template longer than N is truncated to its first N rows (no overflow)."""
    out = Featurizer()._build_single_chain_template_features([_make_template(200)], 146)
    assert out["template_aatype"][0].tolist() == list(range(146))


def test_single_chain_expanded_residue_gathers_by_query_index():
    """A modified residue tokenizes per atom; its 4 atom tokens must all read
    the residue's own template row, and the residues after it must not shift."""
    cols = np.array([0, 1, 2, 2, 2, 2, 3, 4])  # residue 2 expanded to 4 atom tokens
    out = Featurizer()._build_single_chain_template_features([_make_template(5)], len(cols), cols)
    assert out["template_aatype"][0].tolist() == [0, 1, 2, 2, 2, 2, 3, 4]


def test_single_chain_ligand_tokens_take_no_template():
    """Ligand tokens (-1) are outside the query space: masked, never gathered."""
    cols = np.array([0, 1, 2, -1, -1])
    out = Featurizer()._build_single_chain_template_features([_make_template(3)], len(cols), cols)
    assert out["template_aatype"][0].tolist()[:3] == [0, 1, 2]
    assert out["template_aatype"][0].tolist()[3:] == [const.MSA_GAP] * 2
    assert out["template_pseudo_beta_mask"][0].tolist() == [True, True, True, False, False]


# --- per-chain (crop-aware) path — the training path ---


def test_per_chain_contiguous_crop():
    kept = [4, 5, 6, 7]
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist() == kept


def test_per_chain_scattered_crop():
    kept = [2, 9, 4, 11]  # non-contiguous (spatial/interface crop order)
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist() == kept


def test_per_chain_no_crop_positional_fallback():
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(list(range(12))), 12, template_col_idx=None
    )
    assert out["template_aatype"][0].tolist() == list(range(12))


def test_inference_templates_stay_on_their_own_chain():
    """A chain's templates must land on that chain, through the inference entry
    point (tokenize -> query index -> featurize).

    ``from_job`` used to collapse a lone templated chain to the flat builder,
    which is whole-query indexed and put chain B's rows on chain A; it now
    routes every per-chain case here.
    """
    from deepfold3.data.module.dataset import _build_complex_structure, _tokenize_and_featurize
    from deepfold3.data.types import Input

    seq_a, seq_b = "AAAAAAAA", "CCCCCC"
    structure = _build_complex_structure([(seq_a, const.MOL_PROTEIN, "A"), (seq_b, const.MOL_PROTEIN, "B")])
    inp = Input(structure=structure, paired_msa={}, unpaired_msa={})
    feats = _tokenize_and_featurize(inp, template_feats_per_chain={1: [_make_template(len(seq_b))]})

    aatype = feats["template_aatype"][0].tolist()
    assert aatype[: len(seq_a)] == [const.MSA_GAP] * len(seq_a)  # untemplated chain
    assert aatype[len(seq_a) :] == list(range(len(seq_b)))  # chain B's own rows


def test_per_chain_expanded_residue_repeats_row():
    """Atomized / modified residues share one query column, so their atom tokens
    repeat that residue's template row — AF3's ``standard_token_idxs`` gather."""
    cols = [4, 5, 5, 5, 6]  # residue 5 atomized into 3 atom tokens
    tmpl = _make_template(12)
    tmpl.atom_positions[:, :, 0] = np.arange(12)[:, None]  # CB.x == query index
    out = Featurizer()._build_per_chain_template_features(
        {0: [tmpl]}, _make_tokens(cols), len(cols), template_col_idx=np.array(cols)
    )
    assert out["template_aatype"][0].tolist() == cols
    assert out["template_pseudo_beta"][0, :, 0].tolist() == [float(c) for c in cols]


def test_per_chain_out_of_range_dropped():
    kept = [10, 11, 99]  # 99 is beyond the template length -> dropped, masked
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist()[:2] == [10, 11]
    assert out["template_pseudo_beta_mask"][0].tolist() == [True, True, False]


def _build_templates(n_templates: int, *, pad: bool, n_tokens: int = 12):
    kept = list(range(n_tokens))
    per_chain = {0: [_make_template(n_tokens) for _ in range(n_templates)]}
    return Featurizer(max_templates=4, pad_templates=pad)._build_template_features(
        None, n_tokens, _make_tokens(kept), per_chain, template_col_idx=np.array(kept)
    )


@pytest.mark.parametrize("n_templates", [0, 1, 3, 4])
def test_template_mask_always_emitted(n_templates):
    """The mask is part of the contract, padded or not.

    A key present in only some samples is zero-filled by ``collate_fn`` —
    for a mask that reads as "every slot is pad", silently erasing that
    sample's templates. Emitting it unconditionally makes that impossible.
    """
    out = _build_templates(n_templates, pad=False)
    assert out["template_mask"].tolist() == [1.0] * n_templates
    assert out["template_aatype"].shape[0] == n_templates  # unpadded: T is the real count


@pytest.mark.parametrize("n_templates", [0, 1, 3, 4])
def test_pad_templates_fixes_axis_and_marks_real_slots(n_templates):
    """``pad_templates`` yields a constant T with a real-vs-pad mask."""
    out = _build_templates(n_templates, pad=True)
    for key, val in out.items():
        if key != "template_mask":
            assert val.shape[0] == 4, f"{key} not padded for n={n_templates}"
    assert out["template_mask"].tolist() == [1.0] * n_templates + [0.0] * (4 - n_templates)
    # Pad slots must read as empty, not as spurious structure.
    assert not out["template_pseudo_beta_mask"][n_templates:].any()
    assert not out["template_backbone_mask"][n_templates:].any()


def _template_embedding(seed: int = 0):
    """A minimal double-precision TemplateEmbedding for exactness checks."""
    import torch

    from deepfold3.model.config import OptimConfig, TemplateEmbeddingConfig
    from deepfold3.model.template_embedding import TemplateEmbedding

    torch.manual_seed(seed)
    cfg = TemplateEmbeddingConfig()
    cfg.num_blocks = 1
    return TemplateEmbedding(cfg, optim=OptimConfig()).eval().double()


def _template_slots(n_slots: int, n_tokens: int, gen):
    """``n_slots`` populated template slots in the embedding's input layout."""
    import torch

    return {
        "template_aatype": torch.randint(0, 21, (n_slots, n_tokens), generator=gen),
        "template_pseudo_beta": torch.randn(n_slots, n_tokens, 3, generator=gen, dtype=torch.float64),
        "template_pseudo_beta_mask": torch.ones(n_slots, n_tokens, dtype=torch.float64),
        "template_backbone_frame": torch.eye(3, dtype=torch.float64).expand(n_slots, n_tokens, 3, 3).contiguous(),
        "template_backbone_mask": torch.ones(n_slots, n_tokens, dtype=torch.float64),
        "template_ca": torch.randn(n_slots, n_tokens, 3, generator=gen, dtype=torch.float64),
    }


def test_pad_templates_matches_unpadded_output():
    """Padding must not change the template embedding — bit-exact."""
    import torch

    # Wake the zero-init output projections: without this both the padded and
    # the unpadded embedding are exactly zero and the comparison is vacuous.
    mod = wake_zero_init(_template_embedding())
    n_tokens, n_real, t_max = 12, 2, 4
    gen = torch.Generator().manual_seed(1)

    def slots(t):
        return _template_slots(t, n_tokens, gen)

    real = slots(n_real)
    c_pair = mod.single_template_emb.query_norm.normalized_shape[0]
    query_pair = torch.randn(n_tokens, n_tokens, c_pair, dtype=torch.float64)
    pair_mask = torch.ones(n_tokens, n_tokens, dtype=torch.float64)

    # Pad slots carry deliberate garbage: only ``template_mask`` may suppress them.
    pad = t_max - n_real
    junk = slots(pad)
    junk["template_pseudo_beta_mask"] = torch.zeros(pad, n_tokens, dtype=torch.float64)
    junk["template_backbone_mask"] = torch.zeros(pad, n_tokens, dtype=torch.float64)
    padded = {k: torch.cat([real[k], junk[k]]) for k in real}

    with torch.no_grad():
        unpadded_out = mod(query_pair=query_pair, pair_mask=pair_mask, **real)
        padded_out = mod(
            query_pair=query_pair,
            pair_mask=pair_mask,
            template_mask=torch.tensor([1.0] * n_real + [0.0] * pad, dtype=torch.float64),
            **padded,
        )
        # A fully padded sample must equal the legacy T=0 path.
        empty_out = mod(query_pair=query_pair, pair_mask=pair_mask, **slots(0))
        allpad_out = mod(
            query_pair=query_pair,
            pair_mask=pair_mask,
            template_mask=torch.zeros(t_max, dtype=torch.float64),
            **padded,
        )

    assert torch.equal(unpadded_out, padded_out)
    assert torch.equal(empty_out, allpad_out)


def test_pad_templates_reaches_every_dataset_class():
    """Every dataset class must honour the flag, or T still varies by rank.

    of3 and df3 samples share one DataLoader, so padding only one of them
    leaves the template axis variable across ranks and the forward-straggler
    this flag exists to remove survives. Correctness does not depend on the
    two agreeing — ``template_mask`` is emitted either way.
    """
    import inspect

    from deepfold3.data.module.dataset import TrainingDataset
    from deepfold3.data.module.of3_dataset import OF3Dataset
    from deepfold3.training.data import build_dataloaders  # noqa: F401 — import guard

    for cls in (TrainingDataset, OF3Dataset):
        assert "pad_templates" in inspect.signature(cls.__init__).parameters, cls.__name__

    # OF3Dataset is fed through an explicit allow-list, not **kwargs.
    src = inspect.getsource(inspect.getmodule(build_dataloaders))
    assert '"pad_templates",' in src, "pad_templates missing from of3_kwarg_names"


def test_collate_t_padding_does_not_dilute_templates():
    """A sample's template average must not depend on its batch-mates.

    ``collate_fn`` pads every feature's template axis up to the batch
    maximum, so a sample that drew 2 templates alongside one that drew 4
    gains 2 empty slots. Dividing by that padded ``T`` diluted the smaller
    sample's templates by real/padded; the per-sample ``template_mask.sum()``
    denominator is what removes the dependence.
    """
    import torch

    mod = _template_embedding()
    n_tokens, n_real = 8, 2
    gen = torch.Generator().manual_seed(3)
    # One fixed set of real slots, reused verbatim in both containers.
    content = _template_slots(n_real, n_tokens, gen)
    # What collate_fn produces when a batch-mate drew 4 templates: zero-pad to T=4.
    collated = {
        key: torch.cat([val, torch.zeros((2, *val.shape[1:]), dtype=val.dtype)]) for key, val in content.items()
    }

    c_pair = mod.single_template_emb.query_norm.normalized_shape[0]
    query_pair = torch.randn(n_tokens, n_tokens, c_pair, dtype=torch.float64)
    pair_mask = torch.ones(n_tokens, n_tokens, dtype=torch.float64)

    with torch.no_grad():
        alone = mod(query_pair=query_pair, pair_mask=pair_mask, template_mask=torch.ones(n_real), **content)
        # collate zero-pads the mask alongside every other feature.
        with_mate = mod(
            query_pair=query_pair,
            pair_mask=pair_mask,
            template_mask=torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64),
            **collated,
        )
        diluted = mod(query_pair=query_pair, pair_mask=pair_mask, **collated)  # no mask: old behaviour

    assert torch.equal(alone, with_mate), "template output depends on batch-mates' template count"
    # Guard that the test is actually exercising the failure it claims to.
    assert not torch.equal(alone, diluted), "maskless path no longer dilutes — test is vacuous"


class TestTemplateIndexSpaceResolution:
    """Which coordinate frame the loader reads template positions in.

    Structure-space and SEQRES-space differ by the template's unresolved
    N-terminal count, which varies per template. Reading one as the other
    misaligns every hit and raises nothing — the templates simply become noise
    — so the frame is recorded in the index file and the per-dataset config
    flag is only a fallback for files written before that field existed.
    """

    @staticmethod
    def _resolver(config_flag: bool):
        from deepfold3.data.module.dataset import TrainingDataset

        ds = TrainingDataset.__new__(TrainingDataset)
        ds.template_struct_space = config_flag
        ds._warned_index_space = False
        ds.name = "unit"
        ds.data_dir = "/unit"
        return ds

    def test_recorded_value_wins_over_config(self):
        # Config says SEQRES, file says structure -> file wins.
        assert self._resolver(config_flag=False)._indices_are_struct("struct") is True
        # And the other way round.
        assert self._resolver(config_flag=True)._indices_are_struct("seqres") is False

    def test_legacy_index_falls_back_to_config(self):
        assert self._resolver(config_flag=True)._indices_are_struct(None) is True
        assert self._resolver(config_flag=False)._indices_are_struct(None) is False

    def test_fallback_warns_once_per_dataset(self, caplog):
        import logging

        ds = self._resolver(config_flag=True)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                ds._indices_are_struct(None)
        warnings = [r for r in caplog.records if "index_space" in r.getMessage()]
        assert len(warnings) == 1, "legacy fallback must warn, but only once per dataset"

    def test_recorded_value_never_warns(self, caplog):
        import logging

        ds = self._resolver(config_flag=True)
        with caplog.at_level(logging.WARNING):
            ds._indices_are_struct("struct")
            ds._indices_are_struct("seqres")
        assert not [r for r in caplog.records if "index_space" in r.getMessage()]


def test_template_pairformer_checkpoint_matches_and_recomputes_per_block():
    """Per-block checkpointing changes memory, not outputs or gradients."""
    import copy

    import torch

    plain = wake_zero_init(_template_embedding()).train()
    ckpt = copy.deepcopy(plain)
    ckpt.single_template_emb._grad_checkpoint = True
    n_tokens = 8
    c_pair = plain.single_template_emb.query_norm.normalized_shape[0]

    def run(mod):
        gen = torch.Generator().manual_seed(3)
        slots = _template_slots(2, n_tokens, gen)
        query_pair = torch.randn(n_tokens, n_tokens, c_pair, dtype=torch.float64, generator=gen, requires_grad=True)
        calls = []
        for layer in mod.single_template_emb.pairformer:
            fwd = layer.forward

            def counted(*a, _fwd=fwd, **k):
                calls.append(1)
                return _fwd(*a, **k)

            layer.forward = counted
        torch.manual_seed(4)  # same dropout masks in both runs
        out = mod(query_pair=query_pair, pair_mask=torch.ones(n_tokens, n_tokens, dtype=torch.float64), **slots)
        out.pow(2).mean().backward()
        return out, [p.grad for p in mod.parameters() if p.grad is not None] + [query_pair.grad], len(calls)

    out_a, grads_a, calls_a = run(plain)
    out_b, grads_b, calls_b = run(ckpt)
    torch.testing.assert_close(out_b, out_a)
    for a, b in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(b, a)
    blocks = len(plain.single_template_emb.pairformer) * 2  # two templates
    assert (calls_a, calls_b) == (blocks, 2 * blocks)
