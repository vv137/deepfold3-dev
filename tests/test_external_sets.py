"""bench/scripts/external_sets.py: chain-name resolution and ground-truth trimming."""

import sys
from pathlib import Path

import gemmi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "scripts"))
import external_sets as X  # noqa: E402

# label_asym_id -> auth_asym_id, shaped like the RCSB 9qrv assembly: two copies
# of each chain, the copy placed by operator 2 suffixed "-2".
CHAINS = {"A": "D", "A-2": "D-2", "B": "H", "B-2": "H-2", "C": "L", "C-2": "L-2"}


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("A", "A"),  # a label in the file
        ("B.1", "B-2"),  # PXMeter symmetry copy k -> RCSB copy k+1
        ("L", "C"),  # neither a label nor X.k: the one label with that author chain
    ],
)
def test_resolve_label(name, label):
    assert X.resolve_label(name, CHAINS) == label


def test_resolve_label_only_copy():
    # RCSB names a chain placed by a non-identity operator X-k even when alone.
    assert X.resolve_label("E", {"A": "A", "E-3": "E-3"}) == "E-3"


def test_resolve_label_unknown_raises():
    with pytest.raises(KeyError):
        X.resolve_label("Z", CHAINS)


def _toy_cif(path: Path) -> None:
    st = gemmi.Structure()
    st.setup_entities()
    model = gemmi.Model("1")
    for label, auth in (("A", "D"), ("B", "D"), ("C", "H")):
        ch = gemmi.Chain(auth)
        res = gemmi.Residue()
        res.name, res.seqid, res.subchain = "GLY", gemmi.SeqId(1, " "), label
        res.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name, atom.element = "CA", gemmi.Element("C")
        res.add_atom(atom)
        ch.add_residue(res)
        model.add_chain(ch)
    st.add_model(model)
    st.setup_entities()
    doc = st.make_mmcif_document()
    doc.sole_block().set_pair("_test.marker", "kept")
    doc.write_file(str(path))


def test_trim_cif_keeps_only_listed_labels(tmp_path):
    src, dst = tmp_path / "gt.cif", tmp_path / "trim.cif"
    _toy_cif(src)
    assert X.gt_chains(src) == {"A": "D", "B": "D", "C": "H"}
    # A and B share author chain D, so trimming must go by label_asym_id.
    assert X.trim_cif(src, dst, {"A", "C"}) == 2
    assert X.gt_chains(dst) == {"A": "D", "C": "H"}
    # categories other than _atom_site are copied unchanged
    assert gemmi.cif.read(str(dst)).sole_block().find_value("_test.marker") == "kept"
