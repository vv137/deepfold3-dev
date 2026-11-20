"""bench/scripts/eval_ost.py: DockQ is taken from the interface named by the reference chains."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "scripts"))
import eval_ost as E  # noqa: E402


def test_interface_matches_reference_chains_only():
    # 7quv-like: ref A-B is mapped to model A-C, ref A-C to model A-E.
    d = {
        "dockq_interfaces": [["A", "B", "A", "C"], ["A", "C", "A", "E"]],
        "dockq": [0.9, 0.1],
        "irmsd": [1.0, 9.0],
        "lrmsd": [2.0, 20.0],
    }
    assert E.parse_interface(d, "A", "C")["dockq"] == 0.1
    assert E.parse_interface(d, "C", "A")["irmsd"] == 9.0
    assert E.parse_interface(d, "A", "B")["lrmsd"] == 2.0
    assert E.parse_interface(d, "B", "C")["dockq"] is None
