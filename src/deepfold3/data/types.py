import io
from dataclasses import asdict, dataclass

import numpy as np


class NumpySerializable:
    """Serializable to numpy array."""

    def dump(self, file: io.BytesIO) -> np.ndarray:
        """Save to numpy array."""
        np.savez_compressed(file, **asdict(self))

    @classmethod
    def load(cls, file: io.BytesIO) -> "NumpySerializable":
        """Load from numpy array."""
        return cls(**np.load(file, allow_pickle=True))


Atom = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
]


Bond = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]


Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Connection = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray

    @classmethod
    def load(cls, file: io.BytesIO) -> "Structure":
        structure = np.load(file)
        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )
