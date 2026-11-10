"""Featurizer: Tokenized → dict[str, Tensor].

Produces all features needed by the DeepFold3 model forward pass.
Feature specifications follow AlphaFold3 exactly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.feature.ref_atoms import _encode_atom_name, get_ref_atoms
from deepfold3.data.msa.pairing import unpaired_rows_not_in_paired
from deepfold3.data.types import MSA, Tokenized

if TYPE_CHECKING:
    from deepfold3.data.parse.templates import TemplateFeatures

logger = logging.getLogger(__name__)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random 3×3 rotation matrix using QR decomposition."""
    H = rng.standard_normal((3, 3)).astype(np.float64)
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det=+1)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


class Featurizer:
    """Convert Tokenized data into model-ready tensors."""

    def __init__(
        self,
        max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        max_templates: int = 4,
        ligand_perm_cache: dict[str, list[list[int]]] | None = None,
        include_polymer_backbone_bonds: bool = False,
        msa_crop_size: int = 1024,
        n_msa_crops: int = 0,
        ccd_atom_order: dict[str, list[str]] | None = None,
        pad_templates: bool = False,
        dedup_unpaired_against_paired: bool = False,
    ):
        self.max_atoms = max_atoms
        # Drop each chain's unpaired rows that already appear (residue-identical)
        # among its paired rows, as OF3 ``create_main`` / AF3 do after online
        # species pairing. Off by default: precomputed paired MSAs (ColabFold
        # ``pairaln``) were never deduplicated, and training does not do it.
        # ``profile`` / ``deletion_mean`` still use the full unpaired MSA (OF3
        # computes them from the redundant main MSA).
        self.dedup_unpaired_against_paired = dedup_unpaired_against_paired
        self.max_msa_seqs = max_msa_seqs
        self.max_templates = max_templates
        # Emit a fixed ``T == max_templates`` template axis instead of the
        # sampled count, so per-step template compute is constant across DDP
        # ranks. Model output is unchanged. Note ``template_mask`` is emitted
        # either way — see ``_finalize_template_axis``, which explains why the
        # mask (not this flag) is what keeps the average correct.
        # Rationale for the flag itself in ``DataConfig.pad_templates``.
        self.pad_templates = pad_templates
        # MSA pre-crop (training only). When ``n_msa_crops > 0`` the MSA output
        # is a FIXED [n_msa_crops * msa_crop_size, N] stack of pre-sampled crops
        # (CPU resample, mask-aware), one per recycle, so the GPU sees a
        # constant MSA shape every step (no per-step variable allocation /
        # caching-allocator fragmentation) and the model just splits it per
        # recycle instead of resampling. ``profile`` / ``deletion_mean`` are
        # still computed from the FULL pool before cropping. 0 = legacy
        # full-pool output (inference path; model resamples on GPU).
        self.msa_crop_size = msa_crop_size
        self.n_msa_crops = n_msa_crops
        # Optional CCD-keyed ligand automorphism cache (loaded once per
        # process). When set, the featurizer emits ``ligand_swap_pairs``
        # alongside ``atom_swap_pairs`` so the diffusion loss can apply
        # ligand-atom permutations as well.
        self.ligand_perm_cache = ligand_perm_cache
        # Order-preserving CCD ``atoms`` list per code (same index space as the
        # automorphism perms). Lets ``compute_ligand_token_permutations`` place
        # each token by atom name → CCD index, so ligand / modified-residue
        # symmetry resolves regardless of the structure's atom ordering.
        self.ccd_atom_order = ccd_atom_order
        # Bond-loss scope. AF3 §5.3 / OpenFold-3 restrict ``L_bond`` to
        # polymer–ligand token bonds; the Protenix-style extended set (all
        # polymer backbone bonds + ligand intra-bonds) is preserved behind
        # this flag for experimentation but is off by default because it
        # double-counts against the diffusion MSE on shared atom pairs.
        self.include_polymer_backbone_bonds = include_polymer_backbone_bonds

    def process(
        self,
        data: Tokenized,
        training: bool = True,
        random_rot: bool = True,
        seed: int | None = None,
        templates: list[TemplateFeatures] | None = None,
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
        template_col_idx: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Processes tokenized data into a feature dictionary.

        Args:
            data: Tokenized input (tokens, bonds, structure, MSA).
            training: Whether to apply training augmentations.
            random_rot: Whether to apply random rotation to ref_pos.
            seed: Random seed for augmentation reproducibility.
            templates: Flat list of TemplateFeatures (single-chain or whole-complex).
            template_feats_per_chain: Per-chain template features keyed by asym_id.
            template_col_idx: Per-token *original* chain-local column index (the
                pre-rebase ``msa_col_idx``), used to place per-chain template rows
                at the correct query residues after cropping. ``None`` (no crop)
                falls back to contiguous positional placement.

        Returns:
            Dict of tensors with keys:
                res_type [N] int64, residue_index [N] int64,
                token_index [N] int64, asym_id [N] int64,
                entity_id [N] int64, sym_id [N] int64,
                seq_mask [N] bool, pair_mask [N,N] bool,
                frames_mask [N] bool, token_to_rep_atom [N] int64,
                frame_atom_index [N,3] int64,
                frame_token_index [N,3] int64,
                ref_pos [N,M,3] float32, ref_mask [N,M] bool,
                ref_element [N,M] int64, ref_charge [N,M] float32,
                ref_atom_name_chars [N,M,4] int64,
                ref_space_uid [N,M] int64,
                msa [S,N] int64, msa_mask [S,N] bool,
                deletion_matrix [S,N] int64, profile [N,31] float32,
                deletion_mean [N] float32,
                bond_matrix [N,N,1] float32 (if ligand bonds exist),
                template_* [T,N,...] (if templates provided).
        """
        rng = np.random.default_rng(seed)
        tokens = data.tokens
        N = len(tokens)

        if N == 0:
            return self._empty_features()

        M = self.max_atoms

        # Single-atom polymer tokens: modified/non-standard residues (always) and
        # stochastically-atomized standard residues (training regularizer). Both
        # carry ``atom_num == 1`` with a polymer mol_type and are routed through
        # the ligand-style single-atom path below (ref_pos / gt / rep atom /
        # frames) instead of the dense-residue path — which fixes the
        # modified-residue ``ref_pos`` bug (a single-atom token used to receive
        # the whole residue's reference template) and lets atomized standard
        # residues keep a valid representative atom (so they stay in the
        # distogram / confidence masks). Atomized tokens of one residue share a
        # reference-space uid so their conformer is centered / rotated together.
        _polymer_mask = np.isin(tokens["mol_type"], (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA))
        is_atomized = _polymer_mask & (tokens["atom_num"] == 1)
        # Per-residue reference-space grouping for EVERY single-atom token —
        # atomized polymer residues AND ligand atoms alike. All atom tokens of
        # one residue share a ref_space_uid so their conformer is centered /
        # rotated as a unit. Keyed by (asym_id, residue_index): now that ligand
        # atom tokens carry residue_index = res_id (as atomized residues always
        # have), the one grouping serves both, replacing the ligand-specific
        # structure lookup. First surviving token of a group is its uid.
        single_atom_group_first: dict[int, int] = {}
        _single_atom = tokens["atom_num"] == 1
        if _single_atom.any():
            _seen_group: dict[tuple[int, int], int] = {}
            for _i in np.where(_single_atom)[0]:
                _key = (int(tokens[_i]["asym_id"]), int(tokens[_i]["residue_index"]))
                single_atom_group_first[int(_i)] = _seen_group.setdefault(_key, int(_i))

        # ------------------------------------------------------------------
        # Basic token features
        # ------------------------------------------------------------------
        res_type = torch.from_numpy(tokens["res_type"].astype(np.int64))
        residue_index = torch.from_numpy(tokens["residue_index"].astype(np.int64))
        token_index = torch.from_numpy(tokens["token_idx"].astype(np.int64))
        asym_id = torch.from_numpy(tokens["asym_id"].astype(np.int64))
        entity_id = torch.from_numpy(tokens["entity_id"].astype(np.int64))
        sym_id = torch.from_numpy(tokens["sym_id"].astype(np.int64))
        sym_token_index = torch.from_numpy(tokens["sym_token_index"].astype(np.int64))
        seq_mask = torch.ones(N, dtype=torch.bool)

        # ------------------------------------------------------------------
        # Reference atom features
        # ------------------------------------------------------------------
        ref_pos = np.zeros((N, M, 3), dtype=np.float32)
        ref_mask = np.zeros((N, M), dtype=bool)
        ref_element = np.zeros((N, M), dtype=np.int64)
        ref_charge = np.zeros((N, M), dtype=np.float32)
        ref_atom_name_chars = np.zeros((N, M, 4), dtype=np.int64)

        # ref_space_uid: groups atoms sharing the same reference space.
        # For polymer: uid = token_idx. For ligand: uid = first token of the ligand residue.
        ref_space_uid = np.zeros((N, M), dtype=np.int64)

        # Maps each stored structure-atom slot j of a token to its CCD-canonical
        # slot, so ref_* and atom_gt_* agree on which physical atom slot j is.
        # Default identity (single-atom tokens, and residues whose stored order
        # already matches CCD); polymer residues with a missing/reordered atom
        # are remapped by name in the dense branch and reused by the GT loop.
        atom_slot_remap = np.tile(np.arange(M, dtype=np.int64), (N, 1))

        # Pre-compute ligand residue → CCD entry mapping for ref_pos lookup
        _ligand_ccd_cache: dict[str, tuple[list[str], np.ndarray]] = {}

        for i, tok in enumerate(tokens):
            mol_type_tok = int(tok["mol_type"])

            if mol_type_tok == const.MOL_NONPOLYMER or is_atomized[i]:
                # Single-atom token (ligand, or modified / atomized polymer
                # residue): look up the CCD residue for this atom's ideal pos.
                atom_idx = int(tok["atom_idx"])
                res_name = _find_res_name(data, atom_idx)
                if atom_idx < len(data.structure.atoms):
                    atom = data.structure.atoms[atom_idx]
                    atom_name = atom["name"].decode("ascii", errors="replace").strip()
                    elem_int = int(np.clip(atom["element"], 0, 127))
                    chg = float(atom["charge"])

                    # Look up this atom's ideal position from the CCD entry
                    if res_name not in _ligand_ccd_cache:
                        from deepfold3.data.feature import ref_atoms as _rfa

                        entry = _rfa._get_ccd_entry(res_name)
                        if entry is not None:
                            positions = entry.positions
                            # AF3 §2.8: prefer ETKDGv3-generated ref_pos when
                            # cached, matching get_ref_atoms so the single-atom
                            # path uses the same conformer source as the dense path.
                            cache = _rfa._REF_POS_CACHE
                            if (
                                cache is not None
                                and res_name in cache
                                and len(cache[res_name]) == len(entry.atom_names)
                            ):
                                positions = cache[res_name]
                            _ligand_ccd_cache[res_name] = (entry.atom_names, positions)
                        else:
                            _ligand_ccd_cache[res_name] = ([], np.zeros((0, 3), dtype=np.float32))

                    ccd_names, ccd_pos = _ligand_ccd_cache[res_name]
                    ref_atom_data = get_ref_atoms(
                        res_name=res_name,
                        mol_type=mol_type_tok,
                        max_atoms=M,
                        atom_names_override=[atom_name],
                    )

                    # Set ideal position from the CCD ref conformer. When the
                    # atom name isn't in the CCD entry (e.g. a custom SMILES
                    # ligand), fall back to the input structure coordinate ONLY
                    # at inference (a user-provided conformer). In training that
                    # coordinate is the ground truth, so using it as the
                    # reference feature would leak the answer; leave the
                    # (origin-centered) default instead.
                    if atom_name in ccd_names:
                        ccd_idx = ccd_names.index(atom_name)
                        ref_atom_data.positions[0] = ccd_pos[ccd_idx]
                    elif not training:
                        ref_atom_data.positions[0] = [
                            float(atom["x"]),
                            float(atom["y"]),
                            float(atom["z"]),
                        ]

                    ref_atom_data.element[0] = elem_int
                    ref_atom_data.charge[0] = chg
                else:
                    ref_atom_data = get_ref_atoms("UNK", mol_type_tok, M)

                # ref_space_uid groups atoms sharing a reference frame: every
                # single-atom token of one residue (atomized polymer OR ligand)
                # shares one uid so its conformer is centered / rotated together.
                ref_space_uid[i, :] = single_atom_group_first.get(i, i)
            else:
                atom_idx = int(tok["atom_idx"])
                atom_num = int(tok["atom_num"])
                res_name = _find_res_name(data, atom_idx)
                ref_atom_data = get_ref_atoms(res_name, mol_type_tok, M)

                # Structure stores present atoms densely (missing atoms simply
                # absent), while ref_atom_data (positions/mask) is in CCD order.
                # Place each stored atom at its CCD slot by name so ref_* and the
                # GT positions line up even when a non-terminal atom is missing
                # (a residue missing a mid-order atom would otherwise shift every
                # following slot). Fall back to positional j when the name isn't
                # in the CCD ref set (no regression for normal residues).
                name_to_slot = {nm: k for k, nm in enumerate(ref_atom_data.atom_names)}
                for j in range(min(atom_num, M)):
                    if atom_idx + j < len(data.structure.atoms):
                        atom = data.structure.atoms[atom_idx + j]
                        if bool(atom["is_present"]):
                            aname = atom["name"].decode("ascii", errors="replace").strip()
                            slot = name_to_slot.get(aname, j)
                            if slot >= M:
                                slot = j
                            atom_slot_remap[i, j] = slot
                            ref_atom_data.element[slot] = int(np.clip(atom["element"], 0, 127))
                            ref_atom_data.charge[slot] = float(atom["charge"])
                            ref_atom_data.atom_name_chars[slot] = _encode_atom_name(aname)

                # Polymer: ref_space_uid = token index
                ref_space_uid[i, :] = i

            ref_pos[i] = ref_atom_data.positions
            ref_mask[i] = ref_atom_data.mask
            ref_element[i] = ref_atom_data.element
            ref_charge[i] = ref_atom_data.charge
            ref_atom_name_chars[i] = ref_atom_data.atom_name_chars

        # Reference conformers are centered per ref_space_uid group for BOTH
        # training and inference (so embed_ref_pos sees the same distribution).
        # During training the centered coordinates additionally get a random
        # rotation + translation as data augmentation (shared within a ligand's
        # ref_space_uid group); at inference we center only.
        # Polymer tokens each get an independent uid; ligand tokens of one
        # residue share a uid and thus the same transform.
        unique_uids = np.unique(ref_space_uid[:, 0])
        for uid in unique_uids:
            uid_mask = ref_space_uid[:, 0] == uid
            uid_indices = np.where(uid_mask)[0]

            # Collect all valid positions across tokens in this group
            all_valid_pos = []
            for idx in uid_indices:
                valid = ref_mask[idx].astype(bool)
                if valid.any():
                    all_valid_pos.append(ref_pos[idx][valid])

            if not all_valid_pos:
                continue

            all_valid = np.concatenate(all_valid_pos, axis=0)
            center = all_valid.mean(axis=0)

            if random_rot:
                rot = _random_rotation_matrix(rng)
                translation = rng.standard_normal(3).astype(np.float32)
                for idx in uid_indices:
                    ref_pos[idx] = (ref_pos[idx] - center) @ rot.T + translation
            else:
                for idx in uid_indices:
                    ref_pos[idx] = ref_pos[idx] - center

        # ------------------------------------------------------------------
        # MSA features
        # ------------------------------------------------------------------
        msa_tensor, msa_mask_tensor, del_matrix_tensor = self._build_msa_features(data, tokens, N, rng)

        # Profile [N, 31] + raw deletion_mean [N], computed PER CHAIN from each
        # chain's own unpaired MSA (AF3 §2.8 / OF3). Deriving them from the
        # assembled complex MSA instead would let a shallow chain's cross-chain
        # gap padding (mask=True, padded to the deepest chain's depth) dilute
        # its profile toward all-gap and its deletion_mean toward 0 — silent
        # corruption for MSA-depth-imbalanced heteromers. Includes the gap class
        # (index 21) to encode coverage, matching AF3.
        profile, deletion_mean = self._compute_msa_profile_deletion_mean(data, tokens, N)

        # MSA pre-crop (training): replace the variable-depth pool with a FIXED
        # [n_msa_crops * msa_crop_size, N] stack of per-recycle crops. profile +
        # deletion_mean above are already computed from the full pool, so the
        # crop only affects what the MSA module consumes (the model splits it
        # per recycle). See ``__init__`` and ``_build_msa_crops``.
        if self.n_msa_crops > 0:
            msa_tensor, msa_mask_tensor, del_matrix_tensor = self._build_msa_crops(
                msa_tensor, msa_mask_tensor, del_matrix_tensor, training=training
            )

        # Frame features drive PAE (AF3 §4.3.2). ``frames_mask`` is set
        # True only for tokens with a well-defined local rigid frame.
        # ``frame_atom_index [N, 3]`` indexes atoms within a referenced
        # token's dense slot; ``frame_token_index [N, 3]`` selects which
        # token each frame atom comes from:
        #   * polymer (protein/RNA/DNA) → (i, i, i): all three backbone
        #     frame atoms are in the residue's own token.
        #   * ligand / ion / glycan (single-atom token) → (a_tok, i, c_tok)
        #     with a_tok / c_tok the nearest / 2nd-nearest atoms from other
        #     tokens of the same chain. Invalid if the chain has fewer
        #     than 3 atoms or the three atoms fall within 25° of colinear.
        from deepfold3.data.feature.ref_atoms import decode_atom_name_chars

        token_to_rep_atom = np.zeros(N, dtype=np.int64)
        # Token *center* atom (CA for protein, C1' for nucleic, slot-0 for
        # ligand/atomized) — distinct from ``token_to_rep_atom`` (the
        # distogram/PAE pseudo-beta = CB / C4 / C2). AF3 §4.3.1 restricts the
        # pLDDT neighbour set m∈R to these center atoms, NOT the rep atoms.
        token_to_center_atom = np.zeros(N, dtype=np.int64)
        rep_atom_present = np.zeros(N, dtype=bool)
        frames_mask_np = np.zeros(N, dtype=bool)
        frame_atom_index = np.zeros((N, 3), dtype=np.int64)
        frame_token_index = np.tile(np.arange(N, dtype=np.int64)[:, None], (1, 3))

        # Pre-collect one-atom-token positions and per-chain membership for the
        # §4.3.2 nearest-neighbour frame lookup (O(n_chain) per token). Covers
        # ligand / ion / glycan atoms AND atomized polymer atoms (modified
        # residues, stochastic atomization) — both store their single atom at
        # slot 0 and need their frame built from neighbouring atoms.
        single_xyz = np.zeros((N, 3), dtype=np.float32)
        single_present = np.zeros(N, dtype=bool)
        single_by_chain: dict[int, list[int]] = {}
        for i, tok in enumerate(tokens):
            if int(tok["atom_num"]) != 1 and int(tok["mol_type"]) != const.MOL_NONPOLYMER:
                continue
            single_by_chain.setdefault(int(tok["asym_id"]), []).append(i)
            atom_idx = int(tok["atom_idx"])
            if atom_idx < len(data.structure.atoms):
                atom = data.structure.atoms[atom_idx]
                if bool(atom["is_present"]):
                    single_xyz[i] = (float(atom["x"]), float(atom["y"]), float(atom["z"]))
                    single_present[i] = True

        _COS_25_DEG = 0.9063  # cos(25°) — colinearity cutoff (AF3 §4.3.2)

        def _set_single_atom_frame(i: int, asym_id: int) -> None:
            """AF3 §4.3.2 frame for a one-atom token: ``(nearest, self, 2nd nearest)``
            among the one-atom tokens of the same chain.

            No-op when the chain holds fewer than three such atoms or the three
            fall within 25° of colinear, leaving ``frames_mask`` False.
            """
            neighbours = [j for j in single_by_chain.get(asym_id, []) if j != i and single_present[j]]
            if len(neighbours) < 2:
                return  # covers single-atom chains (e.g. Na+, Cl-)
            b_xyz = single_xyz[i]
            dists = np.linalg.norm(single_xyz[neighbours] - b_xyz, axis=-1)
            order = np.argsort(dists)
            a_tok = neighbours[int(order[0])]
            c_tok = neighbours[int(order[1])]
            v_a = single_xyz[a_tok] - b_xyz
            v_c = single_xyz[c_tok] - b_xyz
            denom = float(np.linalg.norm(v_a) * np.linalg.norm(v_c))
            if denom < 1e-6:
                return  # coincident atoms → undefined angle
            if abs(float(np.dot(v_a, v_c)) / denom) > _COS_25_DEG:
                return  # within 25° of colinear
            frame_token_index[i] = (a_tok, i, c_tok)
            # frame_atom_index[i] stays (0, 0, 0) — one-atom tokens store their
            # single atom at slot 0.
            frames_mask_np[i] = True

        for i, tok in enumerate(tokens):
            mt = int(tok["mol_type"])
            rt = int(tok["res_type"])
            if is_atomized[i]:
                # Atomized polymer atom (modified residue, bonded glycan, or a
                # stochastically atomized standard residue): its single atom
                # (slot 0) is both the representative and the token center, so
                # the token stays in the distogram / confidence masks. Its local
                # frame comes from the same nearest-neighbour rule as ligand
                # atoms (AF3 §4.3.2); the nearest one-atom tokens are normally
                # the sibling atoms of its own residue, so the frame recovers
                # the residue-local frame the whole-residue token would have had
                # — keeping the token in the PAE frame set and under FAPE.
                token_to_rep_atom[i] = 0
                rep_atom_present[i] = bool(ref_mask[i, 0])
                if single_present[i]:
                    _set_single_atom_frame(i, int(tok["asym_id"]))
                continue
            if mt == const.MOL_PROTEIN:
                token_to_center_atom[i] = 1  # CA is the pLDDT-neighbour center atom
                if rt == 7:  # GLY
                    token_to_rep_atom[i] = 1  # CA
                    rep_atom_present[i] = bool(ref_mask[i, 1])
                elif ref_mask[i, 4]:
                    token_to_rep_atom[i] = 4  # CB
                    rep_atom_present[i] = True
                elif ref_mask[i, 1]:
                    token_to_rep_atom[i] = 1  # CA fallback
                    rep_atom_present[i] = True
                # else: both CB and CA missing — rep_atom_present stays False
                frame_atom_index[i] = (0, 1, 2)
                if ref_mask[i, 0] and ref_mask[i, 1] and ref_mask[i, 2]:
                    frames_mask_np[i] = True
            elif mt in (const.MOL_RNA, const.MOL_DNA):
                # Representative (pseudo-beta) atom for the distogram + confidence
                # rep-atom geometry is the base atom C4 (purine A/G/DA/DG) or C2
                # (pyrimidine C/U/DC/DT), per AF3 §4.4 (``RES_DISTO_ATOM``) —
                # NOT the ribose C1'. C1' remains the token *center* / frame
                # atom. Unknown nucleic (N/DN) and missing-base cases fall back
                # to C1'.
                disto_name = const.RES_DISTO_ATOM.get(const.RES_TYPE_INDEX_TO_NAME.get(rt, ""), "C1'")
                rep_idx = c1_idx = c3_idx = c4_idx = -1
                for j in range(M):
                    if not ref_mask[i, j]:
                        continue
                    name = decode_atom_name_chars(ref_atom_name_chars[i, j])
                    if name == disto_name:
                        rep_idx = j
                    if name == "C1'":
                        c1_idx = j
                    elif name == "C3'":
                        c3_idx = j
                    elif name == "C4'":
                        c4_idx = j
                if rep_idx < 0:
                    rep_idx = c1_idx  # base atom missing → ribose C1'
                if rep_idx >= 0:
                    token_to_rep_atom[i] = rep_idx
                    rep_atom_present[i] = True
                # pLDDT-neighbour center atom is the ribose C1' (AF3 §4.3.1);
                # fall back to the rep atom if C1' is unresolved.
                token_to_center_atom[i] = c1_idx if c1_idx >= 0 else token_to_rep_atom[i]
                if c1_idx >= 0 and c3_idx >= 0 and c4_idx >= 0:
                    frame_atom_index[i] = (c1_idx, c3_idx, c4_idx)
                    frames_mask_np[i] = True
            elif mt == const.MOL_NONPOLYMER:
                if not single_present[i]:
                    continue
                rep_atom_present[i] = True  # ligand atom at slot 0 (verified present above)
                _set_single_atom_frame(i, int(tok["asym_id"]))

        # Mask tokens without a valid rep atom so distogram / confidence
        # targets don't read garbage coordinates at ``gt_pos[i, rep_idx]``.
        seq_mask = seq_mask & torch.from_numpy(rep_atom_present)
        pair_mask = seq_mask.unsqueeze(1) & seq_mask.unsqueeze(0)  # [N, N]

        frames_mask = torch.from_numpy(frames_mask_np)

        # ------------------------------------------------------------------
        # FAPE atom set — K=5 atoms per token projected into every frame.
        # Slots 0..2 = frame atoms (a, b, c); slot 3 = rep atom; slot 4 =
        # backbone carbonyl O (protein only). AF2 §4.2.7 backbone FAPE
        # projects (N, CA, C, O); we mirror that and add the rep (CB for
        # protein, C1' for nucleic) on top to retain the local-
        # stereocentre chirality channel the rep atom already carried.
        # Slot 4 supervises C=O direction (i.e. ψ commit) — the
        # defining geometry for β-strand H-bond ladders that was
        # missing in the K=4 layout. For tokens where the rep
        # duplicates one of the frame atoms (GLY: rep=CA=frame[1];
        # nucleic: rep=C1'=frame[0]; ligand: rep=self-slot-0=
        # frame[1]'s atom), slot 3 is masked out to avoid double-
        # counting. Slot 4 is masked for non-protein tokens.
        # ------------------------------------------------------------------
        K_FAPE = 5
        fape_atom_index_np = np.zeros((N, K_FAPE), dtype=np.int64)
        fape_token_index_np = np.zeros((N, K_FAPE), dtype=np.int64)
        fape_atom_mask_np = np.zeros((N, K_FAPE), dtype=bool)
        fape_atom_index_np[:, 0:3] = frame_atom_index
        fape_token_index_np[:, 0:3] = frame_token_index
        fape_atom_index_np[:, 3] = token_to_rep_atom
        fape_token_index_np[:, 3] = np.arange(N, dtype=np.int64)
        # Slot 4: backbone O at REF_ATOMS slot 3 for protein. Ref_mask
        # gating handles Gly-without-O / missing-atom; we still hard-
        # mask non-protein tokens since slot index 3 means OP2 (RNA) or
        # similar non-equivalent atoms for non-protein residues.
        fape_atom_index_np[:, 4] = 3
        fape_token_index_np[:, 4] = np.arange(N, dtype=np.int64)
        # Per-slot atom presence via ref_mask gather.
        for k in range(K_FAPE):
            fape_atom_mask_np[:, k] = ref_mask[fape_token_index_np[:, k], fape_atom_index_np[:, k]]
        is_protein_np = tokens["mol_type"] == const.MOL_PROTEIN
        fape_atom_mask_np[~is_protein_np, 4] = False
        # Atomized single-atom tokens have no local frame — drop all FAPE slots
        # so they don't project a garbage (0,1,2) frame onto other tokens.
        fape_atom_mask_np[is_atomized] = False
        # Drop rep slot when it duplicates a frame slot (same token + atom).
        rep_tok = fape_token_index_np[:, 3]
        rep_atom_slot = fape_atom_index_np[:, 3]
        for k in range(3):
            dup = (fape_token_index_np[:, k] == rep_tok) & (fape_atom_index_np[:, k] == rep_atom_slot)
            fape_atom_mask_np[dup, 3] = False

        # ------------------------------------------------------------------
        # Chirality probe — per-token off-plane atom whose projection on
        # the local (N/CA/C) frame normal disambiguates L vs D (or
        # purine vs pyrimidine attachment, future). ``chirality_target``
        # stores the *signed* CCD ideal projection in Å (≈ ±1.20 for
        # natural L/D amino acids); ``0.0`` masks the residue out (Gly,
        # missing atom, ambiguous projection). Consumed by
        # :func:`deepfold3.training.loss.chirality_loss`.
        # Sanity (2026-05-19): an idealised L-aa on a fitted tetrahedral
        # CA gives ``|local_z| = 1.20 ± 0.03`` Å (measured across all 122
        # standard residues of PDB 11CF). Earlier hardcoded ±1.0 target
        # was a 17% underestimate that introduced a fictitious gradient
        # away from the biological ideal once the sign was correct.
        # Currently protein-only; RNA / DNA / ligand left as 0.
        # ------------------------------------------------------------------
        chirality_atom_idx_np = np.zeros(N, dtype=np.int64)
        chirality_target_np = np.zeros(N, dtype=np.float32)
        _CHIRALITY_PROJ_MIN = 0.3  # Å — discard ambiguous near-plane projections
        for i, tok in enumerate(tokens):
            mt = int(tok["mol_type"])
            rt = int(tok["res_type"])
            if mt != const.MOL_PROTEIN:
                continue  # TODO: N1/N9 probe for RNA/DNA
            if rt == 7:
                continue  # Gly — no Cβ
            if not ref_mask[i, 4]:
                continue  # Cβ missing
            if not frames_mask_np[i]:
                continue  # (N, CA, C) not all present
            n_xyz = ref_pos[i, 0]
            ca_xyz = ref_pos[i, 1]
            c_xyz = ref_pos[i, 2]
            cb_xyz = ref_pos[i, 4]
            w1 = n_xyz - ca_xyz
            w2 = c_xyz - ca_xyz
            n1 = float(np.linalg.norm(w1))
            n2 = float(np.linalg.norm(w2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            w1 /= n1
            w2 /= n2
            e1 = w1 + w2
            e2 = w2 - w1
            ne1 = float(np.linalg.norm(e1))
            ne2 = float(np.linalg.norm(e2))
            if ne1 < 1e-6 or ne2 < 1e-6:
                continue
            e1 /= ne1
            e2 /= ne2
            e3 = np.cross(e1, e2)
            proj = float(np.dot(cb_xyz - ca_xyz, e3))
            if abs(proj) < _CHIRALITY_PROJ_MIN:
                continue  # ambiguous — mask out
            chirality_atom_idx_np[i] = 4
            chirality_target_np[i] = proj  # signed value (~±1.20), not just sign

        # ------------------------------------------------------------------
        # mol_type: molecule type per token (protein/DNA/RNA/non-polymer)
        # Used by output layer to determine entity type correctly (ligands
        # have res_type=20 same as UNK protein, so res_type alone is ambiguous).
        # ------------------------------------------------------------------
        mol_type_tensor = torch.from_numpy(tokens["mol_type"].astype(np.int64))

        # ------------------------------------------------------------------
        # token_res_name: residue name per token as [N, 5] int8
        # Encoded as raw ASCII bytes, padded with 0 (null).
        # Needed by output layer to write correct CCD codes for ligands
        # (res_type=20 maps to "UNK" but actual name may be e.g. "PCG").
        # ------------------------------------------------------------------
        token_res_name = np.zeros((N, 5), dtype=np.int8)
        for i, tok in enumerate(tokens):
            atom_idx = int(tok["atom_idx"])
            name = _find_res_name(data, atom_idx)
            name_bytes = name.encode("ascii", errors="replace")[:5]
            for j, b in enumerate(name_bytes):
                token_res_name[i, j] = b

        result = {
            "res_type": res_type,
            "residue_index": residue_index,
            "token_index": token_index,
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
            "sym_token_index": sym_token_index,
            "mol_type": mol_type_tensor,
            "token_res_name": torch.from_numpy(token_res_name),
            "seq_mask": seq_mask,
            "pair_mask": pair_mask,
            "frames_mask": frames_mask,
            "token_to_rep_atom": torch.from_numpy(token_to_rep_atom),
            "token_to_center_atom": torch.from_numpy(token_to_center_atom),
            "frame_atom_index": torch.from_numpy(frame_atom_index),
            "frame_token_index": torch.from_numpy(frame_token_index),
            "chirality_atom_idx": torch.from_numpy(chirality_atom_idx_np),
            "chirality_target": torch.from_numpy(chirality_target_np),
            "fape_atom_index": torch.from_numpy(fape_atom_index_np),
            "fape_token_index": torch.from_numpy(fape_token_index_np),
            "fape_atom_mask": torch.from_numpy(fape_atom_mask_np),
            "ref_pos": torch.from_numpy(ref_pos),
            "ref_mask": torch.from_numpy(ref_mask),
            "ref_element": torch.from_numpy(ref_element),
            "ref_charge": torch.from_numpy(ref_charge),
            "ref_atom_name_chars": torch.from_numpy(ref_atom_name_chars),
            "ref_space_uid": torch.from_numpy(ref_space_uid),
            "msa": msa_tensor.long(),
            "msa_mask": msa_mask_tensor,
            "deletion_matrix": del_matrix_tensor.long(),
            "profile": profile,
            "deletion_mean": deletion_mean,
        }

        # ------------------------------------------------------------------
        # Ground-truth atom positions for training loss
        # Uses actual observed coordinates from the Structure (not CCD ideal).
        # ------------------------------------------------------------------
        if training:
            atom_gt_positions = np.zeros((N, M, 3), dtype=np.float32)
            atom_gt_mask = np.zeros((N, M), dtype=bool)

            for i, tok in enumerate(tokens):
                atom_idx = int(tok["atom_idx"])
                atom_num = int(tok["atom_num"])
                mol_type_tok = int(tok["mol_type"])

                if mol_type_tok == const.MOL_NONPOLYMER or is_atomized[i]:
                    # Single atom per token (ligand / atomized polymer atom)
                    if atom_idx < len(data.structure.atoms):
                        atom = data.structure.atoms[atom_idx]
                        if bool(atom["is_present"]):
                            atom_gt_positions[i, 0] = [float(atom["x"]), float(atom["y"]), float(atom["z"])]
                            atom_gt_mask[i, 0] = True
                else:
                    # Polymer: multiple atoms per token. Place each stored atom
                    # at the same CCD slot the ref loop used (atom_slot_remap),
                    # so GT positions/mask align with ref_pos even when a
                    # non-terminal atom is missing.
                    for j in range(min(atom_num, M)):
                        a_idx = atom_idx + j
                        if a_idx < len(data.structure.atoms):
                            atom = data.structure.atoms[a_idx]
                            if bool(atom["is_present"]):
                                slot = int(atom_slot_remap[i, j])
                                atom_gt_positions[i, slot] = [float(atom["x"]), float(atom["y"]), float(atom["z"])]
                                atom_gt_mask[i, slot] = True

            result["atom_gt_positions"] = torch.from_numpy(atom_gt_positions)
            result["atom_gt_mask"] = torch.from_numpy(atom_gt_mask)

            # Symmetric atom swap pairs (AlphaFold 3 §4.3): per-token
            # ``(token_idx, M_a, M_b)`` triples for residues whose side chain
            # has a 180° equivalence (ASP/GLU/ARG/PHE/TYR/HIS/LEU/VAL).
            # Cheap to compute (microseconds for 12gs); the loss applies the
            # best-of-two swap per (sample, pair).
            from deepfold3.data.symmetry.atom_perms import (
                compute_atomized_residue_token_swap_pairs,
                compute_ligand_token_permutations,
                compute_token_atom_symmetry_pairs,
            )

            atom_swap_pairs = compute_token_atom_symmetry_pairs(data.structure, tokens)
            # ``compute_token_atom_symmetry_pairs`` emits the swap atom slots in
            # *dense structure* order (index into ``structure.atoms[atom_idx+j]``),
            # but the loss ``[N, M, 3]`` tensors place each atom at its
            # CCD-canonical slot via ``atom_slot_remap`` (name -> ``REF_ATOMS``
            # index). These diverge whenever a residue's atoms are stored in
            # non-CCD order — e.g. AFDB writes ARG as ``...NE, NH1, NH2, CZ``,
            # shifting NH1/NH2 by one slot — so the raw dense pairs would swap the
            # wrong atoms (empirically 100% of AFDB ARG). Remap the slots through
            # the same ``atom_slot_remap`` that placed the GT, so the swap table
            # and the loss tensors share the CCD-slot space by construction. A
            # no-op when the structure is already CCD-ordered (identity remap).
            if len(atom_swap_pairs):
                tk = atom_swap_pairs[:, 0]
                atom_swap_pairs[:, 1] = atom_slot_remap[tk, atom_swap_pairs[:, 1]]
                atom_swap_pairs[:, 2] = atom_slot_remap[tk, atom_swap_pairs[:, 2]]
            result["atom_swap_pairs"] = torch.from_numpy(atom_swap_pairs)

            # Ligand CCD automorphisms (AF3 §4.2): the whole permutation group,
            # resolved by joint RMSD in the loss (``apply_ligand_token_permutation``).
            # ``[R, 4]`` (ligand_id, perm_id, dst_token, src_token) over moved
            # atoms only. Empty when no cache is loaded or the crop has no
            # symmetric ligands. Handles 3-fold+ symmetries the pairwise path
            # dropped.
            ligand_perms = compute_ligand_token_permutations(
                data.structure, tokens, self.ligand_perm_cache, ccd_atom_order=self.ccd_atom_order
            )
            result["ligand_perms"] = torch.from_numpy(ligand_perms)

            # Atomized standard residues (training regularizer) whose side chain
            # is symmetric (ASP/GLU/ARG/…) need inter-token swaps — genuine
            # 2-cycles, correctly handled by the pairwise best-of-two path.
            # ``(token_a, token_b, group_id)`` slot-0 triples (group id keeps a
            # PHE/TYR ring flip joint).
            ligand_swap_pairs = np.empty((0, 3), dtype=np.int32)
            if is_atomized.any():
                ligand_swap_pairs = compute_atomized_residue_token_swap_pairs(data.structure, tokens)
            result["ligand_swap_pairs"] = torch.from_numpy(ligand_swap_pairs)

            # Bond features for bond violation loss (AF3 §5.3: polymer–ligand
            # token bonds only by default; extended Protenix-style set
            # gated behind ``include_polymer_backbone_bonds``).
            bond_feats = self._build_bond_features(
                tokens,
                data.structure.atoms,
                atom_gt_positions,
                atom_gt_mask,
                data.bonds,
                N,
                M,
                include_polymer_backbone=self.include_polymer_backbone_bonds,
            )
            result.update(bond_feats)

        # Bond matrix. Emitted unconditionally (all-zero when the crop has no
        # ligand bonds) so every sample carries the same key set: both consumers
        # already treat all-zeros as "no bonds" — ``bond_embedding`` is a
        # ``LinearNoBias`` so zeros map to exact zeros, and
        # ``_inter_chain_bonds_from_features`` returns None on an empty index.
        bond_matrix = torch.zeros(N, N, 1)
        if len(data.bonds) > 0:
            t1 = data.bonds["token_1"].astype(np.intp)
            t2 = data.bonds["token_2"].astype(np.intp)
            bond_matrix[t1, t2, 0] = 1.0
            bond_matrix[t2, t1, 0] = 1.0
        result["bond_matrix"] = bond_matrix

        # ------------------------------------------------------------------
        # Template features
        # ------------------------------------------------------------------
        result.update(self._build_template_features(templates, N, tokens, template_feats_per_chain, template_col_idx))

        # ------------------------------------------------------------------
        # Token mask channels (for loss control)
        # ------------------------------------------------------------------
        if data.token_mask is not None:
            for name, arr in data.token_mask.channels.items():
                result[f"mask_{name}"] = torch.tensor(arr)

        return result

    def _build_msa_features(
        self,
        data: Tokenized,
        tokens: np.ndarray,
        N: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [S, N] MSA tensors with paired + dense unpaired rows.

        MSA layout (following AF3 ``merging_features.py``):
        1. Paired rows (``_all_seq``): cross-chain co-evolutionary signal —
           row *i* from each chain's paired MSA placed on the same full-width
           row.  First row is the query.
        2. Dense unpaired rows: each chain's unpaired MSA is padded to the
           same number of rows (``max_S``), then all chains are placed on
           the **same** rows (each chain fills its own columns, gaps elsewhere).
           This is NOT block-diagonal — row *i* has every chain's *i*-th
           unpaired sequence simultaneously.

        Cross-chain gap positions have msa_mask=True (AF3 treats gaps as
        valid aligned positions).
        """
        # Build chain position maps: asym_id → global token indices.
        chain_pos_map: dict[int, np.ndarray] = {}
        for asym_id in set(tokens["asym_id"]):
            chain_pos_map[int(asym_id)] = np.where(tokens["asym_id"] == asym_id)[0]

        # Pre-compute per-chain scatter indices: (token_positions, msa_columns).
        # When msa_col_idx is available, expanded non-standard residues map
        # multiple tokens to the same MSA column.  Otherwise fall back to
        # positional 1:1 mapping (legacy).
        _chain_scatter: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for aid, pos in chain_pos_map.items():
            if data.msa_col_idx is not None:
                cols = data.msa_col_idx[pos]
                valid = cols >= 0  # ligand tokens have -1
                _chain_scatter[aid] = (pos[valid], cols[valid])
            else:
                _chain_scatter[aid] = (pos, np.arange(len(pos)))

        def _scatter_msa(dst: np.ndarray, aid: int, msa_data: np.ndarray) -> None:
            """Scatter MSA data into token positions for one chain.

            Handles both 1-D (single row) and 2-D (multiple rows) ``msa_data``.
            Columns beyond the MSA width are silently skipped.
            """
            tok_pos, msa_cols = _chain_scatter[aid]
            keep = msa_cols < msa_data.shape[-1]
            if not keep.all():
                tok_pos, msa_cols = tok_pos[keep], msa_cols[keep]
            if msa_data.ndim == 1:
                dst[tok_pos] = msa_data[msa_cols]
            else:
                dst[:, tok_pos] = msa_data[:, msa_cols]

        combined_seqs: list[np.ndarray] = []
        combined_masks: list[np.ndarray] = []
        combined_dels: list[np.ndarray] = []

        # --- 1. Paired rows: zip across chains (up to MAX_PAIRED_MSA_ROWS) ---
        # Use the deepest available chain; chains with shorter or absent paired
        # MSAs leave their columns as gaps for the rows they don't cover.
        n_paired_rows = 0
        if data.paired_msa:
            paired_chains: list[tuple[int, MSA]] = []
            for asym_id, msa in data.paired_msa.items():
                aid = int(asym_id)
                if aid in chain_pos_map and len(chain_pos_map[aid]) > 0 and msa.sequences.shape[0] > 0:
                    paired_chains.append((aid, msa))
            if paired_chains:
                max_depth = max(msa.sequences.shape[0] for _, msa in paired_chains)
                n_paired_rows = min(max_depth, const.MAX_PAIRED_MSA_ROWS + 1)
                for row_i in range(n_paired_rows):
                    row_seq = np.full(N, const.MSA_GAP, dtype=np.int8)
                    row_mask = np.ones(N, dtype=bool)
                    row_del = np.zeros(N, dtype=np.int8)
                    for aid, msa in paired_chains:
                        if row_i >= msa.sequences.shape[0]:
                            continue
                        _scatter_msa(row_seq, aid, msa.sequences[row_i])
                        _scatter_msa(row_mask, aid, msa.mask[row_i])
                        _scatter_msa(row_del, aid, msa.deletion_matrix[row_i])
                    combined_seqs.append(row_seq[None])
                    combined_masks.append(row_mask[None])
                    combined_dels.append(row_del[None])
                # Paired row 0 is the query row. A chain with no paired MSA (DNA,
                # or a chain the search missed) has only gaps there, and the
                # unpaired block below skips its row 0, so copy it in here or the
                # chain's own sequence reaches no row at all.
                paired_aids = {aid for aid, _ in paired_chains}
                for asym_id, msa in data.unpaired_msa.items():
                    aid = int(asym_id)
                    if aid in paired_aids or msa.sequences.shape[0] == 0 or len(chain_pos_map.get(aid, ())) == 0:
                        continue
                    _scatter_msa(combined_seqs[0][0], aid, msa.sequences[0])
                    _scatter_msa(combined_masks[0][0], aid, msa.mask[0])
                    _scatter_msa(combined_dels[0][0], aid, msa.deletion_matrix[0])

        # --- 1b. Auto-synthesise paired rows for homo-multimer entities ---
        # When ColabFold/MMseqs2 didn't produce a paired MSA for this record
        # (typically because it's a homo-oligomer — pure A_n complexes are
        # excluded from the paired-search pipeline since the pairing is
        # trivial), reconstruct paired rows directly from the unpaired MSA:
        # every chain in a homo entity shares the same sequence and therefore
        # the same MSA, so unpaired row i of any one chain is, by
        # construction, the species-aligned row i for every chain in that
        # entity. The synthesised rows are placed in the same paired slot
        # as a real paired MSA would occupy. Heteromer chains in the same
        # crop pass through unchanged (their columns stay as gaps for the
        # synthesised rows, identical to how real paired pipelines treat a
        # missing-from-pairing chain).
        #
        # ``unpaired_skip[asym_id]`` records how many leading rows of each
        # chain's unpaired MSA were already consumed by the paired
        # synthesis, so the dense unpaired block below can advance past
        # them and avoid duplicating rows. For chains not in any homo
        # group it stays at the default of 1 (just the query row).
        unpaired_skip: dict[int, int] = {}
        if n_paired_rows == 0:
            chain_to_entity: dict[int, int] = {}
            for asym_id, positions in chain_pos_map.items():
                if len(positions) == 0:
                    continue
                chain_to_entity[asym_id] = int(tokens[positions[0]]["entity_id"])

            entity_to_chains: dict[int, list[int]] = {}
            for asym_id, eid in chain_to_entity.items():
                entity_to_chains.setdefault(eid, []).append(asym_id)

            homo_groups: list[tuple[list[int], MSA]] = []
            for group_chains in entity_to_chains.values():
                if len(group_chains) <= 1:
                    continue
                shared_msa: MSA | None = None
                for asym_id in group_chains:
                    msa = data.unpaired_msa.get(asym_id)
                    if msa is not None and msa.sequences.shape[0] > 0:
                        shared_msa = msa
                        break
                if shared_msa is not None:
                    homo_groups.append((group_chains, shared_msa))

            if homo_groups:
                max_depth = max(msa.sequences.shape[0] for _, msa in homo_groups)
                n_paired_rows = min(max_depth, const.MAX_PAIRED_MSA_ROWS + 1)
                for row_i in range(n_paired_rows):
                    row_seq = np.full(N, const.MSA_GAP, dtype=np.int8)
                    row_mask = np.ones(N, dtype=bool)
                    row_del = np.zeros(N, dtype=np.int8)
                    for group_chains, msa in homo_groups:
                        if row_i >= msa.sequences.shape[0]:
                            continue
                        for asym_id in group_chains:
                            _scatter_msa(row_seq, asym_id, msa.sequences[row_i])
                            _scatter_msa(row_mask, asym_id, msa.mask[row_i])
                            _scatter_msa(row_del, asym_id, msa.deletion_matrix[row_i])
                    combined_seqs.append(row_seq[None])
                    combined_masks.append(row_mask[None])
                    combined_dels.append(row_del[None])
                # Mark the consumed prefix per homo chain so the dense
                # unpaired block doesn't re-emit the same rows.
                for group_chains, _ in homo_groups:
                    for asym_id in group_chains:
                        unpaired_skip[asym_id] = n_paired_rows

        # If still no paired MSA (monomer or fully-hetero singleton crop),
        # build query row from unpaired row 0 of every chain.
        if n_paired_rows == 0:
            query_seq = np.full(N, const.MSA_GAP, dtype=np.int8)
            query_mask = np.ones(N, dtype=bool)
            query_del = np.zeros(N, dtype=np.int8)
            for asym_id, msa in data.unpaired_msa.items():
                chain_pos = chain_pos_map.get(int(asym_id))
                if chain_pos is None or len(chain_pos) == 0 or msa.sequences.shape[0] == 0:
                    continue
                _scatter_msa(query_seq, int(asym_id), msa.sequences[0])
                _scatter_msa(query_mask, int(asym_id), msa.mask[0])
                _scatter_msa(query_del, int(asym_id), msa.deletion_matrix[0])
            combined_seqs.append(query_seq[None])
            combined_masks.append(query_mask[None])
            combined_dels.append(query_del[None])

        # --- 2. Dense unpaired rows (AF3: pad-to-max then concat along columns) ---
        # For each chain, crop unpaired rows to at most
        # (max_msa_seqs - n_paired_rows) then pad all chains to the same
        # number of rows.  Place each chain's data in its own columns and
        # gap-fill the rest.
        max_unpaired = self.max_msa_seqs - max(n_paired_rows, 1)
        # Collect per-chain unpaired MSA (skip the query row + any rows the
        # paired-synthesis branch already consumed for homo-multimer chains).
        chain_unpaired: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        max_S_unpaired = 0
        n_dropped: dict[int, int] = {}
        for asym_id, msa in data.unpaired_msa.items():
            skip = unpaired_skip.get(int(asym_id), 1)
            paired = data.paired_msa.get(asym_id) if self.dedup_unpaired_against_paired else None
            if paired is not None:
                keep = unpaired_rows_not_in_paired(msa, paired)
                if not keep.all():
                    n_dropped[int(asym_id)] = int((~keep).sum())
                    msa = MSA(
                        sequences=msa.sequences[keep], mask=msa.mask[keep], deletion_matrix=msa.deletion_matrix[keep]
                    )
            if msa.sequences.shape[0] <= skip:
                continue
            end = min(msa.sequences.shape[0], skip + max_unpaired)
            chain_unpaired.append(
                (
                    int(asym_id),
                    msa.sequences[skip:end],
                    msa.mask[skip:end],
                    msa.deletion_matrix[skip:end],
                )
            )
            max_S_unpaired = max(max_S_unpaired, end - skip)
        if self.dedup_unpaired_against_paired:
            logger.info("MSA pairing: dropped unpaired rows already in the paired block, per asym_id %s", n_dropped)

        if max_S_unpaired > 0:
            # Build dense [max_S_unpaired, N] arrays — all chains on same rows
            dense_seqs = np.full((max_S_unpaired, N), const.MSA_GAP, dtype=np.int8)
            dense_masks = np.ones((max_S_unpaired, N), dtype=bool)
            dense_dels = np.zeros((max_S_unpaired, N), dtype=np.int8)
            for asym_id, seqs, masks, dels in chain_unpaired:
                chain_pos = chain_pos_map.get(asym_id)
                if chain_pos is None or len(chain_pos) == 0:
                    continue
                S_chain = seqs.shape[0]
                _scatter_msa(dense_seqs[:S_chain], asym_id, seqs)
                _scatter_msa(dense_masks[:S_chain], asym_id, masks)
                _scatter_msa(dense_dels[:S_chain], asym_id, dels)
            combined_seqs.append(dense_seqs)
            combined_masks.append(dense_masks)
            combined_dels.append(dense_dels)

        if not combined_seqs:
            msa_arr = np.full((1, N), const.MSA_GAP, dtype=np.int8)
            mask_arr = np.ones((1, N), dtype=bool)
            del_arr = np.zeros((1, N), dtype=np.int8)
        else:
            msa_arr = np.concatenate(combined_seqs, axis=0)
            mask_arr = np.concatenate(combined_masks, axis=0)
            del_arr = np.concatenate(combined_dels, axis=0)

        # Cap total rows at max_msa_seqs via uniform subsampling.
        # Always keep row 0 (the query) and uniformly sample the rest. AF3
        # §2.3 specifies the final MSA stack as exactly Nmsa = 16384 rows.
        S = msa_arr.shape[0]
        if S > self.max_msa_seqs:
            keep = self.max_msa_seqs - 1
            rng = np.random.default_rng()
            sel = rng.choice(S - 1, size=keep, replace=False) + 1  # indices 1..S-1
            sel.sort()
            idx = np.concatenate([[0], sel])
            msa_arr = msa_arr[idx]
            mask_arr = mask_arr[idx]
            del_arr = del_arr[idx]

        return (
            torch.from_numpy(msa_arr.astype(np.int16)),
            torch.from_numpy(mask_arr),
            torch.from_numpy(del_arr.astype(np.int16)),
        )

    def _build_msa_crops(
        self,
        msa: torch.Tensor,  # [S, N] int16
        mask: torch.Tensor,  # [S, N] bool
        deln: torch.Tensor,  # [S, N] int16
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a FIXED ``[n_msa_crops * msa_crop_size, N]`` stack of crops.

        Each crop = the query row 0 + ``crop-1`` non-query rows drawn (disjoint,
        wrapping when the real pool is smaller) from one fresh permutation;
        short pools leave the remaining rows masked. The model splits this stack
        and uses crop ``i`` for recycle ``i`` (no on-GPU resample), so the GPU
        MSA shape is constant every step. In training an AF3 §2.3 random-``k``
        mask is applied per crop using the FULL real depth so the masking ratio
        matches the legacy on-GPU path (``k ~ Uniform[1, S]``; if ``k`` < real
        rows kept, the surplus non-query rows are masked).
        """
        n = self.n_msa_crops
        crop = self.msa_crop_size
        msa_np = msa.numpy()
        mask_np = mask.numpy()
        deln_np = deln.numpy()
        S, N = msa_np.shape
        rng = np.random.default_rng()

        out_msa = np.full((n * crop, N), const.MSA_GAP, dtype=msa_np.dtype)
        out_mask = np.zeros((n * crop, N), dtype=mask_np.dtype)
        out_deln = np.zeros((n * crop, N), dtype=deln_np.dtype)

        # Query row 0 heads every crop (rows 0, crop, 2*crop, …) — write once via
        # a strided broadcast rather than re-copying inside the per-crop loop.
        out_msa[::crop] = msa_np[0]
        out_mask[::crop] = mask_np[0]
        out_deln[::crop] = deln_np[0]

        nonquery = np.arange(1, S)
        rng.shuffle(nonquery)
        avail = nonquery.shape[0]  # S - 1
        fill = min(crop - 1, avail)  # real non-query rows placed per crop

        for c in range(n):
            base = c * crop
            if fill > 0:
                start = (c * fill) % avail
                sel = nonquery[(np.arange(fill) + start) % avail]
                rows = slice(base + 1, base + 1 + fill)
                out_msa[rows] = msa_np[sel]
                out_mask[rows] = mask_np[sel]
                out_deln[rows] = deln_np[sel]
            # AF3 §2.3 random-k mask, per crop, ratio-matched to the legacy path.
            if training and fill > 0:
                k = int(rng.integers(1, S + 1))  # Uniform[1, S]
                n_real = fill + 1  # query + real non-query rows in this crop
                if k < n_real:
                    drop = rng.choice(fill, size=n_real - k, replace=False)
                    out_mask[base + 1 + drop] = False

        return (
            torch.from_numpy(out_msa),
            torch.from_numpy(out_mask),
            torch.from_numpy(out_deln),
        )

    def _compute_msa_profile_deletion_mean(
        self,
        data: Tokenized,
        tokens: np.ndarray,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-chain MSA profile ``[N, 31]`` and raw deletion_mean ``[N]``.

        For each chain, the masked per-column class frequency
        ``profile[col, c] = Σ_s mask·1[msa==c] / Σ_s mask`` (classes 0–30) and
        the raw ``mean_s deletion`` are computed from that chain's own unpaired
        MSA — over its *full* depth (before the 16384-row cap, AF3 §2.8) — then
        scattered to the chain's token columns via ``msa_col_idx``. Ligand /
        column-less tokens keep a zero profile and zero deletion_mean.

        Computing per chain (rather than over the assembled complex MSA) is what
        prevents a shallow chain's cross-chain gap padding from diluting these
        statistics; see the call site.
        """
        C = const.RES_TYPE_COUNT
        profile = np.zeros((N, C), dtype=np.float32)
        deletion_mean = np.zeros(N, dtype=np.float32)

        for asym_id, msa in data.unpaired_msa.items():
            aid = int(asym_id)
            pos = np.where(tokens["asym_id"] == aid)[0]
            n_rows, n_cols = msa.sequences.shape
            if len(pos) == 0 or n_rows == 0 or n_cols == 0:
                continue
            if data.msa_col_idx is not None:
                cols = data.msa_col_idx[pos]
                valid = cols >= 0  # ligand / column-less tokens are -1
                tok_pos, msa_cols = pos[valid], cols[valid]
            else:
                tok_pos, msa_cols = pos, np.arange(len(pos))
            keep = msa_cols < n_cols
            tok_pos, msa_cols = tok_pos[keep], msa_cols[keep]
            if len(tok_pos) == 0:
                continue

            seqs = np.clip(msa.sequences, 0, C - 1).astype(np.int64)  # [S, n_cols]
            m = msa.mask.astype(np.float32)  # [S, n_cols]
            # Flat [n_cols·C] histogram via one scatter-add (matches the old
            # _compute_profile accumulation, per chain).
            flat_idx = (np.arange(n_cols) * C)[None, :] + seqs  # [S, n_cols]
            acc = np.zeros(n_cols * C, dtype=np.float32)
            np.add.at(acc, flat_idx.reshape(-1), m.reshape(-1))
            denom = np.clip(m.sum(axis=0), 1.0, None)[:, None]  # [n_cols, 1]
            col_profile = acc.reshape(n_cols, C) / denom
            col_del_mean = msa.deletion_matrix.astype(np.float32).mean(axis=0)  # [n_cols]

            profile[tok_pos] = col_profile[msa_cols]
            deletion_mean[tok_pos] = col_del_mean[msa_cols]

        return torch.from_numpy(profile), torch.from_numpy(deletion_mean)

    @staticmethod
    def _build_bond_features(
        tokens: np.ndarray,
        structure_atoms: np.ndarray,  # ATOM_DTYPE from Structure
        atom_gt_positions: np.ndarray,  # [N, M, 3]
        atom_gt_mask: np.ndarray,  # [N, M]
        token_bonds: np.ndarray,  # TOKEN_BOND_DTYPE from Tokenized
        N: int,
        M: int,
        include_polymer_backbone: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build bond pair features for bond violation loss (AF3 §5.3, Eq. 5).

        AF3 §5.3 and OpenFold-3 restrict ``L_bond`` to polymer–ligand token
        bonds. When ``include_polymer_backbone`` is False (default) that
        scope is applied: only ``Tokenized.bonds`` entries where one side
        is polymer and the other is ligand are emitted.

        When ``include_polymer_backbone`` is True the legacy Protenix-style
        extended set is emitted — preserved for experimentation:

        1. **Intra-residue backbone** (polymer):
           - Protein: N-CA (atoms 0-1), CA-C (atoms 1-2) within each token.
           - RNA/DNA: P-O5', O5'-C5', C5'-C4', C4'-C3', C3'-O3' within each
             token (looked up by atom name since 5'-terminal nucleotides
             may lack the phosphate group).
        2. **Inter-residue peptide/phosphodiester**:
           - Protein: C[i] (atom 2) → N[i+1] (atom 0).
           - RNA/DNA: O3'[i] → P[i+1] (looked up by name).
        3. **All token bonds** (polymer–ligand and ligand–ligand).

        Uses **observed GT distances** as targets (Protenix approach), not
        CCD ideal distances. Only includes bonds where both atoms are
        resolved (``atom_gt_mask`` is True for both).

        Returns:
            Dict with keys:
                ``bond_atom_indices`` ``[B, 4]`` — (token_1, atom_1, token_2, atom_2)
                ``bond_gt_distances`` ``[B]`` — observed distances (Angstroms)
                ``bond_loss_mask``    ``[B]`` — True where both atoms are resolved
        """
        bonds: list[tuple[int, int, int, int]] = []  # (t1, a1, t2, a2)

        # Per-token cache of atom-name → local atom index (only used by the
        # legacy polymer-backbone path; keeps the default AF3 path free of
        # per-token name-lookup overhead on protein-only complexes).
        nucleic_name_map: dict[int, dict[str, int]] = {}

        def _nucleic_atom_map(tok_idx: int) -> dict[str, int]:
            cached = nucleic_name_map.get(tok_idx)
            if cached is not None:
                return cached
            m = _token_atom_name_map(tokens[tok_idx], structure_atoms, M)
            nucleic_name_map[tok_idx] = m
            return m

        # Intra-residue nucleic backbone bond list (names lookup).
        # C4'–O4' and O4'–C1' etc. are intentionally omitted — the five
        # bonds below define the main-chain phosphate–sugar backbone that
        # the bond violation loss needs to constrain.
        _NUC_INTRA_BONDS: tuple[tuple[str, str], ...] = (
            ("P", "O5'"),
            ("O5'", "C5'"),
            ("C5'", "C4'"),
            ("C4'", "C3'"),
            ("C3'", "O3'"),
        )

        if include_polymer_backbone:
            # --- 1. Intra-residue backbone bonds ---
            # Protein: N(0)-CA(1), CA(1)-C(2)
            # RNA/DNA: P-O5', O5'-C5', C5'-C4', C4'-C3', C3'-O3' (by atom name;
            # 5'-terminal nucleotides lack the phosphate group so P may be
            # missing — those bonds are silently skipped).
            for i in range(N):
                mt = int(tokens[i]["mol_type"])
                if mt == const.MOL_PROTEIN:
                    # N-CA bond (atoms 0-1)
                    bonds.append((i, 0, i, 1))
                    # CA-C bond (atoms 1-2)
                    bonds.append((i, 1, i, 2))
                elif mt in (const.MOL_RNA, const.MOL_DNA):
                    amap = _nucleic_atom_map(i)
                    for a_name, b_name in _NUC_INTRA_BONDS:
                        a_idx = amap.get(a_name)
                        b_idx = amap.get(b_name)
                        if a_idx is None or b_idx is None:
                            continue
                        bonds.append((i, a_idx, i, b_idx))

            # --- 2. Inter-residue backbone bonds ---
            for i in range(N - 1):
                mt_i = int(tokens[i]["mol_type"])
                mt_j = int(tokens[i + 1]["mol_type"])
                chain_i = int(tokens[i]["asym_id"])
                chain_j = int(tokens[i + 1]["asym_id"])

                # Must be same chain and both polymer
                if chain_i != chain_j:
                    continue
                if mt_i == const.MOL_NONPOLYMER or mt_j == const.MOL_NONPOLYMER:
                    continue

                if mt_i == const.MOL_PROTEIN and mt_j == const.MOL_PROTEIN:
                    # C[i] (atom 2) → N[i+1] (atom 0)
                    bonds.append((i, 2, i + 1, 0))
                elif mt_i in (const.MOL_RNA, const.MOL_DNA) and mt_j in (const.MOL_RNA, const.MOL_DNA):
                    # O3'[i] → P[i+1] (phosphodiester linkage). Either atom
                    # may be missing at chain termini or when the residue
                    # lacks the phosphate group — skip those cases.
                    amap_i = _nucleic_atom_map(i)
                    amap_j = _nucleic_atom_map(i + 1)
                    o3_idx = amap_i.get("O3'")
                    p_idx = amap_j.get("P")
                    if o3_idx is not None and p_idx is not None:
                        bonds.append((i, o3_idx, i + 1, p_idx))

        # --- 3. Token bonds ---
        # Default (AF3 §5.3): polymer–ligand only. Extended path keeps all
        # token bonds (polymer–ligand + ligand–ligand) to match the legacy
        # Protenix behaviour.
        mol_types = tokens["mol_type"].astype(np.int64)
        for bond in token_bonds:
            t1, t2 = int(bond["token_1"]), int(bond["token_2"])
            a1, a2 = int(bond["atom_1"]), int(bond["atom_2"])
            if not (0 <= t1 < N and 0 <= t2 < N and a1 < M and a2 < M):
                continue
            if not include_polymer_backbone:
                is_polymer_1 = mol_types[t1] != const.MOL_NONPOLYMER
                is_polymer_2 = mol_types[t2] != const.MOL_NONPOLYMER
                if is_polymer_1 == is_polymer_2:
                    # Skip polymer–polymer and ligand–ligand token bonds.
                    continue
            bonds.append((t1, a1, t2, a2))

        if not bonds:
            return {
                "bond_atom_indices": torch.zeros(0, 4, dtype=torch.long),
                "bond_gt_distances": torch.zeros(0, dtype=torch.float32),
                "bond_loss_mask": torch.zeros(0, dtype=torch.bool),
            }

        indices = np.array(bonds, dtype=np.int64)  # [B, 4]
        t1, a1, t2, a2 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        # Compute GT distances and mask
        pos1 = atom_gt_positions[t1, a1]  # [B, 3]
        pos2 = atom_gt_positions[t2, a2]  # [B, 3]
        gt_dist = np.linalg.norm(pos1 - pos2, axis=-1)  # [B]
        mask = atom_gt_mask[t1, a1] & atom_gt_mask[t2, a2]  # [B]

        # AF3 §2.8: exclude token bonds with GT distance >= 2.4 A
        mask = mask & (gt_dist < 2.4)

        return {
            "bond_atom_indices": torch.from_numpy(indices),
            "bond_gt_distances": torch.from_numpy(gt_dist.astype(np.float32)),
            "bond_loss_mask": torch.from_numpy(mask),
        }

    def _build_template_features(
        self,
        templates: list[TemplateFeatures] | None,
        N: int,
        tokens: np.ndarray | None = None,
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
        template_col_idx: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build template feature tensors from TemplateFeatures list.

        For multimer: if template_feats_per_chain is provided, each chain's
        templates populate only that chain's region of the [T, N] tensors.
        Single-chain (or single chain with templates): uses flat list as before.

        ``template_col_idx`` is the query index each token gathers its template
        row by. Its space follows the path taken: chain-local residue index for
        the per-chain path, whole-query index for the flat path. Without it both
        paths fall back to token position, which is only correct when every
        token is one residue (no atomized / modified residues, no ligand) and
        nothing was cropped away.

        Returns dict with keys:
            template_aatype            [T, N]         int64
            template_pseudo_beta       [T, N, 3]      float32
            template_pseudo_beta_mask  [T, N]         bool
            template_backbone_frame    [T, N, 3, 3]   float32
            template_backbone_mask     [T, N]         bool

        With ``pad_templates`` the template axis is fixed at ``max_templates``
        and an extra ``template_mask [T]`` marks real vs pad slots.
        """
        if not templates and not template_feats_per_chain:
            feats = {
                "template_aatype": torch.zeros(0, N, dtype=torch.int64),
                "template_ca": torch.zeros(0, N, 3),
                "template_pseudo_beta": torch.zeros(0, N, 3),
                "template_pseudo_beta_mask": torch.zeros(0, N, dtype=torch.bool),
                "template_backbone_frame": torch.zeros(0, N, 3, 3),
                "template_backbone_mask": torch.zeros(0, N, dtype=torch.bool),
            }
        elif template_feats_per_chain and tokens is not None:
            feats = self._build_per_chain_template_features(template_feats_per_chain, tokens, N, template_col_idx)
        else:
            # Single chain / flat list path (backward compatible)
            feats = self._build_single_chain_template_features(templates, N, template_col_idx)

        return self._finalize_template_axis(feats)

    def _finalize_template_axis(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attach ``template_mask``, and with ``pad_templates`` fix ``T``.

        ``template_mask`` is emitted **unconditionally** — 1.0 for a real
        template slot, 0.0 for a pad slot — because it is what makes the
        template average correct per sample rather than per batch:

        * ``collate_fn`` pads every feature's ``T`` axis up to the batch
          maximum with zeros, so a sample that drew 2 templates next to one
          that drew 4 ends up with 2 empty slots. Without a mask the average
          divides by the *padded* ``T`` and that sample's templates are
          diluted 2/4. The mask is zero-padded by the very same collate step,
          so its sum stays the real count.
        * Emitting it always also keeps the feature contract identical across
          dataset classes. A key present in only some samples is zero-filled
          by ``collate_fn``, which for a mask means "all slots are pad" —
          silently erasing that sample's templates.

        With ``pad_templates`` the axis is additionally padded out to
        ``max_templates`` so every rank's template compute is the same size.
        Pad slots are all-masked (``template_pseudo_beta_mask`` /
        ``template_backbone_mask`` False, aatype = GAP), so the only thing
        that changes is the shape — never the model output.

        Args:
            feats: Template feature dict with a variable leading ``T`` axis.

        Returns:
            The same dict plus ``template_mask``, with ``T == max_templates``
            when ``pad_templates`` is set.
        """
        t = feats["template_aatype"].shape[0]
        t_max = self.max_templates
        # Guard the inference path, which may hand over more hits than
        # ``max_templates``: the fixed axis is the contract, so truncate.
        if self.pad_templates and t > t_max:
            feats = {k: v[:t_max] for k, v in feats.items()}
            t = t_max

        out_t = t_max if self.pad_templates else t
        mask = torch.zeros(out_t, dtype=torch.float32)
        mask[:t] = 1.0

        pad = out_t - t
        if pad > 0:
            # Trailing shape / dtype come from the tensor being extended, so a
            # new template feature is padded correctly without touching this
            # method. aatype pads with GAP (not ALA / zeros), matching the
            # uncovered-position convention of the builders above.
            padded = {}
            for key, val in feats.items():
                fill = const.MSA_GAP if key == "template_aatype" else 0
                padded[key] = torch.cat([val, torch.full((pad, *val.shape[1:]), fill, dtype=val.dtype)])
            feats = padded

        return {**feats, "template_mask": mask}

    def _build_single_chain_template_features(
        self,
        templates: list[TemplateFeatures] | None,
        N: int,
        col_idx: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build template features from a flat list (single-chain case).

        ``col_idx`` is the per-token query index into the (whole-query) template
        arrays, with -1 for tokens outside that space (ligands). When omitted,
        template row *k* goes to token *k* — correct only while token position
        and query residue coincide.
        """
        if not templates:
            return {
                "template_aatype": torch.zeros(0, N, dtype=torch.int64),
                "template_ca": torch.zeros(0, N, 3),
                "template_pseudo_beta": torch.zeros(0, N, 3),
                "template_pseudo_beta_mask": torch.zeros(0, N, dtype=torch.bool),
                "template_backbone_frame": torch.zeros(0, N, 3, 3),
                "template_backbone_mask": torch.zeros(0, N, dtype=torch.bool),
            }

        T = len(templates)
        # Uncovered query positions get the GAP class (AF3 §2.8 / OF3), not ALA
        # (np.zeros): the template aatype stream is not coverage-masked, so an
        # ALA-everywhere one-hot would leak a spurious residue signal into
        # embed_aatype_row/col over off-template regions.
        t_aatype = np.full((T, N), const.MSA_GAP, dtype=np.int64)
        t_pb = np.zeros((T, N, 3), dtype=np.float32)
        t_pb_mask = np.zeros((T, N), dtype=bool)
        t_frame = np.zeros((T, N, 3, 3), dtype=np.float32)
        t_frame_mask = np.zeros((T, N), dtype=bool)
        t_ca = np.zeros((T, N, 3), dtype=np.float32)  # CA = unit-vector frame origin

        cols = None if col_idx is None else np.asarray(col_idx)
        for t, tmpl in enumerate(templates):
            if cols is None:
                # Single-chain, one token per residue: row k → query position k.
                idx = np.arange(min(len(tmpl.aatype), N))
                dst_pos, src_idx = idx, idx
            else:
                # Gather by query index: expanded residues (modified / atomized)
                # repeat their row across their atom tokens, ligands (-1) get none.
                valid = (cols >= 0) & (cols < len(tmpl.aatype))
                dst_pos, src_idx = np.where(valid)[0], cols[valid]
            if len(dst_pos) == 0:
                continue
            self._fill_template_slot(
                t_aatype[t],
                t_pb[t],
                t_pb_mask[t],
                t_frame[t],
                t_frame_mask[t],
                tmpl,
                dst_pos,
                src_idx,
                ca_row=t_ca[t],
            )

        return {
            "template_ca": torch.from_numpy(t_ca),
            "template_aatype": torch.from_numpy(t_aatype),
            "template_pseudo_beta": torch.from_numpy(t_pb),
            "template_pseudo_beta_mask": torch.from_numpy(t_pb_mask),
            "template_backbone_frame": torch.from_numpy(t_frame),
            "template_backbone_mask": torch.from_numpy(t_frame_mask),
        }

    def _build_per_chain_template_features(
        self,
        template_feats_per_chain: dict[int, list[TemplateFeatures]],
        tokens: np.ndarray,
        N: int,
        template_col_idx: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build template features with per-chain template population.

        Each chain's templates fill only that chain's region of the [T, N] tensors.
        """
        T = min(self.max_templates, max((len(v) for v in template_feats_per_chain.values()), default=0))
        if T == 0:
            return {
                "template_aatype": torch.zeros((0, N), dtype=torch.long),
                "template_ca": torch.zeros((0, N, 3)),
                "template_pseudo_beta": torch.zeros((0, N, 3)),
                "template_pseudo_beta_mask": torch.zeros((0, N), dtype=torch.bool),
                "template_backbone_frame": torch.zeros((0, N, 3, 3)),
                "template_backbone_mask": torch.zeros((0, N), dtype=torch.bool),
            }
        # Uncovered query positions → GAP class (see _build_single_chain_...):
        # chains without a template, and crop-away positions, must read as gap,
        # not ALA, in the (un-coverage-masked) template aatype stream.
        t_aatype = np.full((T, N), const.MSA_GAP, dtype=np.int64)
        t_pb = np.zeros((T, N, 3), dtype=np.float32)
        t_pb_mask = np.zeros((T, N), dtype=bool)
        t_frame = np.zeros((T, N, 3, 3), dtype=np.float32)
        t_frame_mask = np.zeros((T, N), dtype=bool)
        t_ca = np.zeros((T, N, 3), dtype=np.float32)  # CA = unit-vector frame origin

        col_idx = None if template_col_idx is None else np.asarray(template_col_idx)
        for asym_id, chain_templates in template_feats_per_chain.items():
            chain_mask = tokens["asym_id"] == asym_id
            chain_pos = np.where(chain_mask)[0]
            if len(chain_pos) == 0:
                continue

            # Map each surviving token to the template row for its query residue.
            # ``template_col_idx`` is the token's original chain-local column
            # (== the template array's 0-based query index). Without it (no crop),
            # the chain's tokens are contiguous 0..k-1, so a positional range is
            # the correct chain-local index.
            if col_idx is not None:
                local = col_idx[chain_pos]
            else:
                local = np.arange(len(chain_pos))

            for t, tmpl in enumerate(chain_templates[:T]):
                valid = (local >= 0) & (local < len(tmpl.aatype))
                dst_pos = chain_pos[valid]
                src_idx = local[valid]
                if len(dst_pos) == 0:
                    continue
                self._fill_template_slot(
                    t_aatype[t],
                    t_pb[t],
                    t_pb_mask[t],
                    t_frame[t],
                    t_frame_mask[t],
                    tmpl,
                    dst_pos,
                    src_idx,
                    ca_row=t_ca[t],
                )

        return {
            "template_ca": torch.from_numpy(t_ca),
            "template_aatype": torch.from_numpy(t_aatype),
            "template_pseudo_beta": torch.from_numpy(t_pb),
            "template_pseudo_beta_mask": torch.from_numpy(t_pb_mask),
            "template_backbone_frame": torch.from_numpy(t_frame),
            "template_backbone_mask": torch.from_numpy(t_frame_mask),
        }

    @staticmethod
    def _fill_template_slot(
        aatype_row: np.ndarray,
        pb_row: np.ndarray,
        pb_mask_row: np.ndarray,
        frame_row: np.ndarray,
        frame_mask_row: np.ndarray,
        tmpl: TemplateFeatures,
        dst_pos: np.ndarray,
        src_idx: np.ndarray,
        ca_row: np.ndarray | None = None,
    ) -> None:
        """Scatter template rows ``src_idx`` onto query token positions ``dst_pos``.

        ``dst_pos`` and ``src_idx`` are paired element-wise, so this handles both
        contiguous (no-crop) and non-contiguous (spatial/interface crop) token
        layouts, placing each template row at its correct query residue.
        """
        aatype_row[dst_pos] = tmpl.aatype[src_idx]
        pos = tmpl.atom_positions[src_idx]  # [L, M, 3]
        mask = tmpl.atom_mask[src_idx]  # [L, M]

        # Pseudo-beta: CB (index 3) for all, CA (index 1) for GLY (type 7)
        is_gly = tmpl.aatype[src_idx] == 7
        cb_pos = pos[:, 3, :]
        ca_pos = pos[:, 1, :]
        pb_row[dst_pos] = np.where(is_gly[:, None], ca_pos, cb_pos)
        pb_mask_row[dst_pos] = np.where(is_gly, mask[:, 1], mask[:, 3])

        # Backbone frames from N(0), CA(1), C(2) via Gram-Schmidt
        n_pos = pos[:, 0, :]
        ca = pos[:, 1, :]
        c_pos = pos[:, 2, :]

        # CA is the frame origin; the template unit-vector feature is the
        # displacement of CA (not CB) in each residue's local frame (AF3 §2.8).
        if ca_row is not None:
            ca_row[dst_pos] = ca

        fm = mask[:, 0] & mask[:, 1] & mask[:, 2]
        frame_mask_row[dst_pos] = fm

        eps = 1e-8
        v1 = c_pos - ca
        v2 = n_pos - ca
        v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + eps
        e1 = v1 / v1_norm
        proj = np.sum(v2 * e1, axis=-1, keepdims=True) * e1
        u2 = v2 - proj
        u2_norm = np.linalg.norm(u2, axis=-1, keepdims=True) + eps
        e2 = u2 / u2_norm
        e3 = np.cross(e1, e2)
        frame_row[dst_pos] = np.stack([e1, e2, e3], axis=1)

    def _empty_features(self) -> dict[str, torch.Tensor]:
        M = self.max_atoms
        return {
            "res_type": torch.zeros(0, dtype=torch.int64),
            "residue_index": torch.zeros(0, dtype=torch.int64),
            "token_index": torch.zeros(0, dtype=torch.int64),
            "asym_id": torch.zeros(0, dtype=torch.int64),
            "entity_id": torch.zeros(0, dtype=torch.int64),
            "sym_id": torch.zeros(0, dtype=torch.int64),
            "sym_token_index": torch.zeros(0, dtype=torch.int64),
            "mol_type": torch.zeros(0, dtype=torch.int64),
            "token_res_name": torch.zeros(0, 5, dtype=torch.int8),
            "seq_mask": torch.zeros(0, dtype=torch.bool),
            "pair_mask": torch.zeros(0, 0, dtype=torch.bool),
            "frames_mask": torch.zeros(0, dtype=torch.bool),
            "token_to_rep_atom": torch.zeros(0, dtype=torch.int64),
            "token_to_center_atom": torch.zeros(0, dtype=torch.int64),
            "frame_atom_index": torch.zeros(0, 3, dtype=torch.int64),
            "frame_token_index": torch.zeros(0, 3, dtype=torch.int64),
            "chirality_atom_idx": torch.zeros(0, dtype=torch.int64),
            "chirality_target": torch.zeros(0),
            "ref_pos": torch.zeros(0, M, 3),
            "ref_mask": torch.zeros(0, M, dtype=torch.bool),
            "ref_element": torch.zeros(0, M, dtype=torch.int64),
            "ref_charge": torch.zeros(0, M),
            "ref_atom_name_chars": torch.zeros(0, M, 4, dtype=torch.int64),
            "ref_space_uid": torch.zeros(0, M, dtype=torch.int64),
            "msa": torch.zeros(1, 0, dtype=torch.int64),
            "msa_mask": torch.zeros(1, 0, dtype=torch.bool),
            "deletion_matrix": torch.zeros(1, 0, dtype=torch.int64),
            "profile": torch.zeros(0, const.RES_TYPE_COUNT),
            "deletion_mean": torch.zeros(0),
            "template_aatype": torch.zeros(0, 0, dtype=torch.int64),
            "template_ca": torch.zeros(0, 0, 3),
            "template_pseudo_beta": torch.zeros(0, 0, 3),
            "template_pseudo_beta_mask": torch.zeros(0, 0, dtype=torch.bool),
            "template_backbone_frame": torch.zeros(0, 0, 3, 3),
            "template_backbone_mask": torch.zeros(0, 0, dtype=torch.bool),
        }


def _token_atom_name_map(
    token: np.void,
    structure_atoms: np.ndarray,
    M: int,
) -> dict[str, int]:
    """Return a mapping from atom name → **CCD-canonical slot** within a token.

    The returned index is the slot the featurizer places the atom at in the
    ``[N, M, 3]`` GT tensors — i.e. its position in ``const.REF_ATOMS`` for the
    token's residue, matching ``atom_slot_remap`` (name -> ``REF_ATOMS`` index).
    Callers (the nucleic backbone bond builder) use these as M-slots into
    ``atom_gt_positions``, so returning the *dense structure position* here
    would mis-place the bond atoms whenever the structure stores the residue in
    non-CCD order (e.g. AFDB's atom convention) — the same class of bug the
    intra-token symmetry remap fixes. Falls back to the dense position ``j`` for
    residues without a ``REF_ATOMS`` entry or atom names absent from it.

    Normalizes the trailing ``*`` prime convention (``O3*``) to the canonical
    ``O3'`` form, so nucleic-acid atom lookups work regardless of the source
    naming style. Only the first ``min(atom_num, M)`` atoms are considered —
    anything past the per-token slot budget is silently dropped (matching
    the rest of the featurizer's token-atom handling).
    """
    atom_idx = int(token["atom_idx"])
    atom_num = int(token["atom_num"])
    res_name = const.RES_TYPE_INDEX_TO_NAME.get(int(token["res_type"]))
    ref_names = const.REF_ATOMS.get(res_name) if res_name is not None else None
    name_to_slot = {nm: k for k, nm in enumerate(ref_names)} if ref_names else None
    n = min(atom_num, M)
    out: dict[str, int] = {}
    total = len(structure_atoms)
    for j in range(n):
        a_idx = atom_idx + j
        if a_idx >= total:
            break
        raw = structure_atoms[a_idx]["name"].decode("ascii", errors="replace").strip()
        if not raw:
            continue
        # CCD-canonical slot (name -> REF_ATOMS index), matching atom_slot_remap;
        # dense-position fallback for unknown residues / names or slots past M.
        slot = j
        if name_to_slot is not None:
            s = name_to_slot.get(raw)
            if s is not None and s < M:
                slot = s
        out[raw] = slot
        # Be defensive about the prime vs star convention used in legacy PDB
        # files: structure.py normalizes to one form, but we accept both so
        # lookups for "O3'" still find an "O3*" entry (and vice versa).
        if "*" in raw:
            out[raw.replace("*", "'")] = slot
        elif "'" in raw:
            out[raw.replace("'", "*")] = slot
    return out


def _find_res_name(data: Tokenized, atom_idx: int) -> str:
    """Find residue name from atom index by searching residues array."""
    for res in data.structure.residues:
        start = int(res["atom_idx"])
        end = start + int(res["atom_num"])
        if start <= atom_idx < end:
            return res["name"].decode("ascii", errors="replace").strip()
    return "UNK"
