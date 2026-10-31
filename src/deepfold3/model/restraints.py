"""Distance restraints steered into the diffusion sampler.

A restraint says "these residues must touch those residues". It is not
conditioning: nothing is fed to the trunk and nothing was trained for it. At
each sampler step the predicted clean structure ``D`` is nudged down the
gradient of a flat-bottom penalty before the Euler update uses it, so the
restraint acts on coordinates directly rather than on what the model believes.

That distinction is the point. Measurement on the antibody-antigen w grid says
the model builds physically sound interfaces (no clashes, 87-90 % of the
native packing density, native side-chain involvement) on the *wrong* patch,
and that classifier-free guidance only shifts how often the sampler lands in
one basin or another. Amplifying a belief cannot fix a wrong belief; a
restraint does not need the belief to be right.

One ``--restraint`` value is ``SEL:SEL[@DIST[@LO..HI]]`` with the selector
syntax of :mod:`deepfold3.model.guidance` (``A/25-40+88:B@6``): every residue
named on the left must come within ``DIST`` A of the group on the right.

Relation to Boltz's steering potentials, from which the idea is taken: their
flat-bottom energy is linear in the violation, so the force has the same
magnitude at 1 A over the bound as at 10 A and the constraint rings rather
than settles; their soft-min sharpness decays to zero, which spreads the pull
over every pair of a group exactly when it should be committing to the closest
one; and the guidance is applied every fourth step. Here the penalty is
quadratic (the force falls off as the restraint is met), the soft-min sharpens
as the noise falls, and it applies on every step inside its sigma range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepfold3.model.guidance import ChainSel, _parse_sel, _sel_mask

if TYPE_CHECKING:
    import torch

DEFAULT_DISTANCE = 6.0  # A, as Boltz's pocket/contact constraints default to
MIN_DISTANCE, MAX_DISTANCE = 2.0, 30.0


@dataclass(frozen=True)
class RestraintSpec:
    """One parsed restraint value, before chain names are resolved.

    Attributes:
        groups: The two selectors; the left one's residues are restrained.
        distance: The flat bottom, in A.
        sigma_range: ``(lo, hi)`` in absolute sigma-hat, or ``None`` for every step.
        paired: Pair the selectors residue by residue, in the order written,
            instead of restraining each left residue to the right group.
        repel: Push apart instead of together: penalise distances BELOW
            ``distance`` rather than above it.
    """

    groups: tuple[ChainSel, ChainSel]
    distance: float = DEFAULT_DISTANCE
    sigma_range: tuple[float, float] | None = None
    paired: bool = False
    repel: bool = False


@dataclass(frozen=True)
class Restraint:
    """A resolved restraint the sampler can apply.

    Attributes:
        tokens_a: ``[N]`` bool, the residues that must make the contact.
        tokens_b: ``[N]`` bool, what they must touch.
        distance: The flat bottom, in A.
        sigma_range: ``(lo, hi)`` in absolute sigma-hat, or ``None``.
    """

    tokens_a: torch.Tensor
    tokens_b: torch.Tensor
    distance: float = DEFAULT_DISTANCE
    sigma_range: tuple[float, float] | None = None
    chain_a: torch.Tensor | None = None  # [N] bool: all of the left selector's chain
    chain_b: torch.Tensor | None = None  # [N] bool: all of the right selector's chain
    pair_index: tuple[tuple[int, int], ...] = ()  # token pairs, when paired
    repel: bool = False


def parse_restraint_spec(text: str, paired: bool = False, repel: bool = False) -> RestraintSpec:
    """Parse ``SEL:SEL[@DIST[@LO..HI]]``.

    Args:
        text: The option value.

    Returns:
        The parsed restraint.

    Raises:
        ValueError: If ``text`` does not follow the syntax.
    """
    from deepfold3.model.guidance import _parse_sigma_range

    fields = text.split("@")
    if len(fields) > 3:
        raise ValueError(f"expected SEL:SEL[@DIST[@LO..HI]], got {text!r}")
    sides = fields[0].split(":")
    if len(sides) != 2 or not all(sides):
        raise ValueError(f"expected two selectors like A/25-40:B, got {fields[0]!r}")
    a, b = (_parse_sel(s) for s in sides)
    if a.chain == b.chain:
        raise ValueError(f"a restraint needs two chains, got {fields[0]!r}")
    distance = float(fields[1]) if len(fields) > 1 else DEFAULT_DISTANCE
    if not MIN_DISTANCE <= distance <= MAX_DISTANCE:
        raise ValueError(f"restraint distance must be {MIN_DISTANCE:g}..{MAX_DISTANCE:g} A, got {distance:g}")
    sigma_range = _parse_sigma_range(fields[2]) if len(fields) > 2 else None
    if paired:
        na, nb = _count(a), _count(b)
        if not (na and nb):
            raise ValueError(f"a paired restraint needs residues on both sides, got {fields[0]!r}")
        if na != nb:
            raise ValueError(f"paired selectors must name the same number of residues, got {na} and {nb}")
    return RestraintSpec((a, b), distance, sigma_range, paired, repel)


def _count(sel: ChainSel) -> int:
    """How many residues a selector names; 0 for a whole chain."""
    return sum(hi - lo + 1 for lo, hi in sel.ranges)


def _residues(sel: ChainSel) -> list[int]:
    """The residue numbers a selector names, in the order written."""
    return [n for lo, hi in sel.ranges for n in range(lo, hi + 1)]


def describe_restraint(spec: RestraintSpec) -> str:
    """One line for the run log and the manifest."""
    a, b = spec.groups
    sigma = f" σ∈[{spec.sigma_range[0]:g}, {spec.sigma_range[1]:g}]" if spec.sigma_range else ""
    op = "≥" if spec.repel else "≤"
    kind = f" ({_count(a)} pairs)" if spec.paired else ""
    return f"{a}:{b} {op} {spec.distance:g} Å{kind}{sigma}"


def build_restraints(
    specs: Sequence[RestraintSpec],
    asym_id: torch.Tensor,  # [N]
    chain_index: Mapping[str, int],
    n_tokens: int | None = None,
    residue_index: torch.Tensor | None = None,  # [N]
) -> tuple[Restraint, ...]:
    """Resolve the selectors into token masks.

    Args:
        specs: The parsed restraints.
        asym_id: Token chain ids, possibly bucket-padded.
        chain_index: Chain name to ``asym_id`` for this input.
        n_tokens: Tokens before bucket padding; padding is never restrained.
        residue_index: Token residue numbers, for selectors that name residues.

    Returns:
        One resolved restraint per spec, in order.

    Raises:
        ValueError: If a selector names something the input does not have.
    """
    out = []
    for spec in specs:
        a, b = (_sel_mask(sel, asym_id, chain_index, residue_index, n_tokens) for sel in spec.groups)
        # The whole chain of each side, for the rigid mode: the restraint names
        # residues, but what moves as one body is the chain they sit on.
        whole = [_sel_mask(ChainSel(sel.chain), asym_id, chain_index, None, n_tokens) for sel in spec.groups]
        pair_index: tuple[tuple[int, int], ...] = ()
        if spec.paired:
            pair_index = _pair_tokens(spec, asym_id, chain_index, residue_index, n_tokens)
        out.append(Restraint(a, b, spec.distance, spec.sigma_range, whole[0], whole[1], pair_index, spec.repel))
    return tuple(out)


def _pair_tokens(
    spec: RestraintSpec,
    asym_id: torch.Tensor,
    chain_index: Mapping[str, int],
    residue_index: torch.Tensor | None,
    n_tokens: int | None,
) -> tuple[tuple[int, int], ...]:
    """Token index for each written residue, zipped into pairs.

    A paired restraint says which residue touches which, so the two selectors
    are read in the order written and matched position by position. A residue
    the input does not have is an error rather than a silent shift of the rest
    of the pairing — getting the k-th pair wrong would mis-state every pair
    after it.
    """
    import torch

    if residue_index is None:
        raise ValueError("a paired restraint needs the input's residue_index, got none")
    picked = []
    for sel in spec.groups:
        if sel.chain not in chain_index:
            raise ValueError(f"unknown chain {sel.chain!r}, this input has {sorted(chain_index)}")
        in_chain = asym_id == chain_index[sel.chain]
        if n_tokens is not None:
            in_chain = in_chain.clone()
            in_chain[n_tokens:] = False
        idx = []
        for num in _residues(sel):
            hit = torch.nonzero(in_chain & (residue_index == num), as_tuple=False).flatten()
            if not hit.numel():
                raise ValueError(f"{sel.chain}/{num} is not a residue of this input")
            idx.append(int(hit[0]))  # a residue is one token unless it was atomised
        picked.append(idx)
    return tuple(zip(*picked, strict=True))


def _soft_min(d: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Softmax-weighted mean of ``d`` along its last axis.

    Lies between the minimum and the mean, tends to the minimum as
    ``sharpness`` grows, and — unlike ``-logsumexp(-λd)/λ`` — does not shift
    with the number of entries, so a restraint on a large group means the same
    thing as one on a small group.
    """
    import torch

    w = torch.softmax(-sharpness * d, dim=-1)
    return (w * d).sum(-1)


def restraint_gradient(
    positions: torch.Tensor,  # [..., N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    restraints: Sequence[Restraint],
    sharpness: float,
) -> torch.Tensor:
    """``dU/dx`` of the flat-bottom penalty, same shape as ``positions``.

    For every restrained residue i, ``d_i`` is a soft minimum over the
    distances from its atoms to the atoms of the other group, and

        U = Σ_i max(0, d_i − distance)²

    so a residue already inside the bound contributes nothing and the force on
    a violating one falls off as it comes in. ``sharpness`` is the soft-min's
    lambda in 1/A: the larger it is, the more the pull concentrates on the
    closest atom pair rather than on the group's centre.

    The gradient is taken with autograd on a copy, so this is safe to call
    from the sampler's ``no_grad`` block — and from ``inference_mode``, which
    needs more than ``enable_grad``: a tensor made under inference mode records
    no graph even inside ``enable_grad`` (``requires_grad_(True)`` succeeds and
    the energy still comes back with ``requires_grad=False``), so the copy is
    made under ``inference_mode(False)``. Without that the restraint is a
    silent no-op, which is exactly how it first shipped.
    """
    import torch

    if not restraints:
        return torch.zeros_like(positions)
    with torch.inference_mode(False):
        x = torch.empty_like(positions).copy_(positions).requires_grad_(True)
    with torch.inference_mode(False), torch.enable_grad():
        energy = x.new_zeros(())
        flat = x.reshape(*x.shape[:-3], -1, 3)
        m = x.shape[-2]
        flat_mask = atom_mask.reshape(-1)
        for r in restraints:
            if r.pair_index:
                # Residue to residue: the distance between THIS pair, not to
                # the other group. A group restraint says "be near that chain",
                # which a wrong pose satisfies — on 7uvs half the native
                # epitope sits inside 6 Å of the antibody while fnat is 0.000.
                # Naming the partner is what fixes the orientation, and three
                # well-separated pairs already pin all six rigid freedoms.
                dists = []
                for ta, tb in r.pair_index:
                    ra = flat_mask.new_zeros(flat_mask.shape)
                    rb = flat_mask.new_zeros(flat_mask.shape)
                    ra[ta * m : (ta + 1) * m] = atom_mask[ta]
                    rb[tb * m : (tb + 1) * m] = atom_mask[tb]
                    if not (bool(ra.any()) and bool(rb.any())):
                        continue
                    d = torch.cdist(flat[..., ra, :], flat[..., rb, :])
                    dists.append(_soft_min(_soft_min(d, sharpness), sharpness))
                if not dists:
                    continue
                dist = torch.stack(dists, dim=-1)
            else:
                rows = (r.tokens_a.unsqueeze(-1) & atom_mask).reshape(-1)  # [N*M]
                cols = (r.tokens_b.unsqueeze(-1) & atom_mask).reshape(-1)
                if not (bool(rows.any()) and bool(cols.any())):
                    continue
                a = flat[..., rows, :]  # [..., A, 3]
                b = flat[..., cols, :]  # [..., B, 3]
                d = torch.cdist(a, b)  # [..., A, B]
                # Soft minimum over the other group, then over each residue's
                # atoms. A softmax-weighted mean of the distances, NOT
                # -logsumexp(-λd)/λ: the latter sits below the true minimum by
                # ln(count)/λ, so a group of many atoms would read as satisfied
                # while it is still ln(count)/λ too far away, and the bias would
                # grow with the selection's size.
                token_of_row = torch.nonzero(rows, as_tuple=False).squeeze(-1) // m
                per_atom = _soft_min(d, sharpness)  # [..., A]
                per_res = [_soft_min(per_atom[..., token_of_row == t], sharpness) for t in torch.unique(token_of_row)]
                dist = torch.stack(per_res, dim=-1)  # [..., R]
            # Repel penalises coming closer than the bound; the default
            # penalises staying further than it.
            gap = (r.distance - dist) if r.repel else (dist - r.distance)
            energy = energy + (torch.clamp(gap, min=0.0) ** 2).sum()
        if not energy.requires_grad:
            # Every restraint is already satisfied, so the clamp cut the graph.
            return torch.zeros_like(positions)
        (grad,) = torch.autograd.grad(energy, x)
    return grad


def rigid_displacement(
    positions: torch.Tensor,  # [..., N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    restraints: Sequence[Restraint],
    weight: float,
    sharpness: float,
    max_step: float,
) -> torch.Tensor:
    """Displacement that moves each restrained chain as a rigid body.

    The per-atom gradient is projected onto the six rigid degrees of freedom of
    the chain it acts on, and the chain is translated and rotated by the
    result. This is geometry, not dynamics: there is no mass, no velocity and
    nothing is integrated. Each atom has a displacement it would like to make,
    ``u_i = -weight/2 * dU/dx_i``, and the projection is the least-squares fit
    of that field to a rigid motion,

        min over (t, omega) of  sum_i |t + omega x r_i - u_i|^2

    whose normal equations give ``t = mean(u)`` and
    ``(sum_i |r_i|^2 I - r_i r_i^T) omega = sum_i r_i x u_i``. The matrix on
    the left has the form of an inertia tensor of unit masses, but it is the
    Gram matrix of the rotation generators; the names below say force and
    torque because the formulas coincide, not because anything is accelerating.
    With no momentum in the loop, nothing overshoots on its own: instability
    could only come from the step size, which is why the penalty is quadratic
    and the step is clipped. Internal geometry is untouched by
    construction, which the per-atom pull does not promise: steering a subset
    of residues stretches the chain at the edge of the selection, and on an
    8weo sample whose CDRs started 19 A away it pulled two peptide bonds out to
    2.99 A. The sampler can absorb a rigid move, since the next denoiser call
    re-predicts everything from the moved coordinates; it cannot always repair
    a stretched bond.

    The rotation is applied as a rotation, not as its linearisation, so the
    move stays exactly rigid however large the angle. Both the translation and
    the largest displacement the rotation produces are clipped to ``max_step``
    angstroms per call — steering needs that clip: the force is unbounded in
    the violation, and one big step early leaves the trajectory somewhere the
    denoiser has never seen.

    Returns:
        ``positions``-shaped displacement to ADD to the coordinates.
    """
    import torch

    grad = restraint_gradient(positions, atom_mask, restraints, sharpness)
    out = torch.zeros_like(positions)
    if float(grad.abs().max()) == 0.0:
        return out
    flat_pos = positions.reshape(*positions.shape[:-3], -1, 3)
    flat_grad = grad.reshape(*grad.shape[:-3], -1, 3)
    flat_out = out.reshape(*out.shape[:-3], -1, 3)

    seen: set[tuple] = set()
    for restraint in restraints:
        for chain in (restraint.chain_a, restraint.chain_b):
            if chain is None:
                continue
            sel = (chain.unsqueeze(-1) & atom_mask).reshape(-1)
            key = tuple(sel.nonzero(as_tuple=False).flatten().tolist()[:1]) + (int(sel.sum()),)
            if key in seen or not bool(sel.any()):
                continue
            seen.add(key)
            p = flat_pos[..., sel, :]  # [..., K, 3]
            g = flat_grad[..., sel, :]
            centre = p.mean(-2, keepdim=True)
            r = p - centre
            # Least-squares rigid fit of the desired displacement field -g.
            # Normalised by the atoms the restraint actually pulls on, not by
            # the whole chain. A least-squares rigid fit would divide by every
            # atom, which dilutes a 22-residue selection on a 126-residue chain
            # about six-fold: the move then stays a couple of angstroms while
            # the denoiser puts the chain back, and the restraint reads as
            # inert. Dividing by the pulled atoms means "move the body by what
            # those residues are asking for", and weight means the same thing
            # in both modes.
            pulled = (g.abs().sum(-1) > 0).sum(-1).clamp(min=1).unsqueeze(-1)  # [..., 1]
            force = (-g).sum(-2) / pulled  # the fit's translation
            torque = torch.cross(r, -g, dim=-1).sum(-2)  # sum r x u, the fit's rhs
            # The rotation solves the normal equation above; its matrix is an
            # inertia tensor only by coincidence of form. Scaled like the
            # translation so a patch can turn the body it sits on.
            eye = torch.eye(3, dtype=p.dtype, device=p.device)
            inertia = (r.pow(2).sum(-1)[..., None, None] * eye) - r.unsqueeze(-1) * r.unsqueeze(-2)
            inertia = inertia.sum(-3) + 1e-6 * eye
            omega = torch.linalg.solve(inertia, torque.unsqueeze(-1)).squeeze(-1)  # [..., 3]
            omega = omega * (p.shape[-2] / pulled)

            step = 0.5 * weight
            t = force * step
            omega = omega * step
            # Clip: the translation, and the arc the rotation sweeps at the
            # atom furthest from the centroid.
            radius = r.norm(dim=-1).amax(-1, keepdim=True)  # [..., 1]
            t = t * (max_step / t.norm(dim=-1, keepdim=True).clamp(min=max_step))
            angle = omega.norm(dim=-1, keepdim=True)
            cap = (max_step / radius.clamp(min=1e-6)).clamp(max=float("inf"))
            omega = omega * (cap / angle.clamp(min=cap))
            angle = omega.norm(dim=-1, keepdim=True)

            axis = omega / angle.clamp(min=1e-12)
            cos, sin = torch.cos(angle).unsqueeze(-2), torch.sin(angle).unsqueeze(-2)
            axis_e = axis.unsqueeze(-2)
            rotated = (  # Rodrigues
                r * cos
                + torch.cross(axis_e.expand_as(r), r, dim=-1) * sin
                + axis_e * (axis_e * r).sum(-1, keepdim=True) * (1 - cos)
            )
            flat_out[..., sel, :] = rotated - r + t.unsqueeze(-2)
    return out


def active_restraints(restraints: Sequence[Restraint], sigma_hat: float) -> tuple[Restraint, ...]:
    """The restraints whose sigma range covers this step."""
    return tuple(r for r in restraints if r.sigma_range is None or r.sigma_range[0] <= sigma_hat <= r.sigma_range[1])
