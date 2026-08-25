"""Heuristic rules and the weighted arbiter (the "grab-bag", made principled).

A rule reads what it needs from the :class:`~mimcs.factory.evidence.Evidence` and emits zero
or more :class:`Proposal` objects, each targeting a named *slot* of the spec with a *weight*
(the rule's confidence). The :func:`arbitrate` step groups proposals by slot and lets the
highest-weight proposal win (the spec's current value is the weight-0 baseline); ties break
by registration order, so analysis is reproducible. Both winners and the losers they beat
are recorded in ``spec.rationale``. Conflicts thus resolve by weight, not by rule order, and
new rules can be added without any rule knowing about the others.

The main structural rule is :func:`block_partition_rule`. See
``docs/design/09_sampler_factory.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

#: block size at/below which a dense (within-block) mass is feasible to adapt; above -> low-rank.
DENSE_MAX_DIM = 50
#: block size (above DENSE_MAX_DIM) up to which a low-rank mass is preferred over diagonal; above
#: -> diagonal. PLACEHOLDER --- a rough guess with no evidence yet, pending closer investigation.
LOWRANK_MAX_DIM = 1000
#: default number of low-rank directions J for a lowrank block. PLACEHOLDER --- pending evidence.
LOWRANK_DEFAULT_RANK = 8
#: low-dimensional parameters (dim <= this) are fused, and a fused block is *capped* at this size
#: (never grown past it). The cap keeps every fused block small enough that a single-parameter block
#: stays eligible for a learned metric (the learned-metric rule fuses nothing past the cap).
FUSE_MIN_DIM = 20
#: minimum evidence rows before the metric-regression rule will run.
LEARNED_METRIC_MIN_ROWS = 200
#: minimum evidence rows before the evidence-based mass-mode rule will run.
MODE_SELECT_MIN_ROWS = 50
#: if more than this fraction of the pilot's transitions diverged, its scores are too pathological
#: (heavy-tailed) to trust for evidence-based mass-mode / metric-shape selection: skip it and keep
#: the dimension-count defaults (constant blocks) / a plain diagonal metric (learned blocks). A
#: coarse safety valve pending a robust/trimmed score-covariance estimator (see docs/design/09).
MODE_SELECT_MAX_DIVERGENCE_RATE = 0.10
#: how much lower a candidate's AIC must be than the constant baseline's to adopt it.
LEARNED_METRIC_AIC_MARGIN = 2.0
#: cheap sub-steps per expensive gradient for the multi-rate (RESPA) integrator. PLACEHOLDER ---
#: doc 06's example value, with no evidence yet; the modest rule weight lets a measured rule win.
MULTIRATE_DEFAULT_N = 4


@dataclass
class Proposal:
    """A weighted vote that a spec slot should take ``value``."""

    slot: str
    value: object
    weight: float
    reason: str
    rule: str = ""


# --- slot addressing -------------------------------------------------------- #

_BLOCK_RE = re.compile(r"blocks\[(\d+)\]\.(\w+)")


def get_slot(spec, slot: str):
    m = _BLOCK_RE.fullmatch(slot)
    if m:
        return getattr(spec.blocks[int(m.group(1))], m.group(2))
    return getattr(spec, slot)


def set_slot(spec, slot: str, value) -> None:
    m = _BLOCK_RE.fullmatch(slot)
    if m:
        setattr(spec.blocks[int(m.group(1))], m.group(2), value)
    else:
        setattr(spec, slot, value)


# --- arbitration ------------------------------------------------------------ #

#: longest value repr a log line will carry (fitted metric parameters are whole arrays).
_MAX_VALUE_REPR = 160


def _fmt_value(value) -> str:
    """A compact one-line repr of a proposal's value for a log message.

    A block list is rendered as its ``names[dim,kind]`` summary rather than a page of
    dataclasses, and anything else is truncated --- a fitted ``metric_init`` is an array pytree.
    """
    if isinstance(value, list) and value and hasattr(value[0], "coord_slices"):
        return "[" + ", ".join(str(b) for b in value) + "]"    # BlockSpec.__str__
    text = " ".join(repr(value).split())
    return text if len(text) <= _MAX_VALUE_REPR else text[:_MAX_VALUE_REPR - 3] + "..."


def arbitrate(spec, proposals: list[Proposal]):
    """Resolve ``proposals`` into ``spec``, recording the reasoning in ``spec.rationale``.

    Each adopted proposal is logged at INFO (what the factory chose, and why) and each proposal
    it beat at DEBUG (what was rejected, and by how much weight) --- the same content as
    ``spec.rationale``, available without holding on to the spec.
    """
    by_slot: dict[str, list[Proposal]] = {}
    for p in proposals:
        by_slot.setdefault(p.slot, []).append(p)

    for slot, props in by_slot.items():
        winner = max(props, key=lambda p: p.weight)   # first max wins -> order tie-break
        if winner.weight <= 0:
            log.debug("rejected %s = %s [weight %.2f, %s: %s]: no proposal beats the spec's "
                      "current value", slot, _fmt_value(winner.value), winner.weight,
                      winner.rule, winner.reason)
            continue
        set_slot(spec, slot, winner.value)
        log.info("chose %s = %s [weight %.2f, %s: %s]",
                 slot, _fmt_value(winner.value), winner.weight, winner.rule, winner.reason)
        for p in props:
            if p is not winner:
                log.debug("rejected %s = %s [weight %.2f, %s: %s]: beaten by %s (weight %.2f)",
                          slot, _fmt_value(p.value), p.weight, p.rule, p.reason,
                          winner.rule, winner.weight)
        # Through ``_fmt_value``, like the log line above: a raw ``!r`` here put whole fitted
        # metric arrays into ``spec.rationale``, which the reference doc tells users to print.
        line = (f"{slot} = {_fmt_value(winner.value)} [weight {winner.weight:.2f}, "
                f"{winner.rule}: {winner.reason}]")
        losers = [p for p in props if p is not winner]
        if losers:
            line += " | over " + ", ".join(
                f"{_fmt_value(p.value)} [weight {p.weight:.2f}, {p.rule}: {p.reason}]"
                for p in losers)
        spec.rationale.append(line)
    return spec


# --- the rules -------------------------------------------------------------- #

def _merge_slices(slices):
    """Coalesce adjacent ``(s, e)`` slices, so a contiguous fusion is a single slice while a
    genuinely scattered one keeps several."""
    out: list = []
    for s, e in sorted(slices):
        if out and s == out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _kind_for(dim: int) -> str:
    """The mass kind a block of this many coordinates gets, before any refinement rule.

    One coordinate has no correlations to model, and a dense mass above ``DENSE_MAX_DIM`` is not
    worth adapting (``lowrank_block_rule`` upgrades those); everything between is dense.
    """
    if dim > DENSE_MAX_DIM or dim <= 1:
        return "diagonal"
    return "dense"


def normalize_block_override(groups, model) -> list[tuple]:
    """Validate a user-supplied block partition and put each group in declaration order.

    ``groups`` is a list whose entries are parameter names or iterables of them --- ``"mu"`` is
    shorthand for ``("mu",)``. Each entry becomes one block; parameters left unnamed are
    partitioned by the usual rule, so a partial specification is the normal case.

    Groups are sorted into **model declaration order** because ``block.names`` is not just a
    label: :func:`_block_columns` and the ``__``-joined kinetic id in
    :func:`mimcs.factory.build._block_kinetic` both follow it, and the default rule builds fused
    blocks in declaration order. A forced block should be indistinguishable from one the rule
    could have produced.

    Raises:
        ValueError: on an unknown parameter name, a name in two groups, or an empty group.
    """
    known = {p.name: i for i, p in enumerate(model.parameters)}
    seen: dict[str, int] = {}
    out: list[tuple] = []
    for position, group in enumerate(groups):
        names = [group] if isinstance(group, str) else list(group)
        if not names:
            raise ValueError(
                f"blocks[{position}] is empty; every block needs at least one parameter")
        for name in names:
            if name not in known:
                raise ValueError(
                    f"blocks[{position}] names {name!r}, which is not a parameter of this model "
                    f"(it has {sorted(known)})")
            if name in seen:
                raise ValueError(
                    f"blocks[{position}] names {name!r}, which is already in blocks[{seen[name]}]; "
                    f"a parameter belongs to exactly one block")
            seen[name] = position
        out.append(tuple(sorted(names, key=known.__getitem__)))
    return out


def block_partition_rule(spec, evidence, model) -> list[Proposal]:
    """Partition the parameters into coordinate blocks and pick each block's mass kind.

    * a parameter of dimension ``> DENSE_MAX_DIM`` (50) is its own **diagonal** block (a dense
      mass that large is not worth adapting) --- which :func:`lowrank_block_rule` then upgrades to
      a low-rank mass up to ``LOWRANK_MAX_DIM``;
    * a parameter of dimension in ``(FUSE_MIN_DIM, DENSE_MAX_DIM]`` (21..50) is its own
      **dense** block (already dense-worthy);
    * the low-dimensional (``<= 20``) parameters are **fused** --- in declaration order but
      regardless of coordinate adjacency (each fused block carries a *list of slices*) --- into
      **dense** blocks, each **capped** at ``FUSE_MIN_DIM``: a parameter joins the current block
      only if it fits within the cap, else the block is flushed and a new one started.

    The point (hierarchical models): the correlations that matter most for a mass matrix are
    among the naturally few, high-level parameters --- which are often declared *separately*
    (e.g. a mean and a variance parameter, possibly with lower-level parameters between them in
    the coordinate vector) and so are individually low-dimensional. Fusing them into dense
    blocks captures those cross-parameter correlations even when the parameters are not adjacent
    in the coordinate vector, while high-dimensional parameters stay diagonal.

    Why *cap* the fusion rather than fuse until just over the threshold: a fused multi-parameter
    block can never carry a learned (position-dependent) metric --- that rule works only on a
    single-parameter block. Letting fusion grow a block past the cap can swallow a whole vector
    parameter (e.g. a 20-dim ``a`` glued onto two scalars -> 22 dim), which then can never be given
    the funnel-whitening metric it needs. Capping trades a little cross-parameter correlation for
    keeping such parameters eligible; on hierarchical funnels the eligibility is worth far more.

    ``spec.block_override`` overrides all of that for the parameters it names: each group becomes
    one block, and the rule above runs on whatever is left --- so a partial specification is the
    normal case. The size heuristic is only a heuristic, and the override is how a user states the
    grouping their model actually needs, most often to *un*-fuse parameters so that a
    single-parameter block becomes eligible for a learned metric. Only the grouping is fixed: the
    refinement rules still choose each block's kind from the evidence.
    """
    from .spec import BlockSpec
    blocks: list = []
    low: list = []          # (name, (s, e)) low-dimensional params to fuse (maybe scattered)

    forced = list(getattr(spec, "block_override", None) or ())
    for group in forced:
        slices = [model.coord_block(name) for name in group]
        blocks.append(BlockSpec(list(group), _merge_slices(slices),
                                _kind_for(sum(e - s for s, e in slices))))
    claimed = {name for group in forced for name in group}

    for p in model.parameters:
        if p.name in claimed:
            continue
        s, e = model.coord_block(p.name)
        d = e - s
        if d > DENSE_MAX_DIM:
            blocks.append(BlockSpec([p.name], [(s, e)], "diagonal"))
        elif d > FUSE_MIN_DIM:
            blocks.append(BlockSpec([p.name], [(s, e)], "dense"))
        else:
            low.append((p.name, (s, e)))

    # Fuse the low-dimensional parameters into dense blocks, each capped at FUSE_MIN_DIM: a
    # parameter joins the current block only if it fits within the cap, else the current block is
    # flushed and a new one begun. (Every param here has dim <= FUSE_MIN_DIM, so it always fits a
    # fresh block.) A lone scalar block stays diagonal.
    names, slices, size = [], [], 0
    for name, sl in low:
        d = sl[1] - sl[0]
        if names and size + d > FUSE_MIN_DIM:        # would exceed the cap: flush, then start anew
            blocks.append(BlockSpec(list(names), _merge_slices(slices), _kind_for(size)))
            names, slices, size = [], [], 0
        names.append(name); slices.append(sl); size += d
    if names:               # leftover low-dim params (a lone scalar stays diagonal)
        blocks.append(BlockSpec(list(names), _merge_slices(slices), _kind_for(size)))

    blocks.sort(key=lambda b: b.coord_slices[0][0])   # coordinate order (deterministic)
    summary = ", ".join(str(b) for b in blocks)          # BlockSpec.__str__
    forced_note = ""
    if forced:
        forced_note = (f"; {len(forced)} block(s) fixed by the caller "
                       f"({', '.join('+'.join(g) for g in forced)})"
                       + (f", {len(model.parameters) - len(claimed)} param(s) left to the rule"
                          if len(claimed) < len(model.parameters) else ""))
    return [Proposal("blocks", blocks, 0.8,
                     f"{len(model.parameters)} params -> {len(blocks)} block(s): {summary}"
                     f"{forced_note}",
                     "block_partition")]


def _diverged_too_much(evidence) -> bool:
    """True when the evidence's run diverged on more than ``MODE_SELECT_MAX_DIVERGENCE_RATE`` of its
    transitions --- a signal its scores are too heavy-tailed for the Gaussian/Marchenko-Pastur null
    behind :func:`mimcs.factory.mode_select.select_mass_mode` to hold. Evidence without a divergence
    rate (raw samples, an explicit bundle) does not trip the gate."""
    diag = getattr(evidence, "diagnostics", None)
    rate = getattr(diag, "divergence_rate", None) if diag is not None else None
    return rate is not None and rate > MODE_SELECT_MAX_DIVERGENCE_RATE


def _block_columns(model, block) -> list[int]:
    """The block's coordinate columns, in parameter-declaration order (matching how a learned
    block gathers a dependency via ``_resolve_dep`` on the ``__``-joined name)."""
    cols: list[int] = []
    for name in block.names:
        s, e = model.coord_block(name)
        cols.extend(range(s, e))
    return cols


def learned_metric_rule(spec, evidence, model) -> list[Proposal]:
    """Regress candidate position-dependent metrics on the evidence and adopt the best fit.

    For each single-parameter block, fit the mini-language candidates (see
    :func:`mimcs.factory.regression.select_metric`) to that block's conditional score covariance
    over the *other* blocks' coordinates, and --- when the AIC-best candidate is genuinely
    position-dependent and beats the constant baseline by ``LEARNED_METRIC_AIC_MARGIN`` --- turn
    the block into a ``"learned_metric"`` kinetic, stashing the chosen expression and its fitted
    parameters in ``block.params`` (so it is inspectable and manually overridable on the spec).

    A refinement rule: it reads the *already-partitioned* ``spec.blocks``. Inert without
    row-aligned coordinate + gradient evidence, so evidence-free factory calls are unchanged.
    """
    ev = evidence
    if ev is None or ev.coordinates is None or ev.gradients is None:
        log.debug("learned_metric rule inert: no row-aligned coordinate + gradient evidence")
        return []
    if len(ev.coordinates) < LEARNED_METRIC_MIN_ROWS:
        log.debug("learned_metric rule inert: %d evidence row(s) < the %d it needs",
                  len(ev.coordinates), LEARNED_METRIC_MIN_ROWS)
        return []
    from .regression import select_metric, whitened_scores
    from .mode_select import select_mass_mode

    blocks = spec.blocks
    proposals: list[Proposal] = []
    for i, block in enumerate(blocks):
        # scope: only a single (contiguous) parameter can be a learned block for now
        if len(block.names) != 1 or len(block.coord_slices) != 1:
            log.debug("learned_metric: block %d ('%s') skipped --- a learned metric needs a "
                      "single contiguous parameter", i, "+".join(block.names))
            continue
        dep_cols = {"__".join(b.names): _block_columns(model, b)
                    for j, b in enumerate(blocks) if j != i}
        if not dep_cols:
            log.debug("learned_metric: block %d ('%s') skipped --- nothing for the metric to "
                      "depend on (it is the only block)", i, "+".join(block.names))
            continue
        ranked = select_metric(_block_columns(model, block), dep_cols,
                               ev.coordinates, ev.gradients)
        best = ranked[0]
        baseline = next(r for r in ranked if r.expr.deps() == set())
        if best.expr.deps() and best.aic < baseline.aic - LEARNED_METRIC_AIC_MARGIN:
            reason = (f"regressed {best.expr!r} on block '{block.names[0]}' score covariance "
                      f"(AIC {best.aic:.1f} < baseline {baseline.aic:.1f})")
            params = {"metric": best.expr, "metric_init": best.params}
            # Shape selection: choose the constant shape A -- diagonal (none) / low-rank(J) / dense --
            # from the D(x)^{-1/2}-whitened *conditional* scores' correlation (the same selector as
            # the plain mass mode). Only added when a shape beats a bare diagonal metric, and skipped
            # entirely on a too-divergent pilot (heavy-tailed scores over-select a spurious shape ---
            # the learned metric then stays a plain diagonal D(x)).
            if not _diverged_too_much(ev):
                used = {d: dep_cols[d] for d in best.expr.deps()}
                h = whitened_scores(best.expr, best.params, _block_columns(model, block), used,
                                    ev.coordinates, ev.gradients)
                skind, J = select_mass_mode(h)
                if skind != "diagonal":
                    shape = ("lowrank", J) if skind == "lowrank" else "dense"
                    params["shape"] = shape
                    reason += f"; shape {skind}" + (f" J={J}" if skind == "lowrank" else "")
            proposals.append(Proposal(f"blocks[{i}].kind", "learned_metric", 0.7, reason,
                                      "learned_metric"))
            proposals.append(Proposal(f"blocks[{i}].params", params, 0.7, reason, "learned_metric"))
        else:
            log.debug("learned_metric: block %d ('%s') keeps a constant mass --- best candidate "
                      "%r (AIC %.1f) does not beat the constant baseline (AIC %.1f) by the "
                      "%.1f margin", i, "+".join(block.names), best.expr, best.aic,
                      baseline.aic, LEARNED_METRIC_AIC_MARGIN)
    return proposals


def lowrank_block_rule(spec, evidence, model) -> list[Proposal]:
    """Default a mid-sized diagonal block to a low-rank mass (placeholder, dimension-count only).

    A parameter block too large for a dense mass (``> DENSE_MAX_DIM``) but still moderate
    (``<= LOWRANK_MAX_DIM``) becomes a :class:`~mimcs.hmc.LowRankQuadraticKinetic` with
    ``LOWRANK_DEFAULT_RANK`` directions --- a diagonal scale plus a few learned correlation
    directions, cheaper than a dense mass and better than a pure diagonal. Above
    ``LOWRANK_MAX_DIM`` a block stays diagonal.

    A refinement rule (reads the already-partitioned ``spec.blocks``) and evidence-free. The
    thresholds and the rank are rough placeholders with no evidence yet; the modest weight lets an
    evidence-based refinement (e.g. :func:`learned_metric_rule`) override it.
    """
    proposals: list[Proposal] = []
    for i, block in enumerate(spec.blocks):
        if block.kind != "diagonal":
            continue
        d = sum(e - s for s, e in block.coord_slices)
        if DENSE_MAX_DIM < d <= LOWRANK_MAX_DIM:
            reason = (f"block '{'+'.join(block.names)}' dim {d} in ({DENSE_MAX_DIM}, "
                      f"{LOWRANK_MAX_DIM}] -> low-rank mass (rank {LOWRANK_DEFAULT_RANK})")
            proposals.append(Proposal(f"blocks[{i}].kind", "lowrank", 0.6, reason, "lowrank_block"))
            proposals.append(Proposal(f"blocks[{i}].params", {"rank": LOWRANK_DEFAULT_RANK},
                                      0.6, reason, "lowrank_block"))
    return proposals


def mass_mode_rule(spec, evidence, model) -> list[Proposal]:
    """Choose each block's mass mode --- diagonal / low-rank(J) / dense --- from the **evidence**
    score covariance rather than dimension counting. The score covariance is the target precision
    (the ideal mass); its *whitened* spectrum says whether a diagonal is enough, a few directions
    are isolated above the Marchenko-Pastur bulk (low-rank, J of them), or the structure is dense
    (see :func:`mimcs.factory.mode_select.select_mass_mode`, default rule ``"aic"``).

    A refinement rule: reads the already-partitioned ``spec.blocks`` and needs row-aligned gradient
    evidence. Inert without it, so evidence-free calls keep the dimension-count placeholders
    (:func:`lowrank_block_rule`). Weight 0.65 --- above that placeholder, below
    :func:`learned_metric_rule` (0.7), so a block chosen for a learned metric keeps it.
    """
    ev = evidence
    if ev is None or ev.gradients is None or len(ev.gradients) < MODE_SELECT_MIN_ROWS:
        log.debug("mass_mode rule inert: %s",
                  "no gradient evidence" if ev is None or ev.gradients is None
                  else f"{len(ev.gradients)} evidence row(s) < the {MODE_SELECT_MIN_ROWS} it needs")
        return []
    if _diverged_too_much(ev):        # pathological pilot: keep the dimension-count defaults
        log.debug("mass_mode rule inert: the pilot diverged on %.1f%% of its transitions "
                  "(> %.1f%%), so its scores are too heavy-tailed to select a mass mode from",
                  100.0 * ev.diagnostics.divergence_rate, 100.0 * MODE_SELECT_MAX_DIVERGENCE_RATE)
        return []
    from .mode_select import select_mass_mode
    grads = np.asarray(ev.gradients, dtype=float)
    proposals: list[Proposal] = []
    for i, block in enumerate(spec.blocks):
        cols = _block_columns(model, block)
        kind, J = select_mass_mode(grads[:, cols])
        reason = (f"evidence spectrum of '{'+'.join(block.names)}' (d={len(cols)}, n={len(grads)}) "
                  f"-> {kind}" + (f" J={J}" if kind == "lowrank" else ""))
        proposals.append(Proposal(f"blocks[{i}].kind", kind, 0.65, reason, "mass_mode"))
        params = {"rank": J} if kind == "lowrank" else {}
        proposals.append(Proposal(f"blocks[{i}].params", params, 0.65, reason, "mass_mode"))
    return proposals


def multirate_integrator_rule(spec, evidence, model) -> list[Proposal]:
    """Use the multi-rate (RESPA) integrator when the model declares a real cheap/expensive split.

    The gate is **both** sides of the split, over *model components only*:

    * nothing cheap --- an undeclared model (the default for every hand-written one) has nothing
      to sub-step, so plain leapfrog stands. In particular the chart ``JacobianPotential`` is
      cheap by construction but is *not* a model component and can never be named in
      ``cheap_components``, so it can never trigger the multi-rate path on its own. That is the
      intent, not an accident: sub-stepping a lone Jacobian would change the dynamics of **every
      constrained model** for a stiffness gain we have not measured, while costing ``n`` extra
      Jacobian gradients and drifts per step. The Jacobian rides *along* with a split that a
      model component already justified (:func:`mimcs.hmc.split_potentials` puts it in the cheap
      group there).
    * nothing expensive --- a DSL program with no large data comes out all-cheap
      (:mod:`mimcs.dsl.cost`); there is nothing to sub-step *against*, so leapfrog stands.

    Structural and evidence-free: it reads the model's declared costs only.
    """
    cheap = sorted(model.cheap_components)
    expensive = sorted(model.expensive_components)
    if not cheap or not expensive:
        log.debug("multirate rule inert: the model declares %s cheap and %s expensive "
                  "component(s) --- a multi-rate split needs both",
                  cheap or "no", expensive or "no")
        return []
    reason = (f"components {cheap} are cheap, {expensive} expensive -> multi-rate leapfrog with "
              f"n={MULTIRATE_DEFAULT_N} cheap sub-steps per expensive gradient (the chart "
              f"Jacobian, if any, joins the cheap inner loop)")
    return [Proposal("integrator", "multirate", 0.6, reason, "multirate_integrator"),
            Proposal("integrator_params", {"n": MULTIRATE_DEFAULT_N}, 0.6, reason,
                     "multirate_integrator")]


#: Structural rules (run first, then arbitrated so refinement rules see the final partition).
#: ``multirate_integrator_rule`` writes slots disjoint from the partition's, so order is moot.
RULES = [block_partition_rule, multirate_integrator_rule]
#: Refinement rules, run against the already-partitioned spec (a second arbitration pass).
REFINEMENT_RULES = [lowrank_block_rule, mass_mode_rule, learned_metric_rule]


def analyze_proposals(spec, evidence, model, rules=None) -> list[Proposal]:
    proposals: list[Proposal] = []
    for rule in (RULES if rules is None else rules):
        emitted = rule(spec, evidence, model) or []
        log.debug("rule %s emitted %d proposal(s)", rule.__name__, len(emitted))
        proposals.extend(emitted)
    return proposals
