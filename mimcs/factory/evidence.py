"""Normalize heterogeneous "earlier sampling results" into one :class:`Evidence`.

The sampler factory accepts results in several forms --- raw samples, a testing
``SamplerOutput``, a live :class:`~mimcs.samplers.base.BaseSampler`, or an explicit
``(samples, coordinates, gradients[, discrete])`` bundle (tuple or dict) --- and the heuristic rules
want a single uniform view of them. :func:`normalize` funnels every accepted form into an
:class:`Evidence` by ``isinstance``/duck-type dispatch; any field that a particular input
does not carry stays ``None``. A live sampler additionally yields the *coordinates* and
*scores* of its draws, so a prior run can drive the metric-regression rule: coordinates are
always recomputed from the ambient samples (cheap); scores come from the sampler's **saved**
gradients (on by default --- the score is already cached each step, so saving is nearly free)
or, if it did not save them, are recomputed here --- a vmapped gradient pass that
``recompute_gradients=False`` skips for a very expensive model. Multiple results merge
(samples/coordinates/gradients/discrete are concatenated, diagnostics filled field-wise). No
results at all yields an empty ``Evidence``, which drives the default sampler.

A model with integer parameters also carries its **labels** (``Evidence.discrete``), and that is
not bookkeeping: its coordinate-space density is conditional on them, so a recomputed score has
no meaning without the row's own ``z``. The labels are what lets the draws be warm-started as a
whole state rather than as a position paired with the wrong assignment (doc 14).

See ``docs/design/09_sampler_factory.md`` for the normalization table.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from .._chunked import map_rows
from .._logging import get_logger

log = get_logger(__name__)


@dataclass
class Diagnostics:
    """Run diagnostics extracted from a result (any field ``None`` when unavailable)."""

    divergence_count: int | None = None
    divergence_rate: float | None = None
    mean_tree_depth: float | None = None
    mean_refinements: float | None = None
    mean_n_leaves: float | None = None
    grad_evals: float | None = None
    accept_rate: float | None = None
    ess: np.ndarray | None = None
    warmup_step_sizes: np.ndarray | None = None


@dataclass
class Evidence:
    """Normalized earlier sampling results. Every field is ``None`` when not provided."""

    samples: np.ndarray | None = None       # (n, ambient_dim)
    coordinates: np.ndarray | None = None    # (n, coord_dim)
    gradients: np.ndarray | None = None      # (n, coord_dim), per-iteration score
    #: (n, discrete_dim) **integer** labels, row-aligned with ``coordinates``. A separate array
    #: rather than extra columns of ``samples`` for the reason the state keeps two: folding labels
    #: into the float matrix loses the one property that makes them labels. Needed because the
    #: coordinate-space score of a discrete model is a *conditional* one --- it exists only at
    #: given labels --- so recomputing it needs the row's own ``z`` (doc 14).
    discrete: np.ndarray | None = None
    diagnostics: Diagnostics | None = None


def _shape(a) -> str:
    """``"(n, d)"`` or ``"none"`` --- what a log message wants to say about an evidence array."""
    return "none" if a is None else str(tuple(np.shape(a)))


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _as_2d_int(a) -> np.ndarray:
    """``_as_2d`` for the label block. Deliberately **not** routed through ``_as_2d``: a label is
    an index into a support, and a float round-trip is exactly the corruption the two-array state
    exists to prevent."""
    a = np.asarray(a, dtype=np.int32)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _merge_diag(a: Diagnostics | None, b: Diagnostics | None) -> Diagnostics | None:
    if a is None:
        return b
    if b is None:
        return a
    updates = {f.name: getattr(b, f.name) for f in fields(b)
               if getattr(b, f.name) is not None}
    return replace(a, **updates)


def _safe_call(fn):
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        log.debug("evidence: %s() failed; that diagnostic stays None",
                  getattr(fn, "__name__", fn), exc_info=True)
        return None


def _coordinates(model, sampler, samples):
    """Coordinates of the ambient ``samples`` under the sampler's final (frozen) chart state ---
    the regime the samples were drawn in. Coordinates are cheap **per row**, so they are always
    recomputed; the whole ``(n, coord_dim)`` map is not cheap, so it is done a chunk at a time.

    That also fixes a quieter defect. This used to end in ``np.asarray`` of the vmapped device
    array, which is **zero-copy** on the CPU backend: ``Evidence.coordinates`` was a non-owning
    view pinning the whole JAX buffer for as long as the evidence lived (``owns_data=False``,
    measured). :func:`~mimcs._chunked.map_rows` copies each chunk into a preallocated array, so the
    result owns its data and the device buffers are released as the loop advances.
    """
    import jax.numpy as jnp
    st = sampler.state
    chp, ci = st.chart_hyperparams, st.chart_indices
    itemsize = np.dtype(jnp.result_type(float)).itemsize
    return map_rows(lambda s: model.sample_to_coordinate(jnp.asarray(s, float), chp, ci),
                    np.asarray(samples), extra_row_bytes=model.coord_dim * itemsize)


def _recomputed_scores(model, sampler, coordinates, discrete=None):
    """Per-sample scores (gradient of the log-density in coordinate space) at ``coordinates``,
    under the sampler's frozen charts --- a one-time vmapped gradient pass. Used only when the
    sampler did not save its gradients (see :meth:`BaseSampler.get_gradients`) and recompute is
    allowed; for an expensive model this pass can be skipped (``recompute_gradients=False``).

    ``discrete`` supplies each row's own labels, which a model with integer parameters *requires*:
    its coordinate-space density is conditional on them, so there is no score to take without one.
    Omitting them used to raise inside ``Model._require_discrete``, which the caller swallowed ---
    leaving ``gradients=None`` and silently disabling the mass-mode and metric-regression rules.

    Chunked for the same reason as :func:`_coordinates`: one gradient per row is cheap, ``n`` of
    them at once is not. Both branches chunk identically --- the labels staying ``int32`` the whole
    way, since a float round-trip is exactly the corruption the two-array state exists to prevent.
    """
    import jax
    import jax.numpy as jnp
    st = sampler.state
    chp, ci = st.chart_hyperparams, st.chart_indices
    itemsize = np.dtype(jnp.result_type(float)).itemsize
    if discrete is None:
        score = jax.grad(lambda c: model.log_prob_at_coordinate(c, chp, ci))
        return map_rows(lambda c: score(jnp.asarray(c, float)), np.asarray(coordinates),
                        extra_row_bytes=model.coord_dim * itemsize)
    score = jax.grad(lambda c, z: model.log_prob_at_coordinate(c, chp, ci, z))
    return map_rows(lambda c, z: score(jnp.asarray(c, float), jnp.asarray(z, jnp.int32)),
                    np.asarray(coordinates), np.asarray(discrete, dtype=np.int32),
                    extra_row_bytes=model.coord_dim * itemsize)


def _saved_gradients(sampler, n):
    """The sampler's saved per-sample scores, if present and aligned with the ``n`` samples."""
    g = _safe_call(getattr(sampler, "get_gradients", None))
    if g is not None and len(g) == n:
        return np.asarray(g, dtype=float)
    return None


def _from_sampler(model, sampler, recompute_gradients: bool = True) -> dict:
    """Pull samples, coordinates, scores, and diagnostics from a live sampler.

    The dimension-match guard ensures the sampler ran on the same target. Coordinates are always
    recomputed from the ambient samples (cheap). Scores are taken from the sampler's *saved*
    gradients when available (the default --- see ``save_gradients``); otherwise they are
    recomputed here so a prior run still drives the metric-regression rule, unless
    ``recompute_gradients`` is ``False`` (skip the potentially expensive gradient pass). Anything
    that fails degrades to what could be obtained.
    """
    # ``summary_model``, not ``model``: it is defined as the model the **retained** draws belong
    # to, which is exactly what this function reads. They differ only for a sampler that narrows
    # what it keeps --- parallel tempering runs on the K-fold product but retains the cold chain
    # (``ProductSpaceMixin``), whose ``get_samples_flat`` / ``get_discrete_flat`` / ``get_gradients``
    # are already narrowed. Reading ``model`` here compared the *product* dimensions against the
    # base model's and rejected every tempered run as "a different target", K times too wide.
    sm = sampler.summary_model
    # ``discrete_dim`` joins the guard: two models can agree on every continuous dimension and
    # still be different targets, and the labels are what the score is conditional on.
    if (sm.coord_dim != model.coord_dim or sm.ambient_dim != model.ambient_dim
            or getattr(sm, "discrete_dim", 0) != getattr(model, "discrete_dim", 0)):
        raise ValueError(
            "make_sampler: the provided sampler was run on a model with different "
            f"dimensions (coord {sm.coord_dim} vs {model.coord_dim}, "
            f"ambient {sm.ambient_dim} vs {model.ambient_dim}, "
            f"discrete {getattr(sm, 'discrete_dim', 0)} vs {getattr(model, 'discrete_dim', 0)})")
    samples = sampler.get_samples_flat()
    if samples is not None and len(samples) == 0:
        samples = None
    coordinates = gradients = discrete = None
    if samples is not None and getattr(model, "discrete_dim", 0):
        z = _safe_call(getattr(sampler, "get_discrete_flat", None))
        if z is not None and len(z) == len(samples):
            discrete = np.asarray(z, dtype=np.int32)
    if samples is not None:
        try:
            coordinates = _coordinates(model, sampler, samples)
        except Exception:
            log.warning("evidence: could not map the sampler's %d draw(s) back to coordinates; "
                        "the evidence-based rules will not fire", len(samples), exc_info=True)
            coordinates = None
        gradients = _saved_gradients(sampler, len(samples))
        if gradients is None and recompute_gradients and coordinates is not None:
            log.debug("evidence: the sampler saved no gradients; recomputing %d score(s)",
                      len(coordinates))
            try:
                gradients = _recomputed_scores(model, sampler, coordinates, discrete)
            except Exception:
                log.warning("evidence: recomputing the scores failed; the metric-regression and "
                            "mass-mode rules will not fire", exc_info=True)
                gradients = None
    diag = Diagnostics(
        accept_rate=_safe_call(getattr(sampler, "acceptance_rate", None)),
        warmup_step_sizes=_safe_call(getattr(sampler, "warmup_step_sizes", None)),
        divergence_count=_safe_call(getattr(sampler, "divergence_count", None)),
        divergence_rate=_safe_call(getattr(sampler, "divergence_rate", None)),
        mean_tree_depth=_safe_call(getattr(sampler, "mean_tree_depth", None)),
        mean_refinements=_safe_call(getattr(sampler, "mean_refinements", None)),
        mean_n_leaves=_safe_call(getattr(sampler, "mean_n_leaves", None)),
        grad_evals=_safe_call(getattr(sampler, "total_grad_evals", None)))
    return {"samples": samples, "coordinates": coordinates, "gradients": gradients,
            "discrete": discrete, "diagnostics": diag}


def normalize(model, *results, recompute_gradients: bool = True) -> Evidence:
    """Collapse ``*results`` into one :class:`Evidence` (see module docstring).

    ``recompute_gradients`` (default ``True``) controls only the fallback for a live sampler
    that did not *save* its gradients: recompute them (so the metric-regression rule can fire)
    or leave them out (skip the gradient pass, for a very expensive model).
    """
    from ..samplers import BaseSampler

    samples_parts: list = []
    coord_parts: list = []
    grad_parts: list = []
    discrete_parts: list = []
    diag: Diagnostics | None = None

    def add(*, samples=None, coordinates=None, gradients=None, discrete=None, diagnostics=None,
            owned: bool = False):
        """``owned`` says these arrays were built by this module and are aliased by nobody, which
        is what lets :func:`cat` skip its copy for them. Only the live-sampler branch can claim
        it: everything else here is handed in by a caller."""
        nonlocal diag
        if samples is not None:
            samples_parts.append((_as_2d(samples), owned))
        if coordinates is not None:
            coord_parts.append((_as_2d(coordinates), owned))
        if gradients is not None:
            grad_parts.append((_as_2d(gradients), owned))
        if discrete is not None:
            discrete_parts.append((_as_2d_int(discrete), owned))
        if diagnostics is not None:
            diag = _merge_diag(diag, diagnostics)

    for r in results:
        if r is None:
            continue
        if isinstance(r, np.ndarray):
            add(samples=r)
        elif isinstance(r, BaseSampler):
            add(**_from_sampler(model, r, recompute_gradients=recompute_gradients), owned=True)
        elif hasattr(r, "samples") and hasattr(r, "ess"):
            # a testing SamplerOutput (duck-typed to avoid a production -> testing dep)
            add(samples=getattr(r, "samples"),
                diagnostics=Diagnostics(
                    accept_rate=getattr(r, "accept_rate", None),
                    ess=getattr(r, "ess", None),
                    warmup_step_sizes=getattr(r, "warmup_step_sizes", None)))
        elif isinstance(r, dict):
            add(samples=r.get("samples"), coordinates=r.get("coordinates"),
                gradients=r.get("gradients"), discrete=r.get("discrete"))
        elif isinstance(r, (tuple, list)):
            # ``discrete`` is a fourth *optional* element: a 3-tuple still means exactly what it
            # always did, so no existing caller changes.
            keys = ("samples", "coordinates", "gradients", "discrete")
            add(**{k: v for k, v in zip(keys, r) if v is not None})
        else:
            raise TypeError(
                f"make_sampler: cannot interpret a result of type {type(r).__name__!r} "
                "(expected a samples array, a SamplerOutput, a sampler, or a "
                "(samples, coordinates, gradients[, discrete]) tuple/dict)")

    def cat(parts):
        # A single part is the common case and ``np.concatenate`` merely copies it. That copy is
        # load-bearing for a **caller's** array: ``_as_2d`` returns an already-float64 one
        # *unchanged*, so skipping it would leave the evidence aliasing something the caller may
        # still mutate. An ownership guard cannot tell the two cases apart --- a caller's own array
        # owns its data too --- which is why provenance is threaded through ``add`` instead. A part
        # this module built (the live-sampler branch: a fresh ``np.stack``, a fresh ``map_rows``)
        # is aliased by nobody, so its copy is pure waste: 320 MB of it on a 6000-draw run.
        if not parts:
            return None
        if len(parts) == 1 and parts[0][1]:
            return parts[0][0]
        return np.concatenate([p for p, _ in parts], axis=0)

    evidence = Evidence(samples=cat(samples_parts), coordinates=cat(coord_parts),
                        gradients=cat(grad_parts), discrete=cat(discrete_parts),
                        diagnostics=diag)
    log.debug("normalized %d result(s) into evidence: samples %s, coordinates %s, gradients %s, "
              "discrete %s, diagnostics %s", len(results), _shape(evidence.samples),
              _shape(evidence.coordinates), _shape(evidence.gradients),
              _shape(evidence.discrete), "yes" if diag is not None else "no")
    return evidence
