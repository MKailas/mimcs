"""Fit and compare mass-matrix mini-language expressions against sampling evidence.

Given row-aligned evidence (coordinates and per-iteration scores, i.e. potential gradients),
a block's *conditional score covariance* is estimated by fitting a
:class:`~mimcs.hmc.metric_expr.MetricExpr` --- a candidate diagonal metric ``M_i(q_{-i})`` ---
to minimise the batch KL loss

    mean_n  1/2 sum_d ( log M_i[d](coord_n) + g_{n,i,d}^2 / M_i[d](coord_n) ),

the same objective :class:`mimcs.adaptation.MetricAdaptation` descends online, whose minimiser
is ``E[g_i[d]^2 | q_{-i}]``. The offline fit uses the L-BFGS in :mod:`mimcs.optim`.

Candidate forms are enumerated dimension-aware (bounded parameter count, capped count,
simplest first) and compared by **AIC** (``2 k + 2 N * mean_loss``; the shared Gaussian-NLL
constant ``1/2 N d log 2*pi`` cancels between a block's candidates). AIC is a pragmatic first
choice: it scores goodness-of-fit against parameter count only.

NOTE (a known limitation of pure goodness-of-fit, for a future cost-aware criterion): a
*position-dependent* mass that departs exponentially from a good value degrades sampling
efficiency roughly exponentially, so a poorly-fit **unbounded** form (e.g. ``Exp("x")``, whose
linear exponent extrapolates without bound) is far costlier than a bounded one with a few more
parameters (e.g. ``Exp()*Sigmoid("x")``, gated into a finite range). AIC does not see this; it
should eventually be folded into the comparison.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .._chunked import CHUNK_BYTES, map_rows, sum_rows
from ..hmc.metric_encode import encode_discrete, encoded_width
from ..hmc.metric_expr import MetricExpr, Exp, Sigmoid, SpExp, SpSigmoid
from ..optim import minimize
from .._logging import get_logger

log = get_logger(__name__)

#: parameter budget for a block's candidates, as a multiple of the block's coordinate dim.
PARAM_BUDGET_MULT = 20
#: hard cap on the number of regressions (candidates fitted) per block --- a first-pass limit.
MAX_REGRESSIONS = 50
#: AIC penalty per parameter (2 is standard AIC).
AIC_PENALTY = 2.0
#: offer each position-dependent form **bare** as well as with the additive ``+ Exp()`` floor.
INCLUDE_BARE_CANDIDATES = True
#: score working set (``N * block_dim * itemsize``) above which :func:`fit_metric_expr` accumulates
#: its loss in chunks (:func:`mimcs._chunked.sum_rows`) instead of one whole-array ``vmap``.
#:
#: A **gate**, not a tuning knob, and the one number in this module that is not free to move
#: downwards. Chunked accumulation reorders the sum, so the two paths agree only to ~1e-13 on the
#: loss --- negligible against the ``1/N``-nats margin that decides bare-vs-floored candidates, but
#: not bit-identical. Set above the largest fit the test suite performs so the suite provably stays
#: on the whole-array path and no seed-pinned expectation can shift underneath it.
#:
#: **Measured, not guessed**: instrumenting every fit across the factory and metric test files (215
#: of them) puts the largest at **0.4 MB** (4000 rows x 25 coordinates; the horseshoe's, often
#: assumed to be the big one, is 0.3 MB over a 30-coordinate block). 64 MiB clears that by ~160x,
#: while the run that motivated this --- 6000 draws x 2000 coordinates, 91.6 MB --- is well above it.
#:
#: Distinct from :data:`mimcs._chunked.CHUNK_BYTES`, which is the *size* of a chunk once chunking
#: is on: this one has to be large to be safe, that one small to be useful.
CHUNK_LOSS_BYTES = 64 * 1024 * 1024


class MetricCandidate(NamedTuple):
    """One fitted candidate metric for a block."""

    expr: MetricExpr
    params: object          # fitted parameter pytree (in the expression's structure)
    loss: float             # achieved batch KL loss (mean over evidence rows)
    n_params: int
    aic: float              # 2 * n_params + 2 * N * loss (shared NLL constant dropped)


#: floor on the per-coordinate init scale (avoids ``log 0`` for a coordinate with no score signal).
INIT_SCALE_FLOOR = 1e-30


def typed_discrete(discrete_cols: dict | None, expr) -> dict:
    """Kind-free ``{name: (cols, lo, hi)}`` + an expression -> ``{name: (cols, kind, lo, hi)}``.

    Two tuple shapes exist on purpose. The **kind-free** one is the data: which columns hold a
    label and what its support is, facts about the model. The **typed** one is a data-plus-coding
    pair, and the coding is the *expression's* declaration --- so it is per candidate, and the same
    label is one column in one candidate and ``k - 1`` in another. Keeping them distinct is what
    lets one column map serve a whole pool.
    """
    out = {}
    for d, (cols, lo, hi) in (discrete_cols or {}).items():
        kind = expr.dep_kind(d) if expr is not None else None
        if kind is not None:
            out[d] = (cols, kind, lo, hi)
    return out


def typed_dims(dep_cols: dict, typed: dict | None = None) -> dict:
    """``{name: width}`` from a **typed** discrete map (see :func:`typed_discrete`)."""
    dims = {d: int(np.asarray(c).shape[0]) for d, c in dep_cols.items()}
    for d, (cols, kind, lo, hi) in (typed or {}).items():
        dims[d] = encoded_width(int(np.asarray(cols).shape[0]), kind, lo, hi)
    return dims


def dependency_dims(dep_cols: dict, discrete_cols: dict | None = None, expr=None) -> dict:
    """``{name: width}`` as the expression sees each dependency.

    A continuous dependency's width is its column count. A discrete one's is the width of its
    **encoded** design columns, which sets ``W``'s shape and therefore the parameter count AIC is
    computed from --- and that depends on the *coding*, so it is read from ``expr`` and the widths
    are **per candidate**, not shared across the pool: the same label is one column as an ordinal
    and ``k - 1`` as a categorical.
    """
    return typed_dims(dep_cols, typed_discrete(discrete_cols, expr))


def _dep_data(dep_cols: dict, coords, discrete_cols: dict | None = None, discrete=None):
    """Every dependency's per-row feature array: ``{name: (N, dep_dim)}``.

    The single place dependency data is materialised, which is why the discrete half belongs here
    too. A discrete dependency is encoded --- reference-coded indicators or a support-standardized
    value --- by :func:`~mimcs.hmc.metric_encode.encode_discrete`, the *same* function the runtime
    block calls. Two spellings of that encoding would make a metric fitted here mean something
    else when the sampler evaluated it, with no error either way.
    """
    out = {d: jnp.asarray(coords, float)[:, jnp.asarray(np.asarray(c, dtype=int))]
           for d, c in dep_cols.items()}                               # each (N, dep_dim)
    for d, (cols, kind, lo, hi) in (discrete_cols or {}).items():
        z = jnp.asarray(np.asarray(discrete)[:, np.asarray(cols, dtype=int)])
        out[d] = jax.vmap(lambda row, k=kind, a=lo, b=hi: encode_discrete(row, k, a, b))(z)
    return out


def _constant_metric(expr: MetricExpr, dep_cols: dict, discrete_cols: dict | None):
    """Is ``expr`` constant in the row? Then ``M`` is one vector, not an ``(N, block_dim)`` array.

    Worth its own branch in each caller below rather than a broadcast: a dep-less candidate is the
    baseline every block fits, and evaluating it once is both cheaper and clearer than mapping a
    constant over ``N`` rows.
    """
    return not dep_cols and not discrete_cols


def fit_metric_expr(expr: MetricExpr, block_cols, dep_cols: dict, coords, grads,
                    discrete_cols: dict | None = None, discrete=None, **opt):
    """Fit ``expr`` to the block's conditional score covariance; return ``(loss, params)``.

    The fit is **initialised at the evidence's own scale**: each coordinate's bias starts at the
    empirical second moment ``mean_n g_{n,d}^2`` (the exact minimiser of the constant candidate),
    rather than at ``M = I``. This is load-bearing, not a nicety --- the per-coordinate loss
    ``f(b) = 1/2 (b + g^2 e^{-b})`` is exponentially steep below its optimum and almost exactly
    linear with slope ``1/2`` above it, so a fit started orders of magnitude too low takes one
    enormous (correctly Armijo-accepted) L-BFGS step into the flat region and then has to crawl
    back at slope ``1/2``, which ``max_iter`` does not allow. On a target whose scores are ~1e5
    that left the *constant* baseline fitted at ``b ~ 1e4`` instead of ``~11``, wrecking both the
    coefficients and the AIC comparison every other candidate is judged against
    (``docs/design/09``).

    Args:
        expr: the candidate metric expression.
        block_cols: coordinate/score column indices of the block being fitted.
        dep_cols: ``{dep_name: column indices}`` for the blocks ``expr`` depends on.
        coords, grads: ``(N, coord_dim)`` evidence (positions and scores), row-aligned.
        **opt: forwarded to :func:`mimcs.optim.minimize`.
    """
    block_cols = jnp.asarray(np.asarray(block_cols, dtype=int))
    block_dim = int(block_cols.shape[0])
    dep_dims = typed_dims(dep_cols, discrete_cols)      # `discrete_cols` here is the typed form
    g = jnp.asarray(grads, float)[:, block_cols]                       # (N, block_dim)
    dep_data = _dep_data(dep_cols, coords, discrete_cols, discrete)
    n_rows = int(g.shape[0])
    working = int(g.size) * g.dtype.itemsize

    def row(params, g_row, dep_row):
        M = expr.evaluate(params, dep_row)
        return 0.5 * jnp.sum(jnp.log(M) + g_row ** 2 / M)

    # Reverse-mode AD through a whole-array ``vmap`` keeps every row's residuals live at once ---
    # O(N * block_dim) per intermediate. Above the gate the same sum is accumulated over
    # rematerialised chunks instead. Measured on this whole function at N=6000, block_dim=2000
    # under x64: peak **924 -> 551 MB** and **14.6 -> 2.7 s**, the speed-up free because the memory
    # traffic dominated the arithmetic. (In isolation the loss alone goes 740 -> 90 MB; the rest of
    # the 551 is `g` and `dep_data`, 96 MB each, which neither path can avoid.) Below the gate the
    # original whole-array expression is kept **verbatim**, so every existing fit reproduces
    # bit-for-bit --- checked, since no test in the suite is tight enough to notice if it did not.
    log.debug("fit_metric_expr: %r over %d row(s) x %d coordinate(s), score working set %.1f MB",
              expr, n_rows, block_dim, working / 2 ** 20)
    if working <= CHUNK_LOSS_BYTES:
        def mean_loss(params):
            return jnp.mean(jax.vmap(lambda a, b: row(params, a, b))(g, dep_data))
    else:
        def mean_loss(params):
            return sum_rows(lambda r: row(params, r[0], r[1]), (g, dep_data),
                            budget=CHUNK_BYTES) / n_rows

    scale = jnp.maximum(jnp.mean(g ** 2, axis=0), INIT_SCALE_FLOOR)    # (block_dim,)
    res = minimize(mean_loss, expr.init_params(block_dim, dep_dims, target=scale), **opt)
    return float(res.fun), res.x


def fit_is_usable(expr: MetricExpr, params, dep_cols: dict, coords, loss: float,
                  discrete_cols: dict | None = None, discrete=None) -> bool:
    """Is a fitted candidate safe to hand to a sampler?

    Rejects a fit whose loss or parameters are non-finite, and --- the case that matters --- one
    whose **metric ``M`` is not finite and positive** at the evidence. A log-linear metric can
    carry finite parameters that still exponentiate to ``inf`` (``exp(1e4)``), which becomes a
    non-finite mass the moment the sampler starts and an ``inf``/``nan`` in the very first
    adaptation step. A rejected candidate is scored ``AIC = inf`` by :func:`select_metric` so it
    is ranked last rather than dropped (the constant baseline must always remain available)."""
    if not np.isfinite(loss):
        return False
    if not all(bool(np.all(np.isfinite(np.asarray(leaf))))
               for leaf in jax.tree_util.tree_leaves(params)):
        return False
    # Reduced **per row** rather than over a materialised ``(N, block_dim)`` matrix: the question
    # is a conjunction, so it needs no matrix, and a boolean ``and`` has no summation order to
    # change. ``map_rows`` returns one flag per row --- ``N`` bytes instead of ``N * block_dim``
    # floats.
    if _constant_metric(expr, dep_cols, discrete_cols):
        M = np.asarray(expr.evaluate(params, {}))
        return bool(np.all(np.isfinite(M)) and np.all(M > 0.0))

    def row_ok(dep_row):
        M = expr.evaluate(params, dep_row)
        return jnp.all(jnp.isfinite(M) & (M > 0.0))

    dep_data = _dep_data(dep_cols, coords, discrete_cols, discrete)
    return bool(map_rows(row_ok, dep_data).all())


def whitened_scores(expr: MetricExpr, params, block_cols, dep_cols: dict, coords, grads,
                    discrete_cols: dict | None = None, discrete=None):
    """The ``D(x)^{-1/2}``-whitened block scores ``h_i = g_i / sqrt(M_i(dep_i))`` (``(N, block_dim)``)
    for a fitted diagonal metric ``expr`` with parameters ``params``.

    ``h``'s *constant* correlation is exactly the shaped-metric shape ``A``'s target (what
    :class:`mimcs.adaptation.ShapedMetricAdaptation` fits online), so passing ``h`` to
    :func:`mimcs.factory.mode_select.select_mass_mode` chooses the shape (diagonal / low-rank(J) /
    dense). Reuses the per-row ``expr.evaluate`` evaluation of :func:`fit_metric_expr`.

    The ``(N, block_dim)`` result is genuinely needed --- ``select_mass_mode`` forms a covariance
    from it --- but the ``M`` it is divided by is not, so the whitening is done a row at a time and
    only the result is kept. Elementwise throughout, hence bit-identical to the whole-array form.
    """
    block_cols = jnp.asarray(np.asarray(block_cols, dtype=int))
    g = jnp.asarray(grads, float)[:, block_cols]                          # (N, block_dim)
    if _constant_metric(expr, dep_cols, discrete_cols):
        M = expr.evaluate(params, {})
        return np.asarray(g / jnp.sqrt(jnp.maximum(M, 1e-30)))
    dep_data = _dep_data(dep_cols, coords, discrete_cols, discrete)
    return map_rows(
        lambda g_row, dep_row: g_row / jnp.sqrt(
            jnp.maximum(expr.evaluate(params, dep_row), 1e-30)),
        g, dep_data)


def aic(loss: float, n_params: int, n_rows: int) -> float:
    """AIC score (lower is better): ``2 k + 2 N * mean_loss`` (shared NLL constant dropped)."""
    return AIC_PENALTY * n_params + 2.0 * n_rows * loss


def discrete_factors(block_dim: int, discrete_cols: dict) -> list[MetricExpr]:
    """Candidate **discrete-only** metric factors, one group per discrete dependency.

    These are the terms a label can contribute on its own; the two-pass selection then also tries
    multiplying the best of them onto the best continuous candidate, which is what
    "labels modulate the continuous metric multiplicatively" means operationally (doc 14).

    Sparsity is gated on the **encoded** width, not the label count. That distinction is the whole
    point of the gate here: a binary indicator vector of the block's own length is the case that
    matters (spike-and-slab, ``z_j`` for ``beta_j``) and encodes to width ``n``, while a 3-level
    categorical over the same coordinates encodes to ``2n`` and must not be treated as an
    elementwise correspondence.
    """
    out: list[MetricExpr] = []
    for d in sorted(discrete_cols):
        cols, lo, hi = discrete_cols[d]
        size = int(np.asarray(cols).shape[0])
        # Both codings above two values; only one for a binary parameter, where the reference
        # indicator is an affine function of the standardized value --- the same model, and fitting
        # it twice would just spend a regression to rediscover that.
        kinds = ("ordinal",) if hi - lo <= 1 else ("ordinal", "categorical")
        for kind in kinds:
            kw = {kind: [d]}
            out.append(Exp(**kw))
            if encoded_width(size, kind, lo, hi) == block_dim:
                out.append(SpExp(**kw))
                out.append(Exp() * SpSigmoid(**kw))
            out.append(Exp() * Sigmoid(**kw))
    return out


def enumerate_candidates(block_dim: int, dep_dims: dict[str, int], *,
                         param_budget: int, max_candidates: int,
                         include_bare: bool = None,
                         discrete_cols: dict | None = None) -> list[MetricExpr]:
    """Simple candidate metric expressions for a block, dimension-aware and capped.

    Ordered simplest first --- the constant baseline ``Exp()`` (always included, = the current
    score mass), then per-dependency additive and gated forms (including **sparse** elementwise
    forms when a dependency shares the block's dimension), then pairwise additive and joint forms
    --- keeping only those within ``param_budget`` parameters and stopping at ``max_candidates``.
    So high-dimensional blocks/dependencies, or models with many small parameters, degrade
    gracefully to the cheapest candidates.

    A ``dep_dims[d] == block_dim`` match triggers sparse candidates: a bijective row
    correspondence (e.g. a horseshoe's per-element scale ``lambda_j`` for ``x_j``) is common
    between equal-dimension arrays, and its ``2 * block_dim`` parameters fit the budget where the
    dense ``block_dim * dep_dim`` form cannot --- often the only viable position-dependent form
    for equal *large* dimensions. (A coincidental total-dim match with incompatible shapes just
    fits poorly and is AIC-rejected, so triggering on the dimension alone is safe.)

    Every position-dependent form is offered **twice**: bare (``SpExp(d)``) and with an additive
    constant floor (``SpExp(d) + Exp()``), bare first since it is the cheaper of the pair. The
    floor buys a likelihood term the block's own conditional variance often does have (a
    hierarchical scale plus a data-driven one), but where the truth has *no* floor it is a spare
    term with nowhere to go: driving it to zero needs its bias to run to ``-inf``, which neither a
    capped L-BFGS nor a Robbins--Monro warmup reliably reaches, so it inflates the fit exactly
    where the true metric is smallest. Offering both lets AIC pay for the extra parameters only
    when they earn it. ``include_bare=False`` restores the floor-only pool (the pre-2026-08
    behaviour, and the control arm of the study that motivated the change); the default follows
    :data:`INCLUDE_BARE_CANDIDATES`.
    """
    if include_bare is None:
        include_bare = INCLUDE_BARE_CANDIDATES
    # `dep_dims` carries both namespaces (one width map keeps `n_params` a single call), so the
    # continuous enumeration must subtract the discrete names --- otherwise a label would be
    # enumerated a second time as a continuous dependency and then resolved against the coordinate
    # layout it is not in.
    deps = sorted(set(dep_dims) - set(discrete_cols or {}))
    forms: list[MetricExpr] = []
    for d in deps:                                        # single dependency
        forms.append(Exp(d))                             # log-linear
        if dep_dims[d] == block_dim:                     # equal dims -> sparse (elementwise)
            forms.append(SpExp(d))                       # elementwise log-linear (horseshoe form)
            forms.append(Exp() * SpSigmoid(d))           # bounded (gated) elementwise alternative
        forms.append(Exp() * Sigmoid(d))                 # bounded (gated) dense alternative
    for d1, d2 in itertools.combinations(deps, 2):       # dependency pairs
        forms.append(Exp(d1) + Exp(d2))                  # separable additive
        forms.append(Exp(d1, d2))                        # joint log-linear
    forms += discrete_factors(block_dim, discrete_cols or {})

    tiers: list[MetricExpr] = []
    for f in forms:                                      # bare first: it is the cheaper of the two
        if include_bare:
            tiers.append(f)
        tiers.append(f + Exp())

    out: list[MetricExpr] = [Exp()]                      # constant baseline, always
    for c in tiers:
        if len(out) >= max_candidates:
            break
        if c.n_params(block_dim, dependency_dims(
                {d: [0] * n for d, n in dep_dims.items()}, discrete_cols, c)) <= param_budget:
            out.append(c)
    return out


def _fit_and_log(fit_one, expr) -> MetricCandidate:
    """``fit_one(expr)`` with the per-candidate debug line, in one place for both passes."""
    r = fit_one(expr)
    if np.isfinite(r.aic):
        log.debug("  candidate %r: loss %.6g, %d parameter(s), AIC %.1f",
                  expr, r.loss, r.n_params, r.aic)
    else:
        log.debug("  candidate %r: rejected --- the fit is not usable (non-finite loss / "
                  "parameters, or a metric that is not finite and positive at the evidence; "
                  "loss %.6g). Ranked last, not dropped.", expr, r.loss)
    return r


def select_metric(block_cols, dep_cols: dict, coords, grads, *,
                  param_budget_mult: int = PARAM_BUDGET_MULT,
                  max_candidates: int = MAX_REGRESSIONS,
                  include_bare: bool = None,
                  discrete_cols: dict | None = None, discrete=None,
                  **opt) -> list[MetricCandidate]:
    """Enumerate, fit, and AIC-rank candidate metrics for one block; best (lowest AIC) first.

    ``dep_cols`` maps each candidate dependency-block name to its coordinate columns (the block
    being fitted is excluded by the caller). Returns every fitted :class:`MetricCandidate`,
    sorted by AIC, so a rule can compare the winner against the constant baseline.
    """
    block_dim = int(np.asarray(block_cols).shape[0])
    n_rows = int(np.asarray(coords).shape[0])
    budget = param_budget_mult * block_dim
    # Resolved here rather than left to `enumerate_candidates`, because pass 2 below reads it too
    # and `None` would quietly read as False there --- silently dropping every bare discrete factor.
    if include_bare is None:
        include_bare = INCLUDE_BARE_CANDIDATES

    cont_dims = dependency_dims(dep_cols)
    candidates = enumerate_candidates(block_dim, cont_dims, param_budget=budget,
                                      max_candidates=max_candidates, include_bare=include_bare,
                                      discrete_cols=discrete_cols)
    log.debug("metric regression on a %d-dim block over %d evidence row(s): %d candidate(s) "
              "within a %d-parameter budget, dependencies %s%s", block_dim, n_rows,
              len(candidates), budget, cont_dims,
              f", discrete {sorted(discrete_cols)}" if discrete_cols else "")
    def fit_one(expr):
        """Fit one candidate and score it; an unusable fit is ranked last, never dropped (the
        constant baseline must stay available to compare against)."""
        used = {d: dep_cols[d] for d in expr.deps()}
        z_typed = typed_discrete(
            {d: discrete_cols[d] for d in expr.discrete_deps()} if discrete_cols else {}, expr)
        loss, params = fit_metric_expr(expr, block_cols, used, coords, grads,
                                       z_typed, discrete, **opt)
        k = expr.n_params(block_dim, typed_dims(dep_cols, z_typed))
        usable = fit_is_usable(expr, params, used, coords, loss, z_typed, discrete)
        return MetricCandidate(expr, params, loss, k,
                               aic(loss, k, n_rows) if usable else float("inf"))

    results = []
    for expr in candidates:
        results.append(_fit_and_log(fit_one, expr))
    results.sort(key=lambda r: r.aic)

    # Pass 2: multiply the best **continuous** candidate by each discrete factor. A label is
    # expected to scale a mode rather than reshape it (doc 14), so the product is the form worth
    # trying --- and trying it only against the winner keeps this a handful of extra fits instead
    # of a pool multiplied by the number of discrete dependencies. AIC decides each on its own; a
    # discrete factor that buys nothing is charged for its parameters and loses.
    factors = discrete_factors(block_dim, discrete_cols or {})
    if factors and results:
        best_cont = next((r for r in results if not r.expr.discrete_deps()), None)
        if best_cont is not None and np.isfinite(best_cont.aic):
            for f in factors:
                for factor in ((f, f + Exp()) if include_bare else (f + Exp(),)):
                    combined = best_cont.expr * factor
                    dims = dependency_dims(dep_cols, discrete_cols, combined)
                    if combined.n_params(block_dim, dims) > budget:
                        continue
                    results.append(_fit_and_log(fit_one, combined))
            results.sort(key=lambda r: r.aic)

    log.debug("metric regression ranked %r best (AIC %.1f)", results[0].expr, results[0].aic)
    return results
