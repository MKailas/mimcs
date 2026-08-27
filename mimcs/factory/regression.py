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


class MetricCandidate(NamedTuple):
    """One fitted candidate metric for a block."""

    expr: MetricExpr
    params: object          # fitted parameter pytree (in the expression's structure)
    loss: float             # achieved batch KL loss (mean over evidence rows)
    n_params: int
    aic: float              # 2 * n_params + 2 * N * loss (shared NLL constant dropped)


#: floor on the per-coordinate init scale (avoids ``log 0`` for a coordinate with no score signal).
INIT_SCALE_FLOOR = 1e-30


def _dep_data(dep_cols: dict, coords):
    return {d: jnp.asarray(coords, float)[:, jnp.asarray(np.asarray(c, dtype=int))]
            for d, c in dep_cols.items()}                              # each (N, dep_dim)


def evaluate_metric(expr: MetricExpr, params, dep_cols: dict, coords):
    """The fitted diagonal metric ``M`` at every evidence row: ``(N, block_dim)``."""
    n_rows = int(np.asarray(coords).shape[0])
    if not dep_cols:                       # dep-less (constant) expression: one value, broadcast
        M = expr.evaluate(params, {})
        return jnp.broadcast_to(M, (n_rows,) + M.shape)
    dep_data = _dep_data(dep_cols, coords)
    return jax.vmap(lambda dep_row: expr.evaluate(params, dep_row))(dep_data)


def fit_metric_expr(expr: MetricExpr, block_cols, dep_cols: dict, coords, grads, **opt):
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
    dep_dims = {d: int(np.asarray(c).shape[0]) for d, c in dep_cols.items()}
    g = jnp.asarray(grads, float)[:, block_cols]                       # (N, block_dim)
    dep_data = _dep_data(dep_cols, coords)

    def mean_loss(params):
        def row(g_row, dep_row):
            M = expr.evaluate(params, dep_row)
            return 0.5 * jnp.sum(jnp.log(M) + g_row ** 2 / M)
        return jnp.mean(jax.vmap(row)(g, dep_data))

    scale = jnp.maximum(jnp.mean(g ** 2, axis=0), INIT_SCALE_FLOOR)    # (block_dim,)
    res = minimize(mean_loss, expr.init_params(block_dim, dep_dims, target=scale), **opt)
    return float(res.fun), res.x


def fit_is_usable(expr: MetricExpr, params, dep_cols: dict, coords, loss: float) -> bool:
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
    M = np.asarray(evaluate_metric(expr, params, dep_cols, coords))
    return bool(np.all(np.isfinite(M)) and np.all(M > 0.0))


def whitened_scores(expr: MetricExpr, params, block_cols, dep_cols: dict, coords, grads):
    """The ``D(x)^{-1/2}``-whitened block scores ``h_i = g_i / sqrt(M_i(dep_i))`` (``(N, block_dim)``)
    for a fitted diagonal metric ``expr`` with parameters ``params``.

    ``h``'s *constant* correlation is exactly the shaped-metric shape ``A``'s target (what
    :class:`mimcs.adaptation.ShapedMetricAdaptation` fits online), so passing ``h`` to
    :func:`mimcs.factory.mode_select.select_mass_mode` chooses the shape (diagonal / low-rank(J) /
    dense). Reuses the per-row ``expr.evaluate`` evaluation of :func:`fit_metric_expr`."""
    block_cols = jnp.asarray(np.asarray(block_cols, dtype=int))
    g = jnp.asarray(grads, float)[:, block_cols]                          # (N, block_dim)
    Dx = evaluate_metric(expr, params, dep_cols, coords)                   # (N, block_dim)
    return np.asarray(g / jnp.sqrt(jnp.maximum(Dx, 1e-30)))


def aic(loss: float, n_params: int, n_rows: int) -> float:
    """AIC score (lower is better): ``2 k + 2 N * mean_loss`` (shared NLL constant dropped)."""
    return AIC_PENALTY * n_params + 2.0 * n_rows * loss


def enumerate_candidates(block_dim: int, dep_dims: dict[str, int], *,
                         param_budget: int, max_candidates: int,
                         include_bare: bool = None) -> list[MetricExpr]:
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
    deps = sorted(dep_dims)
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

    tiers: list[MetricExpr] = []
    for f in forms:                                      # bare first: it is the cheaper of the two
        if include_bare:
            tiers.append(f)
        tiers.append(f + Exp())

    out: list[MetricExpr] = [Exp()]                      # constant baseline, always
    for c in tiers:
        if len(out) >= max_candidates:
            break
        if c.n_params(block_dim, dep_dims) <= param_budget:
            out.append(c)
    return out


def select_metric(block_cols, dep_cols: dict, coords, grads, *,
                  param_budget_mult: int = PARAM_BUDGET_MULT,
                  max_candidates: int = MAX_REGRESSIONS,
                  include_bare: bool = None, **opt) -> list[MetricCandidate]:
    """Enumerate, fit, and AIC-rank candidate metrics for one block; best (lowest AIC) first.

    ``dep_cols`` maps each candidate dependency-block name to its coordinate columns (the block
    being fitted is excluded by the caller). Returns every fitted :class:`MetricCandidate`,
    sorted by AIC, so a rule can compare the winner against the constant baseline.
    """
    block_dim = int(np.asarray(block_cols).shape[0])
    dep_dims = {d: int(np.asarray(c).shape[0]) for d, c in dep_cols.items()}
    n_rows = int(np.asarray(coords).shape[0])
    budget = param_budget_mult * block_dim

    candidates = enumerate_candidates(block_dim, dep_dims, param_budget=budget,
                                      max_candidates=max_candidates, include_bare=include_bare)
    log.debug("metric regression on a %d-dim block over %d evidence row(s): %d candidate(s) "
              "within a %d-parameter budget, dependencies %s", block_dim, n_rows,
              len(candidates), budget, dep_dims)
    results = []
    for expr in candidates:
        used = {d: dep_cols[d] for d in expr.deps()}
        loss, params = fit_metric_expr(expr, block_cols, used, coords, grads, **opt)
        k = expr.n_params(block_dim, dep_dims)
        usable = fit_is_usable(expr, params, used, coords, loss)
        score = aic(loss, k, n_rows) if usable else float("inf")
        if usable:
            log.debug("  candidate %r: loss %.6g, %d parameter(s), AIC %.1f", expr, loss, k, score)
        else:
            log.debug("  candidate %r: rejected --- the fit is not usable (non-finite loss / "
                      "parameters, or a metric that is not finite and positive at the evidence; "
                      "loss %.6g). Ranked last, not dropped.", expr, loss)
        results.append(MetricCandidate(expr, params, loss, k, score))
    results.sort(key=lambda r: r.aic)
    log.debug("metric regression ranked %r best (AIC %.1f)", results[0].expr, results[0].aic)
    return results
