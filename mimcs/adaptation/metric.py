"""Learned-metric adaptation for explicit (block) Riemannian HMC.

A mixin (``docs/design/02_sampler_classes.md``) that adapts a
:class:`~mimcs.hmc.block_riemannian.LearnedDiagonalBlock`'s parameters online during
warmup by stochastic gradient descent on the KL objective

    L_i(phi) = 1/2 sum_d ( log M_i[d](q_{-i}; phi) + g_i[d]^2 / M_i[d](q_{-i}; phi) ),

summed over learned blocks, where ``g_i`` is the current potential gradient (the
"score") restricted to block ``i``. The per-sample minimiser is ``M_i[d] = g_i[d]^2`` and
the expected-loss minimiser is the conditional gradient second moment
``E[g_i[d]^2 | q_{-i}]`` --- the metric that whitens the local geometry. The parametric
log-linear form (sum of exponentials) keeps every ``M_i[d]`` positive automatically.

SGD details (defaults follow what works well in practice, and mirror
:class:`mimcs.adaptation.ScoreMassAdaptation` --- the two implement the same
Kailas--Vihola--Wallin regularizations):

* step size ``(n + n0)^{-kappa}``, ``kappa = 0.75``, ``n0 = 5`` (Robbins--Monro decay);
* gradient clipping at an *adaptive* threshold tracked so that a target fraction (default 10%)
  of steps are clipped --- a stochastic-approximation estimate of the ``(1 - frac)`` quantile of
  the gradient norm, maintained on the log scale (scale-free), updated with the decreasing gain
  ``(n + n0)^{-kappa}`` (the same schedule as :class:`mimcs.adaptation.ScoreMassAdaptation`, which
  worked better than a ``beta``-scaled gain on the hard Poisson random-effects case). The clip is
  **per target coordinate**, not per block: coordinate ``d``'s own gradient (the loss depends on
  ``M_i[d]`` only through that coordinate's own weights/bias, over every additive/multiplicative
  term of the metric expression) is clipped against its OWN threshold, tracked independently of
  every other coordinate's. This decouples the adaptation trajectories -- one coordinate with a
  large gradient no longer scales down every other coordinate's update, exactly mirroring
  :class:`mimcs.adaptation.ScoreMassAdaptation`'s diagonal (non-block-Riemannian) mass, which
  learns as ``d`` independent one-coordinate problems for the same reason. Each coordinate's
  threshold is initialised dimension-aware at ``log p`` (``p`` the number of scalar parameters
  touching that one coordinate -- e.g. 2 for a single ``Exp("v")`` atom's weight+bias -- summed
  over every atom in the expression, the per-coordinate analogue of the old whole-block ``log d``);
* gradient mean estimation (``metric_center_grad``, default **off**): a running mean
  ``mean_grad`` of the score is maintained (same schedule) and the loss uses the *centred*
  score ``g - mean_grad``. This is the same regularization as
  :class:`mimcs.adaptation.ScoreMassAdaptation`, and it is the right one for a *marginal*
  metric (a constant, ``depends_on=[None]`` block): the marginal score is mean-zero at
  stationarity (integration by parts), so ``mean_grad -> 0`` and the centring is inert there
  while removing the transient downhill offset early. It is **off by default** because a
  learned block metric is usually *conditional* (``M_i(q_{-i})``), whose conditional mean
  ``E[g_i | q_{-i}]`` is already zero; subtracting a single marginal mean then injects a
  constant floor ``mean_grad^2`` that distorts the conditional fit (e.g. it flattens the
  funnel metric's ``e^{-v}`` toward a constant). Enable it only for effectively marginal /
  constant blocks.

The clip thresholds, mean gradient and step counter live on the Python object; only the
resulting parameters cross into the JAX state. Adaptation runs during warmup only and is
frozen for sampling. By default (``mass_polyak=True``) the raw SGD iterate drives warmup, but
the parameters *frozen for sampling* are their Polyak--Ruppert running mean (a linear average
of these log-linear metric parameters; see :mod:`mimcs.adaptation._polyak`).
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)


def _per_coord_size(params) -> int:
    """How many scalar parameters (summed over each leaf's trailing axes) apply to ONE
    coordinate of a learned metric's block --- every leaf's axis 0 is the block's own ``d``
    coordinates (``mimcs/hmc/metric_expr.py``: every atom's ``W``/``b`` is shaped
    ``(block_dim, ...)``, and ``Sum``/``Product`` only nest already-atom-shaped params), so this
    is the same for every coordinate and can be computed once from any one params pytree."""
    return sum(int(np.prod(leaf.shape[1:])) for leaf in jax.tree_util.tree_leaves(params))


class MetricAdaptation:
    """Mixin: SGD-adapt the kinetic's learned diagonal-metric parameters during warmup."""

    def _init_hooks(self, **kwargs):
        self._metric_kappa = float(kwargs.get("metric_adapt_kappa", DEFAULT_KAPPA))
        self._metric_n0 = float(kwargs.get("metric_adapt_n0", DEFAULT_N0))
        self._metric_clip_frac = float(kwargs.get("metric_clip_frac", 0.1))
        self._metric_center_grad = bool(kwargs.get("metric_center_grad", False))
        self._metric_polyak = bool(kwargs.get("mass_polyak", True))
        self._metric_count = 0
        self._metric_log_clip: dict[str, float] = {}   # running log-quantile per block id
        self._metric_mean_grad = None                  # running mean of the score (centring)
        self._metric_step_fns: dict = {}               # jitted grad step per block id
        self._metric_params: dict = {}                 # raw SGD iterate per block id (Python-side)
        self._metric_avg: dict = {}                    # Polyak average of the params per block id
        self._metric_nonfinite: dict = {}              # skipped (non-finite) updates per block id
        super()._init_hooks(**kwargs)

    def metric_nonfinite_count(self, block_id: str | None = None) -> int:
        """How many (iteration, coordinate) pairs were skipped because that coordinate's KL-loss
        gradient was not finite --- the clip is now per coordinate (see the module docstring), so
        this counts individual coordinate-skips, not whole-block-step skips: one warmup iteration
        with 5 pathological coordinates out of 2000 adds 5, not 1.

        Non-zero means some coordinate's metric evaluated to ``inf``/``nan`` --- almost always a
        pathological *initial* metric (e.g. a regression fit whose ``exp`` overflows), since the
        guard keeps every coordinate's parameters finite thereafter. A persistently rising count
        says (some of) the metric never became usable, not that the guard is working."""
        if block_id is not None:
            return int(self._metric_nonfinite.get(block_id, 0))
        return int(sum(self._metric_nonfinite.values()))

    def _learned_blocks(self):
        return [k for k in self.kinetics if getattr(k, "is_learned", False)]

    def _make_step(self, block):
        """A jitted step for one learned block: its KL-loss gradient and, per target coordinate,
        that coordinate's OWN gradient norm (a ``(block_dim,)`` vector, not one scalar for the
        whole block). Every leaf's axis 0 is the block's own coordinates (see
        :func:`_per_coord_size`), so summing each leaf's trailing axes and then summing across
        leaves gives a well-defined per-coordinate norm."""
        def step(params, q, score, lr):
            g = jax.grad(lambda p: block.metric_loss(p, q, score))(params)
            row_sq = sum(jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1)
                        for leaf in jax.tree_util.tree_leaves(g))
            gnorm = jnp.sqrt(row_sq)                      # (block_dim,)
            return g, gnorm
        return jax.jit(step)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state
        blocks = self._learned_blocks()
        if not blocks:
            return state

        self._metric_count += 1
        lr = rm_gain(self._metric_count, self._metric_n0, self._metric_kappa)
        score = sum(state.potential_grads.values())     # total potential gradient at q

        # Centre the score by its running mean (E[score] -> 0 at stationarity) so the metric
        # fits the gradient covariance rather than its second moment.
        if self._metric_center_grad:
            score_np = np.asarray(score, dtype=float)
            if self._metric_mean_grad is None:
                self._metric_mean_grad = np.zeros_like(score_np)
            delta = score_np - self._metric_mean_grad
            score = jnp.asarray(delta)
            self._metric_mean_grad += lr * delta          # SA running mean

        q = state.coordinate
        lr_j = jnp.asarray(lr, float)
        new_ham = dict(state.ham_params)
        for k in blocks:
            if k.id not in self._metric_log_clip:
                p = _per_coord_size(state.ham_params[k.id])
                # ONE clip threshold PER TARGET COORDINATE, not one for the whole block: each
                # coordinate's own KL-loss gradient touches only its own `p` parameters (the
                # weight(s)/bias of every atom at that coordinate's row), independent of every
                # other coordinate's. Tracking `d` independent thresholds (dimension-aware init
                # log(p), the per-coordinate analogue of the old log(block_dim)) means one
                # coordinate's large gradient no longer scales down every other coordinate's
                # update -- see the module docstring.
                self._metric_log_clip[k.id] = math.log(p) * np.ones(k.size)
                self._metric_step_fns[k.id] = self._make_step(k)
                self._metric_params[k.id] = state.ham_params[k.id]   # seed the raw iterate
                log.debug("learned-metric adaptation started on block %r (%d coordinate(s), "
                          "%d parameter(s)/coordinate, per-coordinate clip threshold init "
                          "log %d)", k.id, k.size, p, p)

            # SGD advances the raw iterate (kept Python-side so Polyak averaging of the *written*
            # params does not feed back into the descent).
            params = self._metric_params[k.id]
            g, gnorm = self._metric_step_fns[k.id](params, q, score, lr_j)
            gn = np.asarray(gnorm, dtype=float)                       # (block_dim,)
            thr = np.exp(self._metric_log_clip[k.id])                 # (block_dim,)

            # Guard: a non-finite loss gradient AT COORDINATE j means M_j itself went inf/nan (an
            # overflowing exp, typically from a pathological init). Descending on it would poison
            # that coordinate's parameters permanently, so skip only THAT coordinate -- every
            # other coordinate keeps adapting normally, which is the point of decoupling the clip:
            # one pathological coordinate should not also stall the rest of the block.
            finite = np.isfinite(gn)
            n_bad = int((~finite).sum())
            if n_bad:
                self._metric_nonfinite[k.id] = self._metric_nonfinite.get(k.id, 0) + n_bad
                log.debug("learned metric %r: skipped %d non-finite KL-gradient coordinate(s) "
                          "at warmup iteration %d (%d skipped so far); those coordinates' "
                          "metric evaluated to inf/nan", k.id, n_bad, self._metric_count,
                          self._metric_nonfinite[k.id])
            if not finite.any():
                new_ham[k.id] = params
                continue

            # Clip each coordinate's KL-loss gradient at its OWN adaptive threshold, then
            # descend -- a non-finite coordinate's update is discarded via `jnp.where`, not by
            # zeroing its scale (0 * nan/inf is nan/inf, not 0).
            clip_factor = np.where(finite, np.minimum(1.0, thr / (gn + 1e-12)), 0.0)
            scale_j = jnp.asarray(lr * clip_factor, float)             # (block_dim,)
            finite_j = jnp.asarray(finite)

            def _apply(w, gw, s=scale_j, f=finite_j):
                shape = (s.shape[0],) + (1,) * (gw.ndim - 1)
                updated = w - s.reshape(shape) * gw
                return jnp.where(f.reshape(shape), updated, w)

            params = jax.tree_util.tree_map(_apply, params, g)
            self._metric_params[k.id] = params
            if self._metric_polyak:
                self._metric_accumulate(k.id, params)
            new_ham[k.id] = params                     # warmup uses the raw iterate

            # Move each coordinate's running log-quantile of its (raw) gradient norm toward
            # `1 - frac`; a skipped (non-finite) coordinate counts as "not exceeded" -- its
            # threshold is diagnostic bookkeeping only once skipped, since its scale is zero
            # regardless of the threshold.
            exceeded = np.where(finite, np.asarray(gn > thr, dtype=float), 0.0)
            self._metric_log_clip[k.id] = self._metric_log_clip[k.id] + rm_gain(
                self._metric_count, self._metric_n0, self._metric_kappa) * (
                    exceeded - self._metric_clip_frac)

        return state._replace(ham_params=new_ham)

    def _metric_accumulate(self, block_id, params):
        """Fold the raw iterate into the block's Polyak--Ruppert running mean of the (log-linear)
        metric parameters."""
        n = self._metric_count                        # updates so far for this block
        if block_id not in self._metric_avg:
            self._metric_avg[block_id] = params
        else:
            avg = self._metric_avg[block_id]
            self._metric_avg[block_id] = jax.tree_util.tree_map(
                lambda a, x: a + (x - a) / n, avg, params)

    def _finalize_hooks(self, state):
        """Freeze the Polyak-averaged metric parameters for sampling (see :mod:`._polyak`)."""
        state = super()._finalize_hooks(state)
        if self._metric_polyak and self._metric_avg:
            state = state._replace(ham_params={**state.ham_params, **self._metric_avg})
            log.debug("froze the Polyak-averaged metric of block(s) %s after %d update(s)",
                      sorted(self._metric_avg), self._metric_count)
        skipped = self.metric_nonfinite_count()
        if skipped:
            log.warning(
                "learned metric: %d adaptation step(s) over %d warmup iteration(s) were skipped "
                "for a non-finite KL gradient (per block: %s). The metric evaluated to inf/nan "
                "there --- usually a pathological initial fit; a count that kept rising means it "
                "never became usable.", skipped, self._metric_count,
                dict(self._metric_nonfinite))
        return state
