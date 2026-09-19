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
the parameters *frozen for sampling* are their Polyak--Ruppert running mean --- a *uniform* mean
from the first update (a linear average of these log-linear metric parameters).

``metric_ema_warmup`` (default **off**, experimental) replaces that with an **exponential moving
average** of the parameters, ``ema_n = ema_{n-1} + eta_n (theta_n - ema_{n-1})`` with the same
Robbins--Monro gain ``eta_n = (n + n0)^{-kappa}`` as the SGD step (the Kailas--Vihola--Wallin
smoother :class:`~mimcs.adaptation.ScoreMassAdaptation` uses for its mass), and lets the EMA
**drive warmup** as well as being frozen for sampling: the SGD still advances the raw iterate, but
the chain is simulated with the EMA. :class:`~mimcs.adaptation.ShapedMetricAdaptation` reads the
same key for its ``D(x)`` and then also whitens the shape ``A`` by the EMA, so the shape is fitted
under the metric the chain actually runs with.
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


def _is_shared(leaf, block_dim: int) -> bool:
    """Is this leaf **shared** across the block's coordinates (axis 0 of length 1)?

    Every atom's ``W``/``b`` is shaped ``(rows, ...)`` with ``rows`` either ``block_dim`` (one
    value per coordinate) or ``1`` (one value serving them all --- ``shared_weights`` /
    ``shared_bias`` in ``mimcs/hmc/metric_expr.py``); ``Sum``/``Product`` only nest
    already-atom-shaped params. A block of dimension 1 is read as per-coordinate, since there is
    then nothing to share it with.
    """
    return block_dim != 1 and int(leaf.shape[0]) == 1


def _coords_served(leaf, block_dim: int) -> int:
    """How many of the block's coordinates one row of this leaf serves: 1, or ``block_dim``."""
    return block_dim // int(leaf.shape[0])


def _per_coord_size(params, block_dim: int) -> int:
    """How many scalar parameters apply to ONE coordinate of a learned metric's block.

    Counts the **per-coordinate** leaves only (summed over their trailing axes): a shared leaf
    applies to every coordinate at once and is its own update unit, with its own clip threshold,
    so counting it here would inflate the per-coordinate dimension-aware threshold init."""
    return sum(int(np.prod(leaf.shape[1:])) for leaf in jax.tree_util.tree_leaves(params)
               if not _is_shared(leaf, block_dim))


def _make_kl_step(loss, block_dim: int):
    """A jitted KL-SGD gradient step: ``step(params, *args) -> (g, per_coord_norm, shared_norms)``.

    ``loss(params, *args)`` is the block's metric loss. Shared by the plain learned metric
    (:class:`MetricAdaptation`) and the shaped metric's ``D(x)``
    (:class:`~mimcs.adaptation.shaped_metric.ShapedMetricAdaptation`), so the two cannot drift
    apart again: the shaped adapter used to take a single global-norm step with an undivided
    shared-leaf gradient, and on `irt_2pl` that path, not the plain one, broke stage 2
    (``tests/experiments/writeups/irt_two_stage_v2.md``).

    Two kinds of update unit, and they must not be mixed. A per-coordinate leaf's row ``j``
    belongs to coordinate ``j`` alone, so summing its trailing axes and then across such leaves
    gives that coordinate's own norm --- a ``(block_dim,)`` vector. A **shared** leaf belongs to no
    single coordinate: its gradient is the sum over every coordinate it serves, so it gets its own
    scalar norm. Letting it into the per-coordinate sum is not a shape error, it is a silent one
    --- Python's ``sum`` broadcasts the ``(1,)`` term into the ``(block_dim,)`` accumulator and every
    coordinate's "own" norm is inflated by the whole block's shared gradient, which is exactly the
    coupling the per-coordinate clip exists to remove.

    A shared leaf's gradient is divided by the number of coordinates it serves. ``L(w) =
    sum_d l_d(w)`` scales the gradient *and* the curvature with that count, so a first-order step
    needs ``eta < 2 / (n h_1)`` where a per-coordinate one needs ``eta < 2 / h_1``; the adaptive
    clip cannot absorb it, because its threshold tracks the observed norm and both sides scale
    together. Dividing is a **per-unit learning rate**, not a change of objective --- the online and
    offline losses are the same *data* term --- and a per-coordinate leaf divides by 1, so nothing
    about the unshared path moves. (The offline fit adds a ridge toward its scale-aware init that
    this does not; see :data:`mimcs.factory.regression.RIDGE_SIGMA`. That asymmetry is deliberate:
    this SGD descending the unpenalised loss through warmup is what removes the ridge's bias.)
    """
    bd = block_dim
    def step(params, *args):
        g = jax.grad(lambda p: loss(p, *args))(params)
        g = jax.tree_util.tree_map(
            lambda leaf: leaf / _coords_served(leaf, bd), g)
        leaves = jax.tree_util.tree_leaves(g)
        per_coord = [leaf for leaf in leaves if not _is_shared(leaf, bd)]
        row_sq = sum(jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1)
                     for leaf in per_coord) if per_coord else jnp.zeros(bd)
        shared = [jnp.sqrt(jnp.sum(leaf ** 2))
                  for leaf in leaves if _is_shared(leaf, bd)]
        return g, jnp.sqrt(row_sq), shared
    return jax.jit(step)


class _PerUnitClip:
    """The per-unit adaptive clip, the non-finite guard and the descent for one learned block.

    One threshold **per target coordinate** plus one **per shared leaf**, each a running
    log-quantile of its own gradient norm (see the module docstring), initialised dimension-aware at
    the log of the unit's parameter count. Kept on the Python side, like the rest of the adaptation
    state. Used by :class:`MetricAdaptation` and by the shaped metric's ``D(x)``.
    """

    def __init__(self, params, block_dim: int):
        self.block_dim = block_dim
        # ONE clip threshold PER TARGET COORDINATE, not one for the whole block: each coordinate's
        # own KL-loss gradient touches only its own `p` parameters (the weight(s)/bias of every
        # atom at that coordinate's row), independent of every other coordinate's. Tracking `d`
        # independent thresholds (dimension-aware init log(p), the per-coordinate analogue of the
        # old log(block_dim)) means one coordinate's large gradient no longer scales down every
        # other coordinate's update -- see the module docstring.
        self.p = _per_coord_size(params, block_dim)
        self.log_clip = math.log(max(self.p, 1)) * np.ones(block_dim)
        # A shared leaf is its own update unit, so it gets its own threshold, initialised
        # dimension-aware on ITS parameter count for the same reason.
        self.shared_sizes = [int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)
                             if _is_shared(leaf, block_dim)]
        self.shared_clip = np.array([math.log(max(n, 1)) for n in self.shared_sizes], dtype=float)

    def update(self, params, g, gnorm, shared_norms, lr: float, gain: float, clip_frac: float):
        """One clipped descent step; returns ``(params, n_bad, applied)``.

        ``n_bad`` counts the non-finite units skipped; ``applied`` is False when nothing at all
        was finite, in which case ``params`` is returned untouched and the thresholds do not move.
        """
        gn = np.asarray(gnorm, dtype=float)                       # (block_dim,)
        thr = np.exp(self.log_clip)                               # (block_dim,)
        s_gn = np.asarray([float(v) for v in shared_norms], dtype=float)
        s_thr = np.exp(self.shared_clip)

        # Guard: a non-finite loss gradient AT COORDINATE j means M_j itself went inf/nan (an
        # overflowing exp, typically from a pathological init). Descending on it would poison that
        # coordinate's parameters permanently, so skip only THAT coordinate -- every other
        # coordinate keeps adapting normally, which is the point of decoupling the clip: one
        # pathological coordinate should not also stall the rest of the block.
        finite = np.isfinite(gn)
        s_finite = np.isfinite(s_gn)
        # A shared leaf's skip counts as one unit, not as `block_dim` coordinate-skips: it is one
        # update that did not happen.
        n_bad = int((~finite).sum()) + int((~s_finite).sum())
        # Only bail out when there is nothing at all left to update. A bad *shared* gradient must
        # not freeze the block's per-coordinate leaves (nor the reverse) -- that is the decoupling
        # the per-unit clip exists for.
        if not finite.any() and not s_finite.any():
            return params, n_bad, False

        # Clip each unit's KL-loss gradient at its OWN adaptive threshold, then descend -- a
        # non-finite unit's update is discarded via `jnp.where`, not by zeroing its scale
        # (0 * nan/inf is nan/inf, not 0).
        clip_factor = np.where(finite, np.minimum(1.0, thr / (gn + 1e-12)), 0.0)
        scale_j = jnp.asarray(lr * clip_factor, float)             # (block_dim,)
        finite_j = jnp.asarray(finite)
        s_scale = jnp.asarray(
            lr * np.where(s_finite, np.minimum(1.0, s_thr / (s_gn + 1e-12)), 0.0), float)
        s_finite_j = jnp.asarray(s_finite)

        def _apply(w, gw, s_i):
            """Descend one leaf. The scale is keyed off the LEAF's own row count, never off the
            per-coordinate vector's: reshaping `(block_dim,)` against a `(1, feat)` leaf broadcasts
            the result up to `(block_dim, feat)` and silently un-shares it on the first warmup step
            -- no exception, the declared sharing simply evaporates."""
            if s_i is not None:                       # a shared leaf: one scalar step
                return jnp.where(s_finite_j[s_i], w - s_scale[s_i] * gw, w)
            shape = (gw.shape[0],) + (1,) * (gw.ndim - 1)
            return jnp.where(finite_j.reshape(shape), w - scale_j.reshape(shape) * gw, w)

        # Paired explicitly rather than through a stateful counter inside `tree_map`: the shared
        # norms were collected in `tree_leaves` order, and this is the same order, stated once
        # instead of relied on implicitly.
        p_leaves, treedef = jax.tree_util.tree_flatten(params)
        g_leaves = jax.tree_util.tree_leaves(g)
        slots, nxt = [], 0
        for leaf in p_leaves:
            if _is_shared(leaf, self.block_dim):
                slots.append(nxt)
                nxt += 1
            else:
                slots.append(None)
        params = jax.tree_util.tree_unflatten(
            treedef, [_apply(w, gw, i) for w, gw, i in zip(p_leaves, g_leaves, slots)])

        # Move each unit's running log-quantile of its (raw) gradient norm toward `1 - frac`; a
        # skipped (non-finite) unit counts as "not exceeded" -- its threshold is diagnostic
        # bookkeeping only once skipped, since its scale is zero regardless of the threshold.
        exceeded = np.where(finite, np.asarray(gn > thr, dtype=float), 0.0)
        self.log_clip = self.log_clip + gain * (exceeded - clip_frac)
        if s_gn.size:
            s_exceeded = np.where(s_finite, np.asarray(s_gn > s_thr, dtype=float), 0.0)
            self.shared_clip = self.shared_clip + gain * (s_exceeded - clip_frac)
        return params, n_bad, True


def _running_mean(avg, params, n: int):
    """Fold ``params`` into a Polyak--Ruppert running mean over ``n`` updates."""
    return jax.tree_util.tree_map(lambda a, x: a + (x - a) / n, avg, params)


def _ema(avg, params, eta: float):
    """Fold ``params`` into an exponential moving average with gain ``eta`` (``metric_ema_warmup``)."""
    return jax.tree_util.tree_map(lambda a, x: a + eta * (x - a), avg, params)


class MetricAdaptation:
    """Mixin: SGD-adapt the kinetic's learned diagonal-metric parameters during warmup."""

    def _init_hooks(self, **kwargs):
        self._metric_kappa = float(kwargs.get("metric_adapt_kappa", DEFAULT_KAPPA))
        self._metric_n0 = float(kwargs.get("metric_adapt_n0", DEFAULT_N0))
        self._metric_clip_frac = float(kwargs.get("metric_clip_frac", 0.1))
        self._metric_center_grad = bool(kwargs.get("metric_center_grad", False))
        self._metric_polyak = bool(kwargs.get("mass_polyak", True))
        # An EMA of the params drives warmup and is frozen for sampling (off: the raw iterate drives
        # warmup, the uniform Polyak mean is frozen). ShapedMetricAdaptation reads the same key.
        self._metric_ema_warmup = bool(kwargs.get("metric_ema_warmup", False))
        self._metric_count = 0
        self._metric_clips: dict = {}                  # _PerUnitClip per block id
        self._metric_log_clip: dict[str, float] = {}   # running log-quantile per block id (view)
        self._metric_shared_clip: dict = {}            # ditto, one per shared leaf, per block id
        self._metric_mean_grad = None                  # running mean of the score (centring)
        self._metric_step_fns: dict = {}               # jitted grad step per block id
        self._metric_params: dict = {}                 # raw SGD iterate per block id (Python-side)
        self._metric_avg: dict = {}                    # Polyak average of the params per block id
        self._metric_ema: dict = {}                    # EMA of the params per block id (ema_warmup)
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
        """A jitted step for one learned block: the KL-loss gradient, the per-coordinate gradient
        norm, and one norm per **shared** leaf (see :func:`_make_kl_step`)."""
        return _make_kl_step(
            lambda p, q, labels, score, lr: block.metric_loss(p, q, labels, score), block.size)

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
        # The labels are the *current* draw's and consistent with the score: the Gibbs sweep runs
        # inside `kernel`, and `BaseHMC._after_discrete` re-seeds `potential_grads` under the new
        # labels before this hook sees them --- so (q, labels, score) is a coherent triple.
        labels = getattr(state, "discrete", None)
        lr_j = jnp.asarray(lr, float)
        new_ham = dict(state.ham_params)
        for k in blocks:
            if k.id not in self._metric_clips:
                clip = _PerUnitClip(state.ham_params[k.id], k.size)
                self._metric_clips[k.id] = clip
                self._metric_log_clip[k.id] = clip.log_clip
                self._metric_shared_clip[k.id] = clip.shared_clip
                self._metric_step_fns[k.id] = self._make_step(k)
                self._metric_params[k.id] = state.ham_params[k.id]   # seed the raw iterate
                log.debug("learned-metric adaptation started on block %r (%d coordinate(s), "
                          "%d parameter(s)/coordinate, per-coordinate clip threshold init "
                          "log %d%s)", k.id, k.size, clip.p, max(clip.p, 1),
                          f", plus {len(clip.shared_sizes)} shared leaf/leaves "
                          f"{clip.shared_sizes}" if clip.shared_sizes else "")
            clip = self._metric_clips[k.id]

            # SGD advances the raw iterate (kept Python-side so Polyak averaging of the *written*
            # params does not feed back into the descent).
            params = self._metric_params[k.id]
            g, gnorm, shared_norms = self._metric_step_fns[k.id](params, q, labels, score, lr_j)
            gain = rm_gain(self._metric_count, self._metric_n0, self._metric_kappa)
            params, n_bad, applied = clip.update(params, g, gnorm, shared_norms, lr, gain,
                                                 self._metric_clip_frac)
            self._metric_log_clip[k.id] = clip.log_clip
            self._metric_shared_clip[k.id] = clip.shared_clip
            # `metric_nonfinite_count`'s "(iteration, coordinate)" contract is about the
            # per-coordinate units; a shared leaf's skip counts as one (see `_PerUnitClip`).
            if n_bad:
                self._metric_nonfinite[k.id] = self._metric_nonfinite.get(k.id, 0) + n_bad
                log.debug("learned metric %r: skipped %d non-finite KL-gradient unit(s) "
                          "at warmup iteration %d (%d skipped so far); those units' "
                          "metric evaluated to inf/nan", k.id, n_bad, self._metric_count,
                          self._metric_nonfinite[k.id])
            if not applied:
                new_ham[k.id] = (self._metric_ema.get(k.id, params) if self._metric_ema_warmup
                                 else params)
                continue
            self._metric_params[k.id] = params
            if self._metric_ema_warmup:
                # The EMA, not the raw iterate, drives warmup (and is what gets frozen).
                self._metric_ema[k.id] = (params if k.id not in self._metric_ema else
                                          _ema(self._metric_ema[k.id], params, lr))
                new_ham[k.id] = self._metric_ema[k.id]
                continue
            if self._metric_polyak:
                self._metric_accumulate(k.id, params)
            new_ham[k.id] = params                     # warmup uses the raw iterate

        return state._replace(ham_params=new_ham)

    def _metric_accumulate(self, block_id, params):
        """Fold the raw iterate into the block's Polyak--Ruppert running mean of the (log-linear)
        metric parameters."""
        n = self._metric_count                        # updates so far for this block
        if block_id not in self._metric_avg:
            self._metric_avg[block_id] = params
        else:
            self._metric_avg[block_id] = _running_mean(self._metric_avg[block_id], params, n)

    def _finalize_hooks(self, state):
        """Freeze the Polyak-averaged metric parameters for sampling --- or, under
        ``metric_ema_warmup``, the EMA that drove warmup."""
        state = super()._finalize_hooks(state)
        if self._metric_ema_warmup and self._metric_ema:
            state = state._replace(ham_params={**state.ham_params, **self._metric_ema})
            log.debug("froze the EMA metric of block(s) %s after %d update(s)",
                      sorted(self._metric_ema), self._metric_count)
        elif self._metric_polyak and self._metric_avg:
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
