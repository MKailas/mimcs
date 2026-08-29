"""Unit-vector chart adaptation: the sphere analogue of centering.

A mixin (``docs/design/02_sampler_classes.md``) that places the stereographic chart of every
:class:`~mimcs.model.UnitVectorParameter` created with ``adaptive=True``. It is to a unit
vector what :class:`~mimcs.adaptation.CenteringAdaptation` is to a Euclidean parameter: fit the
chart to where the draws actually are, so the coordinate target comes out centred and
unit-scaled. The two hyperparameters (:class:`~mimcs.model.SphereChart`) split exactly the
way location and scale do, and together carry ``d`` degrees of freedom --- the same count as the
von Mises--Fisher family's ``(mu, kappa)``:

* **Pole** (``d - 1`` dof) --- *location*. The chart is singular at its pole, so the pole belongs
  where the density is lowest. Each draw ``x`` votes for its own antipode, and the pole is
  **stepped along the geodesic** toward that vote on the shared Robbins--Monro schedule:

      pole <- Exp_pole(gain * Log_pole(-x))

  This is Riemannian SGD on ``F(p) = 0.5 E[d(p, -x)^2]``, so what it estimates is the **Frechet
  (Karcher) mean of the antipodal draws** --- an intrinsic quantity, defined by a variational
  problem on the sphere rather than by borrowing the ambient linear structure.

  The intrinsic formulation is the whole point, because the obvious extrinsic estimator ---
  ``-normalize(E[x])``, the anti-mean-direction --- is **unstable exactly where it matters**. A
  running mean tends to ``0`` on any diffuse target, and normalizing it is then ``0/0``: the
  magnitude vanishes but the direction does not converge, so the estimate wanders over the whole
  sphere. Nothing rescues that, because no amount of gating fixes an estimator whose *value* is
  undefined; a floor test on the mean resultant only converts the wander into a flicker, since a
  hard threshold on a fluctuating quantity leaks on every crossing. The geodesic step never forms
  that ratio. Each step is a bounded move (``gain * theta``, at most ``gain * pi``) toward a
  perfectly well-defined point, whatever the distribution looks like.

  What the Frechet mean buys, concretely:

  * **Axial targets** work. Mass spread over a great circle has ``E[x] = 0`` no matter where the
    circle lies, so the extrinsic estimator is blind to it; the Frechet mean lands on the
    circle's axis (``F = pi^2/4`` there against ``pi^2/3`` on the circle itself, a strict
    minimum) --- exactly the pole one wants, since the axis is where a girdle is emptiest.
  * **Concentrated targets** still work: on a vMF it recovers ``-mu``. It is the only one of the
    candidates that gets both regimes right --- the second moment ``E[x x^T]`` also finds the
    axis, via its smallest eigenvector, but cannot tell ``+mu`` from ``-mu`` and so is wrong for
    the concentrated case.

  The price is variance: stepping toward each *draw's* antipode averages once, where the
  extrinsic estimator averaged twice (a running mean of ``x``, then a step toward its direction),
  so at moderate concentration the pole carries a few more degrees of residual jitter about
  ``-mu``. That costs nothing worth having --- a few degrees off the antipode changes the density
  at the pole by well under a percent --- and it buys an estimator that is defined everywhere.

  **Uniform on the sphere** has no Frechet mean --- every point minimizes ``F`` --- so the pole
  keeps drifting there, and that is accepted. The step size is ``O(gain)``, so the chart's change
  per iteration vanishes: the chain satisfies *diminishing adaptation*, which is what adaptive
  MCMC needs for validity, whether or not the pole converges. (NUTS may still struggle near the
  pole of a target with mass there; no chart placement fixes that, only an atlas would.)

* **Scale** (1 dof) --- ``log_scale``. Targets a quantile: ``unit_vector_target_frac`` of draws
  should land inside the unit circle of coordinate space. Because ``|u| < 1`` holds exactly when
  ``<x, pole> < c`` (see :class:`~mimcs.model.UnitVectorParameter`), the plane offset ``c``
  *is* the targeted quantile of ``<x, pole>``, and the default ``0.5`` makes it the median. The
  fraction inside is decreasing in ``s``, so the Robbins--Monro step is
  ``log_s <- log_s + gain (1[|u| < 1] - frac)``: this is the same stochastic-quantile idiom used
  for the gradient-norm clips in :mod:`mimcs.adaptation.score_mass` and for the MAD in
  :class:`~mimcs.adaptation.RobustCenteringAdaptation`. A concentrated target projects to a tiny
  ``|u|``, so ``s`` grows until the bulk fills the unit circle.

The two co-adapt (the pole moves under the scale's feet and vice versa), exactly as the mean and
variance do in :class:`~mimcs.adaptation.CenteringAdaptation`.

Changing a chart relabels the coordinate of a fixed physical point, so on each update the mixin
recomputes the coordinate from the (unchanged) sample and refreshes the cached potential
values/gradients there. Adaptation runs during warmup only. HMC-family only (it uses
``state_at_coordinate``), as centering is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._logging import get_logger
from ..model import UnitVectorParameter, SphereChart
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)


@jax.jit
def _geodesic_step(pole, target, frac):
    """``Exp_pole(frac * Log_pole(target))``: move ``frac`` of the way along the geodesic.

    Compiled: pure, shape-stable, and called once per adaptive unit vector per warmup iteration,
    where the dozen eager dispatches cost 15x the compiled call.

    Both arguments are unit vectors, batched over their leading axes; the result is the point
    at geodesic distance ``frac * d(pole, target)`` from ``pole`` along the great circle
    through the two. ``frac = 0`` stays put and ``frac = 1`` lands on ``target``.
    """
    cos_theta = jnp.clip(jnp.sum(pole * target, axis=-1, keepdims=True), -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    tangent = target - cos_theta * pole                          # norm = sin(theta)
    sin_theta = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    # The geodesic *direction* is undefined where sin(theta) = 0. Two ways that happens, both
    # harmless: the points coincide, where the step is zero regardless; or they are antipodal,
    # where every direction is a geodesic, so any choice is as good as another. Falling back to
    # a zero tangent leaves ``cos(frac * pi) * pole``, i.e. the pole unmoved -- the gain never
    # reaches 1/2, so that cosine never turns negative and flips it.
    safe = jnp.where(sin_theta > 1e-6, sin_theta, 1.0)
    direction = jnp.where(sin_theta > 1e-6, tangent / safe, 0.0)
    stepped = jnp.cos(frac * theta) * pole + jnp.sin(frac * theta) * direction
    return stepped / jnp.linalg.norm(stepped, axis=-1, keepdims=True)


@jax.jit
def _householder_for(pole):
    """A unit ``v`` with ``H_v e_d = pole``, batched over ``pole``'s leading axes.

    Compiled, for the same reason as :func:`_geodesic_step` and more so: eagerly this is two
    ``jnp.zeros(...).at[...].set(...)`` allocations plus a norm, a ``where`` and a divide, each its
    own dispatch, and it measured **200x** the compiled call.

    ``v ~ pole - e_d``, which degenerates as ``pole -> e_d``: there the reflection must fix
    ``e_d``, which *every* ``v`` orthogonal to ``e_d`` does, so fall back to ``e_1`` (the same
    vector the chart initializes with). This lives on the Python side precisely so the
    degenerate branch never reaches the chart's gradient path.
    """
    e_d = jnp.zeros(pole.shape, float).at[..., -1].set(1.0)
    e_1 = jnp.zeros(pole.shape, float).at[..., 0].set(1.0)
    v = pole - e_d
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    # The denominator is guarded as well as the branch: `where` already discards the dead
    # branch's value, but not a NaN's derivative, so this stays safe under `grad`.
    safe = jnp.where(norm > 1e-6, norm, 1.0)
    return jnp.where(norm > 1e-6, v / safe, e_1)


class UnitVectorCenteringAdaptation:
    """Mixin: fit each adaptive unit vector's stereographic chart (pole + scale) to the chain."""

    def _init_hooks(self, **kwargs):
        self._uv_target_frac = float(kwargs.get("unit_vector_target_frac", 0.5))
        self._uv_min_samples = int(kwargs.get("unit_vector_min_samples", 50))
        self._uv_n0 = float(kwargs.get("unit_vector_adapt_n0", DEFAULT_N0))
        self._uv_kappa = float(kwargs.get("unit_vector_adapt_kappa", DEFAULT_KAPPA))
        # (param index, param, ambient slice, coordinate slice) per adaptive unit vector. Both
        # slices are needed, and they are different lengths: the draw is read from the *sample*
        # (d components; the pole is a direction in ambient space) and the relabelled coordinate
        # is written back to the *coordinate* (d-1 of them).
        aoffs, coffs = self.model._ambient_offsets, self.model._coord_offsets
        self._uv_params = [
            (i, p, (int(aoffs[i]), int(aoffs[i + 1])), (int(coffs[i]), int(coffs[i + 1])))
            for i, p in enumerate(self.model.parameters)
            if isinstance(p, UnitVectorParameter) and p.adaptive]
        # Estimator state, on the Python object (as every other mixin keeps it): the pole and the
        # log-scale. Both are *published* into state.chart_hyperparams only once there are enough
        # draws to be worth recharting.
        self._uv_pole = {i: p.pole(p.init_hyperparams()) for i, p, _, _ in self._uv_params}
        self._uv_log_scale = {i: jnp.zeros(p.batch_shape, float)
                              for i, p, _, _ in self._uv_params}
        self._uv_count = 0
        super()._init_hooks(**kwargs)
        log.debug("UnitVectorCenteringAdaptation: fitting the chart of %d adaptive unit "
                  "vector(s) %s (target inside-fraction %.2f, recharting from iteration %d)",
                  len(self._uv_params), [p.name for _, p, _, _ in self._uv_params],
                  self._uv_target_frac, self._uv_min_samples + 1)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP or not self._uv_params:
            return state

        self._uv_count += 1
        gain = rm_gain(self._uv_count, self._uv_n0, self._uv_kappa)
        new_hyperparams = list(state.chart_hyperparams)

        for i, p, (a_lo, a_hi), _ in self._uv_params:
            x = jnp.reshape(state.sample[a_lo:a_hi], p.ambient_shape)

            # The quantile is read under the chart the estimator currently believes in, which
            # is what keeps the two estimators consistent before the first chart is published.
            # (Once published this is exactly `state.coordinate`'s slice.)
            chart = SphereChart(householder=_householder_for(self._uv_pole_step(i, gain, x)),
                                log_scale=self._uv_log_scale[i])
            u = jnp.reshape(p.to_coordinate(x, chart), p.batch_shape + (p.d - 1,))
            inside = (jnp.sum(u * u, axis=-1) < 1.0).astype(float)
            self._uv_log_scale[i] = (
                self._uv_log_scale[i] + gain * (inside - self._uv_target_frac))
            new_hyperparams[i] = SphereChart(householder=chart.householder,
                                             log_scale=self._uv_log_scale[i])

        if self._uv_count <= self._uv_min_samples:
            return state

        # Relabel: hold the sample fixed and re-derive its coordinate under the new charts.
        new_coordinate = state.coordinate
        for i, p, (a_lo, a_hi), (c_lo, c_hi) in self._uv_params:
            new_coordinate = new_coordinate.at[c_lo:c_hi].set(
                p.to_coordinate(state.sample[a_lo:a_hi], new_hyperparams[i]))
        return self.state_at_coordinate(
            state, new_coordinate, sample=state.sample,
            hyperparams=tuple(new_hyperparams))

    def _uv_pole_step(self, i, gain, x):
        """One Robbins--Monro step of the pole along the geodesic toward ``-x``; returns the pole.

        Riemannian SGD on ``F(p) = 0.5 E[d(p, -x)^2]``, whose gradient is ``-Log_p(-x)``: the
        step is ``Exp_p(gain * Log_p(-x))``. Vectorized over an array of unit vectors, each with
        its own pole and its own draw.
        """
        self._uv_pole[i] = _geodesic_step(self._uv_pole[i], -x, gain)
        return self._uv_pole[i]

    def unit_vector_chart(self, name: str):
        """The fitted ``(pole, plane_offset)`` of parameter ``name`` --- for diagnostics/tests."""
        for i, p, _, _ in self._uv_params:
            if p.name == name:
                hyper = self.state.chart_hyperparams[i]
                return p.pole(hyper), p.plane_offset(hyper)
        raise KeyError(f"no adaptive unit-vector parameter named {name!r}")
