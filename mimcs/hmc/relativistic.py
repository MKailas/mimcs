"""Relativistic kinetic energy for HMC / NUTS.

**[experimental]** Not reachable from the sampler factory, and not recommended as a default. The
reason is the light speed ``c``: it is a fixed scalar, and the argument that one value suffices
(below) is conditional on a *centering* reparametrization standardizing the coordinates. Centering
has not been a default for a long time --- it is opt-in via ``spec.centering`` and off --- so on a
target whose coordinates span orders of magnitude a single ``c`` caps the velocity at the wrong
scale for most of them. Making this a first-class option most likely needs an adaptation for ``c``
of its own, which is open research rather than a wiring job.

The relativistic kinetic energy of a particle of rest mass ``m`` and light speed ``c`` with
momentum vector ``p`` is ``T = sqrt(m^2 c^4 + c^2 |p|^2)`` --- equivalently
``m c^2 sqrt(1 + |p|^2 / (m^2 c^2))``. Its velocity ``dT/dp = c^2 p / T`` is *bounded* by
``c``: however large the momentum, the particle moves at most at the speed of light. That
cap stops the integrator from shooting off in light tails or funnel-like geometry, which is
the point of relativistic Monte Carlo (Lu, Perrone, Hernandez-Lobato, Hasenclever, Vollmer,
AISTATS 2017).

What is a "particle"? The energy is a sum over particles ``i`` of
``sqrt(m_i^2 c^4 + c^2 sum_j p_{ij}^2)`` --- an inner sum over each particle's momentum
components and an outer sum over particles. Reshaping the flat coordinate momentum to a
given ``shape``, the ``inner_axes`` are summed *inside* the square root (one particle spans
them) and the remaining axes index particles. The extremes are ``inner_axes=()`` (every
coordinate is its own 1-D particle) and ``inner_axes`` = all axes (a single particle over
the whole block).

The per-particle rest mass ``m_i`` lives in ``ham_params[id]`` and may be adapted (see
:class:`mimcs.adaptation.RelativisticMassAdaptation`); ``c`` is a fixed scalar, which suffices
*if* a centering reparametrization has standardized the coordinates --- see the experimental note
above, since centering is no longer a default. The
momentum marginal ``exp(-T(p))`` is refreshed exactly: per particle the magnitude
``r = |p|`` has radial density ``prop r^{d-1} exp(-sqrt(m_i^2 c^4 + c^2 r^2))``, sampled by a
2-D inverse-CDF table over ``(u, m)`` (so it tracks the adapting mass), and the direction is
uniform on the sphere.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..rng import DrawComponent
from .hamiltonians import KineticHamiltonian


class RelativisticKinetic(KineticHamiltonian):
    """Separable relativistic kinetic ``T = sum_i sqrt(m_i^2 c^4 + c^2 |p_i|^2)``.

    **[experimental]** --- see the module docstring: the fixed light speed's justification rests
    on centering, which is no longer a default.

    Args:
        shape: shape the flat coordinate momentum is reshaped to.
        inner_axes: axes summed *inside* the square root (one particle spans them); the
            remaining axes index particles. ``()`` makes every coordinate a 1-D particle.
        mass: initial (scalar) rest mass for every particle.
        light_speed: the (scalar) light speed ``c``.
        id: kinetic component id.
        n_u, n_m, m_min, m_max: resolution / extent of the ``(u, m)`` radius inverse-CDF
            table (``m`` log-spaced in ``[m_min, m_max]``).
    """

    separable = True

    def __init__(self, shape, *, inner_axes=(), mass: float = 1.0, light_speed: float = 1.0,
                 id: str = "T", slices=None, n_u: int = 2000,
                 n_m: int = 128, m_min: float = 0.05, m_max: float = 20.0):
        self.shape = tuple(int(s_) for s_ in shape)
        self.ndim = len(self.shape)
        self.inner_axes = tuple(sorted(int(a) % self.ndim for a in inner_axes))
        self.outer_axes = tuple(a for a in range(self.ndim) if a not in self.inner_axes)
        self.m = float(mass)
        self.c = float(light_speed)
        self.id = id
        self.slices = slices                # coordinate block (prod(shape) must equal its size)

        self._inner_dim = int(np.prod([self.shape[a] for a in self.inner_axes])) \
            if self.inner_axes else 1
        self._outer_shape = tuple(self.shape[a] for a in self.outer_axes)
        self._n_particles = int(np.prod(self._outer_shape)) if self._outer_shape else 1
        self.dim = int(np.prod(self.shape))

        self._n_u, self._n_m = int(n_u), int(n_m)
        self._radius_table, self._log_m_lo, self._log_m_hi = self._build_radius_table(
            m_min, m_max)

    # --- per-particle mass (from ham_params, falling back to the scalar) ------ #

    def _masses(self, ctx):
        if ctx is not None and ctx.ham_params.get(self.id) is not None:
            return ctx.ham_params[self.id]                  # (n_particles,)
        return jnp.full((self._n_particles,), self.m)

    def _mass_broadcast(self, ctx):
        m = self._masses(ctx).reshape(self._outer_shape)
        return jnp.expand_dims(m, self.inner_axes)          # 1 on the inner axes

    def initial_mass_params(self, dim):
        return jnp.full((self._n_particles,), self.m)

    # --- (u, m) -> radius inverse-CDF table --------------------------------- #

    def _build_radius_table(self, m_min, m_max):
        d = self._inner_dim
        log_m = np.linspace(np.log(m_min), np.log(m_max), self._n_m)
        u_grid = np.linspace(0.0, 1.0, self._n_u)
        table = np.empty((self._n_u, self._n_m), np.float64)
        for j, m in enumerate(np.exp(log_m)):
            r_max = m * self.c + (d + 14.0 * np.sqrt(d + 1.0) + 50.0) / self.c
            r = np.linspace(0.0, r_max, 4000)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_g = (d - 1) * np.log(r) - np.sqrt(m**2 * self.c**4 + self.c**2 * r**2)
            log_g[0] = -(m * self.c**2) if d == 1 else -np.inf
            g = np.exp(log_g - np.nanmax(log_g))
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))])
            cdf /= cdf[-1]
            table[:, j] = np.interp(u_grid, cdf, r)
        return jnp.asarray(table, float), float(log_m[0]), float(log_m[-1])

    def _sample_radius(self, u, m):
        row = u * (self._n_u - 1)
        col = ((jnp.log(m) - self._log_m_lo) / (self._log_m_hi - self._log_m_lo)
               * (self._n_m - 1))
        col = jnp.clip(col, 0.0, self._n_m - 1)
        return jax.scipy.ndimage.map_coordinates(
            self._radius_table, [row, col], order=1, mode="nearest")

    # --- Hamiltonian interface ---------------------------------------------- #

    def _particle_norm_sq(self, p_shaped):
        return jnp.sum(p_shaped ** 2, axis=self.inner_axes, keepdims=True)

    def _p_shaped(self, istate):
        return self._gather(istate.p).reshape(self.shape)

    def energy(self, istate, ctx):
        s = self._particle_norm_sq(self._p_shaped(istate))
        mc2 = self._mass_broadcast(ctx) * self.c**2
        return jnp.sum(jnp.sqrt(mc2**2 + self.c**2 * s))

    def velocity_into(self, v, istate, ctx):
        p = self._p_shaped(istate)
        mc2 = self._mass_broadcast(ctx) * self.c**2
        denom = jnp.sqrt(mc2**2 + self.c**2 * self._particle_norm_sq(p))
        return self._scatter(v, (self.c**2 * p / denom).reshape(-1))

    def make_draw_components(self, dim):
        return [DrawComponent(f"{self.id}_mom_direction", (self.dim,), jax.random.normal),
                DrawComponent(f"{self.id}_mom_radius", (self._n_particles,), jax.random.uniform)]

    def sample_into(self, p, draw, q, ctx):
        z = getattr(draw, f"{self.id}_mom_direction").reshape(self.shape)
        direction = z / jnp.sqrt(self._particle_norm_sq(z))      # uniform on each sphere
        r = self._sample_radius(getattr(draw, f"{self.id}_mom_radius"), self._masses(ctx))
        r = jnp.expand_dims(r.reshape(self._outer_shape), self.inner_axes)
        return self._scatter(p, (r * direction).reshape(-1))

    # --- score aggregation for mass adaptation ------------------------------ #

    @property
    def particle_dim(self) -> int:
        return self._inner_dim

    def particle_grad_sq(self, g_flat):
        """Per-particle squared score ``|g_i|^2`` (sum of ``g^2`` over inner axes)."""
        return jnp.sum(g_flat.reshape(self.shape) ** 2, axis=self.inner_axes).reshape(-1)
