"""The Steihaug conjugate-gradient trust-region subproblem (SciPy's ``trust-ncg``), in pure JAX.

``steihaug_cg`` approximately minimises the quadratic model ``m(p) = g.p + p.Hp / 2`` over the ball
``|p| <= radius``, given only Hessian-vector products. This is Nocedal & Wright (2nd ed.),
Algorithm 7.2, as SciPy implements it (``scipy.optimize._trustregion_ncg.CGSteihaugSubproblem``):
CG from ``p = 0`` with the forcing tolerance ``min(1/2, sqrt|g|) |g|`` on the residual, stopping
at the boundary when an iterate leaves the ball, or along a direction of non-positive curvature
(the boundary point of lower ``m``). The Hessian need not be positive definite.

Departures from SciPy, neither of which changes a step:

* ``H p`` is carried through the CG recurrences (``H z += alpha H d``), so the caller can evaluate
  ``m(p)`` without the extra product SciPy spends on it.
* An optional iteration cap (``max_iters``): at the cap the current CG iterate is returned (an
  interior step, ``capped``). SciPy has none; in exact arithmetic CG ends within ``n`` iterations.
  SciPy's early return ``|g| < tolerance`` is omitted: the tolerance is at most ``|g| / 2``, so it
  can never trigger. A zero gradient returns ``p = 0`` without a product.

The solver is one ``lax.while_loop``, jittable and usable inside an outer loop
(:mod:`mimcs.vi.deterministic` drives it).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["steihaug_cg", "SteihaugResult"]


class SteihaugResult(NamedTuple):
    """``p`` the step, ``Hp`` its Hessian product, ``hits_boundary`` whether ``|p| = radius``,
    ``n_hvp`` the Hessian-vector products spent, ``capped`` whether ``max_iters`` stopped it."""
    p: Array
    Hp: Array
    hits_boundary: Array
    n_hvp: Array
    capped: Array


class _State(NamedTuple):
    z: Array
    Hz: Array
    r: Array
    d: Array
    j: Array
    done: Array
    hits: Array
    capped: Array
    p: Array
    Hp: Array


def _boundaries(z, d, radius):
    """The two roots ``ta <= tb`` of ``|z + t d| = radius`` (SciPy's stable form; ``|z| < radius``
    makes the discriminant positive)."""
    a = jnp.dot(d, d)
    b = 2.0 * jnp.dot(z, d)
    c = jnp.dot(z, z) - radius * radius
    disc = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))
    aux = b + jnp.where(b >= 0, disc, -disc)          # copysign(disc, b), with +0 -> +disc
    aux = jnp.where(aux != 0, aux, 1.0)
    t1, t2 = -aux / (2.0 * jnp.where(a > 0, a, 1.0)), -2.0 * c / aux
    return jnp.minimum(t1, t2), jnp.maximum(t1, t2)


def steihaug_cg(g: Array, hvp: Callable[[Array], Array], radius, max_iters: int) -> SteihaugResult:
    """Minimise ``g.p + p.H p / 2`` over ``|p| <= radius`` by Steihaug's CG (``H p = hvp(p)``),
    at most ``max_iters`` products."""
    dtype = g.dtype
    radius = jnp.asarray(radius, dtype)
    gnorm = jnp.sqrt(jnp.dot(g, g))
    tol = jnp.minimum(0.5, jnp.sqrt(gnorm)) * gnorm
    zero = jnp.zeros_like(g)
    false = jnp.asarray(False)

    def model(p, Hp):
        return jnp.dot(g, p) + 0.5 * jnp.dot(p, Hp)

    def body(s: _State) -> _State:
        Bd = hvp(s.d)
        dBd = jnp.dot(s.d, Bd)
        ta, tb = _boundaries(s.z, s.d, radius)
        # non-positive curvature: the boundary point along d of lower model value
        pa, Hpa, pb, Hpb = s.z + ta * s.d, s.Hz + ta * Bd, s.z + tb * s.d, s.Hz + tb * Bd
        a_wins = model(pa, Hpa) < model(pb, Hpb)
        p_neg, Hp_neg = jnp.where(a_wins, pa, pb), jnp.where(a_wins, Hpa, Hpb)
        neg = dBd <= 0
        r2 = jnp.dot(s.r, s.r)
        alpha = r2 / jnp.where(neg, 1.0, dBd)
        z_next, Hz_next = s.z + alpha * s.d, s.Hz + alpha * Bd
        out = ~neg & (jnp.sqrt(jnp.dot(z_next, z_next)) >= radius)
        p_out, Hp_out = s.z + tb * s.d, s.Hz + tb * Bd     # the crossing, t >= 0
        r_next = s.r + alpha * Bd
        r2_next = jnp.dot(r_next, r_next)
        conv = ~neg & ~out & (jnp.sqrt(r2_next) < tol)
        j = s.j + 1
        cap = ~neg & ~out & ~conv & (j >= max_iters)
        done = neg | out | conv | cap
        p = jnp.where(neg, p_neg, jnp.where(out, p_out, z_next))
        Hp = jnp.where(neg, Hp_neg, jnp.where(out, Hp_out, Hz_next))
        d_next = -r_next + (r2_next / jnp.where(r2 > 0, r2, 1.0)) * s.d
        return _State(z=z_next, Hz=Hz_next, r=r_next, d=d_next, j=j, done=done,
                      hits=neg | out, capped=cap, p=p, Hp=Hp)

    init = _State(z=zero, Hz=zero, r=g, d=-g, j=jnp.asarray(0), done=gnorm == 0, hits=false,
                  capped=false, p=zero, Hp=zero)
    s = jax.lax.while_loop(lambda s: ~s.done, body, init)
    return SteihaugResult(p=s.p, Hp=s.Hp, hits_boundary=s.hits, n_hvp=s.j, capped=s.capped)
