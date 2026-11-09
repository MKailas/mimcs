"""Gaussian-process covariance functions for the model DSL --- Stan's six, under Stan's names.

These are the one place the builtin table departs from JAX/NumPy names wholesale, and for the
reason the naming rule gives (see :mod:`mimcs.dsl.builtins`): neither ``jax.numpy`` nor
``jax.scipy`` has a GP kernel, so there is no JAX name to transcribe into and Stan's is the one a
user already knows. The same goes for :func:`add_diag`, the jitter every GP needs before a
``cholesky``: JAX's nearest relative, ``fill_diagonal``, *overwrites* the diagonal rather than adding
to it.

**Signatures follow Stan's Functions Reference exactly**, overloads included::

    gp_exp_quad_cov(x, sigma, length_scale)            gp_exp_quad_cov(x1, x2, sigma, length_scale)
    gp_exponential_cov(...)   gp_matern32_cov(...)     gp_matern52_cov(...)         (the same four)
    gp_periodic_cov(x, sigma, length_scale, period)    gp_periodic_cov(x1, x2, sigma, length_scale, period)
    gp_dot_prod_cov(x, sigma)                          gp_dot_prod_cov(x1, x2, sigma)

The one-input form is the ``n x n`` covariance of ``x`` with itself; the two-input form the
``n1 x n2`` cross-covariance, which is what a GP prediction needs. The argument count alone tells
them apart, as it does in Stan.

**Inputs.** The DSL has no ``vector`` type, so Stan's two input kinds map onto the array's rank: a
1-d ``array[n] real x`` is ``n`` scalar inputs (Stan's ``array[] real``), and a 2-d
``array[n, D] real x`` is ``n`` points in ``D`` dimensions, **one per row** (Stan's ``vectors``).
A 1-d input is treated as ``(n, 1)``. Any other rank is an error.

**ARD.** For the four stationary kernels whose Stan signatures allow it, ``length_scale`` may be an
array of ``D`` values instead of a scalar --- one length scale per input dimension (automatic
relevance determination). As in Stan's implementation the kernel is then the isotropic one with
unit length scale, applied to the inputs divided coordinate-wise by ``length_scale``. The periodic
and dot-product kernels have no ARD form in Stan, and none here.

**One departure from Stan:** ``gp_periodic_cov`` refuses points of dimension ``D > 1``, which
Stan accepts. Its formula is not positive definite there; see :func:`gp_periodic_cov`.

**Distances are formed from differences, never from** ``|x|^2 + |x'|^2 - 2 x.x'``. The expanded
form cancels catastrophically in float32 and can go slightly *negative*, which a ``sqrt`` turns
into ``NaN``; differences cost an ``(n1, n2, D)`` intermediate instead, which is cheap at any size a
dense GP is usable at. They also make the one-input matrix exactly symmetric with a diagonal of
exactly ``sigma^2``.

**The distance's square root is guarded at zero.** Every kernel but the exponentiated quadratic
and the dot product reads the distance ``r = sqrt(s)`` itself, and ``sqrt`` has an infinite
derivative at ``s = 0`` --- which is every diagonal element. The inner derivative there is zero,
but ``inf * 0`` is ``NaN``, so a plain ``sqrt`` makes the gradient with respect to
``length_scale`` ``NaN`` even when ``x`` is data: a GP whose length scale is a parameter would be
unsampleable. :func:`_distance` uses the double-``where`` guard, which gives the zero gradient the
kernel genuinely has there (``k(x, x) = sigma^2`` depends on neither ``x`` nor ``length_scale``).

**Hyperparameter constraints are the declaration's job.** Stan rejects a non-positive ``sigma``,
``length_scale`` or ``period`` at run time; a traced value cannot raise here, so these compute
through instead (``sigma`` enters only squared). Declare them ``<lower=0>``, as in Stan.
"""

from __future__ import annotations

import jax.numpy as jnp


def _points(x, fn: str, arg: str):
    """``x`` as an ``(n, D)`` array of points: a 1-d input is ``n`` scalar points."""
    x = jnp.asarray(x, float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(
        f"{fn}: {arg} must be a 1-d array of scalar inputs or a 2-d array with one point per row, "
        f"got shape {x.shape}")


def _scalar(v, fn: str, arg: str):
    if jnp.ndim(v) != 0:
        raise ValueError(f"{fn}: {arg} must be a scalar, got shape {jnp.shape(v)}")
    return v


def _split(fn: str, args, hypers: tuple):
    """Stan's overload by count: ``(x, *hypers)`` or ``(x1, x2, *hypers)`` -> ``x1, x2, hypers``."""
    k = len(hypers)
    if len(args) == k + 1:
        x1 = x2 = _points(args[0], fn, "x")
    elif len(args) == k + 2:
        x1, x2 = _points(args[0], fn, "x1"), _points(args[1], fn, "x2")
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"{fn}: x1 and x2 have points of different dimension "
                             f"({x1.shape[1]} and {x2.shape[1]})")
    else:
        h = ", ".join(hypers)
        raise TypeError(f"{fn}() takes {k + 1} or {k + 2} arguments (x, {h}) or (x1, x2, {h}), "
                        f"got {len(args)}")
    return x1, x2, args[-k:]


def _scaled_sq_distance(fn: str, x1, x2, length_scale, *, ard: bool):
    """``sum_d ((x1_d - x2_d) / l_d)^2`` for every pair, as an ``(n1, n2)`` array."""
    if ard:
        if jnp.ndim(length_scale) > 1 or (jnp.ndim(length_scale) == 1
                                          and jnp.shape(length_scale)[0] != x1.shape[1]):
            raise ValueError(f"{fn}: length_scale must be a scalar or have one entry per input "
                             f"dimension ({x1.shape[1]}), got shape {jnp.shape(length_scale)}")
    else:
        _scalar(length_scale, fn, "length_scale")
    diff = (x1[:, None, :] - x2[None, :, :]) / length_scale
    return jnp.sum(diff * diff, axis=-1)


def _distance(s):
    """``sqrt(s)`` with a zero (not ``NaN``) gradient at ``s = 0`` --- see the module docstring.

    The inner ``where`` keeps ``sqrt`` away from zero so its derivative is finite everywhere; the
    outer one selects the true value. A single ``where`` would not do: ``where`` differentiates both
    branches, and ``inf * 0`` from the branch it did not pick still poisons the gradient.
    """
    pos = s > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, s, 1.0)), 0.0)


def _stationary(fn: str, args, profile):
    """The four ARD-capable kernels: ``sigma^2 * profile(s)``, ``s`` the scaled squared distance."""
    x1, x2, (sigma, length_scale) = _split(fn, args, ("sigma", "length_scale"))
    s = _scaled_sq_distance(fn, x1, x2, length_scale, ard=True)
    return jnp.square(_scalar(sigma, fn, "sigma")) * profile(s)


def gp_exp_quad_cov(*args):
    """``sigma^2 exp(-|x - x'|^2 / (2 l^2))`` --- the squared-exponential (RBF) kernel."""
    return _stationary("gp_exp_quad_cov", args, lambda s: jnp.exp(-0.5 * s))


def gp_exponential_cov(*args):
    """``sigma^2 exp(-|x - x'| / l)`` --- Matern 1/2, the Ornstein--Uhlenbeck kernel."""
    return _stationary("gp_exponential_cov", args, lambda s: jnp.exp(-_distance(s)))


def _matern32(s):
    r = jnp.sqrt(3.0) * _distance(s)
    return (1.0 + r) * jnp.exp(-r)


def gp_matern32_cov(*args):
    """``sigma^2 (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)``, ``r = |x - x'|``."""
    return _stationary("gp_matern32_cov", args, _matern32)


def _matern52(s):
    r = jnp.sqrt(5.0) * _distance(s)
    return (1.0 + r + (5.0 / 3.0) * s) * jnp.exp(-r)


def gp_matern52_cov(*args):
    """``sigma^2 (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)``, ``r = |x - x'|``."""
    return _stationary("gp_matern52_cov", args, _matern52)


def gp_periodic_cov(*args):
    """``sigma^2 exp(-2 sin^2(pi |x - x'| / p) / l^2)`` --- no ARD form, as in Stan.

    **Scalar inputs only --- a deliberate departure from Stan**, which also accepts points in
    ``D > 1`` dimensions. There ``sin^2`` is applied to a *Euclidean* distance, which is not
    positive definite: the matrix was indefinite in 200/200 random draws at ``D = 2`` and ``3`` (and
    PSD in 200/200 at ``D = 1``). The symptom would be a ``NaN`` from ``cholesky`` or
    ``multi_normal`` far from the call that caused it, so the shape is refused here instead. The
    valid multi-dimensional periodic kernel is the product of one 1-d kernel per dimension.
    """
    fn = "gp_periodic_cov"
    x1, x2, (sigma, length_scale, period) = _split(fn, args, ("sigma", "length_scale", "period"))
    if x1.shape[1] > 1:
        raise ValueError(
            f"{fn}: inputs must be scalar (got points of dimension {x1.shape[1]}). On "
            "multi-dimensional points this kernel is not positive definite, so its matrix is not "
            "a covariance; multiply one 1-d periodic kernel per dimension instead, e.g. "
            "gp_periodic_cov(x[:, 1], ...) .* gp_periodic_cov(x[:, 2], ...)")
    r = _distance(_scaled_sq_distance(fn, x1, x2, 1.0, ard=False))
    l, p = _scalar(length_scale, fn, "length_scale"), _scalar(period, fn, "period")
    return jnp.square(_scalar(sigma, fn, "sigma")) * jnp.exp(
        -2.0 * jnp.square(jnp.sin(jnp.pi * r / p)) / jnp.square(l))


def gp_dot_prod_cov(*args):
    """``sigma^2 + x . x'`` --- Stan's dot-product kernel, ``sigma`` the bias (no length scale)."""
    fn = "gp_dot_prod_cov"
    x1, x2, (sigma,) = _split(fn, args, ("sigma",))
    return jnp.square(_scalar(sigma, fn, "sigma")) + x1 @ x2.T


def add_diag(m, d):
    """Stan's ``add_diag(m, d)``: ``m`` with ``d`` (a scalar, or one value per row) added to its
    diagonal --- the jitter or noise variance a GP covariance needs before ``cholesky``."""
    m = jnp.asarray(m)
    if m.ndim != 2:
        raise ValueError(f"add_diag: m must be a matrix, got shape {m.shape}")
    k = min(m.shape)
    if jnp.ndim(d) not in (0, 1) or (jnp.ndim(d) == 1 and jnp.shape(d)[0] != k):
        raise ValueError(f"add_diag: d must be a scalar or have {k} entries, got shape "
                         f"{jnp.shape(d)}")
    i = jnp.arange(k)
    return m.at[i, i].add(d)


GP_BUILTINS = {
    "gp_exp_quad_cov": gp_exp_quad_cov, "gp_dot_prod_cov": gp_dot_prod_cov,
    "gp_exponential_cov": gp_exponential_cov, "gp_matern32_cov": gp_matern32_cov,
    "gp_matern52_cov": gp_matern52_cov, "gp_periodic_cov": gp_periodic_cov,
    "add_diag": add_diag,
}
