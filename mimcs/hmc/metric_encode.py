"""Design columns for a **discrete** dependency of a learned metric.

A metric may depend on integer parameters as well as continuous ones
(``docs/design/14_discrete_parameters.md``). The expression language itself stays untouched: an
atom computes ``link(W @ feat + b)`` over a real feature vector, and a discrete dependency is
turned into such a vector *here*, before it ever reaches the expression. That is why adding
discrete metrics needed no change to :mod:`mimcs.hmc.metric_expr`'s arithmetic.

**One definition, two callers.** The runtime block
(:class:`~mimcs.hmc.block_riemannian.LearnedDiagonalBlock`) and the offline regression
(:mod:`mimcs.factory.regression`) must produce *identical* columns for the same labels, or a
metric fitted from a pilot would mean something different when the sampler evaluates it --- with
no error and a perfectly plausible number either way. So both call this, and a test asserts they
agree rather than trusting that they do.

Two codings, and the distinction only matters above two values:

* **ordinal** --- the value itself, standardized by its **declared support** rather than by any
  observed draws: ``mean = (lo + hi) / 2`` and the sd of the uniform distribution on
  ``{lo..hi}``. Support-based because the transform has to be defined with **no evidence at all**:
  :class:`~mimcs.adaptation.MetricAdaptation` starts from scratch and a hand-written metric never
  sees a pilot, and a metric that meant one thing when fitted and another when built cold would be
  a bad kind of surprise. One column per coordinate.
* **categorical** --- **reference coding**: ``k - 1`` indicator columns per coordinate, the lowest
  value being the reference. Not full one-hot: those columns sum to one and so duplicate the
  additive ``+ Exp()`` constant every candidate is paired with, leaving a direction the optimizer
  cannot resolve and AIC still charges for.

For ``k = 2`` the two coincide --- the single reference indicator *is* an affine function of the
standardized value --- so they span the same models and the candidate enumeration offers only one.
That is a fact the enumeration relies on, and it is asserted in the tests.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

#: the coding kinds a discrete dependency may declare.
KINDS = ("categorical", "ordinal")


def support_moments(lower: int, upper: int) -> tuple[float, float]:
    """Mean and sd of the uniform distribution on ``{lower..upper}`` --- the ordinal transform.

    The sd of a discrete uniform over ``n`` consecutive integers is ``sqrt((n^2 - 1) / 12)``, and
    is **floored at 1** for a degenerate one-value support so the transform stays finite: such a
    coordinate carries no information either way, and dividing by zero would poison the whole
    metric rather than that one column.
    """
    n = int(upper) - int(lower) + 1
    mean = (int(lower) + int(upper)) / 2.0
    var = (n * n - 1) / 12.0
    return mean, math.sqrt(var) if var > 0.0 else 1.0


def encoded_width(size: int, kind: str, lower: int, upper: int) -> int:
    """How many design columns ``size`` coordinates of this dependency contribute.

    This is the number the expression sees as the dependency's ``dep_dim``, so it is what sets
    ``W``'s shape and what the parameter count --- and therefore AIC --- is computed from.
    """
    _check_kind(kind)
    if kind == "ordinal":
        return int(size)
    return int(size) * max(int(upper) - int(lower), 0)     # k - 1 columns per coordinate


def encode_discrete(values: Array, kind: str, lower: int, upper: int) -> Array:
    """``(size,)`` integer labels -> ``(encoded_width,)`` float design columns.

    Traceable: ``values`` may be a JAX array under ``jit``/``vmap``, and the width is static
    (it comes from the declared support, never from the data).
    """
    _check_kind(kind)
    x = jnp.asarray(values)
    if kind == "ordinal":
        mean, sd = support_moments(lower, upper)
        return (jnp.asarray(x, float) - mean) / sd
    levels = jnp.arange(int(lower) + 1, int(upper) + 1, dtype=x.dtype)   # drop the reference
    if levels.shape[0] == 0:                       # a one-value support encodes to nothing
        return jnp.zeros((0,), float)
    # (size, k-1) indicators, flattened coordinate-major so column order matches `encoded_width`.
    return jnp.asarray(x[:, None] == levels[None, :], float).reshape(-1)


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown discrete metric coding {kind!r} (use one of {list(KINDS)})")
