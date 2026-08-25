"""Builtin functions for the model DSL: a curated table over ``jax.numpy``.

**Names follow JAX/NumPy, not Stan.** A model written here should read like the same model written
as a plain JAX function over a :class:`~mimcs.model.Model` --- that is the translation a user of this
library actually makes, whereas a Stan program already needs rewriting for the array semantics. So
it is ``det`` and ``slogdet`` rather than ``determinant`` and ``log_determinant``, ``solve`` rather
than ``mdivide_left``, and ``solve_triangular`` rather than ``mdivide_left_tri_low``.

Everything here is differentiable in its arguments, which is what lets a sampler take gradients of
a target that uses it. Three carry sharp edges, all documented in
``docs/reference/model_dsl.md``:

* ``slogdet`` returns a **pair** ``(sign, logabsdet)``, as in JAX. Destructure it ---
  ``(real s, real ld) = slogdet(A);`` --- since the DSL has no way to index a tuple.
* ``solve_triangular`` takes ``lower`` as an optional **third positional argument** (the DSL has no
  keyword arguments), and it defaults to **true**, unlike JAX. Both Cholesky types in this library
  --- :class:`~mimcs.model.CholeskyFactorCovParameter` and
  :class:`~mimcs.model.CholeskyFactorCorrParameter` --- and the ``cholesky`` builtin all produce
  *lower* triangular factors, so the lower solve is the one nearly every call wants; keeping JAX's
  default here would make the common case the one that needs an argument, and the silent failure
  mode is reading the wrong triangle. Pass ``0`` for an upper solve. It must be a literal: the flag
  selects a triangle at trace time, so a computed value is rejected.
* ``eigvals`` returns **complex** values for a general matrix, and the DSL has no complex type.
  ``abs(eigvals(A))`` --- the moduli, and so the spectral radius under ``max`` --- is the usable
  form. For a symmetric matrix use ``eigvalsh``, which is real.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy.special as jsp


def _solve_triangular(a, b, lower=1):
    """``solve_triangular(a, b[, lower])`` --- ``lower`` positional, since the DSL has no kwargs.

    **The default is ``lower=1``, which is not JAX's.** Deliberate: ``cholesky`` and both of this
    library's Cholesky parameter types produce lower-triangular factors, so a lower solve is what
    nearly every call here wants, and a default that silently reads the *other* triangle is a wrong
    answer rather than an error. Pass ``0`` for an upper solve.

    ``bool(lower)`` raises on a traced value rather than silently reading the wrong triangle,
    which is the failure mode worth having: the flag is a structural choice, not data.
    """
    return jsl.solve_triangular(a, b, lower=bool(lower))


BUILTINS = {
    "exp": jnp.exp, "log": jnp.log, "log1p": jnp.log1p, "expm1": jnp.expm1,
    "sqrt": jnp.sqrt, "abs": jnp.abs, "fabs": jnp.abs, "square": jnp.square,
    "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "tanh": jnp.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
    "sum": jnp.sum, "prod": jnp.prod, "mean": jnp.mean, "min": jnp.min, "max": jnp.max,
    "dot": jnp.dot, "transpose": jnp.transpose, "inverse": jnp.linalg.inv, "diag": jnp.diag,
    "floor": jnp.floor, "ceil": jnp.ceil, "lgamma": jsp.gammaln,
    # --- linear algebra ---
    "solve": jnp.linalg.solve, "solve_triangular": _solve_triangular,
    "cholesky": jnp.linalg.cholesky, "trace": jnp.trace,
    "det": jnp.linalg.det, "slogdet": jnp.linalg.slogdet,
    "eigvals": jnp.linalg.eigvals, "eigvalsh": jnp.linalg.eigvalsh,
}
