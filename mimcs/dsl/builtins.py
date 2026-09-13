"""Builtin functions for the model DSL: a curated table over ``jax.numpy``.

**Names follow JAX/NumPy, not Stan.** A model written here should read like the same model written
as a plain JAX function over a :class:`~mimcs.model.Model` --- that is the translation a user of this
library actually makes, whereas a Stan program already needs rewriting for the array semantics. So
it is ``det`` and ``slogdet`` rather than ``determinant`` and ``log_determinant``, ``solve`` rather
than ``mdivide_left``, and ``solve_triangular`` rather than ``mdivide_left_tri_low``.

**Most of these are differentiable in their arguments**, which is what lets a sampler take gradients
of a target that uses them. The predicates are the exception and are here for a different job:
``any``, ``all``, the ``logical_*`` family, ``isfinite`` and ``isnan`` return booleans, and ``where``
is differentiable in the two values it chooses between but **not** in its condition. That is not a
defect --- a comparison has no useful derivative --- but it does mean a gradient can be identically
zero in a direction the model appears to depend on.

Four carry sharp edges, all documented in ``docs/reference/model_dsl.md``:

* ``where`` **evaluates both branches, so a NaN in the branch it does not pick still poisons the
  gradient.** ``where(x > 0, sqrt(x), 0.0)`` at ``x = -1`` returns the right value, ``0.0``, and a
  gradient of ``NaN``: JAX differentiates ``sqrt`` at a negative argument and the ``NaN`` survives
  being multiplied by zero. It survives batching too --- under ``vmap(grad(...))`` the whole batch
  is poisoned, and this library vmaps densities in its own hot paths. In a sampler a ``NaN``
  gradient surfaces as a divergence or a stuck chain, a long way from the line that caused it. Guard
  the *argument* (``sqrt(abs(x))``), or use ``cond``, which does not evaluate the branch it does not
  take. See :mod:`mimcs.dsl.loops`.
* ``max`` and ``min`` are **reductions**, as in NumPy, so ``max(x, 0)`` reads the ``0`` as an *axis*
  and returns the largest element rather than clamping at zero --- a wrong answer with no error.
  ``maximum`` and ``minimum`` are the elementwise two-argument forms, and ``clip(x, lo, hi)`` is the
  usual way to say both at once.

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
* ``norm`` is ``jnp.linalg.norm`` with JAX's own positional signature, ``norm(x, ord, axis)``.
  Since ``None`` is an ordinary value here, ``norm(A, None, 1)`` gives per-row norms of a matrix ---
  the default ``ord`` with an explicit ``axis`` --- and ``norm(x, inf)`` the max-norm.
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


def where(condition, x, y):
    """``where(condition, x, y)`` --- elementwise choice, with the one-argument form refused.

    A thin wrapper purely for the arity. ``jnp.where(condition)`` with a single argument is *legal*
    JAX --- it returns index arrays whose shape depends on the data --- so the mistake surfaces as a
    ``ConcretizationTypeError`` about "the size argument of" from deep inside a trace, saying
    nothing about the program. Three required positional arguments turn that into
    ``where() missing 2 required positional arguments: 'x' and 'y'``, which names the call the user
    actually wrote. (Defining it as ``where`` rather than ``_where`` is what puts the right name in
    that message.)

    **Both branches are evaluated**, which is JAX's semantics and the sharp edge documented above:
    a value that is fine where it is selected and ``NaN`` where it is not will still poison the
    *gradient*. ``cond`` is the form that does not.
    """
    return jnp.where(condition, x, y)


BUILTINS = {
    "exp": jnp.exp, "log": jnp.log, "log1p": jnp.log1p, "expm1": jnp.expm1,
    "sqrt": jnp.sqrt, "abs": jnp.abs, "fabs": jnp.abs, "square": jnp.square,
    "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "tanh": jnp.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
    "sum": jnp.sum, "prod": jnp.prod, "mean": jnp.mean, "min": jnp.min, "max": jnp.max,
    "dot": jnp.dot, "transpose": jnp.transpose, "inverse": jnp.linalg.inv, "diag": jnp.diag,
    "floor": jnp.floor, "ceil": jnp.ceil, "lgamma": jsp.gammaln,
    # --- conditional values and predicates ---
    # `where` is wrapped only to fix its arity; everything else is its JAX callable verbatim.
    "where": where, "clip": jnp.clip,
    "maximum": jnp.maximum, "minimum": jnp.minimum,     # ELEMENTWISE -- `max`/`min` are reductions
    "any": jnp.any, "all": jnp.all,
    "logical_and": jnp.logical_and, "logical_or": jnp.logical_or,
    "logical_not": jnp.logical_not, "logical_xor": jnp.logical_xor,
    "isfinite": jnp.isfinite, "isnan": jnp.isnan,
    # --- linear algebra ---
    "solve": jnp.linalg.solve, "solve_triangular": _solve_triangular,
    "cholesky": jnp.linalg.cholesky, "trace": jnp.trace,
    "det": jnp.linalg.det, "slogdet": jnp.linalg.slogdet,
    "eigvals": jnp.linalg.eigvals, "eigvalsh": jnp.linalg.eigvalsh,
    "norm": jnp.linalg.norm,
}
