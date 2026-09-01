"""A mini-language for position-dependent diagonal mass matrices (block RMHMC).

A learned block metric is a small algebraic *expression* over other blocks' coordinates,
built from two atoms and two combinators::

    Exp("v") + Exp()                        # exp(W v + b) + exp(b0)
    Exp() * Sigmoid("v", "x") + Exp()       # exp(b1) sigma(W v + U x + c) + exp(b2)

The value of an expression, evaluated for a block of ``block_dim`` coordinates, is a
positive vector (the diagonal of ``M_i``), a function of the dependency coordinates.

Atoms (``Exp``, ``Sigmoid``) are ``link(sum_d W_d @ feat(coord_d) + b)`` with
``link in {exp, sigma}``; a dep-less atom (``Exp()``/``Sigmoid()``) is the pure-bias case
``link(b)`` --- a learnable per-coordinate constant (an ``Exp()`` factor is thus a positive
scale, an ``Exp()`` term a baseline). Every atom carries a per-coordinate bias, so the bias
is a built-in per-term scale and the sum-of-exp form generalizes the previous
``depends_on=``-based ``LearnedDiagonalBlock`` exactly. ``feat`` is the identity by default
or per-coordinate quadratic (``[x, x^2]``, no interactions).

**Positivity is structural**: ``exp > 0``, ``sigma in (0, 1)``, and sums/products of
positives are positive, so *every* expression is a valid (positive) diagonal mass.

Each node implements a uniform interface:

* ``deps()`` -- the set of referenced dependency-block names;
* ``init_params(block_dim, dep_dims)`` -- a parameter pytree (nested dict/list mirroring the
  expression) with weights zero and biases set so the whole expression is ~ ``I`` at init;
* ``evaluate(params, dep_coords)`` -- the ``(block_dim,)`` diagonal, ``dep_coords`` a
  ``{name: coordinate_vector}`` map;
* ``n_params(block_dim, dep_dims)`` -- the parameter count (for the factory's dimension-aware
  candidate budget).

The parameters are adapted online by :class:`mimcs.adaptation.MetricAdaptation` (SGD on the KL
objective) or fitted offline by the factory's regression --- both differentiate the same pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

_CLIP = 1e-3          # keep sigmoid init targets inside (0, 1)


def _as_names(value) -> tuple:
    """``None`` / a bare name / an iterable of names -> a tuple of names.

    A lone string is one name, not four characters --- which is what anyone writing
    ``categorical="z"`` intends, and the reading that would otherwise fail much later as an
    unresolvable dependency called ``'z'``'s first letter.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _feat(x: Array, features: str) -> Array:
    """Feature map of a dependency coordinate: identity, or per-coordinate quadratic."""
    return x if features == "identity" else jnp.concatenate([x, x ** 2])


def _feat_dim(n: int, features: str) -> int:
    return n if features == "identity" else 2 * n


def _sparse_feat(x: Array, features: str) -> Array:
    """Per-coordinate feature for a sparse (elementwise) dependency: ``(n,) -> (n, per_feat)``.

    Identity gives ``[x_j]`` per coordinate; quadratic gives ``[x_j, x_j^2]`` (no interactions).
    """
    return x[:, None] if features == "identity" else jnp.stack([x, x ** 2], axis=-1)


def _sparse_feat_dim(features: str) -> int:
    """Per-coordinate feature dimension for a sparse atom (1 identity, 2 quadratic)."""
    return 1 if features == "identity" else 2


class MetricExpr:
    """Base class for mass-matrix expressions; ``+`` builds :class:`Sum`, ``*`` :class:`Product`."""

    def __add__(self, other):
        if not isinstance(other, MetricExpr):
            return NotImplemented
        return Sum(self, other)

    def __mul__(self, other):
        if not isinstance(other, MetricExpr):
            return NotImplemented
        return Product(self, other)

    # interface (subclasses implement) ------------------------------------- #

    def deps(self) -> set[str]:
        raise NotImplementedError

    def discrete_deps(self) -> set[str]:
        """The **discrete** (integer-parameter) dependencies, if any.

        Kept separate from :meth:`deps` rather than folded into it, and deliberately so: every
        existing caller of ``deps()`` resolves a name against the *continuous* coordinate layout
        (``build_block``, ``select_metric``, ``learned_metric_rule``), and a discrete name is not
        in that layout at all. Widening ``deps()`` would have made each of those silently wrong;
        a second accessor makes each of them explicitly incomplete until it is taught.
        """
        return set()

    def dep_kind(self, name: str) -> str | None:
        """``"categorical"`` / ``"ordinal"`` for a discrete dependency, else ``None``."""
        return None

    def _n_add(self) -> int:
        """Number of additive terms (through :class:`Sum`; a product/atom counts as one)."""
        raise NotImplementedError

    def init_params(self, block_dim: int, dep_dims: dict[str, int], target=1.0):
        """Initial parameters: weights zero, biases set so the expression evaluates to ``target``.

        ``target`` is a scalar (default 1.0 --- the metric starts at ``I``) or a ``(block_dim,)``
        array giving a **per-coordinate** scale. The latter matters whenever the target's score
        magnitude is far from 1: the KL loss is exponentially steep below its optimum and nearly
        linear (slope 1/2) above it, so a fit started orders of magnitude too low overshoots into
        the flat region and cannot walk back (see ``docs/design/09``).
        """
        raise NotImplementedError

    def evaluate(self, params, dep_coords: dict[str, Array]) -> Array:
        raise NotImplementedError

    def n_params(self, block_dim: int, dep_dims: dict[str, int]) -> int:
        raise NotImplementedError


class _Atom(MetricExpr):
    """``link(sum_d W_d @ feat(coord_d) + b)`` --- an ``Exp`` or ``Sigmoid`` term.

    ``params`` is ``{"W": [W_d, ...], "b": b}`` with ``W_d`` of shape
    ``(block_dim, feat_dim_d)`` (one per dependency, in declaration order) and ``b`` of shape
    ``(block_dim,)``. A dep-less atom has ``W = []`` and value ``link(b)``.
    """

    def __init__(self, *deps: str, features: str = "identity", categorical=None, ordinal=None):
        cat, ordi = _as_names(categorical), _as_names(ordinal)
        clash = sorted((set(deps) & (set(cat) | set(ordi))) | (set(cat) & set(ordi)))
        if clash:
            raise ValueError(
                f"dependency name(s) {clash} given more than once to {type(self).__name__}: a "
                f"dependency is continuous, categorical or ordinal, not two of them")
        # One ordered tuple, continuous first: `params["W"][k]` is indexed positionally against
        # `enumerate(self.dep_names)`, so this slot order *is* the parameter layout.
        self.dep_names = tuple(deps) + cat + ordi
        self._kinds = {**{d: "categorical" for d in cat}, **{d: "ordinal" for d in ordi}}
        self.features = features

    # link -------------------------------------------------------------- #

    def _link(self, x: Array) -> Array:
        raise NotImplementedError

    def _inv_link(self, target) -> Array:
        """Bias making ``link(b) == target`` at zero weights (so init hits its share of the
        target scale). ``target`` is a scalar or a ``(block_dim,)`` array (a *per-coordinate*
        scale, e.g. the empirical score second moment --- see :func:`mimcs.factory.regression
        .fit_metric_expr`); the returned bias broadcasts to the block."""
        raise NotImplementedError

    # interface --------------------------------------------------------- #

    def deps(self) -> set[str]:
        return set(self.dep_names) - set(self._kinds)

    def discrete_deps(self) -> set[str]:
        return set(self._kinds)

    def dep_kind(self, name: str) -> str | None:
        return self._kinds.get(name)

    def _n_add(self) -> int:
        return 1

    def init_params(self, block_dim, dep_dims, target=1.0):
        W = [jnp.zeros((block_dim, _feat_dim(dep_dims[d], self.features)))
             for d in self.dep_names]
        b = jnp.zeros((block_dim,)) + self._inv_link(target)   # scalar or per-coordinate target
        return {"W": W, "b": b}

    def evaluate(self, params, dep_coords):
        pre = params["b"]
        for k, d in enumerate(self.dep_names):
            pre = pre + params["W"][k] @ _feat(dep_coords[d], self.features)
        return self._link(pre)

    def n_params(self, block_dim, dep_dims):
        w = sum(block_dim * _feat_dim(dep_dims[d], self.features) for d in self.dep_names)
        return w + block_dim

    def __repr__(self):
        parts = [repr(d) for d in self.dep_names if d not in self._kinds]
        if self.features != "identity":
            parts.append(f"features={self.features!r}")
        for kind in ("categorical", "ordinal"):
            named = [d for d in self.dep_names if self._kinds.get(d) == kind]
            if named:
                parts.append(f"{kind}={named!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


class Exp(_Atom):
    """``exp(sum_d W_d @ feat(coord_d) + b)`` --- a positive log-linear term/scale."""

    def _link(self, x):
        return jnp.exp(x)

    def _inv_link(self, target):
        return jnp.log(jnp.asarray(target, float))


class Sigmoid(_Atom):
    """``sigma(sum_d W_d @ feat(coord_d) + b)`` --- a smooth gate in ``(0, 1)``."""

    def _link(self, x):
        return jax.nn.sigmoid(x)

    def _inv_link(self, target):
        t = jnp.clip(jnp.asarray(target, float), _CLIP, 1.0 - _CLIP)
        return jnp.log(t / (1.0 - t))


class _SparseAtom(_Atom):
    """A *sparse* (elementwise) atom --- ``link(sum_d W_d[j,:] . feat(dep_d,j) + b_j)``.

    Coordinate ``j`` of the block depends only on coordinate ``j`` of each dependency (a
    bijective row correspondence between equal-dimension arrays, e.g. a horseshoe's per-element
    scale ``lambda_j`` for ``x_j``), with **no sum over the other dependency coordinates**. Each
    dependency must therefore have ``dep_dim == block_dim``. Parameters are ``{"W": [W_d, ...],
    "b": b}`` with ``W_d`` of shape ``(block_dim, per_feat)`` (per_feat = 1 identity / 2
    quadratic) and ``b`` of shape ``(block_dim,)``.

    Only the numeric methods differ from the dense :class:`_Atom`; the link and the
    ``deps``/``_n_add``/``__repr__`` machinery are inherited. Concrete classes multiply-inherit a
    link (``SpExp(_SparseAtom, Exp)``), so no link code is duplicated.
    """

    def init_params(self, block_dim, dep_dims, target=1.0):
        for d in self.dep_names:
            if dep_dims[d] != block_dim:
                raise ValueError(
                    f"sparse metric {self!r} needs dependency '{d}' to match the block "
                    f"dimension ({dep_dims[d]} != {block_dim})")
        pf = _sparse_feat_dim(self.features)
        W = [jnp.zeros((block_dim, pf)) for _ in self.dep_names]
        b = jnp.zeros((block_dim,)) + self._inv_link(target)   # scalar or per-coordinate target
        return {"W": W, "b": b}

    def evaluate(self, params, dep_coords):
        pre = params["b"]
        for k, d in enumerate(self.dep_names):
            pre = pre + jnp.sum(params["W"][k] * _sparse_feat(dep_coords[d], self.features),
                                axis=-1)
        return self._link(pre)

    def n_params(self, block_dim, dep_dims):
        pf = _sparse_feat_dim(self.features)
        return len(self.dep_names) * block_dim * pf + block_dim


class SpExp(_SparseAtom, Exp):
    """``exp(sum_d W_d[j] . feat(dep_d,j) + b_j)`` --- an elementwise (sparse) log-linear term."""


class SpSigmoid(_SparseAtom, Sigmoid):
    """``sigma(sum_d W_d[j] . feat(dep_d,j) + b_j)`` --- an elementwise (sparse) gate."""


class Sum(MetricExpr):
    """``a + b`` --- elementwise sum over the block's coordinates. Params ``[a_params, b_params]``."""

    def __init__(self, a: MetricExpr, b: MetricExpr):
        self.a, self.b = a, b

    def deps(self):
        return self.a.deps() | self.b.deps()

    def discrete_deps(self):
        return self.a.discrete_deps() | self.b.discrete_deps()

    def dep_kind(self, name):
        return self.a.dep_kind(name) or self.b.dep_kind(name)

    def _n_add(self):
        return self.a._n_add() + self.b._n_add()

    def init_params(self, block_dim, dep_dims, target=1.0):
        na, nb = self.a._n_add(), self.b._n_add()
        n = na + nb
        return [self.a.init_params(block_dim, dep_dims, target * na / n),
                self.b.init_params(block_dim, dep_dims, target * nb / n)]

    def evaluate(self, params, dep_coords):
        return self.a.evaluate(params[0], dep_coords) + self.b.evaluate(params[1], dep_coords)

    def n_params(self, block_dim, dep_dims):
        return self.a.n_params(block_dim, dep_dims) + self.b.n_params(block_dim, dep_dims)

    def __repr__(self):
        return f"{self.a!r} + {self.b!r}"


class Product(MetricExpr):
    """``a * b`` --- elementwise product. Params ``[a_params, b_params]``.

    Init puts the whole target scale on the first factor and neutral (~1) on the rest, so a
    ``Exp() * Sigmoid(...)`` term initialises near its ``Exp()`` scale (write the scale first).
    """

    def __init__(self, a: MetricExpr, b: MetricExpr):
        self.a, self.b = a, b

    def deps(self):
        return self.a.deps() | self.b.deps()

    def discrete_deps(self):
        return self.a.discrete_deps() | self.b.discrete_deps()

    def dep_kind(self, name):
        return self.a.dep_kind(name) or self.b.dep_kind(name)

    def _n_add(self):
        return 1

    def init_params(self, block_dim, dep_dims, target=1.0):
        return [self.a.init_params(block_dim, dep_dims, target),
                self.b.init_params(block_dim, dep_dims, 1.0)]

    def evaluate(self, params, dep_coords):
        return self.a.evaluate(params[0], dep_coords) * self.b.evaluate(params[1], dep_coords)

    def n_params(self, block_dim, dep_dims):
        return self.a.n_params(block_dim, dep_dims) + self.b.n_params(block_dim, dep_dims)

    def __repr__(self):
        return f"{_paren(self.a)}*{_paren(self.b)}"


def _paren(e: MetricExpr) -> str:
    return f"({e!r})" if isinstance(e, Sum) else repr(e)
