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
* ``log_evaluate(params, dep_coords)`` -- its log, computed stably from the atoms' pre-activations
  (``Exp``: the pre-activation itself; ``Sigmoid``: ``log_sigmoid``; ``Sum``: ``logaddexp``;
  ``Product``: a sum of logs) rather than as ``log(evaluate(...))``, so a diagonal far below
  float range still has a finite log, and anything differentiated through it never forms
  ``M^{-2}``;
* ``n_params(block_dim, dep_dims)`` -- the parameter count (for the factory's dimension-aware
  candidate budget).

The parameters are adapted online by :class:`mimcs.adaptation.MetricAdaptation` (SGD on the KL
objective) or fitted offline by the factory's regression --- both differentiate the same pytree.
"""

from __future__ import annotations

import numpy as np
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

    def log_evaluate(self, params, dep_coords: dict[str, Array]) -> Array:
        """``log`` of :meth:`evaluate`, computed stably node by node (see the module docstring).

        This fallback is only for a node that does not override it."""
        return jnp.log(self.evaluate(params, dep_coords))

    def n_params(self, block_dim: int, dep_dims: dict[str, int]) -> int:
        raise NotImplementedError

    def with_sharing(self, shared_weights=(), shared_bias=()) -> "MetricExpr":
        """A copy of this expression with every atom's sharing set to the given axes.

        Sharing is declared per atom but *selected* per expression --- the factory offers a whole
        form with its weights pooled or not --- so one method rebuilds the tree rather than every
        caller reaching into the atoms. A dep-less atom has no weights, so
        ``with_sharing(shared_weights=(0,))`` leaves an ``Exp()`` floor per-coordinate: that is
        what makes ``SpExp(d) + Exp()`` mean "pool the funnel slope, keep a per-coordinate floor".
        """
        raise NotImplementedError

    def relabel(self, mapping: dict[str, str]) -> "MetricExpr":
        """A copy with every dependency name ``d`` replaced by ``mapping.get(d, d)``.

        Only the *names* change: link, sparsity, features, coding and sharing are kept, and so is
        each atom's continuous/categorical/ordinal slot order --- ``params["W"]`` is indexed
        positionally against that order, so a relabelled expression takes the original's parameter
        pytree unchanged.
        """
        raise NotImplementedError

    def dep_order(self) -> list[str]:
        """Every dependency name (continuous and discrete) in first-appearance order, each once."""
        raise NotImplementedError

    def canonical(self) -> tuple["MetricExpr", dict[str, str]]:
        """``(expr with dependencies renamed _d0, _d1, ... in first-appearance order, the map)``.

        Two expressions with the same canonical form are the same *computation* on different
        data: ``Exp('a') + Exp()`` and ``Exp('b') + Exp()`` differ only in which columns feed the
        dependency slot. The metric regression keys its compiled fits on this (see
        :func:`mimcs.factory.regression.structure_key`), passing the data per slot as arguments,
        so the pair shares one compilation. Widths are deliberately *not* part of it: they are
        array shapes, and the compiled function's own shape cache already distinguishes them.
        """
        mapping = {d: f"_d{k}" for k, d in enumerate(self.dep_order())}
        return self.relabel(mapping), mapping


#: the only block axis a weight can be shared over today (see :func:`_check_shared`).
BLOCK_AXES = (0,)


def _check_shared(axes, what: str) -> tuple:
    """Validate and normalise a ``shared_weights`` / ``shared_bias`` axis tuple.

    The axes index the **block** axes of the parameter --- currently exactly one, the block's
    coordinate axis --- and *not* the trailing feature axis. Two deliberate restrictions:

    * A block's coordinates are a flat contiguous slice by the time they reach a metric
      (``Model.coord_block``), so axis 0 is the only block axis that exists. Spelling it as a tuple
      rather than a bool is what lets a future coordinate *shape* add axes 1, 2, ... without
      reinterpreting anything already written.
    * The feature axis is not shareable. A dense atom computes ``W @ feat``, and a matmul cannot
      broadcast its contracted axis (it raises), so supporting it would mean replacing that matmul
      with a multiply-sum and changing the numerics of every existing fitted metric --- to buy two
      models nobody wants ("regress on the sum of the dependency coordinates" for a dense atom,
      "the same coefficient on x and x^2" for a quadratic sparse one).
    """
    if axes is None or axes is False:
        return ()
    if axes is True:
        return BLOCK_AXES
    axes = tuple(int(a) for a in axes)
    bad = sorted(set(axes) - set(BLOCK_AXES))
    if bad:
        raise ValueError(
            f"{what}={axes}: axis/axes {bad} cannot be shared. Only the block's coordinate axis "
            f"{BLOCK_AXES} is shareable today --- a block's coordinates are one flat axis by the "
            f"time they reach a metric, and the trailing feature axis is deliberately excluded.")
    return tuple(sorted(set(axes)))


class _Atom(MetricExpr):
    """``link(sum_d W_d @ feat(coord_d) + b)`` --- an ``Exp`` or ``Sigmoid`` term.

    ``params`` is ``{"W": [W_d, ...], "b": b}`` with ``W_d`` of shape
    ``(block_dim, feat_dim_d)`` (one per dependency, in declaration order) and ``b`` of shape
    ``(block_dim,)``. A dep-less atom has ``W = []`` and value ``link(b)``.
    """

    def __init__(self, *deps: str, features: str = "identity", categorical=None, ordinal=None,
                 shared_weights=(), shared_bias=()):
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
        # A dep-less atom has no weights, so it cannot share any: normalise that away rather than
        # carrying it, or `with_sharing` would stamp a meaningless `shared_weights=(0,)` onto every
        # `Exp()` floor and two identical expressions would print (and compare) differently.
        self.shared_weights = (_check_shared(shared_weights, "shared_weights")
                               if self.dep_names else ())
        self.shared_bias = _check_shared(shared_bias, "shared_bias")

    # link -------------------------------------------------------------- #

    def _link(self, x: Array) -> Array:
        raise NotImplementedError

    def _bias_init(self, target, rows: int) -> Array:
        """The bias value(s) that make the atom evaluate to ``target`` at zero weights.

        A **shared** bias (``rows == 1``) has to reduce a per-coordinate ``target`` first. The
        regression passes ``target`` as the ``(block_dim,)`` empirical second moment
        (:func:`mimcs.factory.regression.fit_metric_expr`), and ``jnp.zeros((1,)) +
        _inv_link(target)`` broadcasts straight back up to ``(block_dim,)`` --- so a bias declared
        shared would come back per-coordinate, be fitted as a dense one, and be charged the shared
        parameter count by AIC. The reduction is a **mean in link space**: for ``Exp`` that is the
        log-mean, i.e. the geometric mean of the per-coordinate scales, which is exactly the single
        value minimising the KL loss over the block at zero weights.
        """
        v = jnp.asarray(self._inv_link(target), float)
        return v if rows != 1 else jnp.reshape(jnp.mean(v), (1,))

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

    def _rows(self, block_dim: int) -> int:
        """Rows a weight/bias actually carries: ``1`` when the coordinate axis is shared."""
        return 1 if 0 in self.shared_weights else block_dim

    def _bias_rows(self, block_dim: int) -> int:
        return 1 if 0 in self.shared_bias else block_dim

    def param_shapes(self, block_dim, dep_dims):
        """``{"W": [shape, ...], "b": shape}`` --- the single source of truth for what
        :meth:`init_params` emits and what :meth:`n_params` counts, so the two cannot drift."""
        r = self._rows(block_dim)
        return {"W": [(r, _feat_dim(dep_dims[d], self.features)) for d in self.dep_names],
                "b": (self._bias_rows(block_dim),)}

    def init_params(self, block_dim, dep_dims, target=1.0):
        shapes = self.param_shapes(block_dim, dep_dims)
        b_shape = shapes["b"]
        return {"W": [jnp.zeros(sh) for sh in shapes["W"]],
                "b": jnp.zeros(b_shape) + self._bias_init(target, b_shape[0])}

    def _pre(self, params, dep_coords):
        """The pre-activation ``sum_d W_d @ feat(coord_d) + b``."""
        pre = params["b"]
        for k, d in enumerate(self.dep_names):
            pre = pre + params["W"][k] @ _feat(dep_coords[d], self.features)
        return pre

    def evaluate(self, params, dep_coords):
        return self._link(self._pre(params, dep_coords))

    def log_evaluate(self, params, dep_coords):
        return self._log_link(self._pre(params, dep_coords))

    def n_params(self, block_dim, dep_dims):
        shapes = self.param_shapes(block_dim, dep_dims)
        return sum(int(np.prod(sh)) for sh in shapes["W"]) + int(np.prod(shapes["b"]))

    def with_sharing(self, shared_weights=(), shared_bias=()):
        cont = [d for d in self.dep_names if d not in self._kinds]
        cat = [d for d in self.dep_names if self._kinds.get(d) == "categorical"]
        ordi = [d for d in self.dep_names if self._kinds.get(d) == "ordinal"]
        # Rebuilt as continuous + categorical + ordinal, the same order the constructor produces,
        # so `params["W"]`'s positional layout is preserved exactly.
        return type(self)(*cont, features=self.features,
                          categorical=cat or None, ordinal=ordi or None,
                          shared_weights=shared_weights, shared_bias=shared_bias)

    def relabel(self, mapping):
        def names(kind):
            return [mapping.get(d, d) for d in self.dep_names if self._kinds.get(d) == kind]
        # Same continuous + categorical + ordinal rebuild as `with_sharing`, so `params["W"]`'s
        # positional layout is untouched; the sharing is carried over as declared.
        return type(self)(*names(None), features=self.features,
                          categorical=names("categorical") or None,
                          ordinal=names("ordinal") or None,
                          shared_weights=self.shared_weights, shared_bias=self.shared_bias)

    def dep_order(self):
        return list(self.dep_names)

    def __repr__(self):
        parts = [repr(d) for d in self.dep_names if d not in self._kinds]
        if self.features != "identity":
            parts.append(f"features={self.features!r}")
        for kind in ("categorical", "ordinal"):
            named = [d for d in self.dep_names if self._kinds.get(d) == kind]
            if named:
                parts.append(f"{kind}={named!r}")
        # The sharing belongs in the repr: `block.params["metric"]` is the user-facing record of
        # what was selected, and two candidates differing only in sharing would otherwise print
        # identically in the spec, the logs and every study table.
        for name, axes in (("shared_weights", self.shared_weights),
                           ("shared_bias", self.shared_bias)):
            if axes:
                parts.append(f"{name}={axes!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


class Exp(_Atom):
    """``exp(sum_d W_d @ feat(coord_d) + b)`` --- a positive log-linear term/scale."""

    def _link(self, x):
        return jnp.exp(x)

    def _log_link(self, x):
        return x

    def _inv_link(self, target):
        return jnp.log(jnp.asarray(target, float))


class Sigmoid(_Atom):
    """``sigma(sum_d W_d @ feat(coord_d) + b)`` --- a smooth gate in ``(0, 1)``."""

    def _link(self, x):
        return jax.nn.sigmoid(x)

    def _log_link(self, x):
        return jax.nn.log_sigmoid(x)

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

    def param_shapes(self, block_dim, dep_dims):
        pf = _sparse_feat_dim(self.features)
        return {"W": [(self._rows(block_dim), pf) for _ in self.dep_names],
                "b": (self._bias_rows(block_dim),)}

    def init_params(self, block_dim, dep_dims, target=1.0):
        for d in self.dep_names:
            if dep_dims[d] != block_dim:
                raise ValueError(
                    f"sparse metric {self!r} needs dependency '{d}' to match the block "
                    f"dimension ({dep_dims[d]} != {block_dim})")
        shapes = self.param_shapes(block_dim, dep_dims)
        b_shape = shapes["b"]
        # A shared sparse weight stays well defined: `sum(W(1, pf) * feat(n, pf), -1)` is `(n,)`,
        # so the atom still evaluates over the whole block.
        return {"W": [jnp.zeros(sh) for sh in shapes["W"]],
                "b": jnp.zeros(b_shape) + self._bias_init(target, b_shape[0])}

    def _pre(self, params, dep_coords):
        pre = params["b"]
        for k, d in enumerate(self.dep_names):
            pre = pre + jnp.sum(params["W"][k] * _sparse_feat(dep_coords[d], self.features),
                                axis=-1)
        return pre

    def n_params(self, block_dim, dep_dims):
        shapes = self.param_shapes(block_dim, dep_dims)
        return sum(int(np.prod(sh)) for sh in shapes["W"]) + int(np.prod(shapes["b"]))


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

    def log_evaluate(self, params, dep_coords):
        return jnp.logaddexp(self.a.log_evaluate(params[0], dep_coords),
                             self.b.log_evaluate(params[1], dep_coords))

    def n_params(self, block_dim, dep_dims):
        return self.a.n_params(block_dim, dep_dims) + self.b.n_params(block_dim, dep_dims)

    def with_sharing(self, shared_weights=(), shared_bias=()):
        return type(self)(self.a.with_sharing(shared_weights, shared_bias),
                          self.b.with_sharing(shared_weights, shared_bias))

    def relabel(self, mapping):
        return type(self)(self.a.relabel(mapping), self.b.relabel(mapping))

    def dep_order(self):
        return _merge_order(self.a.dep_order(), self.b.dep_order())

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

    def log_evaluate(self, params, dep_coords):
        return self.a.log_evaluate(params[0], dep_coords) + self.b.log_evaluate(params[1], dep_coords)

    def n_params(self, block_dim, dep_dims):
        return self.a.n_params(block_dim, dep_dims) + self.b.n_params(block_dim, dep_dims)

    def with_sharing(self, shared_weights=(), shared_bias=()):
        return type(self)(self.a.with_sharing(shared_weights, shared_bias),
                          self.b.with_sharing(shared_weights, shared_bias))

    def relabel(self, mapping):
        return type(self)(self.a.relabel(mapping), self.b.relabel(mapping))

    def dep_order(self):
        return _merge_order(self.a.dep_order(), self.b.dep_order())

    def __repr__(self):
        return f"{_paren(self.a)}*{_paren(self.b)}"


def check_params(expr: MetricExpr, params, block_dim: int, dep_dims: dict, *,
                 what: str = "metric parameters") -> None:
    """Raise unless ``params`` matches what ``expr.init_params(block_dim, dep_dims)`` would emit.

    Nothing else in the library validates a metric parameter pytree, and that absence is precisely
    why a wrong one is *silent*: a leaf of the wrong leading length broadcasts rather than raising,
    so a metric fitted under one sharing pattern and paired with another expression samples happily
    from the wrong Hamiltonian. A supplied ``metric_init`` is user-reachable by hand
    (``mimcs.factory.spec.BlockSpec.params`` invites exactly that), so it is checked where it
    enters the sampler rather than trusted.

    Structure *and* per-leaf shape, because they fail differently: a wrong structure is a
    ``tree_map`` error somewhere later, a wrong shape is no error at all.
    """
    want = expr.init_params(block_dim, dep_dims)
    got_t, want_t = (jax.tree_util.tree_structure(params),
                     jax.tree_util.tree_structure(want))
    if got_t != want_t:
        raise ValueError(f"{what} do not match {expr!r}: expected pytree {want_t}, got {got_t}")
    for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(params),
                                   jax.tree_util.tree_leaves(want))):
        if jnp.shape(a) != jnp.shape(b):
            raise ValueError(
                f"{what} do not match {expr!r}: leaf {i} has shape {jnp.shape(a)}, expected "
                f"{jnp.shape(b)} for a block of {block_dim} coordinate(s). A leading axis of 1 "
                f"means a parameter shared across the block --- declare it on the atom "
                f"(shared_weights=/shared_bias=) rather than supplying a differently shaped init.")


def _paren(e: MetricExpr) -> str:
    return f"({e!r})" if isinstance(e, Sum) else repr(e)


def _merge_order(first: list[str], second: list[str]) -> list[str]:
    """``first`` then the names of ``second`` not already in it --- first-appearance order."""
    return first + [d for d in second if d not in first]
