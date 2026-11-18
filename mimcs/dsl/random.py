"""Random variates for ``generated quantities``: ``<name>_rng(...)``, one per ``jax.random`` sampler.

Stan names its variates ``normal_rng`` and so on; the DSL keeps that suffix, which is what marks
a call as consuming randomness. The **signatures are JAX's**, not Stan's: ``<name>_rng(args)`` is
``jax.random.<name>(key, args)`` with the key supplied by the program, and ``dtype`` and the other
implementation knobs (``mode``, ``out_sharding``, ``unroll``, ``method``) left at JAX's defaults
--- the dtype is the canonical float, so x64 needs nothing extra. Hence ``normal_rng()`` is a
*standard* normal (write ``mu + sigma * normal_rng(shape)``), ``exponential_rng(shape)`` has unit
rate, and ``gamma_rng(a)`` unit scale. A Stan port's ``normal_rng(mu, sigma)`` is an arity error
here, never a silent wrong answer.

Arguments are positional, in JAX's order, and trailing ones may be omitted to take JAX's default.
A ``shape`` argument must be a compile-time constant: an integer ``n`` means ``(n,)``, a tuple
``(n, k)`` is taken as is, and ``None`` means JAX's default --- a scalar where that default is
``()``, and a shape broadcast from the other arguments where it is ``None``.

**Where the key comes from.** A ``generated quantities`` evaluation receives one key per draw
(``docs/design/03``); each ``_rng`` call takes the next key off it by a sequential split, in
evaluation order. An unrolled ``for`` therefore draws fresh variates per iteration, and the same
program on the same key reproduces exactly. Calls are allowed only in ``generated quantities`` ---
the evaluation context of every other block has no key, and ``compile_model`` rejects a call
there statically (:func:`mimcs.dsl.semantics.check_no_rng`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from .errors import DslError

#: Argument kinds. ``array``: any value, passed through. ``shape``: a static shape (see the module
#: docstring). ``int`` / ``float`` / ``bool``: a compile-time constant of that type.
_KINDS = ("array", "shape", "int", "float", "bool")
_REQUIRED = object()


@dataclass(frozen=True)
class RngForm:
    """One ``<name>_rng``: the ``jax.random`` function and its argument list after ``key``."""

    name: str                 # the DSL name, ``<jax name>_rng``
    fn: object                # the jax.random sampler
    params: tuple             # ((name, kind, default | _REQUIRED), ...)

    @property
    def n_required(self) -> int:
        return sum(1 for _, _, d in self.params if d is _REQUIRED)

    @property
    def signature(self) -> str:
        parts = [n if d is _REQUIRED else f"{n}={_show(d)}" for n, _, d in self.params]
        return f"{self.name}({', '.join(parts)})"


def _show(default) -> str:
    return "None" if default is None else str(default)


def _form(jax_name: str, *params) -> RngForm:
    for _, kind, _ in params:
        assert kind in _KINDS, kind
    return RngForm(f"{jax_name}_rng", getattr(jax.random, jax_name), tuple(params))


R = _REQUIRED
_SHAPE0 = ("shape", "shape", ())          # JAX default shape ()
_SHAPEN = ("shape", "shape", None)        # JAX default shape None (broadcast the arguments)

_FORMS = [
    _form("uniform", _SHAPE0, ("minval", "array", 0.0), ("maxval", "array", 1.0)),
    _form("normal", _SHAPE0),
    _form("bernoulli", ("p", "array", 0.5), _SHAPEN),
    _form("beta", ("a", "array", R), ("b", "array", R), _SHAPEN),
    _form("binomial", ("n", "array", R), ("p", "array", R), _SHAPEN),
    _form("categorical", ("logits", "array", R), ("axis", "int", -1), _SHAPEN,
          ("replace", "bool", True)),
    _form("cauchy", _SHAPE0),
    _form("chisquare", ("df", "array", R), _SHAPEN),
    _form("dirichlet", ("alpha", "array", R), _SHAPEN),
    _form("double_sided_maxwell", ("loc", "array", R), ("scale", "array", R), _SHAPE0),
    _form("exponential", _SHAPE0),
    _form("f", ("dfnum", "array", R), ("dfden", "array", R), _SHAPEN),
    _form("gamma", ("a", "array", R), _SHAPEN),
    _form("generalized_normal", ("p", "float", R), _SHAPE0),
    _form("geometric", ("p", "array", R), _SHAPEN),
    _form("gumbel", _SHAPE0),
    _form("laplace", _SHAPE0),
    _form("loggamma", ("a", "array", R), _SHAPEN),
    _form("logistic", _SHAPE0),
    _form("lognormal", ("sigma", "array", 1.0), _SHAPEN),
    _form("maxwell", _SHAPE0),
    _form("multinomial", ("n", "array", R), ("p", "array", R), _SHAPEN),
    _form("multivariate_normal", ("mean", "array", R), ("cov", "array", R), _SHAPEN),
    _form("pareto", ("b", "array", R), _SHAPEN),
    _form("poisson", ("lam", "array", R), _SHAPEN),
    _form("rademacher", _SHAPE0),
    _form("randint", ("shape", "shape", R), ("minval", "array", R), ("maxval", "array", R)),
    _form("rayleigh", ("scale", "array", R), _SHAPEN),
    _form("t", ("df", "array", R), _SHAPE0),
    _form("triangular", ("left", "array", R), ("mode", "array", R), ("right", "array", R),
          _SHAPEN),
    _form("truncated_normal", ("lower", "array", R), ("upper", "array", R), _SHAPEN),
    _form("wald", ("mean", "array", R), _SHAPEN),
    _form("weibull_min", ("scale", "array", R), ("concentration", "array", R), _SHAPE0),
    _form("ball", ("d", "int", R), ("p", "float", 2.0), _SHAPE0),
    _form("orthogonal", ("n", "int", R), _SHAPE0, ("m", "int", None)),
    _form("choice", ("a", "array", R), _SHAPE0, ("replace", "bool", True), ("p", "array", None),
          ("axis", "int", 0)),
    _form("permutation", ("x", "array", R), ("axis", "int", 0), ("independent", "bool", False)),
]

#: ``{dsl name: RngForm}`` --- every random variate a ``generated quantities`` block may draw.
RNG_FUNCTIONS: dict = {f.name: f for f in _FORMS}


def _static(value, kind: str, what: str, span):
    """A compile-time constant of ``kind``, or a spanned error saying why it is not."""
    if value is None:
        return None
    try:
        concrete = np.asarray(value)          # a tracer raises here: not a compile-time constant
        if concrete.ndim != 0:
            raise TypeError
        if kind == "int":
            n = int(concrete)
            if n != float(concrete):
                raise TypeError
            return n
        return float(concrete) if kind == "float" else bool(concrete)
    except Exception:
        raise DslError(f"{what} must be a compile-time {kind} constant", span) from None


def _shape(value, what: str, span):
    if value is None:
        return None
    parts = value if isinstance(value, tuple) else (value,)
    return tuple(_static(p, "int", what, span) for p in parts)


def call_rng(form: RngForm, key, args: list, span):
    """Draw from ``form`` with ``key`` and the DSL's positional ``args``."""
    if not form.n_required <= len(args) <= len(form.params):
        n = (str(form.n_required) if form.n_required == len(form.params)
             else f"{form.n_required} to {len(form.params)}")
        raise DslError(
            f"{form.name} takes {n} argument(s), {len(args)} given --- `{form.signature}`. The "
            f"signatures are jax.random's without the key, not Stan's: e.g. normal_rng() is a "
            f"standard normal, so write mu + sigma * normal_rng(shape).", span)
    # Passed by keyword, all of them: JAX's positional order interleaves `dtype` (e.g.
    # `uniform(key, shape, dtype, minval, maxval)`), which the DSL signature leaves out.
    keyword = {}
    for (pname, kind, default), value in zip(form.params, args + [default for *_, default
                                                                  in form.params[len(args):]]):
        what = f"the `{pname}` argument of {form.name}"
        if kind == "shape":
            # `None` where JAX's default is `()` means that default: it is the only way to reach
            # a later argument (`uniform_rng(None, 2, 5)`) without writing an empty tuple.
            value = default if value is None and default is not _REQUIRED else _shape(
                value, what, span)
        elif kind in ("int", "float", "bool"):
            value = _static(value, kind, what, span)
        keyword[pname] = value
    try:
        return form.fn(key, **keyword)
    except DslError:
        raise
    except (TypeError, ValueError) as e:
        raise DslError(f"{form.name}: {e} --- `{form.signature}`", span) from None
