"""Which log-density components are cheap: the default rule.

A multi-rate (RESPA) integrator kicks a cheap potential several times per expensive one
(``docs/design/06_hamiltonian_monte_carlo.md``), so it needs to be told which is which. For a
DSL program the question has a structural answer: **a component is expensive exactly when it
touches a large constant or a large parameter.** Size is the proxy for gradient cost --- a term
that sweeps a big array (data, or a high-dimensional parameter it is written over) usually costs
more to differentiate than one that touches only scalars, whether that array is a fixed constant
or a parameter's own value.

"Large" is decided by comparing every measurable constant *and* every parameter's ambient
dimension against each other in one pool, since only their *relative* sizes matter:

* the largest item (constant or parameter) is always large;
* a one-element item never is (a scalar hyperparameter, or a scalar parameter, is not "data");
* anything at least :data:`LARGE_FRACTION` of the largest is large;
* the rest are not.

Parameters are measured by **ambient** dimension, not coordinate dimension --- the two differ for
a manifold parameter (a unit vector has one fewer coordinate than ambient component), and ambient
is what the model is written in and what a gradient with respect to it actually costs.

Two consequences are worth stating because they surprise people:

* a program with **no data and no vector parameter**, or whose constants and parameters are all
  scalars, has nothing large --- so every component comes out cheap, which is right: there is
  nothing for a multi-rate split to buy there;
* a component's statements are ``transformed parameters + <its block>`` (the transformed
  parameters run inside every component's closure), so **if `transformed parameters` reads a
  large constant or parameter, every component is expensive**. That is not a quirk of the rule;
  every closure really does re-execute those statements.

This threshold (1/500) is deliberately much looser than counting large *constants* alone would
suggest: a hierarchical prior over a handful of moderately-sized vector parameters is exactly the
case this rule exists to catch, and on `reg_horseshoe` (n=62, d=2000) it is what reclassifies the
prior as expensive --- its own `beta`/`lambda` are each 2000-dim, next to the 62x2000 design
matrix `x`, so 2000/124000 ≈ 1/62 clears even this loose bar. (Measuring cost by size is still
only a proxy: on this same model a prior gradient measured *more* expensive per-call than the
likelihood's, because the likelihood is one BLAS matmul while the prior runs transcendentals
elementwise --- see the `reg_horseshoe` study in doc 06. Size predicts *when there is anything to
split*, not which side wins a wall-clock race.)

**Refining it.** The threshold is a constant and the classifier is a plain function that mutates
a :class:`~mimcs.dsl.spec.ModelSpec` and records its reasoning in ``spec.rationale``; pass a
different one to :meth:`mimcs.dsl.factory.ModelFactory.analyze` (``classify=...``) to replace it
wholesale --- with, say, one that times a gradient. Deliberately *not* built on the sampler
factory's ``Proposal`` / ``arbitrate`` machinery (``mimcs/factory/rules.py``): that exists to
resolve conflicts between rules competing for one slot, and with a single rule the weighted
arbitration is the identity function. Its slot addressing also understands only
``blocks[i].field``, and importing it here would point the DSL at the sampler heuristics that
consume its output. If a second, disagreeing cost rule ever appears, that is the moment to lift
the arbiter into a shared module --- as a deliberate refactor, not by anticipation.
"""

from __future__ import annotations

import math

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

#: The cost labels a component may carry, cheapest first --- the order a multi-rate integrator
#: would nest them in.
COSTS = ("cheap", "expensive")

#: A constant or parameter is "large" when it has at least this fraction of the largest item's
#: elements (constants and parameters compared in one pool; parameters by ambient dimension).
#: PLACEHOLDER --- deliberately loose (1/500, not the 1/10 an all-constants version used), on the
#: observation that a moderately-sized *parameter* (a few thousand elements, common in a
#: hierarchical prior) should count as large next to a data array that is only one or two orders
#: of magnitude bigger, even though two data arrays usually sit within one order of magnitude of
#: each other. No further evidence yet beyond the `reg_horseshoe` case this was tuned to catch.
LARGE_FRACTION = 1 / 500


def constant_size(value) -> int | None:
    """Number of elements in a bound constant, or ``None`` when it cannot be measured.

    ``np.size`` covers everything the DSL can bind --- a Python ``int`` or ``float`` (1), a numpy
    array, a JAX array (via ``.size``, with no host transfer), a nested list. A *ragged* list
    raises, and ``None`` would be reported by numpy as a 0-d object of size 1, which is a claim
    rather than a measurement; both are returned as unmeasurable so the caller can leave them out
    of the comparison instead of guessing.
    """
    if value is None:
        return None
    try:
        return int(np.size(value))
    except Exception:
        return None


def large_items(sizes: dict) -> frozenset:
    """The large entries among ``{name: element count}``, by the rule in the module docstring.

    Generic over what ``sizes`` counts --- constants, parameters, or (as
    :func:`classify_components` uses it) both pooled together, since only the relative sizes
    within one comparison matter.
    """
    if not sizes:
        return frozenset()                       # nothing to compare: nothing is large
    biggest = max(sizes.values())
    if biggest <= 1:
        return frozenset()                       # everything is a scalar: nothing is large
    threshold = math.ceil(LARGE_FRACTION * biggest)
    return frozenset(n for n, s in sizes.items() if s > 1 and s >= threshold)


def classify_components(spec) -> None:
    """Label every component of ``spec`` cheap or expensive, and record why.

    Mutates ``spec.large_constants``, ``spec.large_parameters``, and each ``ComponentSpec``'s
    ``cost`` / ``touches``, and appends one line per decision to ``spec.rationale``. That is the
    whole contract a replacement classifier has to honour (see the module docstring).
    """
    sizes = {**spec.constant_sizes, **spec.parameter_sizes}
    large = large_items(sizes)
    spec.large_constants = frozenset(large & spec.constant_sizes.keys())
    spec.large_parameters = frozenset(large & spec.parameter_sizes.keys())

    if not large:
        why = ("the program has no measurable constant or parameter" if not sizes
               else f"all {len(sizes)} constant(s)/parameter(s) are scalars")
        for c in spec.components:
            c.cost, c.touches = "cheap", frozenset()
        spec.rationale.append(
            f"every component is cheap: {why}, so nothing is large")
        return

    biggest = max(sizes, key=sizes.__getitem__)
    spec.rationale.append(
        f"large constant(s)/parameter(s) {sorted(large)}: at least {LARGE_FRACTION:.2%} of the "
        f"largest, {biggest!r} ({sizes[biggest]} elements), among {sorted(sizes)}")

    shared = spec.shared_reads & large
    if shared:
        spec.rationale.append(
            f"`transformed parameters` reads {sorted(shared)}, and its statements run inside "
            f"every component's closure --- so every component is expensive")

    for c in spec.components:
        c.touches = frozenset(c.reads & large)
        c.cost = "expensive" if c.touches else "cheap"
        spec.rationale.append(
            f"components[{c.name!r}].cost = {c.cost!r} [large_items: "
            + (f"reads {sorted(c.touches)}]" if c.touches
               else "references no large constant/parameter]"))
