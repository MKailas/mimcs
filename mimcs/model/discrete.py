"""``BaseDiscreteParameter``: the interface a discrete (integer-valued) parameter implements.

The counterpart of :mod:`mimcs.model.parameter` for the discrete half of a model, and a much
smaller contract. A continuous parameter is *defined* by its charts --- the maps between the
ambient value and the unconstrained coordinate a gradient-based sampler moves in. A discrete
parameter has none, and the omission is the point rather than an unfinished edge:

* There is nothing to reparameterize. The sampler proposes an integer and accepts or rejects it;
  there is no smooth coordinate in which the proposal is better conditioned, because there is no
  smoothness at all.
* There is nothing to differentiate. A chart exists so ``log_prob_at_coordinate`` can be handed
  to ``jax.grad``; a discrete coordinate never is.
* So sample space **is** coordinate space, and the change-of-variables log-Jacobian is
  identically zero --- which is why a Gibbs sweep can move a discrete parameter without touching
  the continuous state at all (``docs/design/14_discrete_parameters.md``).

What remains is the naming and *feature* half of :class:`~mimcs.model.BaseParameter` --- the
observables a convergence diagnostic reads --- plus the support the sampler enumerates a proposal
from. The two hierarchies are deliberately separate rather than one class with the chart methods
stubbed out: :class:`~mimcs.model.Model` keeps discrete parameters in their own list with their
own flat layout, and a type that raised ``NotImplementedError`` from three chart methods would
invite exactly the accidental mixing that separation prevents.

**No Stein term.** The Langevin--Stein identity (``docs/design/11``) integrates by parts against a
density and its score; a probability *mass* function has neither. Discrete features therefore
report no ``z``, flagged through :meth:`stein_defined` so the summary prints a gap rather than a
number nobody should read. A discrete Stein operator (a difference operator in place of the
derivative) is sketched in doc 14 as deferred work.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .parameter import _index_prefixes, flat_size


class BaseDiscreteParameter:
    """Abstract base for an integer-valued model parameter.

    A subclass sets ``name`` and ``ambient_shape`` and supplies the support through ``lower`` /
    ``upper`` (both ``(size,)`` integer arrays, elementwise and inclusive). Everything else here
    has a working default.

    Attributes:
        name: parameter name (its key in the model's value dict).
        ambient_shape: shape of the ambient value; ``()`` for a scalar.
        lower: elementwise inclusive lower bound, shape ``(size,)``.
        upper: elementwise inclusive upper bound, shape ``(size,)``.
    """

    name: str
    ambient_shape: tuple
    lower: Array
    upper: Array

    #: Names of parameters this one depends on. Always empty in stage 1 --- a discrete
    #: parameter's *support* is constant, and a discrete parameter may not be a continuous
    #: chart's parent (see :class:`~mimcs.model.Model`).
    parents: tuple = ()

    @property
    def size(self) -> int:
        """Number of integer coordinates this parameter contributes to the flat block."""
        return flat_size(self.ambient_shape)

    @property
    def n_values(self) -> Array:
        """Elementwise size of each coordinate's support, ``upper - lower + 1``, shape ``(size,)``."""
        return self.upper - self.lower + 1

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return self.size

    def feature_names(self) -> list:
        """One name per feature, in the order :meth:`features` returns them."""
        return [f"{self.name}{s}" for s in _index_prefixes(self.ambient_shape)]

    def features(self, sample: Array) -> Array:
        """Observables of this parameter: **the bare value**, and nothing else.

        The continuous default is ``[x, x^2]``, the mean and the spread. The square is not worth
        watching here: for the common binary case ``x^2 == x`` exactly, so the second block would
        be a duplicate column that doubles the multiplicity correction while carrying no
        information, and for a categorical label the square of a category index is not a quantity
        anyone reads. Returns shape ``(n_features,)``, and is vmappable over a stack of draws.
        """
        return jnp.reshape(sample, (-1,)).astype(float)

    def ambient_names(self) -> list:
        """One label per ambient coordinate --- the row labels of the posterior summary."""
        return [f"{self.name}{s}" for s in _index_prefixes(self.ambient_shape)]

    def stein_defined(self) -> Array:
        """Per-feature: is a Langevin--Stein term defined? Always ``False`` --- see the module
        docstring. Same length and order as :meth:`features`."""
        return jnp.zeros((self.n_features,), bool)

    # --- initial value ---

    def default_value(self) -> Array:
        """A valid starting value: the lower bound, in the flat ``(size,)`` layout.

        Valid by construction for any support. It is a poor *starting point* for a label --- every
        observation in cluster 1 --- which is why
        :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` randomizes it in ``initialize()``.
        """
        return jnp.asarray(self.lower, jnp.int32)
