"""Custom jump operators: a deterministic map that moves continuous parameters with a label.

A Metropolis-within-Gibbs sweep moves one discrete coordinate and leaves every continuous
parameter where it is. On a strongly coupled model that is fatal --- switching a spike-and-slab
field on changes the fit too much to ever be accepted --- and the cure is to move the continuous
parameters *alongside* the label, compensating the change rather than fighting it
(``docs/design/14_discrete_parameters.md``).

A :class:`JumpOperator` is the model-level artifact that says how. It is the discrete peer of
:class:`~mimcs.model.ScanComponent`: a callable plus the metadata a sampler needs to use it ---
which parameter it attaches to, which parameters it rewrites, and whether it changes volume.

**The map must be written as a difference against the current value.** That is not a style
preference; it is what makes the two balance conditions below hold by construction::

    eta_new = eta + effect(gamma[j]) - effect(g)

Two conditions, checked numerically at sampler construction rather than assumed
(:mod:`mimcs.samplers.discrete_updates`). Writing ``Phi_{a->v}`` for "set the label to ``v`` and
apply the map, starting from current label ``a``":

* **The involution**, ``Phi_{b->a} . Phi_{a->b} = id``, is what Metropolis needs. The reverse move
  is this same operator run at the *current* value, so the involution is what makes it the inverse
  --- and it is why the acceptance ratio carries a single ``|det|`` rather than two terms.
* **The cocycle**, ``Phi_{b->v} . Phi_{a->b} = Phi_{a->v}``, is what exact conditional Gibbs needs.
  It is the group-action condition of Liu and Sabatti's generalized Gibbs sampler: it makes the
  orbit ``{Phi_{a->v}(x)}`` --- and so the weight vector, up to a common factor --- the same seen
  from every member, which is what the draw is over.

The cocycle is strictly stronger, and the gap is real rather than theoretical. A map that negates a
coordinate whenever the label changes satisfies the involution and not the cocycle; measured, it
samples **correctly** under Metropolis and **wrongly** under exact Gibbs (0.076 against 0.003 on
the label marginal). That is why both are checked, and why the error message names which one
failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class JumpOperator:
    """A deterministic map from one discrete coordinate's proposed value to new continuous values.

    Args:
        parameter: the discrete parameter whose sweep this attaches to. One operator per
            parameter.
        outputs: the parameters the map rewrites, in the order ``fn`` returns them. Continuous
            ones, and **other** discrete (``int``) parameters, which move under counting measure
            and so add no Jacobian; a returned label that is non-integral or outside its support
            makes the proposal invalid, and it is rejected rather than written. The discrete
            parameter itself is not listed --- the sweep moves that coordinate.
        fn: ``(values, c, v) -> tuple`` of new ambient values for ``outputs``.

            ``values`` is the model's ambient value dict, exactly what a log-density component
            receives, so the map sees data, parameters and transformed parameters and reads the
            **current** label as ``values[parameter]``. ``c`` is the coordinate's **flat, 0-based
            offset within its own parameter's block** --- the same index
            :meth:`~mimcs.samplers.DiscreteMetropolisWithinGibbs._discrete_delta` and
            :attr:`~mimcs.model.ScanComponent.element_fn` take, so the runtime has one index
            convention throughout. (A DSL-compiled operator unravels it to the parameter's declared
            shape and shifts it to 1-based on the far side of this boundary; a hand-written model
            supplies a plain Python callable and never sees a shape.) ``v`` is the proposed value.
        volume_preserving: whether ``|det dT/dx| == 1``, so the acceptance ratio carries no
            Jacobian term. True for a compensating shift, which is the common case and the one
            that costs nothing. When False the Jacobian is taken by autodiff over the output
            block, which is ``O(m)`` tangents and an ``O(m^3)`` determinant per candidate.
        reads: the free names the map's body reads. Metadata only: the restricted recomputation
            for a jump plans from the *components'* reads and the operator's outputs
            (:func:`~mimcs.samplers.gibbs.jump_restriction_plan`), since the map itself always runs.
    """

    parameter: str
    outputs: tuple
    fn: Callable
    volume_preserving: bool = True
    reads: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "reads", frozenset(self.reads))
        if not self.outputs:
            raise ValueError(
                f"jump operator for '{self.parameter}' rewrites no parameters. An operator that "
                f"changes nothing continuous is the ordinary Metropolis-within-Gibbs move, which "
                f"the sweep already makes --- drop the operator instead.")
        dupes = sorted({n for n in self.outputs if self.outputs.count(n) > 1})
        if dupes:
            raise ValueError(
                f"jump operator for '{self.parameter}' names output(s) {dupes} more than once")

    def __repr__(self) -> str:
        vol = "" if self.volume_preserving else ", scales"
        return f"JumpOperator({self.parameter!r} -> {list(self.outputs)}{vol})"
