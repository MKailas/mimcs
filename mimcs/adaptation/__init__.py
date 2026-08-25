"""Adaptation mixins --- and the initialization and termination mixins beside them.

Composed onto a base sampler with :func:`~mimcs.samplers.make_sampler_class`, cooperating through
``super()`` chains (``docs/design/02_sampler_classes.md``). Four groups, distinguished by which
chain they hang off and what they write:

* **Adaptation** (``_postprocess_hooks``, warmup only) --- the step size
  (:class:`RobbinsMonroStepSize`, or :class:`LineSearchStepSizeAdaptation` under a line-search
  integrator), the mass (:class:`ScoreMassAdaptation`, :class:`MassMatrixAdaptation`,
  :class:`LowRankAdaptation`), learned metrics (:class:`MetricAdaptation`,
  :class:`ShapedMetricAdaptation`) and chart hyperparameters
  (:class:`RobustCenteringAdaptation`, :class:`UnitVectorCenteringAdaptation`).
* **Initialization** (``_initialize_hooks``, once) --- :class:`UniformInit`,
  :class:`StepSizeLineSearch`. Inert unless ``sampler.initialize()`` is called.
* **Termination** (``should_stop()``) --- :class:`ClassifierTermination` (the default),
  :class:`GelmanRubinTermination`. These end warmup on a mixing criterion (doc 10).

Within a group, entries are often **alternatives rather than additions**: the two mass
adaptations share a block filter and must not be stacked, and
:class:`LineSearchStepSizeAdaptation` subclasses :class:`RobbinsMonroStepSize`, so composing both
is an MRO error. Options are read from ``**kwargs`` by name --- see
``docs/reference/algo_kwargs.md``.
"""

from .step_size import RobbinsMonroStepSize, LineSearchStepSizeAdaptation
from .covariance import DiagonalCovarianceAdaptation
from .mass import MassMatrixAdaptation
from .score_mass import ScoreMassAdaptation
from .metric import MetricAdaptation
from .shaped_metric import ShapedMetricAdaptation
from .centering import CenteringAdaptation, RobustCenteringAdaptation
from .unit_vector import UnitVectorCenteringAdaptation
from .relativistic_mass import RelativisticMassAdaptation
from .lowrank_mass import LowRankAdaptation
from .initialization import UniformInit, StepSizeLineSearch
from .termination import GelmanRubinTermination, ClassifierTermination

__all__ = [
    "RobbinsMonroStepSize", "LineSearchStepSizeAdaptation",
    "DiagonalCovarianceAdaptation", "MassMatrixAdaptation",
    "ScoreMassAdaptation", "MetricAdaptation", "ShapedMetricAdaptation", "CenteringAdaptation",
    "RobustCenteringAdaptation", "UnitVectorCenteringAdaptation",
    "RelativisticMassAdaptation", "LowRankAdaptation",
    "UniformInit", "StepSizeLineSearch",
    "GelmanRubinTermination", "ClassifierTermination"]
