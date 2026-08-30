"""Core sampling loop: ``BaseSampler`` and dynamic class construction.

Implements the preprocess / kernel / postprocess protocol from
``docs/design/01_state_and_kernel.md`` and the base-class + mixin composition from
``docs/design/02_sampler_classes.md``.

Iteration is::

    state = sampler.preprocess(state)   # Python: inject RNG draw, chart admin, pre-hooks
    state = sampler._kernel_jit(state)  # JAX JIT: pure Markov kernel
    state = sampler.postprocess(state)  # Python: adapt, save samples, diagnostics

``preprocess`` and ``postprocess`` are Python so they can make dynamic decisions
(buffer replenishment, adaptation schedules, when to start saving) that need not, or
must not, be JIT-compiled. The kernel is the only JIT-compiled, pure part.

Adaptation strategies are mixins cooperating through ``super()`` chains. Six of them:
``_init_hooks`` (read options), ``_init_state_hooks``, ``_preprocess_hooks``,
``_postprocess_hooks`` (where adaptation runs, warmup only), ``_finalize_hooks`` (once, at the
warmup->sampling transition --- e.g. freezing an averaged mass), and ``_initialize_hooks`` (once,
on :meth:`BaseSampler.initialize`, before warmup). A termination mixin additionally votes through
``should_stop()``, which makes ``warmup(n)``'s ``n`` an upper bound rather than a count. A
concrete sampler class is assembled from a base algorithm class and zero or more mixins via
:func:`make_sampler_class`.
"""

from __future__ import annotations

import enum

import jax
import numpy as np

from .._logging import get_logger
from ..rng import RNGBuffer, make_rng_draw_class, zero_draw

log = get_logger(__name__)


class Phase(enum.Enum):
    """Coarse run phase. WARMUP adapts and discards; SAMPLING freezes and saves."""

    WARMUP = "warmup"
    SAMPLING = "sampling"


def _fmt(x, spec: str = ".4g") -> str:
    """Format a scalar (JAX array, numpy value, float) for a log message; ``"n/a"`` if absent."""
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


class BaseSampler:
    """Base class for all samplers: owns the loop, RNG buffer, and phase.

    Concrete algorithm classes (e.g. :class:`RandomWalkMH`) implement
    ``make_draw_components``, ``make_initial_state``, and ``kernel``. Adaptation
    mixins implement the ``_*_hooks`` cooperative methods.

    Constructor args beyond the ones named here are forwarded to the adaptation
    mixins' ``_init_hooks`` and retained for ``make_initial_state``.
    """

    #: Can this sampler class move a model's discrete parameters? Set by
    #: :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs`; a class without it refuses a model
    #: that has any, rather than sampling with the labels frozen.
    handles_discrete = False

    def __init__(self, model, init_position, *, seed: int = 0, buffer_size: int = 1024,
                 **kwargs):
        self.model = model
        self._kwargs = dict(kwargs)
        self._seed = int(seed)          # for initialization draws (a dedicated PRNG stream)

        # RNG: components are declared by the concrete algorithm class. Called as an
        # instance method so samplers whose draws depend on composed components (e.g.
        # HMC's kinetic) can read them; subclasses that set such attributes must do so
        # before calling super().__init__().
        self._draw_components = self.make_draw_components(model, **kwargs)
        self._rng_draw_class = make_rng_draw_class(
            type(self).__name__ + "RngDraw", self._draw_components)
        self._rng_buffer = RNGBuffer(seed, self._draw_components, buffer_size)

        # Run bookkeeping (Python side; never inside JIT).
        self._phase = Phase.WARMUP
        self._iteration = 0
        self._samples: list = []
        self._discrete_samples: list = []    # per-sample discrete block (only if the model has one)
        self._gradients: list = []           # per-sample total score (if save_gradients)
        # Uniform per-transition diagnostics: every entry of ``state.diagnostics`` (kernel-produced)
        # plus the post-adaptation ``step_size``, one list per name, phase-tagged in parallel.
        self._diag: dict = {}
        self._diag_phase: list = []          # True where the transition was in the SAMPLING phase
        # Save the (already-computed) total score of each retained sample, so a later factory
        # call need not recompute it. On by default: the memory is usually cheaper than the
        # recompute, and gradient-free algorithms simply have nothing to save.
        self._save_gradients = bool(kwargs.get("save_gradients", True))

        if model.discrete_dim and not self.handles_discrete:
            raise TypeError(
                f"{type(self).__name__} cannot move this model's discrete parameter(s) "
                f"{[p.name for p in model.discrete_parameters]}, and would sample it with them "
                f"held frozen. Compose a sampler that can: "
                f"make_sampler_class(..., DiscreteMetropolisWithinGibbs, NUTS). "
                f"(Frozen coordinates are not a visible failure --- zero variance reports a "
                f"perfect ESS and R-hat 1.000 --- which is why this raises rather than warns.)")

        # Cooperative initialization of mixin adaptation state.
        self._init_hooks(**kwargs)

        # Build the initial state, then let mixins inject any initial hyperparameters.
        self.state = self.make_initial_state(init_position)
        self.state = self._init_state_hooks(self.state)

        self._kernel_jit = jax.jit(self.kernel)

        log.debug("%s: %d parameter(s), coord_dim %d, ambient_dim %d, seed %d, "
                  "rng buffer %d; mixins %s", type(self).__name__, len(model.parameters),
                  model.coord_dim, model.ambient_dim, self._seed, buffer_size,
                  [c.__name__ for c in type(self).__mro__[1:-1]])

    # ------------------------------------------------------------------ #
    # Concrete-class responsibilities                                    #
    # ------------------------------------------------------------------ #

    def make_draw_components(self, model, **kwargs):
        raise NotImplementedError

    def make_initial_state(self, init_position):
        raise NotImplementedError

    def init_diagnostics(self) -> dict:
        """The sampler's per-transition ``state.diagnostics`` schema (a dict of zero-filled scalars),
        seeded in ``make_initial_state`` so the initial-state pytree matches the kernel's output.
        Concrete samplers override cooperatively (via ``super().init_diagnostics()``)."""
        return {}

    def kernel(self, state):
        raise NotImplementedError

    def _current_score(self, state):
        """The total score --- the gradient of the log-density in coordinate space --- at the
        current state, or ``None`` if the algorithm exposes none (e.g. gradient-free Metropolis).
        Gradient-based samplers override this to return their cached gradient; it is used to
        optionally save the score of each retained sample (see :meth:`get_gradients`)."""
        return None

    # ------------------------------------------------------------------ #
    # Cooperative hook terminals (mixins override and call super())       #
    # ------------------------------------------------------------------ #

    def _init_hooks(self, **kwargs):
        return None

    def _init_state_hooks(self, state):
        return state

    def _preprocess_hooks(self, state):
        return state

    def _postprocess_hooks(self, state):
        return state

    def _finalize_hooks(self, state):
        """Cooperative hook run once when warmup ends and sampling begins. Adaptation mixins
        use it to freeze into the state the value they want held fixed for sampling (e.g. the
        Polyak--Ruppert average of an adapted mass, rather than the last raw iterate)."""
        return state

    def _initialize_hooks(self, state):
        """Cooperative hook run by :meth:`initialize` (before warmup). Initialization mixins use
        it to set a reasonable starting point --- a starting position, an initial step size, or
        (future) a Pathfinder metric --- into the state before any adaptation runs."""
        return state

    # --- end-of-phase reporting (no state change; mixins log what they know) ---

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        """Cooperative hook run once when :meth:`warmup` returns, with the number of iterations
        actually run and whether ``should_stop()`` ended it. Purely for reporting: a termination
        mixin uses it to say whether its criterion fired or its budget simply ran out."""
        return None

    def _sample_end_hooks(self, state):
        """Cooperative hook run once when :meth:`sample` returns. Purely for reporting --- e.g.
        NUTS warns here about post-warmup divergences, which are a property of the whole sampling
        phase rather than of any one transition."""
        return state

    # --- warmup termination ---

    def should_stop(self) -> bool:
        """Should warmup end now? Cooperative; terminal answer is no.

        A termination mixin overrides this to report that the chain looks to be mixing well
        (``super().should_stop() or ...``), which :meth:`warmup` honours. Consulted once per
        warmup iteration, after ``postprocess``.
        """
        return False

    def _warmup_budget(self) -> int:
        """The iteration cap for a ``warmup()`` called without ``n``. Terminal: there is none."""
        raise RuntimeError(
            "warmup() without n needs a termination mixin to bound it (e.g. "
            "mimcs.adaptation.ClassifierTermination, whose max_warmup is the cap); "
            "otherwise pass n explicitly")

    # ------------------------------------------------------------------ #
    # The loop                                                           #
    # ------------------------------------------------------------------ #

    def preprocess(self, state):
        """Inject the next RNG draw and run pre-step hooks (Python)."""
        raw = self._rng_buffer.next()
        rng_draw = self._rng_draw_class(**raw)
        state = state._replace(rng_draw=rng_draw)
        return self._preprocess_hooks(state)

    def postprocess(self, state):
        """Run adaptation hooks, then record diagnostics uniformly / save the sample.

        Every entry of the kernel-produced ``state.diagnostics`` dict is appended to the uniform
        per-transition store, phase-tagged, along with the post-adaptation ``step_size`` (the most
        crucial adapted quantity, whose warmup trajectory is a useful convergence proxy). Samples
        and saved scores are retained only during SAMPLING.
        """
        self._iteration += 1
        state = self._postprocess_hooks(state)
        rec = dict(getattr(state, "diagnostics", None) or {})
        step = getattr(state, "step_size", None)
        if step is not None:
            rec["step_size"] = step
        # ``np.array``, not ``np.asarray``: on the CPU backend ``np.asarray`` of a JAX array is
        # **zero-copy** --- it returns a non-owning view over a memoryview of the live device
        # buffer, which pins the whole ``jax.Array`` (Python object, PJRT buffer, page-granular
        # allocation) for as long as the store lives. Kept once per diagnostic per iteration for
        # the whole run, that dominates a sampler's memory: measured on a 2-d model, resident
        # growth of 29.6 KiB/iteration against 3.5 KiB/iteration when these are real copies
        # (404 -> 110 MiB over a 12 000-iteration warmup). Copying a handful of scalars costs no
        # measurable time, and the stored values are byte-identical either way.
        for k, v in rec.items():
            self._diag.setdefault(k, []).append(np.array(v))
        self._diag_phase.append(self._phase is Phase.SAMPLING)
        if self._phase is Phase.SAMPLING:
            self._samples.append(np.array(self._retained_sample(state)))
            if self.model.discrete_dim:
                self._discrete_samples.append(np.array(self._retained_discrete(state)))
            if self._save_gradients:
                score = self._current_score(state)
                if score is not None:
                    self._gradients.append(np.array(score))
        return state

    def step(self):
        """Advance the chain by one iteration."""
        self.state = self.preprocess(self.state)
        self.state = self._kernel_jit(self.state)
        self.state = self.postprocess(self.state)
        return self.state

    # ------------------------------------------------------------------ #
    # Drivers                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self):
        """Set a reasonable starting point (position, step size) before warmup.

        Runs the ``_initialize_hooks`` chain of any initialization mixins (e.g. a ``U(-2, 2)``
        coordinate draw and a step-size line search). A no-op with no such mixins. Call it once,
        before ``warmup``; it does not step the chain or run adaptation."""
        if self._iteration != 0:
            raise RuntimeError("initialize() must be called before warmup()/sample()")
        self.state = self._initialize_hooks(self.state)
        log.debug("initialized: log_prob %s, step size %s",
                  _fmt(getattr(self.state, "log_prob", None), ".6g"),
                  _fmt(getattr(self.state, "step_size", None)))
        return self

    def warmup(self, n: int | None = None):
        """Run adapting iterations whose samples are discarded.

        ``n`` is an *upper bound*: a termination mixin (see
        :mod:`mimcs.adaptation.termination`) can end warmup earlier, once the chain looks to be
        mixing well. With no such mixin ``should_stop()`` is always false and this runs exactly
        ``n`` iterations, as it always has. ``n=None`` means "run until the criterion fires or the
        mixin's own budget is exhausted", and needs a termination mixin to say what that budget is.
        """
        self._phase = Phase.WARMUP
        n = self._warmup_budget() if n is None else int(n)
        log.debug("warmup starting, up to %d iteration(s)", n)
        completed, stopped = 0, False
        for _ in range(n):
            self.step()
            completed += 1
            if self.should_stop():
                stopped = True
                break
        self._warmup_end_hooks(completed, stopped)
        log.info("warmup finished after %d/%d iteration(s)%s; step size %s",
                 completed, n, " (stopped early)" if stopped else "",
                 _fmt(getattr(self.state, "step_size", None)))
        return self

    def sample(self, n: int):
        """Run ``n`` iterations with adaptation frozen, retaining the samples.

        Returns what :meth:`get_samples` returns --- the draws keyed by parameter name.
        """
        if self._phase is not Phase.SAMPLING:
            self.state = self._finalize_hooks(self.state)   # freeze adapted quantities
            log.debug("adaptation frozen; sampling phase begins")
        self._phase = Phase.SAMPLING
        for _ in range(int(n)):
            self.step()
        self.state = self._sample_end_hooks(self.state)
        log.info("sampled %d iteration(s): %d draw(s) retained, acceptance rate %.3f",
                 int(n), len(self._samples), self.acceptance_rate())
        return self.get_samples()

    @property
    def is_adapting(self) -> bool:
        return self._phase is Phase.WARMUP

    def get_samples(self) -> dict:
        """The retained draws, keyed by parameter name --- ``{name: (n_draws, *ambient_shape)}``.

        Discrete parameters appear in the same dict, keyed the same way, with an **integer**
        dtype. A scalar parameter comes out ``(n,)``, a vector one ``(n, d)``. This is the shape worth
        working in: the values are in *sample* space, so they are the model's own quantities, and
        naming them removes the need to know where each parameter sits in the flat layout.

        :meth:`get_samples_flat` returns the same draws as one ``(n_draws, ambient_dim)`` array,
        which is what the test harness and anything doing linear algebra across parameters wants.
        """
        draws = self.model.unpack_draws(self.get_samples_flat())
        if self.model.discrete_dim:
            draws.update(self.model.unpack_discrete_draws(self.get_discrete_flat()))
        return draws

    def _after_discrete(self, state, log_prob):
        """Restore whatever the *continuous* kernel caches, after a Gibbs sweep moved the labels.

        Cooperative, and terminal here: a sampler that caches nothing beyond ``log_prob`` just
        records the sweep's own value. :class:`~mimcs.hmc.BaseHMC` overrides it, because it caches
        each potential's value **and gradient** at the current coordinate, and those are gradients
        of ``pi(. | old labels)``. Left stale, the next trajectory integrates the wrong density ---
        with the right shapes, a plausible acceptance rate and nothing raising.

        Only ever called by :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs`; a model with
        no discrete parameters never reaches it.
        """
        return state._replace(log_prob=log_prob)

    def _retained_discrete(self, state):
        """The part of the discrete draw worth storing --- the whole block, ordinarily.

        The discrete counterpart of :meth:`_retained_sample`, and the hook a sampler whose target
        is wider than what it reports would override.
        """
        return state.discrete

    def _retained_sample(self, state):
        """The part of the draw worth storing --- the whole thing, for an ordinary sampler.

        The hook exists for a sampler whose target is wider than what it reports: parallel
        tempering runs on the K-fold product but keeps only the cold chain, and narrowing **here**
        rather than when the draws are read means the other ``K-1`` rungs are never stored at all.
        :meth:`_current_score` has always done this for the gradient; this is its counterpart for
        the sample.
        """
        return state.sample

    def get_samples_flat(self) -> np.ndarray:
        """The retained draws as one ``(n_draws, ambient_dim)`` array, in the model's layout."""
        if not self._samples:
            return np.empty((0, self.model.ambient_dim))
        return np.stack(self._samples)

    def get_discrete_flat(self) -> np.ndarray:
        """The retained discrete draws as one ``(n_draws, discrete_dim)`` **integer** array.

        The counterpart of :meth:`get_samples_flat` for the discrete block, kept separate for the
        same reason the state keeps two arrays: these are labels, and folding them into the float
        matrix would lose the one property that makes them labels. Empty (width 0) for a model
        with no discrete parameters.
        """
        if not self._discrete_samples:
            return np.empty((0, self.model.discrete_dim), dtype=np.int32)
        return np.stack(self._discrete_samples)

    def get_gradients(self):
        """The saved per-sample total scores (gradient of the log-density in coordinate space),
        shape ``(n_samples, coord_dim)`` --- or ``None`` if gradient saving was disabled
        (``save_gradients=False``) or the algorithm has no gradient (e.g. random-walk MH)."""
        if not self._gradients:
            return None
        return np.stack(self._gradients)

    # ------------------------------------------------------------------ #
    # Uniform diagnostics store                                          #
    # ------------------------------------------------------------------ #

    def _diag_values(self, key: str, *, warmup: bool = False,
                     sampling: bool = True) -> np.ndarray:
        """Per-transition values recorded for ``key``, filtered to the selected phase(s)."""
        vals = self._diag.get(key)
        if not vals:
            return np.empty((0,))
        keep = [(sampling and s) or (warmup and not s) for s in self._diag_phase]
        return np.asarray([v for v, k in zip(vals, keep) if k])

    def diagnostics(self, phase: str = "sampling") -> dict:
        """All per-transition diagnostics as a dict of arrays. ``phase`` is ``"sampling"`` (default),
        ``"warmup"``, or ``"all"``. Keys depend on the sampler (e.g. ``accepted``, ``accept_prob``,
        ``step_size``; NUTS adds ``diverging``, ``tree_depth``, ``n_leaves``, ``grad_evals`` ...)."""
        warmup, sampling = phase in ("warmup", "all"), phase in ("sampling", "all")
        return {k: self._diag_values(k, warmup=warmup, sampling=sampling) for k in self._diag}

    def acceptance_rate(self) -> float:
        a = self._diag_values("accepted")
        return float(np.mean(a)) if a.size else float("nan")

    def warmup_step_sizes(self) -> np.ndarray:
        """The step-size value after each warmup iteration (empty if none recorded)."""
        return self._diag_values("step_size", warmup=True, sampling=False).astype(float)

    def mean_n_leaves(self) -> float:
        """Mean trajectory size (leapfrog steps / leaves) over sampling (nan if untracked)."""
        v = self._diag_values("n_leaves")
        return float(np.mean(v)) if v.size else float("nan")

    def total_grad_evals(self, *, include_warmup: bool = False,
                         include_sampling: bool = True) -> float:
        """Total gradient evaluations over the selected phase(s) --- the HMC cost proxy for
        efficiency (ESS per gradient evaluation)."""
        v = self._diag_values("grad_evals", warmup=include_warmup, sampling=include_sampling)
        return float(np.sum(v))

    @property
    def summary_model(self):
        """The model the **retained** draws belong to --- what :meth:`summary` evaluates against.

        Its own model for an ordinary sampler. A sampler that narrows what it keeps overrides
        this to match: parallel tempering runs on the K-fold product but retains the cold chain
        (:class:`~mimcs.pt.ProductSpaceMixin`), so the draws, the scores *and* the model handed to
        ``summarize`` must all be that chain's.
        """
        return self.model

    def summary(self):
        """Evaluate the drawn samples: a posterior summary plus per-feature diagnostics.

        Returns a :class:`mimcs.summary.Summary` (also cached on ``self._summary``) that renders as
        two tables --- per-coordinate means / credible intervals, and per-feature ESS, split-R-hat
        and a target-aware **Stein z**. The saved coordinate-space gradients are reused when
        available (pulled back to the ambient score through the frozen chart); a gradient-free
        sampler recomputes the score from the model. Call after :meth:`sample`.
        """
        from ..summary import summarize
        draws = self.get_samples_flat()
        if len(draws) == 0:
            raise RuntimeError("summary() needs draws: call sample() first")
        st = self.state
        self._summary = summarize(
            self.summary_model, draws, self.acceptance_rate(),
            coord_score=self.get_gradients(),
            chart_hyperparams=st.chart_hyperparams, chart_indices=st.chart_indices,
            discrete_draws=(self.get_discrete_flat()
                            if self.summary_model.discrete_dim else None))
        return self._summary


def make_sampler_class(*bases, name: str | None = None) -> type:
    """Assemble a concrete sampler class from mixins and a base algorithm class.

    List adaptation mixins first (highest MRO priority) and the base algorithm class
    last, e.g. ``make_sampler_class(RobbinsMonroStepSize, DiagonalCovariance, RandomWalkMH)``.
    """
    if name is None:
        name = "".join(b.__name__ for b in bases)
    return type(name, tuple(bases), {})
