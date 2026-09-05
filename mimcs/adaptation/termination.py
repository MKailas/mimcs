"""Warmup-termination mixins: end warmup once the chain looks to be mixing well.

These contribute to the ``should_stop()`` chain that :meth:`mimcs.samplers.BaseSampler.warmup`
consults each iteration. They are inert unless included, and a ``warmup(n)`` without one runs
exactly ``n`` iterations as before. Kept as separate mixins behind one cooperative hook, as the
initialization mixins are, so a third criterion (Geweke, say) slots in the same way.

* :class:`GelmanRubinTermination` --- stop when ``max R-hat < 1.01`` over the features. The
  well-tested comparison point.
* :class:`ClassifierTermination` --- stop when a logistic regression cannot tell a middle-period
  draw from a late-period one better than chance.

**What they look at, and what they deliberately do not.** Not the adapted parameters: by
diminishing adaptation the step size, mass and charts all change by ``O(gain) -> 0`` per
iteration, so watching them answers a question that is already settled by construction. Not the
coordinates either: those are a device for computation, and for a manifold parameter they do not
even have the same dimension as the draw. What is left is the draws themselves --- and, one layer
up, the *features* of a draw (:meth:`mimcs.model.Model.features`), the observables each parameter
declares. Every criterion here is a statement about features, which is what lets one mixin serve
Euclidean, bounded and manifold parameters alike.

**The split.** All warmup draws so far, minus a burn-in prefix, cut into two equal halves: an
*early* period and a *late* one. If the chain has settled, the halves are two samples from the
same distribution and nothing distinguishes them. If it is still travelling, they differ. For
R-hat, the halves are the "chains" --- which makes this Stan's split-R-hat for a single chain.

Adaptation runs throughout; the criterion only decides when to stop it.
"""

from __future__ import annotations

import numpy as np

from .._chunked import map_rows
from .._logging import get_logger
from ..diagnostics import split_rhat
from ..samplers.base import Phase
from . import _burnin
from ._burnin import MODES as BURN_IN_MODES
from ._logistic import DEFAULT_L2, accuracy, class_weights, fit_logistic, log_score, scores

log = get_logger(__name__)


class _WarmupTermination:
    """Shared machinery: the check schedule, the feature history, the split, the stop flag.

    Subclasses supply :meth:`_mixing_stat`, which scores one (early, late) pair of feature blocks
    and says whether that counts as mixed.

    Every check is logged at INFO (the statistic, the verdict, the consecutive-pass count), and
    the end of warmup at INFO when the criterion fired or WARNING when the mixin's ``max_warmup``
    budget ran out first --- warmup ending on the cap means the chain never met the criterion,
    which the draws that follow inherit.
    """

    #: what :meth:`_mixing_stat` returns, for the log message (subclasses override).
    _stat_name = "mixing statistic"

    @property
    def _term_name(self) -> str:
        """This mixin's own class name --- ``type(self)`` is the composed sampler class, whose
        name is every mixin's concatenated. The concrete criterion is the first class in the MRO
        that declares its own ``_stat_name``."""
        for cls in type(self).__mro__:
            if "_stat_name" in cls.__dict__ and cls is not _WarmupTermination:
                return cls.__name__
        return type(self).__name__

    def _last_stat(self) -> str:
        """The most recent criterion value, formatted --- or ``"n/a"`` if no check has run."""
        return f"{self._term_history[-1][1]:.4f}" if self._term_history else "n/a"

    def _init_hooks(self, **kwargs):
        self._term_min_warmup = int(kwargs.get("min_warmup", 500))
        self._term_check_every = int(kwargs.get("check_every", 100))
        self._term_max_warmup = int(kwargs.get("max_warmup", 50_000))
        self._term_burn_frac = float(kwargs.get("burn_in_frac", 0.1))
        self._term_thin = int(kwargs.get("feature_thin", 1))
        self._term_patience = int(kwargs.get("patience", 3))
        self._term_keep_features = bool(kwargs.get("keep_features", False))
        self._term_features: list = []     # one row of model.features per retained draw
        self._term_pending: list = []      # raw draws not yet turned into features
        self._term_pending_discrete: list = []   # their discrete blocks (empty model: unused)
        self._term_seen = 0                # warmup draws observed (before thinning)
        self._term_passes = 0              # consecutive checks that have looked mixed
        self._term_stop = False
        self._term_history: list = []      # (iteration, statistic) at each check
        self._term_burn_hist: list = []    # (iteration, burn-in) at each check
        self._term_last_burn = 0           # the count the last split actually discarded
        super()._init_hooks(**kwargs)

    # --- the base class's warmup loop talks to these ---

    def should_stop(self) -> bool:
        return super().should_stop() or self._term_stop

    def _warmup_budget(self) -> int:
        return self._term_max_warmup

    # --- observation ---

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state

        self._term_seen += 1
        if self._term_seen % self._term_thin == 0:
            # Buffer the raw draw and convert in batches at each check: one vmapped call per
            # check rather than a JAX dispatch per iteration, and the features are what we keep.
            self._term_pending.append(np.array(state.sample))   # a copy: see BaseSampler.postprocess
            if self.model.discrete_dim:
                self._term_pending_discrete.append(np.array(state.discrete))

        if (self._term_seen >= self._term_min_warmup
                and self._term_seen % self._term_check_every == 0):
            self._term_flush()
            self._term_check()
        return state

    def _term_flush(self) -> None:
        if not self._term_pending:
            return
        # ``map_rows``, like the evidence, summary and metric-fit passes: the same row map over
        # the same kind of array, so it uses the same helper and the same byte budget. The batch
        # here is only ``check_every`` rows, so this is consistency rather than a measurable win.
        if self.model.discrete_dim:
            # The labels are part of what has to have converged: a chain still reassigning
            # clusters is not mixed, whatever the continuous block is doing. One caveat worth
            # knowing --- a label that never moves has zero variance, so `ess_1d` returns `n` and
            # `split_rhat` returns 1.0 by their no-variance guards, i.e. a *stuck* coordinate
            # reads as perfectly converged. The `discrete_moves` diagnostic is what catches that.
            rows = map_rows(self.model.features, np.stack(self._term_pending),
                            np.stack(self._term_pending_discrete))
            self._term_pending_discrete.clear()
        else:
            rows = map_rows(self.model.features, np.stack(self._term_pending))
        self._term_features.extend(np.asarray(rows, dtype=np.float32))
        self._term_pending.clear()

    def _term_free_features(self) -> None:
        """Drop the retained feature history, once warmup is over and it can no longer be read.

        The history is one ``model.features`` row per retained draw and it is kept for the whole
        of warmup, so on a long warmup of a wide model it is the largest thing this mixin owns ---
        and it is dead weight for the entire sampling phase that follows, which is usually the
        longer one. Subclasses extend this cooperatively to drop whatever else they derived from
        it.

        Only called when warmup is *finished*, never merely because a ``warmup(n)`` call returned:
        ``warmup(500)`` twice in a row is a supported way to continue, and the second call's checks
        read the history the first call accumulated. Freeing it there would silently restart the
        criterion from an empty history.
        """
        n = len(self._term_features)
        self._term_features = []
        self._term_pending = []
        self._term_pending_discrete = []
        log.debug("%s: freed %d retained feature row(s) at the end of warmup "
                  "(pass keep_features=True to hold on to them)", self._term_name, n)

    def _term_check(self) -> None:
        early, late = self._term_split()
        if early.shape[0] < 2:
            log.debug("%s: skipping the check at warmup iteration %d --- only %d retained "
                      "draw(s) per half", self._term_name, self._term_seen, early.shape[0])
            return
        stat, mixed = self._mixing_stat(early, late)
        self._term_history.append((self._term_seen, float(stat)))
        self._term_burn_hist.append((self._term_seen, self._term_last_burn))
        # Require ``patience`` checks in a row. A single check is a noisy thing to bet on, and
        # they are repeated every ``check_every`` draws, so betting on one would stop a chain
        # that merely got lucky once. (Consecutive checks share almost all their draws, so this
        # buys less than independence would suggest -- but it costs a few hundred iterations and
        # removes the one-unlucky-check failure.)
        self._term_passes = self._term_passes + 1 if mixed else 0
        if self._term_passes >= self._term_patience:
            self._term_stop = True
        log.info("%s check at warmup iteration %d: %s = %.4f over %d draw(s) per half "
                 "(%d discarded as burn-in) -> %s (%d/%d consecutive pass(es))%s",
                 self._term_name, self._term_seen, self._stat_name, float(stat), early.shape[0],
                 self._term_last_burn, "mixed" if mixed else "not mixed", self._term_passes,
                 self._term_patience, "; criterion met, ending warmup" if self._term_stop else "")

    def _term_split(self):
        """``(early, late)``: the feature history minus a burn-in prefix, cut into equal halves.

        Stacked at the store's own float32, **not** promoted to float64. It used to be promoted
        here, purely so the classifier could subtract a mean in float64 --- after which
        ``_logistic._buffered`` rounded the result straight back to float32 for the device. At a
        6000-draw, 6000-feature history that promotion was 288 MB and the gathers taken from it
        another 259 MB, all of it discarded by that rounding. The float64 arithmetic still happens,
        one block of rows at a time, inside :func:`mimcs.adaptation._logistic._fill_standardized`,
        and is bit-for-bit what it was.
        """
        f = np.asarray(self._term_features, dtype=np.float32)
        self._term_last_burn = burn = self._term_burn_count(f)
        rest = f[burn:]
        h = len(rest) // 2
        return rest[:h], rest[h:2 * h]

    def _term_burn_count(self, f) -> int:
        """How many leading rows to discard. A fixed fraction of the history so far.

        Note what "so far" means: the prefix *grows* as warmup runs, so a chain whose transient
        outlasts ``burn_in_frac`` of the current history keeps failing the check until the history
        is long enough to bury it. :class:`ClassifierTermination` can estimate the count instead
        (``mimcs.adaptation._burnin``); this default is what :class:`GelmanRubinTermination` uses.
        """
        return int(self._term_burn_frac * len(f))

    def _mixing_stat(self, early, late):
        """Score one split: ``(statistic, mixed)``."""
        raise NotImplementedError

    # --- reporting ---

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        super()._warmup_end_hooks(completed, stopped)
        # "Finished" means warmup is over for good --- the criterion fired, or the budget ran
        # out --- as opposed to a ``warmup(n)`` call simply returning, which the caller may
        # follow with another one. Only the first two release the feature history.
        finished = True
        if self._term_stop:
            log.info("%s: warmup ended at iteration %d --- criterion met (%s = %s)",
                     self._term_name, completed, self._stat_name, self._last_stat())
        elif completed >= self._term_max_warmup:
            log.warning(
                "%s: warmup hit its maximum of %d iteration(s) without the mixing criterion "
                "firing (last %s = %s over %d check(s)). The chain may still be moving; the "
                "draws that follow are the ones it can produce, not the ones it should.",
                self._term_name, self._term_max_warmup, self._stat_name, self._last_stat(),
                len(self._term_history))
        else:
            finished = False
            log.info("%s: warmup ended after the requested %d iteration(s) before the criterion "
                     "fired (last %s = %s over %d check(s); max_warmup is %d, first check at %d)",
                     self._term_name, completed, self._stat_name, self._last_stat(),
                     len(self._term_history), self._term_max_warmup, self._term_min_warmup)
        if finished and not self._term_keep_features:
            self._term_free_features()

    def warmup_mixing_stats(self) -> np.ndarray:
        """``(k, 2)`` of ``(iteration, statistic)`` at each check --- the criterion's trajectory."""
        return np.asarray(self._term_history, dtype=float).reshape(-1, 2)

    def warmup_burn_in_estimates(self) -> np.ndarray:
        """``(k, 2)`` of ``(iteration, burn-in discarded)`` at each check.

        Kept apart from :meth:`warmup_mixing_stats`, whose ``(k, 2)`` shape callers rely on.
        Under the fixed rule this is just ``burn_in_frac`` times the retained history; under a
        dynamic rule it is the estimate, and its trajectory is the thing to look at --- a burn-in
        that keeps climbing toward the upper bound is the late-excursion failure, not a long
        transient.
        """
        return np.asarray(self._term_burn_hist, dtype=float).reshape(-1, 2)

    def warmup_terminated_early(self) -> bool:
        """Did the criterion fire, as opposed to warmup simply running out of iterations?"""
        return self._term_stop


class GelmanRubinTermination(_WarmupTermination):
    """Mixin: stop warmup when ``max R-hat`` over the features falls below ``rhat_threshold``.

    The comparison point --- R-hat is the popular, well-tested diagnostic. Usually it wants
    several chains, which this library does not run; the split form compares one chain's halves
    instead, and is what Stan reports.

    Two things are worth knowing about the default ``1.01``. It is the threshold **Vehtari et al.
    2021** recommend, and the reason is that the conventional ``1.1`` is very loose: R-hat is
    ``sqrt(1 + delta^2 / 2)`` for a half-to-half gap of ``delta`` standard deviations, so 1.1
    tolerates ``delta = 0.65 sd`` while 1.01 tolerates ``0.2 sd`` --- and that tolerance is
    independent of how many draws there are. And R-hat compares *means*, so on a raw draw it is
    blind to a chain whose scale is still moving --- the defect folded R-hat exists to fix. Here it
    is not a defect: the features include ``x^2``, whose mean does move when the scale does, so the
    plain statistic sees it.
    """

    _stat_name = "max split-R-hat"

    def _init_hooks(self, **kwargs):
        self._rhat_threshold = float(kwargs.get("rhat_threshold", 1.01))
        super()._init_hooks(**kwargs)
        log.debug("GelmanRubinTermination: threshold %.3f, checks every %d draw(s) from %d, "
                  "patience %d, max_warmup %d", self._rhat_threshold, self._term_check_every,
                  self._term_min_warmup, self._term_patience, self._term_max_warmup)

    def _mixing_stat(self, early, late):
        worst = float(np.max(split_rhat(early, late)))
        return worst, worst < self._rhat_threshold


class ClassifierTermination(_WarmupTermination):
    """Mixin: stop warmup when a logistic regression cannot tell early from late draws.

    Label each draw by which half of the split it came from and fit a classifier on the features.
    If the chain has settled the two halves are the same distribution and no rule can beat chance;
    while it is still moving, its position gives the period away. Held-out draws score the fit.

    Linear-with-``[x, x^2]`` makes this close kin to Geweke's test: its power comes from mean
    differences in the features, so it is a two-sample test on the observables with the contrast
    direction estimated rather than fixed. The classifier framing buys the room to grow a
    nonlinear rule later, where no analytic null exists.

    **"Better than chance" is the hard part**, because the null is not 1/2. Under the null each
    half still has its own sampling fluctuation --- inflated by autocorrelation --- and the fit
    finds the direction along which the two halves happen to differ. Validation is every
    ``val_every``-th draw, so held-out draws sit *inside* the same halves and share their
    low-frequency excursions: they are fooled by that spurious direction too, and the null lands
    around 0.52 rather than 0.50.

    Measured, that inflation turns out to be small and near-flat in the number of features (0.512
    to 0.524 from p=2 to p=100), because the ridge caps how many noise directions the fit can
    exploit. That is what makes a fixed ``accuracy_threshold`` viable across models. Holding
    validation out as a *contiguous tail* instead would centre the null on 0.50, but it was tried
    and lost on both counts --- less signal and three times the variance, a tail being one
    excursion and so a tiny effective sample (``docs/design/10_warmup_termination.md``).

    Hence ``rule``: ``"threshold"`` compares the accuracy to ``accuracy_threshold``, and is only
    as good as that constant's calibration. ``"permutation"`` builds the null in situ --- relabel
    contiguous blocks at random, refit, and see how unusual the real split is --- which absorbs
    whatever inflation the setup has, for any model, at the cost of ``n_perm`` refits per check.
    """

    _stat_name = "held-out classifier accuracy"

    def _init_hooks(self, **kwargs):
        self._clf_val_every = int(kwargs.get("val_every", 5))
        self._clf_l2 = float(kwargs.get("l2", DEFAULT_L2))
        self._clf_rule = str(kwargs.get("rule", "threshold"))
        self._clf_threshold = float(kwargs.get("accuracy_threshold", 0.55))
        self._clf_n_perm = int(kwargs.get("n_perm", 20))
        self._clf_block_len = int(kwargs.get("perm_block_len", 100))
        self._clf_alpha = float(kwargs.get("perm_alpha", 0.05))
        self._clf_seed = int(kwargs.get("perm_seed", 0))
        self._clf_init = None              # previous coefficients: a warm start for the next fit
        if self._clf_rule not in ("threshold", "permutation"):
            raise ValueError(f"unknown rule {self._clf_rule!r} (use 'threshold' or 'permutation')")

        # --- dynamic burn-in (``mimcs.adaptation._burnin``) ---
        self._burn_mode = str(kwargs.get("burn_in", "fixed"))
        if self._burn_mode not in BURN_IN_MODES:
            raise ValueError(f"unknown burn_in {self._burn_mode!r} (use one of {BURN_IN_MODES})")
        self._burn_min = int(kwargs.get("burn_in_min", 50))
        self._burn_min_frac = float(kwargs.get("burn_in_min_frac", 0.0))
        self._burn_max_frac = float(kwargs.get("burn_in_max_frac", 0.5))
        self._burn_max_iter = int(kwargs.get("burn_in_max_iter", 4))
        self._burn_tail_tol = float(kwargs.get("burn_in_tail_tol", 0.1))
        self._burn_null = str(kwargs.get("burn_in_null", "scaled"))
        self._burn_null_slack = float(kwargs.get("burn_in_null_slack", 0.5))
        if self._burn_null not in _burnin.NULLS:
            raise ValueError(f"unknown burn_in_null {self._burn_null!r} "
                             f"(use one of {_burnin.NULLS})")
        self._burn_objective = kwargs.get("burn_in_objective", _burnin.additive)
        self._burn_fit_opts = dict(kwargs.get("burn_in_fit_opts", {}))
        self._burn_last: _burnin.BurnIn | None = None
        self._burn_cache: tuple[int, int] = (-1, 0)     # (history length, count) -- see below
        self._burn_matrix = None                        # the standardized history, one per check
        # Warm starts are kept per *role*: within a check the search walks candidate splits whose
        # discriminants are similar, but a prefix-vs-tail direction and a tail-early-vs-late one
        # are not, and seeding either from the other only costs iterations.
        self._burn_init: dict = {"prefix": None, "tail": None, "null": None}

        super()._init_hooks(**kwargs)
        log.debug("ClassifierTermination: rule %r (threshold %.3f / alpha %.3f), burn-in %r, "
                  "checks every %d draw(s) from %d, patience %d, max_warmup %d", self._clf_rule,
                  self._clf_threshold, self._clf_alpha, self._burn_mode, self._term_check_every,
                  self._term_min_warmup, self._term_patience, self._term_max_warmup)

    # --- burn-in ---

    def _term_free_features(self) -> None:
        """Also drop the burn-in search's standardized copy of the history, which is the same
        size again. ``_burn_last`` is a handful of scalars and stays, as the record of what the
        last search chose."""
        super()._term_free_features()
        self._burn_matrix = None
        self._burn_cache = (-1, 0)      # keyed on history length, which is now 0

    def _term_burn_count(self, f) -> int:
        """The burn-in prefix: the fixed fraction, or an estimate from the three-way search.

        Cached on the history length. ``_term_split()`` is public enough that experiment scripts
        call it directly, and under a dynamic mode each call would otherwise repeat the search's
        eight-or-so logistic fits.
        """
        if self._burn_mode == "fixed":
            return super()._term_burn_count(f)
        n = len(f)
        if self._burn_cache[0] == n:
            return self._burn_cache[1]

        limits = _burnin.bounds(n, min_abs=self._burn_min, min_frac=self._burn_min_frac,
                                max_frac=self._burn_max_frac,
                                # The tail must survive its own halving and still leave each half
                                # enough rows to hold out a validation set worth scoring.
                                min_tail=8 * self._clf_val_every)
        if limits is None:
            # Too little history to search. Fall back to the fixed fraction rather than to zero:
            # the early checks are exactly where an unremoved transient does the damage.
            count = super()._term_burn_count(f)
        else:
            lo, hi = limits
            # Standardize once, over the whole history, and hand every candidate the same matrix.
            # ``dtype=np.float64`` is load-bearing: ``f`` is the float32 store, and reducing it in
            # its own dtype would quietly move this search's numbers. Promoting each element to
            # float64 as it is accumulated is bit-for-bit what reducing a float64 copy of ``f``
            # gave when ``_term_split`` still made one. ``_burn_matrix`` itself stays float64 ---
            # ``_logistic.scores`` reads it at float64, so narrowing it would only move the copy.
            mu = np.mean(f, axis=0, dtype=np.float64)
            sd = np.std(f, axis=0, dtype=np.float64)
            self._burn_matrix = (f - mu) / np.where(sd > 1e-12, sd, 1.0)
            self._burn_last = _burnin.estimate_burn_in(
                n, self._burn_fit, mode=self._burn_mode, lo=lo, hi=hi,
                init=self._burn_cache[1] or None, max_iter=self._burn_max_iter,
                objective=self._burn_objective, tail_tol=self._burn_tail_tol,
                null=self._burn_null, null_slack=self._burn_null_slack,
                n_perm=self._clf_n_perm, perm_block=self._clf_block_len, seed=self._clf_seed)
            count = self._burn_last.n
            log.debug("ClassifierTermination: burn-in %d of %d row(s) in [%d, %d] (%s/%s null: "
                      "objective %.4f, prefix %.4f, tail %.4f, %d fit(s) over splits %s)", count,
                      n, lo, hi, self._burn_mode, self._burn_null, self._burn_last.objective,
                      self._burn_last.sep_prefix, self._burn_last.sep_tail,
                      len(self._burn_last.path), self._burn_last.path)
        self._burn_cache = (n, count)
        return count

    def _burn_fit(self, idx_a, idx_b, role):
        """The search's ``fit``: ``(separation, projection)``. See ``_burnin.FitFn``.

        Three deliberate differences from the decision fit below. The classes are **weighted**,
        because a candidate prefix can be a twentieth of the history and an unweighted fit would
        buy its likelihood by predicting the majority. The score is :func:`log_score`, a proper
        scoring rule and therefore smooth in the split point, where held-out accuracy on a short
        prefix moves in visible steps of 1/12. And every candidate is fitted on the *whole*
        standardized history with the rows it excludes carrying zero weight, so the design matrix
        never changes shape --- see :func:`mimcs.adaptation._logistic.class_weights` for why that
        is not merely tidy.
        """
        z = self._burn_matrix
        y = np.zeros(len(z))
        y[idx_b] = 1.0
        # Held out the same way the decision does: every ``val_every``-th row of each block, so
        # validation draws sit inside the same low-frequency excursions as the training ones.
        keep = np.zeros(len(z), dtype=bool)
        is_val = np.zeros(len(z), dtype=bool)
        for idx in (idx_a, idx_b):
            keep[idx] = True
            is_val[idx[::self._clf_val_every]] = True
        train, val = keep & ~is_val, keep & is_val
        if not (val.any() and len(np.unique(y[train])) == 2 and len(np.unique(y[val])) == 2):
            # Nothing to learn from, or nothing to score on. No separation, and a flat projection
            # whose changepoint is wherever the bounds put it.
            return 0.0, np.zeros(len(z))
        fit = fit_logistic(z, y, l2=self._clf_l2, wt=class_weights(y, keep=train),
                           init=self._burn_init[role], **self._burn_fit_opts)
        self._burn_init[role] = (fit.w, fit.b)
        return log_score(fit, z[val], y[val]), scores(fit, z)

    # --- the decision ---

    def _mixing_stat(self, early, late):
        acc = self._fit_and_score(early, late, warm=True)
        if self._clf_rule == "threshold":
            return acc, acc < self._clf_threshold
        p_value = self._permutation_p_value(early, late, acc)
        log.debug("ClassifierTermination: permutation p-value %.3f for accuracy %.4f "
                  "(%d refits, alpha %.3f)", p_value, acc, self._clf_n_perm, self._clf_alpha)
        # Cannot reject "the two halves look alike" => treat the chain as mixed.
        return acc, p_value > self._clf_alpha

    def _fit_and_score(self, early, late, *, warm: bool) -> float:
        x_tr, y_tr, x_va, y_va = self._train_val(early, late)
        if len(x_va) == 0 or len(np.unique(y_tr)) < 2:
            return 0.5
        # ``dtype=np.float64`` accumulates in float64 over the float32 gather, which is bit-for-bit
        # what reducing a float64 copy of it gave: float32 -> float64 is exact, and the reduction
        # order depends only on the shape, which has not changed.
        mu = np.mean(x_tr, axis=0, dtype=np.float64)
        sd = np.std(x_tr, axis=0, dtype=np.float64)
        sd = np.where(sd > 1e-12, sd, 1.0)          # a constant feature carries no information
        # The training half is standardized *while it is buffered* --- one blocked float64 pass
        # straight into the device-dtype array, instead of a whole float64 copy that the buffering
        # would immediately round away. On a long history this block is the largest array the
        # check touches.
        fit = fit_logistic(x_tr, y_tr, l2=self._clf_l2, standardize=(mu, sd),
                           init=self._clf_init if warm else None)
        if warm:
            self._clf_init = (fit.w, fit.b)
        # The validation half is scored in float64, because ``accuracy`` runs it through numpy at
        # float64: rounding it to float32 first would move the score. Promoting it here costs a
        # copy, but it is ``1/val_every`` of the rows --- not where the memory was.
        x_va = np.array(x_va, dtype=np.float64)
        np.subtract(x_va, mu, out=x_va)
        np.divide(x_va, sd, out=x_va)
        return accuracy(fit, x_va, y_va)

    def _train_val(self, early, late):
        """Hold out every ``val_every``-th row of each half; the rest trains.

        Each output is gathered **into one preallocated array** rather than built as two gathers
        that are then concatenated: on a long history those blocks are the largest arrays the
        check touches, and the old form held the halves and their concatenation at once. The row
        order is unchanged (early rows, then late rows), and ``np.compress`` writes exactly what
        ``block[mask]`` would.

        Note the halves are usually *views* into the feature history, and their ``.base`` is the
        whole history rather than the burn-in-trimmed part --- so they must be read through, never
        reconstructed from ``.base``.
        """
        h = len(early)
        is_val = np.zeros(h, dtype=bool)
        is_val[::self._clf_val_every] = True
        keep = ~is_val
        n_tr, n_va = int(keep.sum()), int(is_val.sum())
        x_tr = np.empty((2 * n_tr, early.shape[1]), dtype=early.dtype)
        np.compress(keep, early, axis=0, out=x_tr[:n_tr])
        np.compress(keep, late, axis=0, out=x_tr[n_tr:])
        x_va = np.empty((2 * n_va, early.shape[1]), dtype=early.dtype)
        np.compress(is_val, early, axis=0, out=x_va[:n_va])
        np.compress(is_val, late, axis=0, out=x_va[n_va:])
        y_tr = np.concatenate([np.zeros(n_tr), np.ones(n_tr)])
        y_va = np.concatenate([np.zeros(n_va), np.ones(n_va)])
        return x_tr, y_tr, x_va, y_va

    def _permutation_p_value(self, early, late, observed: float) -> float:
        """How unusual is ``observed`` among relabellings that respect the autocorrelation?

        Labels are shuffled in *contiguous blocks*, not per draw: under the null the chain is
        stationary, so blocks longer than the autocorrelation time are exchangeable, whereas
        shuffling single draws would destroy the very correlation that inflates the statistic and
        hand back a null that is too optimistic.
        """
        rng = np.random.default_rng(self._clf_seed + len(self._term_features))
        f = np.concatenate([early, late])
        n_blocks = max(2, len(f) // max(1, self._clf_block_len))
        blocks = np.array_split(np.arange(len(f)), n_blocks)
        n_ge = 0
        for _ in range(self._clf_n_perm):
            order = rng.permutation(len(blocks))
            half = len(blocks) // 2
            idx_a = np.concatenate([blocks[i] for i in order[:half]])
            idx_b = np.concatenate([blocks[i] for i in order[half:2 * half]])
            m = min(len(idx_a), len(idx_b))
            acc = self._fit_and_score(f[idx_a[:m]], f[idx_b[:m]], warm=False)
            n_ge += int(acc >= observed)
        return (n_ge + 1) / (self._clf_n_perm + 1)
