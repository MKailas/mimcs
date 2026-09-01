"""Sample evaluation and summary: ``sampler.summary()``.

After sampling, the question is whether to *accept* the draws or feed them back to
:func:`mimcs.analyze` for another round. This produces the numbers that inform that call --- and,
unlike ESS and R-hat alone, one that is **target-aware**.

Two tables.

**Posterior summary**, per ambient coordinate: mean, its Monte Carlo standard error, standard
deviation, and a 5/50/95% credible interval --- the estimates a data analysis reports.

Row labels index **from 1** (`x[1]`, `S[2][1,1]`; a scalar has no index), matching the DSL these
tables are read alongside. They are labels only --- the arrays behind them are ordinary 0-based
JAX arrays.

**Diagnostics**, per *feature* (the observables each parameter declares, ``[x, x^2]`` by default;
ESS and R-hat are properly per-feature, not per-parameter):

* **ESS** and **ess/n** --- effective sample size, how much the autocorrelation costs.
* **R-hat** --- split-R-hat over the run's two halves; a single-chain mixing check.
* **Stein z** --- the target-aware one. ESS and R-hat notice only whether the draws look like a
  well-mixed sample from *some* distribution; they pass a sample that is biased with respect to the
  actual target. The Langevin--Stein operator applied to a feature ``phi`` produces a function of
  mean zero *under the target* (:meth:`mimcs.model.BaseParameter.stein_terms`), so its sample
  average, divided by its Monte Carlo standard error, is asymptotically ``N(0, 1)`` *iff* the draws
  are from the target. A large ``|z|`` is evidence of bias that mixing diagnostics cannot see. It
  uses only the score, so it needs no normalizing constant.

There is deliberately no automated accept/reject. With ``m`` features about ``0.05 m`` of the
Stein z's exceed 1.96 by chance, so a per-feature flag with a multiplicity note is shown and the
call is left to the user. (No aggregate Stein test either: that would invert an ``m x m``
covariance, the high-dimensional instability that makes kernel Stein discrepancies unreliable.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax

from ._logging import get_logger
from .diagnostics import ess, mcse_mean, split_rhat

log = get_logger(__name__)

Z95 = 1.959963984540054       # two-sided 95% normal quantile
QUANTILES = (0.05, 0.50, 0.95)
#: split-R-hat above which :func:`summarize` warns that the draws do not look converged.
RHAT_WARN = 1.1


@dataclass
class Summary:
    """The result of :meth:`mimcs.samplers.BaseSampler.summary`. Renders as two tables; every
    column is also an array field, so ``s.stein_z``, ``s.ess`` etc. are available."""

    n_samples: int
    accept_rate: float | None
    # posterior table (per ambient coordinate)
    coord_names: list
    mean: np.ndarray
    mcse: np.ndarray
    sd: np.ndarray
    quantiles: np.ndarray            # (len(QUANTILES), n_coords)
    # diagnostics table (per feature)
    feature_names: list
    ess: np.ndarray
    rhat: np.ndarray
    stein_z: np.ndarray
    stein_est: np.ndarray
    stein_mcse: np.ndarray
    stein_boundary: np.ndarray       # bool: degenerate (near-zero-variance) Stein series
    n_nonfinite: int = 0             # draws dropped from the Stein estimate for a non-finite score
    stein_defined: np.ndarray | None = None   # bool: has this feature a Stein term at all?

    @property
    def stein_available(self) -> np.ndarray:
        """Per feature: is a Stein term defined for it? All ``True`` for a continuous model.

        ``False`` for a discrete parameter's features: the Langevin--Stein identity integrates by
        parts against a density and its score, and a probability mass function has neither, so
        there is no z and the table shows a gap rather than a number
        (``docs/design/14_discrete_parameters.md``).
        """
        if self.stein_defined is None:
            return np.ones(len(self.feature_names), bool)
        return np.asarray(self.stein_defined, bool)

    @property
    def stein_flagged(self) -> np.ndarray:
        """Features whose Stein z exceeds the 95% band (and whose series is neither degenerate
        nor undefined)."""
        return (np.abs(self.stein_z) > Z95) & ~self.stein_boundary & self.stein_available

    def __str__(self) -> str:
        acc = "n/a" if self.accept_rate is None else f"{self.accept_rate:.3f}"
        lines = [f"Sample summary --- {self.n_samples} draws, accept {acc}", ""]

        lines.append("Posterior summary")
        w = max((len(n) for n in self.coord_names), default=9)
        q = {int(100 * p): self.quantiles[i] for i, p in enumerate(QUANTILES)}
        lines.append(f"  {'':>{w}}  {'mean':>10} {'mcse':>9} {'sd':>9} "
                     f"{'5%':>9} {'50%':>9} {'95%':>9}")
        for j, name in enumerate(self.coord_names):
            lines.append(f"  {name:>{w}}  {self.mean[j]:>10.4g} {self.mcse[j]:>9.3g} "
                         f"{self.sd[j]:>9.3g} {q[5][j]:>9.4g} {q[50][j]:>9.4g} {q[95][j]:>9.4g}")

        lines += ["", "Diagnostics (per feature)"]
        wf = max((len(n) for n in self.feature_names), default=9)
        lines.append(f"  {'':>{wf}}  {'ess':>9} {'ess/n':>7} {'R-hat':>7} {'stein-z':>9}  flag")
        available = self.stein_available
        for k, name in enumerate(self.feature_names):
            if not available[k]:
                # A discrete feature: no density, no score, so no Stein identity to test. A dash
                # rather than a number, because a 0.00 here would read as "passed".
                zc, flag = f"{'--':>9}", "discrete"
            elif self.stein_boundary[k]:
                zc, flag = f"{self.stein_est[k]:>+9.3g}", "boundary?"
            else:
                zc = f"{self.stein_z[k]:>+9.2f}"
                flag = "*" if abs(self.stein_z[k]) > Z95 else ""
            lines.append(f"  {name:>{wf}}  {self.ess[k]:>9.0f} "
                         f"{self.ess[k] / self.n_samples:>7.2f} {self.rhat[k]:>7.3f} {zc}  {flag}")

        n_flag = int(self.stein_flagged.sum())
        n_testable = int(available.sum())
        expected = 0.05 * n_testable
        suffix = ("" if n_testable == len(self.feature_names)
                  else f" ({len(self.feature_names) - n_testable} discrete feature(s) have no "
                       f"Stein test)")
        lines += ["", f"  Stein: {n_flag} of {n_testable} features flagged at 95% "
                      f"(~{expected:.0f} expected under the null; not an automated verdict)."
                      + suffix]
        if self.n_nonfinite:
            lines.append(f"  {self.n_nonfinite} draw(s) dropped from the Stein estimate "
                         "(non-finite score).")
        return "\n".join(lines)

    __repr__ = __str__


def summarize(model, draws: np.ndarray, accept_rate: float | None = None, *,
              coord_score: np.ndarray | None = None,
              chart_hyperparams: tuple | None = None,
              chart_indices: tuple | None = None,
              discrete_draws: np.ndarray | None = None) -> Summary:
    """Evaluate ``draws`` (``(n, ambient_dim)``, sample space) under ``model``.

    A pure function of the model and the draws, so it is testable without a sampler.
    ``coord_score`` is the sampler's saved coordinate-space gradients (``(n, coord_dim)``): when
    given, the ambient score is recovered by the chart pullback
    (:meth:`mimcs.model.Model.ambient_score`), reusing the sampler's already-spent compute; when
    ``None`` it is recomputed from the model. ``chart_*`` name the frozen sampling chart (default:
    the model's initial chart).

    ``discrete_draws`` is the ``(n, discrete_dim)`` integer block, required when the model has
    discrete parameters. Those contribute rows to both tables --- a posterior summary of a label
    is a legitimate thing to read, if an unusual one --- but no Stein z, which
    :attr:`Summary.stein_available` marks.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] == 0:
        raise ValueError("summarize needs a non-empty (n, ambient_dim) array of draws")
    n = draws.shape[0]
    jdraws = jax.numpy.asarray(draws)
    if model.discrete_dim:
        if discrete_draws is None:
            raise ValueError(
                f"summarize needs this model's discrete draws "
                f"{[p.name for p in model.discrete_parameters]}: pass `discrete_draws=` "
                f"(BaseSampler.summary() takes them from get_discrete_flat())")
        jdiscrete = jax.numpy.asarray(np.asarray(discrete_draws))
        if jdiscrete.shape != (n, model.discrete_dim):
            raise ValueError(
                f"discrete_draws has shape {tuple(jdiscrete.shape)}, expected "
                f"{(n, model.discrete_dim)}")
    else:
        jdiscrete = None
    log.debug("summarizing %d draw(s) of ambient dim %d; ambient score %s", n, draws.shape[1],
              "pulled back from the sampler's saved coordinate gradients"
              if coord_score is not None else "recomputed from the model")

    # --- posterior table, per ambient coordinate ---
    # The discrete block joins the continuous one here, cast to float: a mean, an interval and an
    # MCSE of a label are ordinary sample statistics, and `ambient_names` already lists both.
    # Nothing downstream of this line treats them as integers, and nothing should --- the *draws*
    # stay integer, in `get_discrete_flat`.
    post = draws if jdiscrete is None else np.concatenate(
        [draws, np.asarray(discrete_draws, dtype=float).reshape(n, -1)], axis=1)
    mean = post.mean(axis=0)
    sd = post.std(axis=0, ddof=1)
    mcse = mcse_mean(post)
    quantiles = np.quantile(post, QUANTILES, axis=0)

    # --- features and the ambient score ---
    feats = np.asarray(jax.vmap(model.features)(jdraws) if jdiscrete is None
                       else jax.vmap(model.features)(jdraws, jdiscrete))
    if coord_score is not None:
        # Straight to the device: the float64 hop this used to take is exactly a no-op on the
        # values (a float32 is exactly representable in float64, and rounding back is the
        # identity) while costing a full ``(n, coord_dim)`` float64 temporary.
        cs = jax.numpy.asarray(coord_score)
        scores = jax.vmap(lambda x, g: model.ambient_score(x, g, chart_hyperparams, chart_indices))(
            jdraws, cs)
    else:
        scores = jax.vmap(lambda x: model.ambient_score(x))(jdraws)
    stein = np.asarray(jax.vmap(model.stein_terms)(jdraws, scores))

    # A non-finite score (e.g. a draw at a bound) poisons its Stein row; drop those draws.
    finite = np.isfinite(stein).all(axis=1)
    n_nonfinite = int((~finite).sum())
    stein_f = stein[finite] if n_nonfinite else stein

    # --- diagnostics table, per feature ---
    ess_f = ess(feats)
    rhat = split_rhat(feats[: n // 2], feats[n // 2: 2 * (n // 2)])
    stein_est = stein_f.mean(axis=0)
    stein_mcse = mcse_mean(stein_f)
    # A degenerate (near-constant) Stein series signals the boundary term did not vanish, not
    # convergence: report the estimate rather than an exploding z.
    boundary = stein_mcse < 1e-8 * (np.abs(stein_est) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        stein_z = np.where(boundary, 0.0, stein_est / stein_mcse)

    # Features with no Stein term at all (a discrete parameter's) are zero-padded in
    # `stein_terms`, so their series is exactly constant --- which the boundary rule above does
    # *not* catch (`0 < 1e-20` is false), leaving a 0/0 nan that would poison `max |z|`. Mask them
    # explicitly instead, and keep them out of `boundary`, which means something different.
    defined = np.asarray(model.stein_defined, bool)
    if not defined.all():
        stein_z = np.where(defined, stein_z, 0.0)
        stein_est = np.where(defined, stein_est, np.nan)
        stein_mcse = np.where(defined, stein_mcse, np.nan)
        boundary = boundary & defined

    if n_nonfinite:
        log.debug("summarize: %d of %d draw(s) dropped from the Stein estimate (non-finite "
                  "score)", n_nonfinite, n)
    worst_rhat, min_ess = float(np.max(rhat)), float(np.min(ess_f))
    log.info("summary of %d draw(s) over %d feature(s): min ESS %.1f, max split-R-hat %.4f, "
             "max |Stein z| %.2f", n, feats.shape[1], min_ess, worst_rhat,
             float(np.max(np.abs(stein_z))))
    if worst_rhat > RHAT_WARN:
        log.warning("summary: max split-R-hat %.4f exceeds %.2f --- the draws do not look "
                    "converged; treat the posterior table as provisional", worst_rhat, RHAT_WARN)

    return Summary(
        n_samples=n, accept_rate=accept_rate,
        coord_names=list(model.ambient_names), mean=mean, mcse=mcse, sd=sd, quantiles=quantiles,
        feature_names=list(model.feature_names), ess=ess_f, rhat=rhat,
        stein_z=stein_z, stein_est=stein_est, stein_mcse=stein_mcse, stein_boundary=boundary,
        stein_defined=defined,
        n_nonfinite=n_nonfinite)
