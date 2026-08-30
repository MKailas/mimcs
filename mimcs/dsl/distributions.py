"""Distribution registry for the model DSL.

Maps a Stan distribution name to a function ``(value, *params) -> elementwise log-density``
over ``jax.scipy.stats``, with **parameterization adapters** where Stan and SciPy differ
(Stan ``normal`` uses the standard deviation = SciPy ``scale``; ``exponential`` / ``gamma``
use the *rate*, SciPy the *scale*; ``multi_normal`` uses the covariance). The ``~`` statement
sums the returned array, so each entry here returns the *elementwise* log-density (joint
distributions like ``multi_normal`` already return a scalar, for which the sum is a no-op).

Both continuous (``logpdf``) and discrete (``logpmf``) distributions are registered the same
way --- the interpreter is agnostic, summing whatever the entry returns. Discrete
distributions are typically used for *observed data* on the left of ``~`` (e.g.
``y ~ poisson(lambda)`` with integer data ``y`` and a continuous parameter ``lambda``); the
returned log-pmf is differentiable in the continuous parameters, which is what the samplers
need.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.scipy.special import gammaln, logsumexp, xlogy


def _normal(x, mu, sigma):
    return jss.norm.logpdf(x, loc=mu, scale=sigma)            # Stan sigma = sd = SciPy scale


def _lognormal(x, mu, sigma):
    return jss.norm.logpdf(jnp.log(x), loc=mu, scale=sigma) - jnp.log(x)


def _uniform(x, lo, hi):
    return jss.uniform.logpdf(x, loc=lo, scale=hi - lo)


def _exponential(x, rate):
    return jss.expon.logpdf(x, scale=1.0 / rate)             # Stan rate -> SciPy scale


def _gamma(x, alpha, rate):
    return jss.gamma.logpdf(x, alpha, scale=1.0 / rate)      # Stan (shape, rate)


def _inv_gamma(x, alpha, beta):
    """Stan ``inv_gamma(alpha, beta)``: shape and *scale* of the inverse-gamma.

    ``alpha log beta - lgamma(alpha) - (alpha + 1) log x - beta / x``, written directly rather
    than via ``jss.gamma.logpdf(1/x, ...)`` plus a Jacobian, so the ``-beta/x`` term stays exact
    for small ``x``.
    """
    return (alpha * jnp.log(beta) - gammaln(alpha)
            - (alpha + 1.0) * jnp.log(x) - beta / x)


def _cauchy(x, loc, scale):
    return jss.cauchy.logpdf(x, loc=loc, scale=scale)


def _beta(x, a, b):
    return jss.beta.logpdf(x, a, b)


def _student_t(x, nu, loc, scale):
    return jss.t.logpdf(x, nu, loc=loc, scale=scale)


def _multi_normal(x, mu, cov):
    return jss.multivariate_normal.logpdf(x, mean=mu, cov=cov)   # scalar


# --- discrete distributions (logpmf) ---------------------------------------- #

def _bernoulli(k, theta):
    return jss.bernoulli.logpmf(k, theta)                        # theta = success probability


def _bernoulli_logit(k, alpha):
    """Stan ``bernoulli_logit(alpha)``: Bernoulli with success probability ``sigmoid(alpha)``,
    parameterized by the **log-odds**.

    Not sugar for ``bernoulli(sigmoid(alpha))`` --- it is the only usable form when ``alpha`` is
    a linear predictor over many covariates. ``sigmoid`` saturates to exactly 0 or 1 in floating
    point for ``|alpha| >~ 37`` (float64), after which ``log(1 - p)`` is ``-inf`` and the whole
    density dies; a random start over 2000 predictors reaches ``|alpha| ~ 200`` easily. Written
    as ``k*alpha - softplus(alpha)``, which is exact for every ``alpha`` and whose gradient
    ``k - sigmoid(alpha)`` is well behaved in the tails.
    """
    return k * alpha - jnp.logaddexp(0.0, alpha)


def _n_categories(p, dist: str, arg: str) -> int:
    """The number of categories, or a message saying what was expected instead.

    Shapes are static, so this runs at trace time. Without it a scalar argument reaches
    ``p.shape[-1]`` and surfaces as a bare ``IndexError: tuple index out of range``, which says
    nothing about the model that caused it.
    """
    if p.ndim == 0:
        raise ValueError(
            f"{dist}({arg}) needs an array of category {'probabilities' if arg == 'theta' else 'log-probabilities'}, "
            f"got a scalar")
    return p.shape[-1]


def _categorical(y, theta):
    """Stan ``categorical(theta)``: ``P(y = k) = theta[k]``, with ``y`` in **1..K**.

    The natural prior over an unordered label, and so the natural companion to an
    ``int<lower=1, upper=K>`` parameter --- which is what this exists for. ``y`` is 1-based, both
    to match Stan and because it is what the DSL's own 1-based indexing produces: the same ``z[n]``
    reads ``mu[z[n]]`` and is declared here.

    Indexed with ``jnp.take`` rather than ``theta[y - 1]`` so that an out-of-range label is
    ``-inf`` rather than a silently clamped gather. JAX clamps out-of-bounds indices instead of
    raising, which would otherwise turn "label 0" (a 1-based off-by-one) into a valid-looking
    density at ``theta[0]``.
    """
    theta = jnp.asarray(theta)
    K = _n_categories(theta, "categorical", "theta")
    idx = jnp.asarray(y) - 1
    inside = (idx >= 0) & (idx < K)
    p = jnp.take(theta, jnp.clip(idx, 0, K - 1), axis=-1)
    return jnp.where(inside, jnp.log(p), -jnp.inf)


def _categorical_logit(y, alpha):
    """Stan ``categorical_logit(alpha)``: categorical with ``theta = softmax(alpha)``.

    The same relationship to :func:`_categorical` that ``bernoulli_logit`` has to ``bernoulli``,
    and for the same reason: with ``alpha`` an unconstrained linear predictor, forming ``softmax``
    and taking its log loses the large-``|alpha|`` tail, while ``alpha[y] - logsumexp(alpha)`` is
    exact everywhere.
    """
    alpha = jnp.asarray(alpha)
    K = _n_categories(alpha, "categorical_logit", "alpha")
    idx = jnp.asarray(y) - 1
    inside = (idx >= 0) & (idx < K)
    a = jnp.take(alpha, jnp.clip(idx, 0, K - 1), axis=-1)
    return jnp.where(inside, a - logsumexp(alpha, axis=-1), -jnp.inf)


def _binomial(k, N, theta):
    return jss.binom.logpmf(k, N, theta)                         # Stan (N, theta) = SciPy (n, p)


def _poisson(k, rate):
    # Direct log-pmf ``k log(rate) - rate - log(k!)`` (Stan rate = SciPy mu), rather than
    # ``jss.poisson.logpmf``: the latter guards with a ``round(k) != k`` integer check that is
    # miscompiled under ``jax.jit`` for a *transposed* large-integer-valued ``k`` array (XLA
    # returns a wrong ``round``), spuriously masking every entry to ``-inf``. Here ``k`` is fixed
    # integer data and differentiation is w.r.t. ``rate``, so the check is unneeded anyway;
    # ``xlogy`` handles ``k = 0`` (``rate`` may be 0 there).
    return xlogy(k, rate) - rate - gammaln(k + 1.0)


def _neg_binomial(k, alpha, beta):
    # Stan neg_binomial(alpha, beta): SciPy nbinom with n = alpha, p = beta / (beta + 1).
    return jss.nbinom.logpmf(k, alpha, beta / (beta + 1.0))


def _neg_binomial_2(k, mu, phi):
    # Stan neg_binomial_2(mu, phi): mean mu, variance mu + mu^2 / phi -> nbinom(n=phi,
    # p = phi / (phi + mu)).
    return jss.nbinom.logpmf(k, phi, phi / (phi + mu))


def _beta_binomial(k, N, alpha, beta):
    return jss.betabinom.logpmf(k, N, alpha, beta)               # Stan (N, alpha, beta)


def _geometric(k, theta):
    return jss.geom.logpmf(k, theta)                            # SciPy convention: support k >= 1


def _multinomial(x, theta):
    # Joint (scalar) log-pmf; the counts must be integral. ``theta`` is a simplex and N is the
    # total count = sum of the outcomes (Stan's convention).
    k = jnp.asarray(x).astype(jnp.int32)
    return jss.multinomial.logpmf(k, jnp.sum(k), theta)


def _lkj_corr(x, eta):
    """LKJ(eta) on a correlation matrix: joint, unnormalized, ``(eta - 1) log det Omega``."""
    return (eta - 1.0) * jnp.linalg.slogdet(x)[1]


def _lkj_corr_cholesky(x, eta):
    """LKJ(eta) on its Cholesky factor: joint, unnormalized.

    The density of ``L`` rather than of ``Omega``, so it carries the ``L -> Omega`` Jacobian:
    ``det Omega = prod L_ii^2`` gives ``2 (eta - 1) sum log L_ii``, and the Jacobian adds
    ``sum_i (K - 1 - i) log L_ii`` (0-based ``i``), for
    ``sum_i (K - 1 - i + 2 eta - 2) log L_ii``. The ``i = 0`` term vanishes since ``L_00 = 1``.
    This is the piece worth having as a named distribution: written by hand it is easy to drop
    the Jacobian and sample the wrong target without anything looking wrong.
    """
    K = jnp.shape(x)[-1]
    i = jnp.arange(K, dtype=float)
    diag = jnp.diagonal(x, axis1=-2, axis2=-1)
    return jnp.sum((K - 1 - i + 2.0 * eta - 2.0) * jnp.log(diag), axis=-1)


DISTRIBUTIONS = {
    "normal": _normal,
    "lognormal": _lognormal,
    "uniform": _uniform,
    "exponential": _exponential,
    "gamma": _gamma,
    "inv_gamma": _inv_gamma,
    "cauchy": _cauchy,
    "beta": _beta,
    "student_t": _student_t,
    "multi_normal": _multi_normal,
    "lkj_corr": _lkj_corr,
    "lkj_corr_cholesky": _lkj_corr_cholesky,
    # discrete
    "bernoulli": _bernoulli,
    "categorical": _categorical,
    "categorical_logit": _categorical_logit,
    "bernoulli_logit": _bernoulli_logit,
    "binomial": _binomial,
    "poisson": _poisson,
    "neg_binomial": _neg_binomial,
    "neg_binomial_2": _neg_binomial_2,
    "beta_binomial": _beta_binomial,
    "geometric": _geometric,
    "multinomial": _multinomial,
}
