"""Phase-space state and context for Hamiltonian integration.

Implements the base ``IntegratorState`` contract and ``HamiltonianContext`` from
``docs/design/06_hamiltonian_monte_carlo.md``.

``IntegratorState`` is what flows through an integrator. Its six base fields ---
position ``q``, momentum ``p``, per-potential caches of values and gradients, an
accumulated ``log_weight``, and an integrator-owned ``integrator_data`` dict --- are the
only things the integrator reads or writes (always via ``_replace``), so sampler-specific
extensions ride through untouched.
Caching the gradient (and, for free, the value) of each potential component lets
leapfrog reuse one gradient per step instead of two, and lets later trajectory
samplers (NUTS) resume integration from a stored endpoint without recomputation.

``log_weight`` is a scalar log-probability weight an integrator may accumulate for the
phase point it produces. Standard symplectic integrators (leapfrog, Yoshida, RESPA)
are deterministic and volume-preserving, so they leave it at ``0``. It exists for
*randomized / adaptive* integrators that must report a correction to stay exact ---
the motivating case being WALNUTS (Bou-Rabee & Carpenter), whose within-orbit adaptive
step-size selection accumulates the log forward/reverse transition ratio here. Because
the weight is produced inside the integrator's step, it belongs in the contract the
integrator may write, i.e. a base field. Axis-3 samplers fold it into acceptance as
``min(1, exp(H0 - H1 + Δlog_weight))`` and into trajectory selection weights.

``integrator_data`` is a dict reserved for *other* integrator-specific outputs that a
sampler may consume: the WALNUTS line-search integrators write a "proxy energy" (a
coarse-level-equivalent energy used to drive step-size adaptation) and a refinement-count
diagnostic here. Deterministic integrators leave it ``{}``. Each integrator declares its
schema (``init_integrator_data``), the samplers seed it once per trajectory, and it rides
through every ``_replace`` untouched (like ``log_weight``) unless the integrator rewrites
it. It is never mutated in place --- always replaced with a fresh dict.

All three are NamedTuples with dict fields, hence valid JAX pytrees: they trace cleanly
through ``jit``/``lax.fori_loop`` as long as the dict keys stay fixed within a run (the
component ids for the caches; the integrator's schema for ``integrator_data``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

from jax import Array


class IntegratorState(NamedTuple):
    """Base integrator contract: position, momentum, per-potential caches, log-weight."""

    q: Array                 # position (coordinate space), shape (n,)
    p: Array                 # momentum, shape (n,)
    potential_values: dict   # {potential_id: scalar}  cached V_i(q)
    potential_grads: dict    # {potential_id: Array(n,)}  cached grad_q V_i(q)
    log_weight: Array        # scalar: accumulated log-weight for this phase point
    integrator_data: dict = {}   # integrator-specific outputs (WALNUTS proxy energy, ...); {} default


class HamiltonianContext(NamedTuple):
    """Per-trajectory constants a Hamiltonian component needs besides (q, p)."""

    chart_hyperparams: tuple   # from state.chart_hyperparams
    chart_indices: tuple       # from state.chart_indices
    ham_params: dict           # adapted component parameters, keyed by component id
    betas: Any = None          # parallel tempering only (doc 13): the inverse-temperature
                               # ladder, as a *traced* value so it can be adapted without
                               # retracing the kernel. ``None`` everywhere else.
    discrete: Any = None       # the model's flat integer block, or ``None`` when it has none.
                               # A trajectory constant in the strictest sense: HMC never moves a
                               # discrete coordinate, so the density it integrates is
                               # ``pi(. | discrete)`` at one fixed value of the labels. Placed
                               # here for the same reason ``betas`` is --- a potential needs it,
                               # and it must not be closed over, or a jitted reseed would bake in
                               # the first call's labels (see ``BaseHMC._reseed_caches``).
    kinetic_cache: Any = None  # ``{kinetic id: whatever its ``precompute`` returned}``, filled
                               # once per kernel call by ``BaseHMC.context``. For quantities that
                               # depend only on ``ham_params`` and so are constant for a whole
                               # trajectory: XLA common-subexpression-eliminates repeats *within*
                               # one loop-body iteration but does not hoist them *out* of the
                               # trajectory ``while_loop``, so a per-leaf recomputation stays a
                               # per-leaf recomputation unless it is lifted to here.
