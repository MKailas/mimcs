"""The sampler factory: what it decided, how to override it, and how to feed a run back in.

``make_sampler(model)`` (example 01) is a shortcut for three steps you can take yourself:

    spec = analyze(model)      # a mutable SamplerSpec -- every decision, and the reason for it
    spec.terminate = None      # override anything
    sampler = spec.build(seed=0)

The spec is a prototype, not a config file: the factory's heuristics fill it in, `spec.rationale`
records which rule set each field and how strongly, and you are free to overrule any of it.

The second thing this shows is the loop the library is built around. Hand a *finished run* back to
the factory and it re-decides against that evidence -- the draws, the coordinates and the saved
gradients -- rather than against the model's shape alone.

The target is Neal's funnel: ``v ~ Normal(0, scale)`` and ``x_i | v ~ Normal(0, exp(v/2))``, whose
conditional variance for ``x`` is exactly ``exp(v)``. What the second round demonstrates is that
the factory *recovers that*: given the first run's draws and gradients it regresses a metric
expression on the score covariance and selects ``Exp('v') + Exp()`` -- a term varying as the
exponential of ``v``, which is the funnel's geometry, read off the samples. No constant mass matrix
can express it, and nothing about the model's *shape* reveals it.

**On what the numbers below do and do not show.** At this size and with a well-tuned step size the
two rounds come out about even (five seeds: 0.71, 1.01, 1.02, 1.18 and 1.37 times the round-1
min ESS, and the same metric selected in all five). A wide-necked funnel is one a constant mass
handles perfectly well, so a fitted metric has little to win back; it matters as the funnel
sharpens (``docs/design/07_riemannian_hmc.md``). An earlier draft of this example ran at the
default ``target_accept`` and showed the second round a flattering 2.7x ahead -- almost all of
which turned out to be a badly-tuned *first* round rather than anything the metric did. The
reproducible claim here is the decision, not a speedup.

Run it with the `jax` conda environment active:

    python examples/03_factory_and_evidence.py
"""

import numpy as np

from mimcs import compile_model, analyze

SOURCE = """
data { int nx; real scale; }
parameters { real v; array[nx] real x; }
model {
  v ~ normal(0, scale);
  x ~ normal(0, exp(v / 2));     // x_i | v ~ Normal(0, exp(v))
}
"""

NX, SCALE = 30, 1.0


def configure(spec):
    """The overrides both rounds share, so the two are a fair comparison."""
    # A decision, recorded on the spec: no warmup-termination criterion, so `warmup(n)` runs
    # exactly n iterations and both rounds get an identical budget. The default, "classifier",
    # would stop each round wherever it judged the chain to be mixing.
    spec.terminate = None
    # Anything in `algo_kwargs` is splatted into the sampler's constructor
    # (docs/reference/algo_kwargs.md lists all 85 options). Unknown keys are ignored, so check
    # spelling against that table rather than expecting an error.
    #
    # Raising `target_accept` is the standard response to a funnel: the step-size adaptation aims
    # for a higher acceptance rate, which means smaller steps, which is what it takes to integrate
    # the narrow neck without diverging. The factory's effective default is 0.8 -- note that this
    # is *not* the 0.234 the constructor's own default says, because the factory sets it.
    spec.algo_kwargs["target_accept"] = 0.9
    return spec


def run(spec, seed=0, warmup=1000, draws=1000):
    """Build the sampler this spec describes and run it. `build` takes the arguments that are
    about *this run* -- seed, starting point, RNG buffer size -- while everything else was
    decided on the spec."""
    sampler = spec.build(seed=seed)
    sampler.initialize()
    sampler.warmup(warmup)
    sampler.sample(draws)
    return sampler


def report(label, summary):
    print(f"  {label:<28} min ESS {np.min(summary.ess):>6.1f}   "
          f"max R-hat {np.max(summary.rhat):.3f}   "
          f"Stein flags {int(summary.stein_flagged.sum())}/{len(summary.feature_names)}")


def main():
    # A gentle funnel: `scale=1` keeps the neck wide enough that NUTS samples it healthily, so the
    # comparison below is between two working chains rather than two broken ones. Sharpen it
    # (`SCALE = 3`) and both rounds struggle at this budget -- and it becomes worth enabling float64
    # (`jax.config.update("jax_enable_x64", True)`, before importing mimcs), since a funnel's density
    # spans the range where float32 quietly corrupts the adaptation.
    model = compile_model(SOURCE, data={"nx": NX, "scale": SCALE})

    # --- 1. What did the factory decide, and why? ------------------------------------------
    spec = analyze(model)
    print("\n--- the spec, with no evidence ---")
    print(spec)
    print("\n--- the reasoning ---")
    print(*spec.rationale, sep="\n")

    # --- 2. Round one: overrule what you want, then build ----------------------------------
    print("\n--- round 1: no evidence ---")
    n_default_rules = len(spec.rationale)
    first = run(configure(spec))
    first_summary = first.summary()
    print()
    report("round 1 (model shape only)", first_summary)

    # --- 3. Round two: the same model, now with the first run as evidence ------------------
    #
    # `analyze(model, first)` accepts a live sampler, a raw array of draws, a testing
    # SamplerOutput, or a (samples, coordinates, gradients) tuple -- in any combination. The
    # evidence buys two things: the decisions below, and a warm start -- with no explicit `init`,
    # `build` starts the chain at the last evidence draw rather than at the charts' origin.
    informed = analyze(model, first)
    print("\n--- the spec, with 1000 draws of evidence ---")
    print(informed)
    print("\n--- what changed ---")
    # Everything past the rules that fired without evidence: the mass-mode choice per block, and
    # the metric regression. `Exp('v')` in a metric expression means a term varying as exp of the
    # parameter v -- which is what makes it a *learned* metric rather than a constant mass: the
    # funnel's conditional variance for x is exactly exp(v), and the regression is searching a
    # family of expressions for one that tracks it.
    # (The `.params` lines are skipped here: they carry the *fitted* metric arrays, which the
    # rationale truncates mid-array. `informed.blocks[1].params` has them in full.)
    print(*[line for line in informed.rationale[n_default_rules:] if ".params" not in line],
          sep="\n")

    print("\n--- round 2: the same budget, informed by round 1 ---")
    second = run(configure(informed))
    second_summary = second.summary()
    print()
    report("round 1 (model shape only)", first_summary)
    report("round 2 (fitted to evidence)", second_summary)

    # --- 4. Did it work? -------------------------------------------------------------------
    #
    # The funnel's marginals are known exactly: v ~ Normal(0, scale), and each x_i has mean 0 and
    # standard deviation exp(scale^2 / 4). The second moment of x is the demanding one -- it is
    # dominated by the rare excursions to large v, which is exactly what a chain that cannot move
    # along the funnel's neck will miss.
    print("\nRecovered vs truth (round 2)")
    draws = second.get_samples()
    v, x = np.asarray(draws["v"]), np.asarray(draws["x"])
    print(f"  {'':>10} {'sampled':>10} {'truth':>10}")
    print(f"  {'mean v':>10} {v.mean():>10.3f} {0.0:>10.3f}")
    print(f"  {'sd v':>10} {v.std(ddof=1):>10.3f} {SCALE:>10.3f}")
    print(f"  {'mean x':>10} {x.mean():>10.3f} {0.0:>10.3f}")
    print(f"  {'sd x':>10} {x.std(ddof=1):>10.3f} {np.exp(SCALE ** 2 / 4):>10.3f}")

    ess_ratio = np.min(second_summary.ess) / np.min(first_summary.ess)
    print(f"\n  round 2 / round 1 min ESS: {ess_ratio:.2f}x  "
          "(seed-dependent, and about even on a funnel this gentle -- the")
    print("  reproducible part is the metric the factory fitted, not the ratio)")


if __name__ == "__main__":
    main()
