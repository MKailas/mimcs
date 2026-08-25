# mimcs

General-purpose Bayesian inference with adaptive MCMC, built on JAX.

```python
import mimcs

model = mimcs.compile_model(source, data)   # a Stan-like DSL, or build a Model by hand
sampler = mimcs.make_sampler(model)         # picks the algorithm, blocks and adaptations
sampler.warmup(2000)                       # adapts; an upper bound, not a count
draws = sampler.sample(4000)               # {parameter name: draws}
print(sampler.summary())                   # per-feature ESS, split-R-hat, Stein z
```

## Where to start

**Running something** — the `examples/` directory in the repository holds four short end-to-end
scripts, in reading order; `examples/01_quickstart.py` is a complete session in twenty lines of
API.

**Using the library** — the three reference manuals:

```{toctree}
:maxdepth: 1
:caption: Reference

reference/model_dsl
reference/sampler_factory
reference/algo_kwargs
api/index
```

**Understanding it** — the design documents, in numbered order. These are architecture and
rationale: what was built, why, and what was rejected. Empirical studies that settled a decision
are summarised where they apply; the full research writeups live outside this tree.

```{toctree}
:maxdepth: 1
:caption: Design

design/00_overview
design/01_state_and_kernel
design/02_sampler_classes
design/03_rng_management
design/04_manifold_parameters
design/05_model_interface
design/06_hamiltonian_monte_carlo
design/07_riemannian_hmc
design/08_model_dsl
design/09_sampler_factory
design/10_warmup_termination
design/11_sample_evaluation
design/12_logging
design/13_parallel_tempering
```
