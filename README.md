# Modular Intelligent Markov Chain Samplers

General-purpose Bayesian inference with adaptive MCMC, built on JAX.

```python
import mimcs

model = mimcs.compile_model(source, data)   # a Stan-like DSL, or build a Model by hand
sampler = mimcs.make_sampler(model)         # picks the algorithm, blocks and adaptations
sampler.warmup(2000)                       # adapts; an upper bound, not a count
draws = sampler.sample(4000)               # {parameter name: draws}
print(sampler.summary())                   # per-feature ESS, split-R-hat, Stein z
```

What is here:

- **A model DSL** — a small Stan-like language, compiled to a differentiable JAX density. Bounded
  and positive scalars, unit vectors, simplexes and ordered vectors are types; you write the
  density in the parameters' natural values and never write a Jacobian.
- **A sampler family** — HMC, randomized HMC, NUTS, WALNUTS (within-orbit adaptive step size),
  explicit block-Riemannian HMC with learned metrics, and parallel tempering over any of them.
  Samplers are composed from a base algorithm and adaptation mixins, so a new adaptation is a new
  class rather than an edit to an existing one.
- **A factory** — `make_sampler(model)` chooses the algorithm, the coordinate blocking, the mass
  adaptation and the warmup-termination criterion, and records why. Feed a finished run back in and
  it re-decides against the evidence.
- **Sample evaluation** — per-feature ESS and split-R̂, plus a target-aware Langevin–Stein
  statistic that asks whether the draws come from *this* distribution rather than merely from a
  well-mixed one.

## Install

Python ≥ 3.10. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/MKailas/mimcs.git
cd mimcs
pip install -e .
```

That pulls JAX and NumPy from PyPI, which gives you a CPU build of JAX. If you want a GPU or TPU
build, or you get JAX from conda, install JAX into the environment yourself first — see
[JAX's install instructions](https://docs.jax.dev/en/latest/installation.html) — and then use

```bash
pip install -e . --no-deps
```

so that pip leaves the JAX you chose alone rather than shadowing it with a wheel.

Add the test dependencies with `pip install -e ".[test]"`.

The library runs in float32 by default and is float64-capable — set
`jax.config.update("jax_enable_x64", True)` *before* importing `mimcs`, which is worth doing for any
density that catastrophically cancels.

## Start here

**[`examples/`](examples/)** — four runnable end-to-end scripts, in reading order. Start with
`01_quickstart.py`; it is a complete session in twenty lines of API.

**Reference** — `docs/reference/model_dsl.md` (the language), `sampler_factory.md` (what the
factory decides and how to override it), `algo_kwargs.md` (every sampler option).

**Design** — `docs/design/`, numbered, starting at `00_overview.md`. Architecture and rationale:
what was built, why, and what was rejected.

Both trees are also a Sphinx site, with the API reference generated from the docstrings:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build
```

## Tests

```bash
python -m pytest tests/ -q
```

About 885 statistical MCMC tests, roughly an hour. Seeds are pinned throughout, so failures are
reproducible. Run a single file for fast feedback (`pytest tests/test_nuts.py -q`, 1–3 minutes).

## License

MIT — see [`LICENSE`](LICENSE).
