# Examples

Four runnable scripts, in reading order. Each is self-contained — copy one and edit it — and each
goes end to end: a model, a sampler, draws, and an evaluation of those draws. None of them takes
more than a minute.

```bash
python examples/01_quickstart.py
```

| | Shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | A model in the DSL, a sampler from the factory, `initialize` / `warmup` / `sample` / `summary`. **Read this one first.** |
| [`02_model_by_hand.py`](02_model_by_hand.py) | The same model built directly from parameter objects and JAX functions, checked against the DSL version. Sample space vs coordinate space. |
| [`03_factory_and_evidence.py`](03_factory_and_evidence.py) | What the factory decided and why (`analyze`, `spec.rationale`), overriding it on a mutable `SamplerSpec`, and feeding a finished run back in as evidence. |
| [`04_sampler_by_hand.py`](04_sampler_by_hand.py) | Composing a sampler yourself with `make_sampler_class` — mixins, hook order, and what each one contributes. Advanced; the factory exists so you need not do this. |

The library logs its decisions at INFO while these run, which is deliberate: the log lines are part
of what the examples are showing (where warmup actually stopped, which sampler was built, what the
mass adaptation settled on). Quiet them with `mimcs.set_log_level("WARNING")`.

Reference material for what the examples use: `docs/reference/model_dsl.md` for the language,
`docs/reference/sampler_factory.md` for the factory's decisions, and
`docs/reference/algo_kwargs.md` for every sampler option.
