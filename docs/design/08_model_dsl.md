# mimcs: Model DSL

A domain-specific language for specifying models, compiling to the existing
[`Model`](05_model_interface.md). Superficially it resembles Stan, but it is *imperative*:
a program builds up JAX expressions that compute the log-density components — it is not a
BUGS-style declarative DAG. This document specifies the language and its compiler. For a
user-facing reference of the *implemented* language, see
[`docs/reference/model_dsl.md`](../reference/model_dsl.md).

## Goals and staging

Two goals shape the design, and pull in slightly different directions:

1. **Near-term (us): test problems.** Run simple Stan programs — with small, deliberate
   modifications — as test cases while we keep building samplers and adaptation strategies.
   We need a *richer, more representative supply of models* than the hand-written ones in
   `mimcs/testing/problems.py`.
2. **Eventual (others): a familiar interface.** When the project matures, let external users
   specify models through something that feels familiar to Stan users.

These call for **stages**: a stage-1 *skeleton* (this document's main subject), built so its
choices do not paint stage 2 into a corner. Stage-2 features are flagged **[stage 2]**
throughout and gathered in [Staging plan](#staging-plan).

The stage-1 milestone is concrete: **express the seven models in `mimcs/testing/problems.py`**
(`correlated_gaussian`, `rosenbrock`, `neal_funnel`, `neal_funnel_blocks`,
`positive_lognormal`, `uniform_interval`, `nested_uniform`). That set fixes the exact feature
subset and the verification target.

## What the compiler emits (the backend contract)

The compiler's only job is to build a `Model(parameters, log_prob_fns)`:

- `parameters: list[BaseParameter]` — `EuclideanParameter` / `BoundedParameter` (and the
  `PositiveParameter` / `IntervalParameter` factories). Bounds may be a constant, **another
  parameter's name-string** (a parent-dependent bound, e.g. `BoundedParameter("b", upper="a")`),
  or a callable of parent values.
- `log_prob_fns: dict[str, Callable[[dict], Array]]` — each closure receives
  `{param_name: ambient_jax_array}` and returns a **scalar**; their sum is the joint
  log-density.

Two facts about this target drive the whole design:

- **The model is written in *ambient* (constrained) space; the chart Jacobian is automatic.**
  `Model.log_prob_at_coordinate` pulls the coordinate back through each parameter's chart and
  adds `log_jacobian_det`. The DSL author writes the density of the constrained quantity and
  never writes a Jacobian — exactly Stan's convention.
- **Each `log_prob_fn` is traced by JAX exactly once.** So the *cost of how we evaluate the
  DSL body* is paid once and amortized to nothing across millions of sampler steps. This is
  why the compiler is a tree-walking interpreter (below), not a source generator.

## A first taste

Three of the seven target models, to make the surface concrete. (`d`, `nx`, `scale`, etc.
are `data` — known at model-construction time.)

```stan
// correlated_gaussian
data { array[d] real mu; array[d, d] real precision; }
parameters { array[d] real x; }
model {
  target += -0.5 * (x - mu) * (precision * (x - mu));   // '*' is matrix multiply
}
```

```stan
// neal_funnel (as separate blocks, cf. neal_funnel_blocks)
data { int nx; real scale; }
parameters { real v; array[nx] real x; }
model {
  v ~ normal(0, scale);
  x ~ normal(0, exp(v / 2));        // x_i | v ~ N(0, exp(v)); '~' sums over x
}
```

```stan
// nested_uniform: a parent-dependent bound
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=a> b;          // upper bound is the parameter 'a'
}
model { target += -log(a); }         // density of Uniform(0, a) is 1/a
```

The last shows the one genuinely non-trivial mapping: `<upper=a>`, where `a` is another
parameter, lowers to `BoundedParameter("b", lower=0.0, upper="a")` — the backend resolves the
name-string to a parent, orders evaluation topologically, and the `log a` from `-log(a)`
cancels `b`'s chart Jacobian, leaving independent standard-logistic coordinates.

## Surface language

### Blocks

The Stan block structure, with the same meanings:

| Block | Meaning | Status |
|---|---|---|
| `functions` | user-defined functions | ✅ (pure, no overloading; see below) |
| `data` | external inputs (constants/arrays), bound at construction | ✅ |
| `transformed data` | deterministic computations on data, run once | ✅ |
| `parameters` | parameter declarations → `BaseParameter`s | ✅ |
| `transformed parameters` | deterministic functions of parameters (and data) | ✅ |
| `model` | accumulate the log-density into `target` | ✅ (one or more; named) |
| `generated quantities` | post-sampling deterministic outputs | parse-accept, then ignore |

Statements end with `;`; `{}` delimits blocks; whitespace (outside tokens) is insignificant;
`//` and `/* */` are comments. `=` is assignment, `~` the distributional statement.

### Number types and arrays — and a deliberate omission

Two scalar number types: **`int`** and **`real`** (`complex` is **[stage 2]**). In `data`,
`transformed data` and function signatures the `int`/`real` distinction is load-bearing for exactly
three things — array sizes, loop bounds, and integer indices — and is otherwise kept light;
everything else leans on JAX at trace time.

In the **`parameters`** block `int` means more: `int<lower=L, upper=U>` declares a genuinely
**discrete parameter**, moved by a Metropolis-within-Gibbs sweep rather than by HMC
(`14_discrete_parameters.md`). Both bounds are required and must be constant. This is the one
place where a type keyword's meaning depends on its block, and it is worth knowing that it did not
always: `int` was a registry alias for `real`, so a declared `int` parameter silently compiled to
a continuous `BoundedParameter` on a logit link — it parsed, it ran, and it was not what anyone
writing it meant.

Shaped reals are **plain arrays** declared with `array[...]`:

```stan
array[d] real x;        // shape (d,)
array[m, n] real A;     // shape (m, n)
```

These are ordinary JAX arrays with **NumPy/JAX broadcasting**. We deliberately **do not**
provide Stan's `vector`, `row_vector`, or `matrix` types in stage 1.

> **⚠️ Care needed when adapting Stan programs.** This omission is intentional and is the main
> place a Stan program needs editing (`vector[d] x;` → `array[d] real x;`). The reason is
> *semantics in the `model` block*: Stan's `vector`/`row_vector`/`matrix` carry column/row
> orientation and matrix-specific operator behavior that differ from NumPy broadcasting in
> subtle ways. Silently accepting those keywords while giving them array semantics would
> invite hard-to-spot numerical discrepancies in exactly the test cases we are trying to
> trust. Forcing explicit `array[...]` declarations makes the intended shapes and broadcasting
> unambiguous. **[stage 2]** We may reintroduce `vector`/`matrix` as *syntactic sugar* for
> plain arrays — a clean way to *signal intent* about how a parameter is to be thought of
> (e.g. a covariance "matrix" vs a generic 2-D array) — but only with semantics chosen to
> coincide with the array semantics, never to diverge from them.

### Operators

| Operator | Meaning |
|---|---|
| `+ - * / ^` | arithmetic; `^` is right-associative |
| `*` | **matrix multiply** when both operands are ≥1-D (so `precision * (x - mu)` is a mat-vec, and a 1-D `·` 1-D is a dot); **scalar multiply** when either operand is a scalar |
| `.* ./ .^ .+ .-` | **elementwise**, with NumPy/JAX broadcasting |
| `'` (postfix) | transpose (a no-op on 1-D, since there is no row/column distinction) |
| `()` | grouping; `[]` | indexing / slicing |

The `*`-is-matmul / `.*`-is-elementwise split follows Stan, and is what lets
`correlated_gaussian`'s quadratic form be written without a distribution call.

### Indexing: 1-based and inclusive

Following Stan, indexing starts at **1** and ranges are **inclusive**; we otherwise stay
close to what JAX indexing supports. The compiler *lowers* to 0-based, half-open JAX indexing
in a single, exhaustively tested pass (it is the classic off-by-one trap):

| Stan | JAX |
|---|---|
| `a[i]` | `a[i - 1]` |
| `a[lo:hi]` (both inclusive) | `a[lo - 1 : hi]` (note the asymmetry) |
| `a[:hi]`, `a[lo:]`, `a[:]` | `a[0:hi]`, `a[lo-1:]`, `a[:]` |

Negative and stepped indices are not Stan and are rejected in stage 1. Stage-1 slice bounds
must be static; dynamic slices (`lax.dynamic_slice`) are **[stage 2]**.

## Compilation pipeline

```
source → lexer → parser → AST → semantic pass → closure builder → Model
```

### Why a tree-walking interpreter, not code generation

The closure builder is a **tree-walking interpreter**: an AST visitor that returns thunks
`(env) -> jax_value`. The `model` block becomes one closure that seeds an environment from
the ambient `params` dict, re-runs `transformed parameters`, accumulates a scalar `target`,
and returns it.

We do **not** generate Python source. Because JAX traces each `log_prob_fn` exactly once, the
interpreter's per-node cost is paid once and the resulting jaxpr is what runs hot — codegen's
only real advantage (trace speed) is irrelevant here. The interpreter is far simpler (no
`exec`, no generated-name hygiene), captures host `data` values trivially (they just live in
the environment), and reports shape/type errors *with the offending AST node's source span*
in hand. The semantic layer is kept as a clean IR so a codegen backend could attach later if
ever wanted.

### Hand-written parser

A hand-written **recursive-descent** parser for statements with a **Pratt
(precedence-climbing)** sub-parser for expressions, over a hand-written lexer producing
`Token(kind, text, span)`. This adds **no dependencies** (the library is pure JAX/NumPy),
gives us full control over **error messages** (`expected ';' after statement, found 'x' at
12:5`, with a source caret) and recovery — which matters for the eventual external users —
and the block-structured, `;`-terminated grammar is a natural fit for recursive descent.
(Rejected: PLY — poor errors, effectively unmaintained; Lark — a dependency, and one learns
the framework rather than parsing.)

### AST

Frozen dataclasses, every node carrying a `SourceSpan` for diagnostics:

- **Program / blocks**: `Program(blocks)`, `Block(kind, body, name)` (`name` is the component of
  a `model <name>` block).
- **Declarations**: `VarDecl(base_type, shape, name, lower, upper, init, base_args)`,
  `TypeExpr(base, dims, base_args)`. Bounds are *expressions*. `TypeExpr.dims` is three-valued —
  `()` scalar, a tuple of `Expr` sized, a tuple of `None`s ranked-but-unsized (`array[] real`),
  `None` for a bare `array real` — because a declaration knows its sizes and a function
  signature does not.
- **Functions**: `FuncDef(name, return_type, params, body)`, `Param(type, name)`.
- **Statements**: `Assign`, `Sample(lhs, dist, args)`, `TargetPlus(value)`, `Return(value)`,
  `For(var, lo, hi, body, unroll)`, `While`, `If`, `ExprStmt`, `BlockStmt`.
- **Expressions**: `IntLit`, `RealLit`, `Name`, `BinOp(op, lhs, rhs, elementwise)`, `UnaryOp`,
  `Call(fn, args)`, `Index(base, indices)` with `IndexArg ∈ {ScalarIndex, Range}`.

### Semantic pass

Name resolution and scoping; minimal `int`/`real` typing (enough to validate indices, loop
bounds, and array sizes — sizes must be build-time constants, evaluated by a small constant
evaluator); and the 1-based→0-based index lowering. Full shape inference is **[stage 2]**;
intermediate shapes are left `Unknown` and JAX reports mismatches at trace time, attributed to
the source span.

## Block → Model mapping

### `parameters` → `BaseParameter`

| Declaration | Emitted |
|---|---|
| `real x;` / `array[d] real x;` | `EuclideanParameter("x", shape)` |
| `real<lower=0> s;` | `PositiveParameter("s")` (= `BoundedParameter(lower=0.0)`) |
| `real<lower=L, upper=U> x;` (constants) | `BoundedParameter("x", lower=L, upper=U)` |
| `real<lower=0, upper=a> b;` (`a` a parameter) | `BoundedParameter("b", lower=0.0, upper="a")` |

**Bound lowering.** A constraint expression that is a numeric literal → a Python `float`; a
bare name resolving to *another parameter* → the **name-string** (drives the backend's
`parents` and topological order); a name resolving to data/constant → its value; anything more
complex → a stage-1 error. **[stage 2]** general bound expressions lower to a **callable**
`lambda parents: ...` — already supported by `BoundedParameter`, so no backend change.

`shape` comes from the declaration: `real` → `()`, `array[d] real` → `(d,)`, `array[m,n] real`
→ `(m, n)`, with `d, m, n` static constants.

### `data` / `transformed data` → a `build(data)` factory

`compile_model(source)` returns a **`ModelFactory`**; `factory.build(data) -> Model` binds the
data and emits the `Model`. `data` declarations name the keys `build` expects (validated and
`jnp.asarray`-coerced); `transformed data` runs once, eagerly, inside `build`, adding constants
to the closure environment. The convenience `compile_model(source, data=...)` does both. The
seven stage-1 models have no `data`, so they exercise `build(data=None)` — but the factory seam
is exactly what stage-2 data-driven models need, at no stage-1 cost.

### `transformed parameters`

Statements that depend on parameters cannot be precomputed; they are **prepended to the
`model` closure** and recomputed on each trace from the raw `params`. (Any `~` / `target +=`
appearing here also contributes to the density, per Stan.)

With **several** components the statements are prepended to *every* component closure — that is
how each component sees the transformed values, and it is forced: the closures are independent
`(dict) -> scalar` functions with no channel to pass a value between them, so shared work cannot
be shared. A density statement in `transformed parameters` is therefore **an error once a
program has more than one component**: it would otherwise be counted once per component, and
attributing it to one of them would be an arbitrary, invisible choice. The fix is always local —
move it into the component it belongs to. With one component nothing changes, so the Stan
behaviour above is preserved exactly.

### `model` → the `target` closure

The `model` block compiles to one `log_prob_fn`. It accumulates a scalar `target`, threaded
through the interpreter as a host-side cell (Python mutation during the trace; the value
flowing through it is a JAX scalar):

- **`target += expr`** → `target ← target + eval(expr)`.
- **`lhs ~ dist(args)`** → `target ← target + sum(dist_logpdf(eval(lhs), *eval(args)))`. The
  `sum` is what gives `~` its vectorized meaning: `x ~ normal(0, s)` over a vector `x` adds the
  elementwise log-densities (Stan semantics). For a scalar `lhs` the sum is trivial; for a
  joint distribution like `multi_normal` whose logpdf is already scalar, it is a no-op.
- An empty `model` block returns `jnp.zeros(())` (e.g. `uniform_interval`: the density is just
  the chart Jacobian, which the backend adds).

### Multiple components: named `model` blocks

The backend's `log_prob_fns` is a *dict* — it decomposes the joint density into named components
(e.g. `prior`, `likelihood`). Stan has no syntax for that, so the DSL **appends the component
name to the `model` keyword**:

```stan
model prior      { v ~ normal(0, 3); }
model likelihood { y ~ normal(v, sigma); }
```

Each `model <name>` block compiles to its own closure keyed by `<name>`; a bare `model { }` is
the component `"target"`, and the two forms may be mixed (so peeling a `likelihood` out of an
existing bare `model` is a legal intermediate state). Names must be unique — `model target { }`
is a synonym for a bare block and collides with one — and `jacobian` is rejected, because
`ModelPotential` keys its gradient cache by `V_<name>` and would collide with
`JacobianPotential`'s `V_jacobian`. Since `Model.log_prob` sums the components, **splitting a
model is a no-op on its density and gradient**; that invariant is what the tests pin.

The point is that the *computations forming each component are kept separate* — so an integrator
can kick with a cheap `prior` potential more often than with an expensive `likelihood` potential
(a multi-rate / RESPA splitting; cf. `RepeatedIntegrator` in
[doc 06](06_hamiltonian_monte_carlo.md)). Designing those samplers is still future work; the DSL
only preserves the separation.

**Until then, splitting costs.** Component order is source order, and `default_potentials`
(doc 06) makes one `ModelPotential` per component, each of which independently unpacks the
coordinate through every chart *and* re-runs the prepended `transformed parameters` statements.
Each is a separate `value_and_grad` trace, so XLA cannot recover the shared work. A split model
therefore reports ~N× the gradient evaluations of the fused one for identical dynamics —
`grad_evals` comparisons between a split and a fused version of the same model are not
apples-to-apples.

## The model spec: which component is cheap, and the chart options

Compiling against a dataset is two steps, mirroring the sampler factory (doc 09) closely enough
that the two read as siblings:

```
*results ──▶ Evidence ──▶ analyze ──▶ SamplerSpec ──▶ .build() ──▶ sampler
data ─────────────────▶ analyze ──▶ ModelSpec   ──▶ .build() ──▶ Model
```

`factory.analyze(data)` binds the data, runs `transformed data`, sizes the resulting constants
and labels every component; the returned `ModelSpec` is a mutable prototype
(`ComponentSpec` / `ParameterSpec`, the counterparts of `BlockSpec`) that the user can override
before `spec.build()` lowers it. `factory.build(data)` is exactly `analyze(data).build()`, so
`compile_model(source, data=...)` is unchanged.

It carries two things the grammar has no syntax for.

### Component cost

A multi-rate integrator needs to know which potential is cheap. For a DSL program the question
has a structural answer: **a component is expensive exactly when it touches a large constant or a
large parameter**, because size is the proxy for gradient cost. "Large" is decided by pooling
every measurable constant *and* every parameter's **ambient** dimension (not coordinate dimension
— they differ for a manifold parameter) into one comparison — the largest item is always large, a
one-element item never is, and anything at least `LARGE_FRACTION` (1/500, a **placeholder** with
no further evidence behind it beyond the case it was tuned to catch) of the largest is. The
result reaches the backend as `Model.cheap_components`; everything unnamed is expensive, and an
unlabelled model (the hand-written case) has nothing cheap, so a future multi-rate builder finds
nothing to nest and falls back to plain leapfrog.

The rule **does** count parameters, deliberately (a revision from an earlier constants-only
version) — a prior over a handful of *large* parameters (a hierarchical model's vector-valued
local shrinkage terms, say) is exactly the case multi-rate should catch, and ignoring parameter
size entirely missed it. The threshold is looser than a constants-only version needed (1/500, not
1/10) precisely so a parameter a couple of orders of magnitude smaller than the data still counts.
Two consequences worth stating:

- a program with no data or vector parameter, or with only scalars, has **every** component
  cheap — right, since there is nothing for a split to buy anything on;
- a component's statements are `transformed parameters + <its block>`, so a large constant or
  parameter read in `transformed parameters` makes **every** component expensive. Not a quirk:
  every closure really does re-execute those statements.

**A known limitation, measured twice, and one loosened by the parameter revision.** Comparing
only against the *largest* item can still hide a genuinely per-observation array, but the looser
1/500 threshold raises the bar this needs to clear: on the real `diamonds` program (`X` 125000,
`Xc` 120000, `Y` 5000) `Y` at 4% of `X` easily clears 1/500 (0.2%) and is now correctly called
large, where the earlier all-constants version's 1/10 (10%) threshold hid it. It still takes only
a wider design matrix to reproduce the failure — a component reading only a response vector `y`
would be called cheap once the design matrix has a few hundred more columns than rows. And size is
still only a proxy for *cost*, not cost itself, which is exactly what this parameter-pooling
revision was for: on `reg_horseshoe` (n=62, d=2000) the pre-revision rule called only `likelihood`
expensive (it touches the 124000-element `x`) and `prior` cheap, although a directly measured
gradient shows the "cheap" prior actually costs *more* per call than the likelihood — one runs
elementwise transcendentals over a 2000-vector, the other is a single optimized 62×2000 matmul.
With parameters pooled in, `beta`/`lambda` (2000-dim each) now also clear the bar, so **both**
components come out expensive, `model.cheap_components` is empty, and multi-rate correctly stays
off this model rather than firing on the backwards split it used to see
(`tests/experiments/writeups/reg_horseshoe_learned_metric.md`, the two-stage studies). Both
remaining issues above are harmless while such programs are single-component and the classifier is
swappable:
`analyze(data, classify=...)` replaces the rule wholesale, with (say) one that times a gradient.
The `Proposal`/`arbitrate` machinery of doc 09 is deliberately *not* reused — it exists to resolve
conflicts between rules competing for a slot, and with a single rule the weighted arbitration is
the identity; importing it would also point the DSL at the sampler heuristics that consume its
output. The day a second, disagreeing cost rule appears is the day to lift the arbiter into a
shared module.

### Chart options

`centered` (Euclidean and bounded parameters) and `adaptive` (unit vectors) are per-parameter
switches that only an adaptation mixin reads, and that the declaration grammar cannot express.
`spec.parameter("beta").centered = True` supplies them; `analyze` leaves every option unset, so
this stage is an interface, not a heuristic. Note that centering a parameter makes its chart
non-Euclidean, which adds the `JacobianPotential` to the model's potentials — itself a cheap
component by construction, since it never touches data.

## Distribution registry

`distributions.py` maps a Stan distribution name to `(logpdf, arity, param_adapter)` over
`jax.scipy.stats`. Adapters are necessary because Stan and SciPy parameterizations differ in
two recurring ways — **which moment** (sd vs variance vs precision) and **rate vs scale**:

| Stan | Backend |
|---|---|
| `normal(mu, sigma)` | `norm.logpdf(x, loc=mu, scale=sigma)` — Stan uses sd = SciPy `scale` |
| `multi_normal(mu, Sigma)` | `multivariate_normal.logpdf(x, mu, Sigma)` — covariance (scalar logpdf) |
| `lognormal(mu, sigma)` | `norm.logpdf(log x, mu, sigma) - log x` |
| `uniform(L, U)` | `uniform.logpdf(x, loc=L, scale=U - L)` |
| `exponential(beta)` | rate `beta` → SciPy `scale = 1/beta` |
| `gamma(alpha, beta)` | rate `beta` → `scale = 1/beta` |
| `cauchy`, `beta`, `student_t` | scale / direct adapters |

The registry is the single source of truth for parameterization and is unit-tested against
known logpdf values. The stage-1 set covers the targets (`normal`, `multi_normal`, `lognormal`,
`uniform`) plus a few extras so it feels real. **[stage 2]**: `multi_normal_prec`, a fuller
library, truncation/`T[,]`.

## Loops and builtins

`for` with **static** bounds **unrolls**: the simplest graph, and right for a three-term sum.
It is hopeless for a thousand-step recursion, though, because the jaxpr then grows with the
loop and so do compile time and memory. For those there are two **higher-order builtins**,
`scan` and `fori_loop` (`loops.py`), which stay a *single* jaxpr equation however long they
run. Measured on an AR(1) recursion, going from 3 to 60 steps: `scan` holds at 5 equations
while the equivalent `for` goes from 32 to 302.

Both keep their **length a compile-time constant**, and that is not a limitation to work
around — it is exactly why they are differentiable. `jax.lax.fori_loop` with static bounds
lowers to a `scan` (reverse-mode differentiable); with *traced* bounds it lowers to a
`while_loop`, which has no reverse-mode rule and so could never carry a model's gradient. We
make that relationship explicit rather than depend on it: **`fori_loop` is implemented as a
`scan`**, so the differentiability is true by construction. A non-constant bound is a
span-carrying error, and so is a fractional one — `as_static_int` is an `int()` call, which
truncates, and a bound quietly off by one is the hardest kind of bug to see in a result.

Three deliberate departures from the JAX originals:

* **`fori_loop`'s range is inclusive**, `[lower, upper]`, matching this language's
  `for (i in 1:n)` and its 1-based indexing — *not* JAX's half-open `[lower, upper)`. The same
  call therefore runs one more iteration here than in JAX. An index that indexes correctly beats
  one that ports silently, but it is a trap worth flagging wherever it appears.
* **Extra arguments are forwarded** to every body call: `scan(f, init, xs, A, k)` calls
  `f(carry, x, A, k)`. A DSL function sees only its arguments, so it cannot capture data the way
  a JAX closure does; forwarding is this language's stand-in, and it costs nothing at trace time.
* **The body must be a user-defined function**, named directly in the call. There are no
  first-class functions: the parser rewrites that one argument slot into an `ast.FuncRef`, which
  is why a function's *name* never has to be resolvable as a value.

`scan` otherwise keeps JAX's signature exactly — the body returns `(carry, y)` and the form
returns `(carry, ys)` — which is what the tuples below are for. Since `init` and `xs` are
pytrees, a tuple `init` carries several values and a tuple `xs` scans several arrays in step,
both for free.

### `None`

`None` is Python's, and earns its place twice for the same reason it does there: `jnp.newaxis`
*is* `None`, and `None` is an empty JAX pytree. So one word covers a reshape (`a[:, None]`, the
NumPy way to line arrays up for broadcasting) and an absent `scan` carry or input. It is an
ordinary expression, valid wherever a value is — which is what lets a body write
`return (None, y);` for an empty carry with no special case anywhere.

The one wrinkle is that `lax.scan` cannot infer a length from an absent array, so with
`xs = None` **the length is the next argument**: `scan(f, init, None, n)`, extras after it. That
rule is declared on the `LoopForm` (`length_after`) rather than special-cased by name, so the
parser, the static check and the interpreter all read it from one place. Because the `None` is a
*literal*, the static check can see it: a missing length is a compile-time error, and the arity
message says explicitly that the argument after `None` is the length rather than an extra —
which is the mistake that rule invites.

`None` is also a declared type, in a parameter list or a tuple type, and its name may be omitted
(`(None, real) f(None, real x)`) since there is nothing to refer to. As with every type here it
is *recorded and never checked*. It is not a parameter type: a `None` parameter would have no
coordinates and no density, so `parameters { None x; }` is rejected explicitly rather than
falling through to a `KeyError`.

### Tuples, and how little of them there is

Tuples exist to make `scan`'s signature expressible, and go no further than that: a tuple
literal `(a, b)`, a tuple type for a function's argument or return, and **destructuring**
declarations, which nest (`((real a, real b), array[n] real ys) = scan(...)`). There is no
`t.1` accessor — the lexer has no `.` token, and destructuring covers the need — and there are
no tuple-typed locals or model parameters.

Dropping tuple-typed locals is what keeps the grammar unambiguous: inside a target list a `(`
can then only open a nested group, never a tuple type, so `(real a, real b) = ...` needs no
backtracking to tell it from `(real, real) t = ...`. As everywhere else in this DSL, the
declared element types are *recorded and never checked*.

`builtins.py` is a curated table over `jax.numpy` (`exp`, `log`, `sqrt`, `abs`, `sum`, `dot`,
`solve`, `cholesky`, `slogdet`, ...), and it uses **JAX/NumPy names, not Stan's** — `det` rather
than `determinant`, `solve` rather than `mdivide_left`. The choice is deliberate: the translation
a user of this library actually performs is between a DSL model and the same model written as a
plain JAX function over a `Model`, and keeping one vocabulary for both makes that a transcription.
A Stan program already needs rewriting for the array semantics, so little is lost by not matching
it here either.

Two consequences fall out of the DSL's own grammar rather than from JAX. `slogdet` returns a pair,
which is reachable only through destructuring (`(real s, real ld) = slogdet(A);`) since there is no
tuple indexing. And `solve_triangular`'s `lower` flag has to be a third *positional* argument,
because there are no keyword arguments; it is passed through `bool()`, so a traced value raises
instead of silently reading the wrong triangle.

## User-defined functions

Stan's C-like declarations, with full input and output types:

```stan
functions {
  array real add_two(array real a, array real b) { return a + b; }
  real scale(real x, real k) { return x * k; }
}
```

Signature types are written without sizes — `array real x` or Stan's `array[] real x`, both
accepted — since a signature does not know them. The types are **recorded, never checked**:
there is no type checker, and JAX reports a real mismatch at trace time against the source span.
What *is* checked, statically, at `compile_model(source)`:

- **Purity.** A body may not contain `~` or `target +=`. Stan's escape hatch is the `_lp` name
  suffix, which we do not implement, so functions are values of their arguments and nothing else.
  This also makes `void` pointless — arguments are by value and JAX arrays are immutable, so a
  void function could have no observable effect at all — and it is rejected with that message
  rather than a parse error.
- **Scope.** A body sees **only its arguments and its own locals** (Stan's rule). It cannot read
  data, parameters or transformed parameters; the call frame is built from the arguments alone,
  so this is structural, not merely checked.
- **No overloading.** One definition per name, and the name may not be a keyword, a builtin or a
  distribution. Function names must be reserved because `Call.fn` and the builtin table share one
  lookup path; ordinary *variable* names need not be, since `Name.id` and `Sample.dist` are
  separate namespaces — a variable called `mean` or `beta` stays legal.
- **Arity**, at every call site (builtins are exempt: `BUILTINS` records no arity).

**Recursion** is allowed — the table is complete before evaluation, so a function may call one
defined below it, and two may call each other. Two consequences follow from tracing: the base
case must be decidable at trace time (`if` needs a compile-time-constant condition, `for`
unrolls), and the recursion is *inlined* into the jaxpr. A depth cap (`MAX_CALL_DEPTH = 64`)
turns runaway recursion into a span-carrying `DslError` instead of a `RecursionError` thrown
from inside the interpreter, where the compiler's error funnel would never see it.

A `return` inside an unrolled `for` stops the unrolling at that iteration — exactly what a
runtime `return` would do. Functions are *not* usable in array sizes or bounds: `const_eval` has
no `Call` case (builtins have the same gap).

## Module layout and public API

```
mimcs/dsl/
  tokens.py        # Token, TokenKind, SourceSpan
  lexer.py         # str -> tokens
  ast.py           # frozen-dataclass node hierarchy
  parser.py        # recursive-descent + Pratt -> Program
  errors.py        # DslError: span-aware message with a source caret
  semantics.py     # name resolution, static checks, const-size eval, index lowering
  interpreter.py   # closure builder: AST -> (env)->jax_value; one closure per component
  distributions.py # the registry
  builtins.py      # jnp op tables
  cost.py          # which components are cheap: the data-size rule
  spec.py          # ModelSpec / ComponentSpec / ParameterSpec: the mutable prototype
  factory.py       # ModelFactory: analyze(data) -> ModelSpec -> build() -> Model
  __init__.py      # public API
```

Public API (exported from `mimcs/__init__.py` alongside `Model`):

```python
from mimcs import compile_model
model = compile_model(source, data=None)          # one-shot -> Model
factory = compile_model(source); model = factory.build(data)   # reusable
```

Errors raise `DslError` with `line:col` and a source caret.

## Staging plan

**Stage 1 (skeleton).** Blocks `data`, `transformed data`, `parameters`, `transformed
parameters`, `model` (parse-accept and reject/ignore `functions` / `generated quantities`).
Types `int`, `real`, `array[...] real`. `EuclideanParameter` / `BoundedParameter` with constant
or single-parameter-name bounds. Statements `=`, `~`, `target +=`, `for` (unrolled +
`fori_loop`), `while`, `if`. Expressions: literals, names, `+ - * / ^`, elementwise `.`-ops,
matmul, calls, indexing/inclusive-range slicing (static), grouping, transpose. Distributions
`normal`, `multi_normal`, `lognormal`, `uniform` (+ extras). `compile_model` / `ModelFactory`;
span-aware `DslError`. Reproduce the seven `problems.py` models.

**Stage 2 (landed).** User-defined `functions` (pure, no overloading, recursion capped) and
multiple named `model` components — the two together, since both are about building closures:
one call table shared by every component's closure.

**Stage 3 (landed).** The `ModelSpec` prototype: `factory.analyze(data)` labels each component
cheap or expensive by the data-size rule and exposes the per-parameter chart options
(`centered`, `adaptive`) the grammar cannot express; `Model.cheap_components` carries the
labels to the backend.

**Stage 4 (landed).** The non-unrolling loops `scan` and `fori_loop` as higher-order builtins,
and the minimal tuples that let `scan` keep JAX's signature (see "Loops and builtins").

**Still deferred.** `generated quantities`; tuple *locals*, tuple parameters and `t.1` element
access; first-class functions beyond a loop form's body slot; `complex`;
`vector`/`matrix` as array sugar; general-expression bounds (→ callable lowering);
`multi_normal_prec` and a fuller distribution library; `dynamic_slice`.

**Forward-compatibility seams chosen now:** the `build(data)` factory; the callable-bound
escape hatch (already a backend feature); the `log_prob_fns` dict (component splitting); the
semantic-IR boundary (a codegen backend could attach); recording array base types so sugar can
be added additively.

## Verification

- **Unit:** the index-lowering function (exhaustive 1-based cases); distribution adapters vs
  known logpdf values; the constant-size evaluator; lexer/parser golden ASTs; `DslError`
  formatting.
- **Density equivalence (the core check):** for each of the seven problems, `compile_model` the
  DSL source and compare the compiled `Model.log_prob_at_coordinate` against the hand-written
  `problems.py` model at many random coordinates. Compare the **gradient exactly** (HMC
  correctness; additive constants drop out), and the **density up to an additive constant**
  (because `~` uses normalized SciPy logpdfs while some hand-written problems drop the
  normalizing constant — irrelevant for sampling). When the DSL source is written with explicit
  `target += <unnormalized>`, the density matches exactly. Comparing the *coordinate-space*
  density specifically exercises the chart Jacobian, including `nested_uniform`'s
  parent-dependent cancellation.
- **End-to-end:** feed a DSL-compiled model into `mimcs.testing.evaluate(...)` and
  `report.assert_correct()` on `correlated_gaussian` and `rosenbrock`, proving the emitted
  `Model` is sampler-compatible (pack/unpack, charts, gradient).

## Open questions for later stages

- **`vector`/`matrix` as intent-signaling sugar** — if reintroduced, with semantics chosen to
  coincide with arrays (the care flagged above).
- **Component-aware samplers** — multi-rate integrators that exploit named `model` components.
  The DSL now emits the separation *and* the cheap/expensive labels
  (`Model.cheap_components`); `RepeatedIntegrator` (doc 06) and the integrator-selection rule in
  the sampler factory have since shipped.
- **A cost rule with evidence behind it** — the 0.1 fraction is a placeholder, and comparing
  against the largest constant alone mislabels a narrow response beside a wide design matrix
  (the `diamonds` numbers above).
- **Builtin arity metadata** — `BUILTINS` records bare `jnp` callables, so only *user* function
  calls can be arity-checked; a wrong builtin call still surfaces as a `TypeError` from JAX.
- **`const_eval` with a `Call` case** — so a function (or builtin) could size an array.
- **Correlation / covariance matrix types.** (`simplex` and `ordered` have since shipped as DSL keywords — see `model/registry.py`'s `PARAMETER_KINDS`.)
  (these map to the manifold-parameter machinery of [doc 04](04_manifold_parameters.md), not
  to bounds).
- **Error recovery / multiple diagnostics per compile** — accumulate errors rather than failing
  on the first, for a better external-user experience.
