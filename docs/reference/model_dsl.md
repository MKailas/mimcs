# Model DSL — Reference Manual

A reference for the model DSL as currently implemented (stage 1). The DSL is a small,
Stan-like language for specifying a probabilistic model; a program is compiled to a
`mimcs.Model` that the samplers can target. For the design and rationale, see
`docs/design/08_model_dsl.md`; features marked **(not yet)** below are planned but not
implemented.

## Quick start

```python
from mimcs import compile_model

source = """
data { real sigma; }
parameters { real<lower=0> s; }
model { s ~ lognormal(0, sigma); }
"""

model = compile_model(source, data={"sigma": 1.0})    # -> mimcs.Model
```

`compile_model(source, data=...)` returns a `Model`. Called without `data`, it returns a
**`ModelFactory`** that you can reuse across datasets:

```python
factory = compile_model(source)
model_a = factory.build({"sigma": 1.0})
model_b = factory.build({"sigma": 2.0})
```

Compile-time problems raise `mimcs.DslError` with a `line:col` location and a source caret.

## Program structure

A program is a sequence of **blocks**, each `name { ... }`. The blocks and their meanings
follow Stan:

| Block | Meaning |
|---|---|
| `data` | external inputs (constants/arrays), supplied to `build(data)` |
| `transformed data` | deterministic computations on data, run once at build time |
| `parameters` | the model's parameters (see the `parameters` block below) |
| `transformed parameters` | deterministic functions of parameters/data, recomputed each evaluation |
| `model` | accumulates the log-density into `target` (see [Model components](#model-components)) |
| `functions` | user-defined functions (see [Functions](#functions)) |
| `generated quantities` | **(not yet)** — accepted but ignored, with a `WARNING` on compile |

A program must contain **at least one `model` block**; several are allowed when they are named
(`model prior { ... }`). All other blocks are optional and may appear at most once.

### How the log-density is written

The `model` block builds up a scalar **`target`** — the log-density — in **constrained
(ambient) space**, i.e. in terms of the parameters' natural values. **You never write a
Jacobian**: the change-of-variables correction for a constrained parameter is added
automatically. For example, with `real<lower=0> s;`, writing `s ~ lognormal(0, 1)` (or
`target += ...` in terms of `s`) is enough; the log-chart Jacobian is supplied by the
compiled `Model`.

### Model components

A bare `model { ... }` is the component named `target`. Naming the block splits the density into
**components**, each compiled to its own closure:

```stan
model prior      { v ~ normal(0, 3); }
model likelihood { y ~ normal(v, sigma); }
```

The joint log-density is their sum, so splitting a model does not change what it is — the point
is that the *computations stay separate*, which is what lets a sampler treat them differently
(the intended use is a multi-rate integrator that evaluates a cheap prior gradient more often
than an expensive likelihood one). They appear as `model.log_prob_fns` in source order, and
`model.log_prob_components(sample)` reports each one's value.

Rules: names must be unique (`model target { }` is a synonym for a bare block, so the two
collide); a bare and named blocks may be mixed; `jacobian` is reserved, as it would collide with
the chart-Jacobian term the backend adds. Statements in `transformed parameters` are recomputed
inside *every* component, so all of them see the transformed values — but for that reason a `~`
or `target +=` written there is rejected once a program has more than one component (it would
otherwise be counted once per component); put it in the component it belongs to.

**Cost:** with N components the sampler evaluates N separate gradients, each of which unpacks
the parameters and recomputes `transformed parameters`. Until a component-aware integrator
exists, splitting is a cost rather than a saving, and it inflates the reported `grad_evals`.

## Configuring a compiled model

`compile_model(source, data=...)` hands back a finished `Model`. To see — or change — the
decisions made along the way, compile in two steps:

```python
factory = compile_model(source)
spec    = factory.analyze(data)      # bind the data, classify the components
spec.component("likelihood").cost = "cheap"     # override a decision
spec.parameter("beta").centered = True          # a flag the grammar cannot express
model   = spec.build()
```

`factory.build(data)` is exactly `analyze(data).build()`, so nothing above is required.
`spec.rationale` explains every choice in words, and `spec.constant_sizes` / `spec.large_constants`
/ `spec.parameter_sizes` / `spec.large_parameters` show what the data and the declared parameters
looked like to the rule.

### Cheap and expensive components

Each component is labelled by whether it **touches a large constant or a large parameter** — size
is the proxy for gradient cost, whether the large thing is a fixed data array or a
high-dimensional parameter the component is written over. Large means: the biggest item (pooling
constants and parameters, parameters counted by *ambient* dimension) always; never a
single-element one; anything at least 1/500 of the biggest. The labels reach the model as
`model.cheap_components`, for the multi-rate integrator that evaluates cheap gradients more often
than expensive ones. They are metadata about *cost* — they never change the density.

A program whose constants and parameters are all scalars (or that has no data and no vector
parameter) has every component cheap, since nothing is large. A large constant or parameter read
in `transformed parameters` makes every component expensive, because those statements run inside
each of them. And because "large" is relative to the *biggest* item, a very wide design matrix
can still push a smaller per-observation array under the threshold — override the label, or pass
your own rule as `analyze(data, classify=...)`. Size is also only a proxy, not a measurement: a
component reading a large *parameter* it processes elementwise (e.g. transcendentals over a
2000-vector) can cost more per gradient than one reading a larger constant it processes with one
BLAS call — the rule tells you when there is something to split, not which side is actually
cheaper (see the `reg_horseshoe` study, `docs/design/06_hamiltonian_monte_carlo.md`).

### Chart options

`centered=True` (for `real` parameters, bounded or not) standardizes a parameter by an adapted
location and scale; `adaptive` (for `unit_vector`) decides whether its stereographic chart is
fitted. Both are per-parameter switches with no declaration syntax, and both are inert unless
the *sampler* also enables the matching adaptation.

## Types

Two scalar number types, plain shaped arrays, and one manifold type:

| Declaration | Meaning |
|---|---|
| `int x;` | an integer (for sizes, loop bounds, indices) |
| `real x;` | a real scalar |
| `array[d] real x;` | a real array of shape `(d,)` |
| `array[m, n] real x;` | a real array of shape `(m, n)` |
| `unit_vector[d] x;` | a unit vector of `d` components: a point on the sphere `S^(d-1)`, shape `(d,)` |
| `array[n] unit_vector[d] x;` | `n` unit vectors, shape `(n, d)` |
| `simplex[d] w;` | a probability vector: `d` positive components summing to 1, shape `(d,)` |
| `ordered[d] c;` | a strictly increasing vector `c[1] < … < c[d]`, shape `(d,)` |
| `ordered<lower=L, upper=U>[d] c;` | the same, with `L < c[1]` and `c[d] < U` |
| `cov_matrix[K] S;` | a `K x K` covariance matrix: symmetric, positive definite, shape `(K, K)` |
| `cholesky_factor_cov[K] L;` | its Cholesky factor: lower triangular, positive diagonal, shape `(K, K)` |
| `corr_matrix[K] Omega;` | a `K x K` correlation matrix: positive definite, **unit diagonal**, shape `(K, K)` |
| `cholesky_factor_corr[K] L;` | its Cholesky factor: lower triangular, **unit row norms**, shape `(K, K)` |
| `array[n] simplex[d] w;` / `array[n] ordered[d] c;` | `n` of them, shape `(n, d)` |
| `array[n] cov_matrix[K] S;` | `n` of them, shape `(n, K, K)` |
| `array real x` / `array[] real x` | an array of unstated size — **only** in a function signature |

Arrays are ordinary JAX arrays with **NumPy/JAX broadcasting**. There are no `vector`,
`row_vector`, or `matrix` types — use `array[...] real`. (See the design doc for why.)
Array sizes (`d`, `m`, `n`) must be compile-time constants: literals, `data`/`transformed
data` integers, or constant arithmetic of those.

### `int`: discrete parameters

A parameter declared `int<lower=L, upper=U>` is an **integer** parameter taking values in
`{L, ..., U}`, moved by a Metropolis-within-Gibbs sweep rather than by HMC. Both bounds are
required and must be compile-time constants: the sweep proposes from the enumerated support.

```
parameters {
  array[N] int<lower=1, upper=K> z;    // a cluster label per observation
  int<lower=0, upper=1> include;       // a spike-and-slab indicator
}
```

A discrete parameter can be used as an **index** — `mu[z[n]]` — which is what makes mixture and
latent-class models expressible. It has no chart, so no `centered` or `adaptive` option applies
to it.

Two things to know:

* **The sampler factory refuses a model with `int` parameters.** Compose the sampler yourself
  with `DiscreteMetropolisWithinGibbs`; see `docs/design/14_discrete_parameters.md`.
* `int` in a `data` block or a function signature is unchanged — it declares an integer *value*,
  not a parameter, and nothing about that moved.

Count-valued integers (`int<lower=0>`, no upper bound) are not supported yet, and a discrete
parameter may not appear in another parameter's bound.

### `unit_vector`

`unit_vector[d]` is a point on `S^(d-1)`, so its value in the model block is an ordinary
`d`-component array constrained to unit norm — write the density with respect to the surface
measure and the change of variables is handled for you:

```stan
data { real kappa; array[3] real mu; }
parameters { unit_vector[3] x; }
model { target += kappa * dot(x, mu); }      // von Mises-Fisher on the sphere
```

It differs from every other type in that its **coordinate space has a lower dimension than its
value**: `d` ambient components but only `d - 1` free coordinates, since the sampler works in a
stereographic chart. That is invisible in the DSL, but it means a `unit_vector[3]` contributes 3
columns to the draws and 2 dimensions to the sampler.

`d` must be at least 2. A `unit_vector` cannot carry `<lower=…>` / `<upper=…>` bounds (unit norm
is its constraint), and may only be declared in the `parameters` block. With
`array[n] unit_vector[d]`, each of the `n` unit vectors is adapted independently.

### `simplex`

`simplex[d]` is a probability vector: `d` positive components summing to exactly 1. Like
`unit_vector` its **coordinate space has a lower dimension than its value** — `d` components,
`d - 1` free coordinates — because the sum-to-one constraint removes one degree of freedom. The
chart is the standard stick-breaking transform, and the coordinate origin is the *uniform* point
`w[k] = 1/d`.

```stan
data { array[3] real alpha; }
parameters { simplex[3] w; }
model { for (k in 1:3) target += (alpha[k] - 1) * log(w[k]); }   // Dirichlet(alpha)
```

`d` must be at least 2. A `simplex` cannot carry bounds (its constraint is the simplex itself)
and may only be declared in the `parameters` block.

### `ordered`

`ordered[d]` is a strictly increasing vector — the type for mixture-component locations or
ordinal cutpoints, where without it the posterior is invariant under relabelling and multimodal
by construction. Ordering costs no dimension: `d` components and `d` free coordinates.

Unlike the other manifold types, `ordered` **may carry bounds**, and they constrain the two
ends — `lower` the first entry, `upper` the last (and hence, with the ordering, every entry).
They are written *before* the size, as in Stan:

```stan
parameters { ordered<lower=0, upper=1>[4] c; }   // 0 < c[1] < c[2] < c[3] < c[4] < 1
model { target += 0.0; }                          // the order statistics of 4 uniforms
```

With both bounds the `d + 1` gaps `(c[1]-L, c[2]-c[1], …, U-c[d])` are positive and sum to
`U - L`, so a doubly-bounded ordered vector is a scaled `(d+1)`-part simplex and uses the same
stick breaking; the coordinate origin is then the evenly spaced point. `d` must be at least 1,
and `ordered` may only be declared in the `parameters` block.

A gap is `exp(z)`, so two entries can be asked to sit closer together than float32 can express
(about `1.2e-7` relative), at which point they come out *equal*. Reach for x64 if your model
takes `log(c[k+1] - c[k])`.


### `cov_matrix` and `cholesky_factor_cov`

A covariance matrix, in either of the two forms a model might want it. `cov_matrix[K]` gives `Sigma`
itself; `cholesky_factor_cov[K]` gives the lower-triangular `L` with `Sigma = L L'`, which is what a
multivariate normal density actually uses. Both are `K x K` arrays, both may only be declared in the
`parameters` block, and neither takes bounds (positive definiteness *is* the constraint).

```stan
data { int n; array[3] real mu; array[n, 3] real y; }
parameters { cov_matrix[3] S; }
model {
  for (i in 1:n) y[i,:] ~ multi_normal(mu, S);
}
```

The same model over the factor, which is what a multivariate normal density actually uses. Written
out with `solve_triangular`, nothing is factorized or inverted at all — the parameter *is* the
factorization, and the log-determinant is a sum over its diagonal:

```stan
data { int n; array[3] real mu; array[n, 3] real y; }
parameters { cholesky_factor_cov[3] L; }
model {
  for (i in 1:n) {
    target += -sum(log(diag(L)))
              - 0.5 * sum(square(solve_triangular(L, y[i,:] - mu)));
  }
}
```

That is the case for having the factor as its own type. Handing `L * L'` to `multi_normal` also
works and is shorter, but it multiplies the factor out only to factorize it again inside.

Both sample in the same **log-Cholesky** coordinates — the strict lower triangle of `L` together
with `log L_ii`, so `K(K+1)/2` unconstrained coordinates for a `K x K` matrix — and the chart
Jacobian is supplied for you (Stan's `K log 2 + sum_i (K - i + 2) log L_ii` for `cov_matrix`, and
`sum_i log L_ii` for the factor). The coordinate origin is the identity matrix, which is where a
chain starts by default.

`K` must be at least 1. Two things worth knowing:

* **The diagnostics are computed from `Sigma`**, not from `L`, whichever type you declare: the
  reported features are the lower triangle of the covariance and its squares. For
  `cholesky_factor_cov` the posterior summary additionally shows the factor's structurally-zero
  upper triangle as rows of exact zeros — those are the parametrization showing through, not a
  stuck chain.
* **The Stein diagnostic uses the geometry of the positive definite matrices** (Brownian motion on
  the affine-invariant symmetric space), which unlike the simplex's or a bounded scalar's needs no
  assumption that the density vanishes at the boundary. See
  `docs/design/04_manifold_parameters.md`.

Stan's rectangular `cholesky_factor_cov[M, N]` is not supported: its `Sigma` is singular, which
that geometry does not admit.


### `corr_matrix` and `cholesky_factor_corr`

A correlation matrix, again in either form: `corr_matrix[K]` gives `Omega` with a unit diagonal,
`cholesky_factor_corr[K]` gives the lower-triangular `L` whose *rows* have unit norm, with
`Omega = L L'`. Both are `K x K` arrays, `parameters`-block only, and take no bounds.

They exist because scales and dependence usually deserve separate priors. The idiom is to keep them
apart and multiply:

```stan
data { int n; array[n, 3] real y; }
parameters {
  cholesky_factor_corr[3] L;
  array[3] real<lower=0> sigma;
}
model {
  L ~ lkj_corr_cholesky(2.0);
  sigma ~ lognormal(0, 1);
  for (i in 1:n) {
    target += -sum(log(sigma)) - sum(log(diag(L)))
              - 0.5 * sum(square(solve_triangular(diag(sigma) * L, y[i,:])));
  }
}
```

`diag(sigma) * L` is the Cholesky factor of `diag(sigma) Omega diag(sigma)`, so one triangular
solve does the whole quadratic form and the log-determinant is a sum over two diagonals — nothing
is factorized, inverted, or multiplied out.

The chart is Stan's: canonical partial correlations through a `tanh` link, then stick breaking on
sums of squares, giving `K(K-1)/2` unconstrained coordinates and a matrix whose rows have unit norm
by construction. The coordinate origin is the identity. `K` must be at least 2.

Three things worth knowing, beyond what `cov_matrix` already documents:

* **`lkj_corr(eta)` and `lkj_corr_cholesky(eta)`** are the standard prior. `eta = 1` is uniform over
  correlation matrices; larger `eta` concentrates toward the identity. Use the `_cholesky` form with
  the `_cholesky` type — it carries the `L -> Omega` Jacobian, which is exactly the part that is
  easy to drop when writing the density by hand.
* **The structural entries show through.** `corr_matrix` reports `K` rows of exact 1s in the
  posterior summary, and `cholesky_factor_corr` reports a zero upper triangle plus a diagonal that
  is *determined* by the rest of its row. They are the parametrization, not a stuck chain, and none
  of them reaches a diagnostic: the reported features are the strict lower triangle of `Omega` and
  its squares.
* **The Stein diagnostic uses the elliptope's own geometry** — Brownian motion on the correlation
  matrices as a submanifold of the positive definite ones — and, like the covariance types, needs no
  assumption that the density vanishes at the boundary. See
  `docs/design/04_manifold_parameters.md`.

## Declarations and constraints

```
[array[ <size>, ... ]] <base-type> [ < lower = <expr> [, upper = <expr>] > ] [ [ <size> ] ]  <name> [ = <expr> ] ;
```

where `<base-type>` is `real`, `int`, `unit_vector`, `simplex`, or `ordered`. The `array[...]`
prefix sizes the array *of* elements; a base type's own `[d]` sizes the element itself — hence
`array[n] unit_vector[d]` carries both. Bounds go **between the type and its size**
(`ordered<lower=0>[d]`, not `ordered[d]<lower=0>`), and only `real`, `int` and `ordered` accept
them.

Bounds use Stan's `<lower=…, upper=…>` syntax and select a link chart on a parameter:

| Bounds | Constrained domain | Link |
|---|---|---|
| `<lower=L>` | `x > L` | log: `x = L + exp(z)` |
| `<upper=U>` | `x < U` | reflected log: `x = U - exp(z)` |
| `<lower=L, upper=U>` | `L < x < U` | logit: `x = L + (U-L)·sigmoid(z)` |

In the **`parameters`** block, a bound must be a numeric constant **or the name of another
parameter** (a *parent-dependent* bound). For example:

```stan
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=a> b;     // b's upper bound is the parameter a
}
```

The parameters are evaluated in dependency order automatically. (General-expression bounds
are **not yet** supported.) Constraints are meaningful only on parameter declarations; on
other declarations they are ignored.

An optional `= <expr>` initializer is allowed on `transformed data` / `transformed
parameters` / local declarations (it is how you compute them); parameters may not have one.

## Expressions

### Literals

- **int**: a run of digits, e.g. `0`, `42`.
- **real**: has a decimal point or an exponent, e.g. `3.0`, `2.5`, `1e2`, `6.022e-23`. A real
  literal must start with a digit (`0.5`, not `.5`).

### Operators

| Operator | Meaning |
|---|---|
| `+ - * /` | arithmetic |
| `^` | exponentiation (**right-associative**: `2^3^2` = `2^(3^2)`) |
| `.* ./ .^ .+ .-` | **elementwise** (with broadcasting) |
| unary `- +` | negation / identity |
| `< > <= >= == !=` | comparisons (used in `if` conditions) |
| `'` (postfix) | transpose (a no-op on scalars and 1-D arrays) |
| `f(...)` | function call |
| `a[...]` | indexing / slicing |
| `( ... )` | grouping |

**The key non-obvious rule:** `*` is **matrix multiplication** when both operands are arrays
(so `1-D · 1-D` is a dot product and `2-D · 1-D` is a matrix–vector product), and **scalar
multiplication** when either operand is a scalar. Use `.*` for elementwise multiplication.
The other arithmetic operators (`+ - / ^` and their `.`-prefixed forms) are elementwise with
broadcasting.

```stan
target += -0.5 * (x - mu) * (precision * (x - mu));   // (x-mu)·P·(x-mu): two matmuls
y = a .* b;                                            // elementwise product
```

Precedence, lowest to highest: comparisons < (`+ - .+ .-`) < (`* / .* ./`) < unary `- +` <
(`^ .^`) < postfix (`[]`, call, `'`). Unary minus binds looser than `^` but tighter than `*`,
so `-x^2` is `-(x^2)` and `-x*y` is `(-x)*y`.

### Indexing

Indices are **1-based** and ranges are **inclusive** (Stan convention):

| Expression | Selects |
|---|---|
| `a[1]` | the first element |
| `a[i]` | the `i`-th element (1-based); `i` may be a loop variable |
| `a[2:4]` | elements 2, 3, 4 (inclusive) |
| `a[3:]`, `a[:4]`, `a[:]` | open ends |
| `A[i, j]` | multi-dimensional indexing |
| `a[:, None]` | insert an axis of length 1 (NumPy's `newaxis`) |

Slice bounds must be compile-time constants (a single scalar index may be dynamic, e.g. a
loop variable). Negative and stepped indices are not supported. Indexed assignment
(`x[i] = ...;`) performs a functional update.

**`None` inserts an axis**, exactly as in NumPy and JAX — `jnp.newaxis` *is* `None`. It is the
usual way to line two arrays up for broadcasting:

```stan
target += sum(x[:, None] .* y[None, :]);    // the outer product of x and y
```

It may appear in any index position and be mixed with ranges (`A[:, None, 1:2]`). It is *not* a
slice bound: `a[None:2]` is an error.

### Function calls

Builtin functions wrap `jax.numpy`, and **use JAX/NumPy names rather than Stan's** — `det` not
`determinant`, `solve` not `mdivide_left`. The intent is that a model here reads like the same
model written as a plain JAX function over a `Model`, which is the translation you actually make;
a Stan program already needs rewriting for the array semantics.

```
exp  log  log1p  expm1  sqrt  abs  fabs  square
sin  cos  tan  tanh  sigmoid
sum  prod  mean  min  max
dot  transpose  inverse  diag
floor  ceil  lgamma
solve  solve_triangular  cholesky  trace  det  slogdet  eigvals  eigvalsh
```

All of them are differentiable in their arguments, which is what lets a sampler take gradients of
a target that uses them.

#### Linear algebra

| Call | Returns |
|---|---|
| `solve(A, b)` | `A^-1 b` — prefer it to `inverse(A) * b` |
| `solve_triangular(A, b)` | the same for triangular `A`, reading the **lower** triangle |
| `solve_triangular(A, b, 0)` | …reading the **upper** triangle |
| `cholesky(A)` | the lower-triangular `L` with `A = L L'` |
| `trace(A)`, `det(A)` | the trace and the determinant |
| `slogdet(A)` | the **pair** `(sign, log|det A|)` |
| `eigvals(A)`, `eigvalsh(A)` | eigenvalues of a general / a symmetric matrix |

Three carry sharp edges, all of them inherited from JAX rather than invented here:

**`slogdet` returns a pair**, so destructure it — the DSL has no way to index a tuple:

```stan
(real sign, real ld) = slogdet(S);
target += -0.5 * ld;
```

It is the numerically sound way to get `log|det|`: `log(det(S))` overflows or underflows for a
matrix of any size, while `slogdet` never forms the determinant itself.

**`solve_triangular` takes `lower` as a third positional argument**, because the DSL has no keyword
arguments — and **it defaults to the lower triangle, unlike JAX**. That is a deliberate departure
from the naming principle above, on the grounds that `cholesky` and both Cholesky parameter types
(`cholesky_factor_cov`, `cholesky_factor_corr`) all produce lower-triangular factors: keeping JAX's
default would put an argument on the common case, and getting it wrong reads the other triangle
and returns a wrong answer rather than an error. Pass `0` for an upper solve. It must be a literal:
the flag selects a triangle when the expression is traced, so a computed value is rejected.

**`eigvals` returns complex numbers** for a general matrix, and the DSL has no complex type. Use
`abs(eigvals(A))` for the moduli — under `max`, that is the spectral radius, which is what a
stability condition needs. For a symmetric matrix use `eigvalsh`, whose values are real. (The real
and imaginary parts are not separately reachable: `real` is a type keyword, so it cannot also be a
function name.)

#### A multivariate normal, done properly

Together these let a covariance parameter be used the way it should be — one factorization, one
triangular solve, no inverse and no determinant:

```stan
data { int n; array[3] real mu; array[n, 3] real y; }
parameters { cov_matrix[3] S; }
model {
  (real sign, real ld) = slogdet(S);
  for (i in 1:n) {
    target += -0.5 * ld - 0.5 * sum(square(solve_triangular(cholesky(S), y[i,:] - mu)));
  }
}
```

With `cholesky_factor_cov[3] L;` instead, the factorization is the parameter itself and
`cholesky(S)` drops out, `ld` becomes `2 * sum(log(diag(L)))`, and nothing is factorized at all.

## Functions

The `functions` block defines your own, in Stan's C-like form — a return type, a name, typed
arguments, and a body ending in `return`:

```stan
functions {
  array real add_two(array real a, array real b) { return a + b; }
  real scale(real x, real k) { return x * k; }
}
```

Because a signature cannot know an array's size, array arguments and return types are written
without one: `array real x`, or Stan's `array[] real x` (`array[,] real x` for two dimensions).
Both spellings are accepted. Types are **recorded but not checked** — a genuine mismatch is
reported by JAX when the model is first evaluated, pointing at your source.

A function is **pure**: it may not contain `~` or `target +=` (Stan's `_lp` functions are not
supported), and it sees **only its arguments and its own locals** — not data, parameters or
transformed parameters. Anything it needs must be passed in. Consequently a `void` function
could do nothing at all, and is rejected.

- **No overloading**: one definition per name, and the name may not be a keyword, a builtin, or
  a distribution. (Ordinary variable names are unrestricted, as before.)
- **Recursion** works, provided the base case is decidable when the model is traced — the same
  restriction that makes `if` conditions compile-time constants. Runaway recursion stops with a
  "call depth exceeded" error rather than a Python crash.
- A `return` inside a `for` exits the function immediately, as you would expect.
- Functions may be called anywhere an expression may appear, including `transformed data` (run
  once, at build time) — but *not* in an array size or a parameter bound.

## Statements

| Statement | Meaning |
|---|---|
| `<lvalue> = <expr>;` | assignment (to a name or an indexed name) |
| `<expr> ~ <dist>(<args>);` | sampling: adds `sum(logpdf)` to `target` |
| `target += <expr>;` | add a term to the log-density directly |
| `return <expr>;` | return a value from a user-defined function |
| `for (<i> in <lo>:<hi>) <body>` | loop, `<i>` from `<lo>` to `<hi>` inclusive |
| `if (<cond>) <body> [else <body>]` | conditional |

A `<body>` is either a single statement or a braced block `{ ... }`. `else if` chains work
(the `else` body is itself an `if`). Sampling and `target +=` are only allowed in the `model`
block.

**Sampling (`~`).** `x ~ dist(args)` is exactly `target += sum(dist_logpdf(x, args))`. The
`sum` means a vectorized statement adds the elementwise log-densities: `x ~ normal(0, 1)` for
an array `x` contributes `sum_i logpdf(x_i)`.

**Loops.** `for` loops are **unrolled** and so require compile-time-constant bounds. That is
fine for a short loop and wasteful for a long one — see [Non-unrolling loops](#non-unrolling-loops)
for `scan` and `fori_loop`, which do not unroll. `while` loops and dynamic-condition `if` are
**not yet** supported (an `if` condition must be a compile-time constant).

## Non-unrolling loops

A `for` loop is written out in full when the model is traced, so a 1000-step recursion becomes
1000 copies in the computation graph and compiles slowly. `scan` and `fori_loop` do not unroll:
the graph stays the same size however long the loop runs.

| Form | Body | Returns |
|---|---|---|
| `scan(f, init, xs, ...extra)` | `f(carry, x, ...extra)` → `(carry, y)` | `(carry, ys)` |
| `fori_loop(lower, upper, f, init, ...extra)` | `f(i, val, ...extra)` → `val` | the final `val` |

```stan
functions {
  (real, real) ar_step(real prev, real e, real phi) {
    real next = phi * prev + e;
    return (next, next);              // (new carry, the value to keep)
  }
}
data { int n; array[n] real e; }
parameters { real phi; }
model {
  (real last, array[n] real path) = scan(ar_step, 0.0, e, phi);
  target += -0.5 * sum(path .* path);
}
```

The loop **length must be a compile-time constant** — a literal, a `data` integer, or constant
arithmetic of those. This is not red tape: a static length is exactly what makes the loop
reverse-mode differentiable, which every model needs. A non-constant bound is an error, and so
is a fractional one.

> **⚠️ `fori_loop`'s range is inclusive.** `fori_loop(1, n, f, init)` runs `n` times, with `i`
> taking `1 … n` — matching `for (i in 1:n)` and this language's 1-based indexing. This differs
> from `jax.lax.fori_loop`, whose range is half-open `[lower, upper)`; **the same call runs one
> more iteration here than in JAX.** Worth remembering when porting JAX code.

**The body must be a function** defined in the `functions` block, named directly in the call —
you cannot pass a builtin or an expression. Since a function sees only its own arguments, it
cannot capture data the way a JAX closure does; instead, **any arguments after the fixed ones
are forwarded to every call** (`phi` in the example above). A body cannot use `~` or
`target +=`: like every function it is pure, so anything it computes must come back through the
carry.

Because `init` and `xs` are pytrees, a tuple `init` carries several values at once and a tuple
`xs` scans several arrays in step:

```stan
((real a, real b), array[n] real ys) = scan(f, (0.0, 1.0), (xs1, xs2));
```

### Empty carries and empty inputs: `None`

Either end of a `scan` may be `None`.

**An empty carry** (`init = None`) turns the scan into a map — there is no state to thread, only
an output per step:

```text
functions {
  (None, real) scale(None, real x, real k) { return (None, k * x); }
}
...
(None, array[n] real ys) = scan(scale, None, xs, k);
```

**Empty inputs** (`xs = None`) run the loop without consuming an array. With no array there is
nothing to take the length from, so **the length is written as the next argument** and the
forwarded extras follow it:

```stan
(real last, array[6] real path) = scan(tick, 0.0, None, 6, phi);
//                                                     ^ the length
```

A missing length is a compile-time error, as is a length that is not a constant whole number —
the length is what keeps the loop static, and therefore differentiable. Note that with `None`
inputs the argument after it is the *length*, not an extra; getting that wrong shows up as an
arity mismatch against the body.

`None` is also a **type**, usable in a parameter list and in tuple types, and its name may be
omitted since there is nothing to refer to — `(None, real) f(None, real x)`. Like every other
type here it is *recorded and never checked*: it documents the signature, it does not enforce
it. `None` is a reserved word, and is not a parameter type (there is nothing to sample).

### Tuples

Tuples exist so `scan` can keep JAX's signature, and go no further. Available: tuple literals
`(a, b)`, tuple types for a function's arguments and return, and destructuring declarations,
which may nest.

```text
functions {
  (real, real) split(real x) { return (x - 1.0, x + 1.0); }
}
...
(real lo, real hi) = split(m);                  // destructuring binds both names
((real a, real b), real c) = ((m, m + 1.0), m + 2.0);   // and it nests
```

Destructuring is the **only** way to take a tuple apart: there is no `t.1` element access, and
no tuple-typed local variable or model parameter. `(x)` with one element is ordinary grouping,
not a 1-tuple. As with every other type here, the declared element types are recorded but never
checked — JAX reports real mismatches at trace time.

## Distributions

Available in `~` (and as the source of `target +=` terms). Parameterizations follow Stan
(note where they differ from SciPy).

**Continuous** (`logpdf`):

| Distribution | Arguments | Notes |
|---|---|---|
| `normal(mu, sigma)` | mean, **standard deviation** | |
| `lognormal(mu, sigma)` | log-mean, log-sd | density of the positive variable |
| `uniform(L, U)` | lower, upper | |
| `exponential(beta)` | **rate** | |
| `gamma(alpha, beta)` | shape, **rate** | |
| `cauchy(loc, scale)` | location, scale | |
| `beta(a, b)` | the two shape parameters | |
| `student_t(nu, loc, scale)` | dof, location, scale | |
| `multi_normal(mu, Sigma)` | mean vector, **covariance** matrix | joint (scalar) density |
| `lkj_corr(eta)` | concentration | joint; on a `corr_matrix`. `eta = 1` is uniform, `eta > 1` favours the identity |
| `lkj_corr_cholesky(eta)` | concentration | the same prior on a `cholesky_factor_corr`, Jacobian included |

**Discrete** (`logpmf`). These are normally used for *observed data* on the left of `~`
(e.g. `y ~ poisson(lambda)` with integer data `y` and a continuous parameter `lambda`); the
log-pmf is differentiable in the continuous parameters, which is what the samplers use:

| Distribution | Arguments | Notes |
|---|---|---|
| `bernoulli(theta)` | success probability | `y` in {0, 1} |
| `categorical(theta)` | simplex of `K` probabilities | `y` in **1..K** (1-based, as Stan); out of range is `-inf` |
| `categorical_logit(alpha)` | `K` unnormalized log-probabilities | as `categorical(softmax(alpha))`, but exact in the tails |
| `binomial(N, theta)` | trials, success probability | |
| `poisson(rate)` | **rate** | |
| `neg_binomial(alpha, beta)` | shape, inverse-scale | Stan's `neg_binomial` |
| `neg_binomial_2(mu, phi)` | **mean**, dispersion | variance `mu + mu^2 / phi` |
| `beta_binomial(N, alpha, beta)` | trials, the two shape parameters | |
| `geometric(theta)` | success probability | SciPy convention: support `k >= 1` |
| `multinomial(theta)` | a simplex | joint (scalar); `N = sum(y)`, counts must be integral |

(Other distributions, precision-parameterized variants, and truncation are **not yet**
available.)

## Worked examples

A correlated Gaussian, written with an explicit quadratic form (`*` as matmul):

```stan
data { int d; array[d] real mu; array[d, d] real precision; }
parameters { array[d] real x; }
model { target += -0.5 * (x - mu) * (precision * (x - mu)); }
```

Neal's funnel, with separate parameter blocks and `~`:

```stan
data { int nx; real scale; }
parameters { real v; array[nx] real x; }
model {
  v ~ normal(0, scale);
  x ~ normal(0, exp(v / 2));     // x_i | v ~ Normal(0, exp(v))
}
```

A parent-dependent bound (nested uniforms); the model writes only the density, the chart
Jacobian is automatic:

```stan
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=a> b;
}
model { target += -log(a); }      // density of Uniform(0, a) is 1/a
```

## Not yet supported

For reference, the following are recognized in the design but not implemented yet: the
`generated quantities` block (parsed but ignored, and warned about at compile time --- the
program samples fine, it just produces none of the quantities); `vector` / `row_vector` / `matrix` types;
tuple *locals* (`(real, real) t = …`), tuple parameters and `t.1` element access (tuple
literals, tuple function types and destructuring **are** supported); first-class functions
beyond a loop form's body slot (`scan` and `fori_loop` **are** supported);
other manifold types such as the Grassmannians — `unit_vector`, `simplex`, `ordered`,
`cov_matrix`, `cholesky_factor_cov`, `corr_matrix` and `cholesky_factor_corr` are supported; `complex`; Stan's `_lp` / `_rng` / `_lpdf` /
`_lpmf` functions and the `data` argument qualifier; function calls in array sizes and
parameter bounds; general-expression parameter bounds; precision-parameterized and truncated
distributions; `while` loops, dynamic-condition `if`, and dynamic slice bounds.

Compile errors for unknown *names* and *distributions* still surface on first evaluation of the
model rather than at `compile_model` time (the interpreter builds its closure lazily). Errors in
the `functions` block, unknown or wrong-arity function calls, `return` outside a function, and
the model-component rules are all reported at `compile_model` time.
