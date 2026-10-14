# CooccurrenceRegression (R)

R interface to the Julia package
[CooccurrenceRegression.jl](https://github.com/EvoArt/CooccurrenceRegression.jl). The
computation is done in Julia through [JuliaCall](https://cran.r-project.org/package=JuliaCall);
the functions and their arguments are those of the Julia package.

## Installation

```r
remotes::install_github("EvoArt/CooccurrenceRegression.jl", subdir = "R")
```

Julia 1.10 or later is needed. If JuliaCall cannot find it, `JuliaCall::install_julia()`
installs it. The Julia package is installed into its own Julia environment the first time a
function is called (see `?cr_setup`).

## Use

```r
library(CooccurrenceRegression)

Y <- matrix(runif(30 * 12) < 0.5, 30, 12)       # 30 sites, 12 species

cc <- cooccurrence_counts(Y)
affinity_posterior(cc$K[2, 1], cc$m[2], cc$m[1], cc$N)
res <- pairwise_affinity(Y)                    # matrices: median, lower, upper, pd, p_positive

X <- standardise(trait_distance(rnorm(12)))
fit <- cooccurrence_regression(X, Y, drop_lambda = TRUE)
posterior::summarise_draws(fit)
```

`cooccurrence_regression()` returns a `posterior::draws_df`. Parameter names are in ASCII
(`alpha`, `beta[1]`, `lambda[1]`, `beta_mu[1]`, `beta_sigma[1]`) and indices start at 1, as
in Julia. Unlike the Julia function it runs several chains (`n_chains = 4`), one after another.

The first call in a session starts Julia and compiles the package, which takes a minute or
more.
