# CooccurrenceRegression

[![Build Status](https://github.com/EvoArt/CooccurrenceRegression.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EvoArt/CooccurrenceRegression.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Bayesian estimation of co-occurrence *affinity*, and regression of affinity on properties of
pairs of species such as trait or phylogenetic distance.

Affinity is the log odds ratio of Fisher's noncentral hypergeometric distribution: a measure of
how much more (or less) often two species are found together than expected from how common
each is ([Mainali et al. 2022](https://doi.org/10.1126/sciadv.abj9204)). This package

- estimates the affinity of pairs of species with a prior, giving finite estimates with
  credible intervals for every pair (`affinity_posterior`, `pairwise_affinity`), alongside the
  maximum likelihood estimate (`affinity_mle`);
- fits regression models in which the co-occurrence counts themselves are the response and
  the explanatory variables are pairwise quantities, with species effects to account for the
  non-independence of pairs that share a species (`cooccurrence_regression`).

The methods are described in [Bayesian estimation of co-occurrence affinity with dyadic regression](https://www.biorxiv.org/content/10.1101/2024.01.16.575941v1).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/CooccurrenceRegression.jl")
```

Julia 1.10 or later.

## Data

All functions take an `N × S` presence/absence matrix `Y` with **sites in rows and species in
columns**. Species found at none or at all of the sites carry no information about
co-occurrence and are ignored.

## Affinity of pairs of species

```julia
using CooccurrenceRegression

Y = rand(Bool, 30, 12)                      # 30 sites, 12 species

K, m, N = cooccurrence_counts(Y)            # co-occurrence counts, prevalences, number of sites

affinity_mle(K[2, 1], m[2], m[1], N)        # maximum likelihood; ±Inf at the extremes
affinity_map(K[2, 1], m[2], m[1], N)        # maximum a posteriori; always finite
affinity_posterior(K[2, 1], m[2], m[1], N)  # (median, mean, lower, upper, pd)

res = pairwise_affinity(Y)                  # S × S matrices: median, lower, upper, pd
```

`pd` is the probability of direction: the posterior probability that the affinity has the
sign of its median. The posterior of a single affinity is one-dimensional and is integrated
numerically, so these functions are fast and involve no sampling.

## Regression

```julia
trait = randn(12)
X = standardise(trait_distance(trait))      # S × S matrix of pairwise distances

chn = cooccurrence_regression(X, Y)         # one explanatory variable
chn = cooccurrence_regression([X1, X2], Y)  # several
```

The model for species `i` and `j` is

```
k[i, j] ~ FNCH(m[i], m[j], N, α + λ[i] + λ[j] + β[1] * X1[i, j] + β[2] * X2[i, j] + …)
α ~ Normal(0, 4),   β[p] ~ Normal(0, 1),   λ[i] ~ Normal(0, 1)
```

and the prior standard deviations can be changed with the keywords `alpha_sd`, `beta_sd` and
`lambda_sd`. The defaults assume standardised explanatory variables.

To study relationships within groups (genera, guilds), pass a vector giving the group of
each species. By default each group has its own coefficients, drawn from a common
distribution; `pooled=true` fits a single set of coefficients to the within-group pairs.

```julia
g = rand(["a", "b", "c"], 12)
chn = cooccurrence_regression(X, Y, g)
chn = cooccurrence_regression(X, Y, g; pooled=true)
```

Sampling uses NUTS from [Turing.jl](https://turinglang.org/), which this package re-exports,
and the result is an [`MCMCChains.Chains`](https://turinglang.org/MCMCChains.jl/stable/) object,
so the usual summaries and diagnostics apply:

```julia
describe(chn)
quantile(vec(chn[Symbol("β[1]")]), [0.025, 0.5, 0.975])
```

See the docstring of `cooccurrence_regression` for all options, including the choice of
automatic differentiation backend (`adtype`).

## The likelihood

`fnch_logpdf(k, m_i, m_j, N, α)` is the log probability of `k` co-occurrences under Fisher's
noncentral hypergeometric distribution, written in terms of the log odds ratio `α`. It is
exact for extreme `α`, and its derivative with respect to `α` is available in closed form
(observed minus expected co-occurrences), with rules for ForwardDiff, Mooncake and
ChainRules. `LogOddsFNCH(m_i, m_j, N, α)` wraps it as a `Distributions.jl` distribution for
use in your own Turing models:

```julia
@model function my_model(k, m_i, m_j, N)
    α ~ Normal(0, 4)
    k ~ LogOddsFNCH(m_i, m_j, N, α)
end
```

## Examples

`examples/beetles.jl` reproduces the dung beetle analyses from the paper.

## Citation

If you use this package please cite the paper above; see `CITATION.cff`.
