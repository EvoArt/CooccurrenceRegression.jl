# cooccurrence-regression (Python)

Python interface to the Julia package
[CooccurrenceRegression.jl](https://github.com/EvoArt/CooccurrenceRegression.jl). The
computation is done in Julia through [juliacall](https://github.com/JuliaPy/PythonCall.jl);
the functions and their arguments are those of the Julia package.

## Installation

```
pip install "git+https://github.com/EvoArt/CooccurrenceRegression.jl#subdirectory=python"
```

juliacall installs Julia and the Julia package the first time the package is used.

## Use

```python
import numpy as np
import cooccurrence_regression as cr

Y = np.random.default_rng(1).random((30, 12)) < 0.5     # 30 sites, 12 species

K, m, N = cr.cooccurrence_counts(Y)
cr.affinity_posterior(K[1, 0], m[1], m[0], N)
res = cr.pairwise_affinity(Y)                    # median, lower, upper, pd, p_positive

X = cr.standardise(cr.trait_distance(np.random.default_rng(2).normal(size=12)))
fit = cr.cooccurrence_regression(X, Y, drop_lambda=True)

import arviz as az                                  # pip install "cooccurrence-regression[arviz]"
az.summary(cr.to_inference_data(fit))
```

`cooccurrence_regression` returns a pandas DataFrame with columns `chain`, `draw` and one per
parameter. Parameter names are in ASCII (`alpha`, `beta[1]`, `lambda[1]`, `beta_mu[1]`,
`beta_sigma[1]`) with indices from 1, as in Julia. Unlike the Julia function it runs several
chains (`n_chains=4`), one after another. If `Y` is a DataFrame, matrices returned by
`pairwise_affinity` and `cooccurrence_counts` are labelled by its columns.

The first call in a session starts Julia and compiles the package, which takes a minute or
more.

To use a local copy of the Julia package:

```python
import juliapkg
juliapkg.add("CooccurrenceRegression", "b11753cd-0d5a-47ad-8b8f-99db3fa4a09b",
             path="/path/to/CooccurrenceRegression", dev=True)
```
