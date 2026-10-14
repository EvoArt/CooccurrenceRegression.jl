import numpy as np
import pandas as pd
import pytest

import cooccurrence_regression as cr

rng = np.random.default_rng(1)
Y = pd.DataFrame(rng.random((20, 8)) < 0.5, columns=[f"sp{i}" for i in range(1, 9)])


def test_counts_and_pairs():
    K, m, N = cr.cooccurrence_counts(Y)
    assert N == 20
    assert (m.to_numpy() == Y.sum(axis=0).to_numpy()).all()
    assert K.loc["sp1", "sp2"] == (Y["sp1"] & Y["sp2"]).sum()
    a = cr.affinity_mle(K.iloc[0, 1], m.iloc[0], m.iloc[1], N)
    assert isinstance(a, float)
    assert cr.fnch_mean(m.iloc[0], m.iloc[1], N, a) == pytest.approx(K.iloc[0, 1], abs=1e-8)
    assert cr.affinity_mle(5, 5, 9, 12) == np.inf
    assert cr.affinity_map([1, 2, 3], 5, 9, 12).shape == (3,)
    post = cr.affinity_posterior(3, 5, 9, 12)
    assert list(post.columns) == ["median", "mean", "lower", "upper", "pd", "p_positive"]
    assert post["lower"][0] < post["median"][0] < post["upper"][0]


def test_pairwise():
    pw = cr.pairwise_affinity(Y)
    assert pw.median.shape == (8, 8)
    assert list(pw.median.index) == list(Y.columns)
    M = pw.median.to_numpy()
    assert np.allclose(M, M.T, equal_nan=True)
    k = (Y["sp1"] & Y["sp2"]).sum()
    assert M[1, 0] == cr.affinity_posterior(k, Y["sp2"].sum(), Y["sp1"].sum(), 20)["median"][0]


def test_distances_nulls_site_effects():
    x = rng.normal(size=8)
    assert np.allclose(cr.trait_distance(x), np.abs(x[:, None] - x[None, :]))
    assert (cr.trait_distance(["a", "b", "a"]) == np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])).all()
    Z = cr.standardise(cr.trait_distance(x))
    assert abs(Z[~np.eye(8, dtype=bool)].mean()) < 1e-10
    assert np.allclose(cr.gower_distance(x), cr.trait_distance(x) / np.ptp(x))
    N0 = cr.swap_null(Y, seed=2)
    assert (N0.sum(axis=0) == Y.sum(axis=0)).all()
    assert (N0.sum(axis=1) == Y.sum(axis=1)).all()
    assert N0.equals(cr.swap_null(Y, seed=2))
    assert cr.site_effects(Y).sites.min() >= 0


def test_regression():
    X = cr.standardise(cr.trait_distance(rng.normal(size=8)))
    fit = cr.cooccurrence_regression(X, Y, n_iter=30, n_chains=2, drop_lambda=True, seed=3)
    assert list(fit.columns) == ["chain", "draw", "alpha", "beta[1]"]
    assert len(fit) == 60
    assert sorted(fit["chain"].unique()) == [0, 1]

    g = ["a"] * 4 + ["b"] * 4
    fit = cr.cooccurrence_regression([X, X], Y, g=g, n_iter=30, n_chains=1, pooled=True)
    assert {"beta[1]", "beta[2]", "lambda[8]"} <= set(fit.columns)
    with pytest.raises(ValueError, match="only when g"):
        cr.cooccurrence_regression(X, Y, pooled=True)
    with pytest.raises(ValueError, match="expected"):
        cr.cooccurrence_regression(X[:4, :4], Y)
