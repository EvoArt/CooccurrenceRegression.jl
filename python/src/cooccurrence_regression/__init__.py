"""Bayesian estimation of co-occurrence affinity with dyadic regression.

Python interface to the Julia package CooccurrenceRegression.jl. Julia and the package are
installed on first use by juliacall; the first call in a session takes a minute or more while
Julia loads and compiles the package.

Presence/absence data are given as a matrix (numpy array or pandas DataFrame) with sites in
rows and species in columns; any value greater than zero counts as a presence.
"""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

__all__ = [
    "setup",
    "cooccurrence_counts",
    "affinity_mle",
    "affinity_map",
    "affinity_posterior",
    "pairwise_affinity",
    "fnch_logpdf",
    "fnch_mean",
    "trait_distance",
    "gower_distance",
    "standardise",
    "swap_null",
    "site_effects",
    "cooccurrence_regression",
    "to_inference_data",
]

_interop = None


def setup():
    """Start Julia and load CooccurrenceRegression.jl.

    Called by every other function, so it only needs calling directly to control when the
    start-up cost is paid. Returns the Julia module of wrapper entry points.
    """
    global _interop
    if _interop is None:
        from juliacall import Main as jl

        jl.seval("import CooccurrenceRegression")
        _interop = jl.CooccurrenceRegression.Interop
    return _interop


def _np(x):
    return np.asarray(x.to_numpy())


def _presence(Y):
    P = np.asarray(Y, dtype=float)
    if P.ndim != 2:
        raise ValueError("Y must be a matrix with sites in rows and species in columns")
    if np.isnan(P).any():
        raise ValueError("Y contains missing values")
    return P > 0


def _species(Y):
    return list(Y.columns) if isinstance(Y, pd.DataFrame) else None


def _square(M, labels):
    return M if labels is None else pd.DataFrame(M, index=labels, columns=labels)


def _labels(x):
    return None if x is None else [str(v) for v in x]


def _trait(x):
    a = np.asarray(x)
    if np.issubdtype(a.dtype, np.number) and a.dtype != bool:
        return a.astype(float)
    return [str(v) for v in x]


def _vector(*args):
    scalar = all(np.ndim(a) == 0 for a in args)
    return scalar, [np.atleast_1d(np.asarray(a, dtype=float)) for a in args]


def _result(x, scalar):
    v = _np(x)
    return float(v[0]) if scalar else v


Counts = namedtuple("Counts", ["K", "m", "N"])
PairwiseAffinity = namedtuple("PairwiseAffinity", ["median", "lower", "upper", "pd", "p_positive"])
SiteEffects = namedtuple("SiteEffects", ["sites", "species", "b", "a", "D"])


def cooccurrence_counts(Y, occupied_only=False):
    """Co-occurrence counts.

    Returns ``(K, m, N)``: the species by species matrix of co-occurrence counts, the number
    of sites at which each species was found, and the number of sites. With
    ``occupied_only=True`` sites at which none of the species was found are left out. If
    ``Y`` is a DataFrame, ``K`` and ``m`` are labelled by its columns.
    """
    K, m, N = setup().counts(_presence(Y), occupied_only=bool(occupied_only))
    labels = _species(Y)
    m = _np(m)
    return Counts(_square(_np(K), labels), m if labels is None else pd.Series(m, index=labels), int(N))


def affinity_mle(k, m_i, m_j, N):
    """Maximum likelihood estimate of the affinity (log odds ratio) of two species.

    The species occupy ``m_i`` and ``m_j`` of ``N`` sites and co-occur at ``k`` of them. The
    estimate is infinite when ``k`` is the largest or smallest value possible. Arguments may
    be arrays, which are broadcast.
    """
    scalar, args = _vector(k, m_i, m_j, N)
    return _result(setup().mle(*args), scalar)


def affinity_map(k, m_i, m_j, N, prior_sd=3.0):
    """Maximum a posteriori estimate of affinity under the prior Normal(0, prior_sd)."""
    scalar, args = _vector(k, m_i, m_j, N)
    return _result(setup().map_estimate(*args, prior_sd=float(prior_sd)), scalar)


def affinity_posterior(k, m_i, m_j, N, prior_sd=4.0, level=0.95, npoints=1001):
    """Posterior summary of affinity under the prior Normal(0, prior_sd).

    The posterior is integrated numerically (no sampling). Returns a DataFrame with one row
    per pair and columns ``median``, ``mean``, ``lower`` and ``upper`` (central credible
    interval of probability ``level``), ``pd`` (probability of direction: posterior
    probability that the affinity has the sign of its median) and ``p_positive``.
    """
    _, args = _vector(k, m_i, m_j, N)
    r = setup().posterior(*args, prior_sd=float(prior_sd), level=float(level), npoints=int(npoints))
    cols = ["median", "mean", "lower", "upper", "pd", "p_positive"]
    return pd.DataFrame({c: _np(v) for c, v in zip(cols, r)})


def pairwise_affinity(Y, prior_sd=4.0, level=0.95, site_weights=False, metric="affinity"):
    """Affinity of every pair of species, each pair analysed on its own.

    ``site_weights`` is ``False``, ``True`` or an array of site effects (one per row of
    ``Y``). With ``True``, sites are weighted by their effects in a logistic model with
    species and site effects, estimated with each pair left out, so that differences in
    richness between sites do not appear as associations. ``metric`` is ``"affinity"`` or
    ``"jaccard"`` (summaries of the logit of the Jaccard index).

    Returns species by species matrices ``median``, ``lower``, ``upper``, ``pd`` and
    ``p_positive`` (see ``affinity_posterior``), NaN on the diagonal and for species found at
    none or all of the sites. DataFrames if ``Y`` is a DataFrame.
    """
    if not isinstance(site_weights, bool):
        site_weights = np.asarray(site_weights, dtype=float)
    r = setup().pairwise(_presence(Y), prior_sd=float(prior_sd), level=float(level),
                         site_weights=site_weights, metric=str(metric))
    labels = _species(Y)
    return PairwiseAffinity(*(_square(_np(x), labels) for x in r))


def fnch_logpdf(k, m_i, m_j, N, alpha):
    """Log probability of ``k`` co-occurrences given affinity ``alpha`` (Fisher's noncentral
    hypergeometric distribution on the log odds scale)."""
    scalar, args = _vector(k, m_i, m_j, N, alpha)
    return _result(setup().fnch_logpdf(*args), scalar)


def fnch_mean(m_i, m_j, N, alpha):
    """Expected number of co-occurrences given affinity ``alpha``."""
    scalar, args = _vector(m_i, m_j, N, alpha)
    return _result(setup().fnch_mean(*args), scalar)


def trait_distance(x):
    """Pairwise distances in one trait.

    ``abs(x[i] - x[j])`` for a numeric trait; for any other trait 0 where two species share a
    value and 1 where they differ.
    """
    return _np(setup().trait_distance(_trait(x)))


def gower_distance(*traits):
    """Gower's distance over one or more traits (one value per species each), or over the
    columns of a single DataFrame. Numeric traits are scaled by their range and the result is
    averaged over traits."""
    if len(traits) == 1 and isinstance(traits[0], pd.DataFrame):
        traits = [traits[0][c] for c in traits[0].columns]
    return _np(setup().gower_distance(*(_trait(x) for x in traits)))


def standardise(X, g=None):
    """Centre and scale the off-diagonal elements of a square matrix to mean 0 and standard
    deviation 1, with the diagonal set to 0. With ``g`` (the group of each species) this is
    done within groups and elements between groups are set to 0."""
    X = np.asarray(X, dtype=float)
    kw = {} if g is None else {"g": _labels(g)}
    return _np(setup().standardise(X, **kw))


def swap_null(Y, n_trades=None, seed=None):
    """Random presence/absence matrix with the same row and column totals as ``Y`` but no
    associations beyond those the totals impose (curveball algorithm, Strona et al. 2014)."""
    kw = {}
    if n_trades is not None:
        kw["n_trades"] = int(n_trades)
    if seed is not None:
        kw["seed"] = int(seed)
    Z = _np(setup().swap_null(_presence(Y), **kw))
    return pd.DataFrame(Z, index=Y.index, columns=Y.columns) if isinstance(Y, pd.DataFrame) else Z


def site_effects(Y, prior_sd=10.0):
    """Fit ``logit P(Y[n, s] = 1) = a[s] + b[n]`` with Normal(0, prior_sd) priors.

    Sites and species present nowhere or everywhere are removed first. Returns the (0-based)
    indices of the ``sites`` and ``species`` kept, their effects ``b`` and ``a``, and ``D``,
    the curvature of the log posterior for each site effect.
    """
    sites, species, b, a, D = setup().site_effects(_presence(Y), prior_sd=float(prior_sd))
    return SiteEffects(_np(sites) - 1, _np(species) - 1, _np(b), _np(a), _np(D))


def cooccurrence_regression(X, Y, g=None, n_iter=1000, n_chains=4, *, alpha_sd=None,
                            beta_sd=None, lambda_sd=None, mu_sd=None, sigma_params=None,
                            pooled=None, group_range=None, strata=None, site_weights=None,
                            metric="affinity", n_adapts=None, delta=None, adtype=None,
                            drop_lambda=False, seed=None):
    """Bayesian dyadic regression of co-occurrence affinity.

    For each pair of species ``i > j``, with prevalences ``m[i]``, ``m[j]`` and ``k[i, j]``
    co-occurrences at ``N`` sites::

        k[i, j] ~ FNCH(m[i], m[j], N, alpha + lambda[i] + lambda[j] + sum_p beta[p] * X_p[i, j])
        alpha ~ Normal(0, alpha_sd), beta[p] ~ Normal(0, beta_sd), lambda[i] ~ Normal(0, lambda_sd)

    where FNCH is Fisher's noncentral hypergeometric distribution with log odds ratio given
    by the linear predictor.

    Parameters
    ----------
    X : array or list of arrays
        Species by species matrix of an explanatory variable (only the lower triangle is
        used), or a list of them.
    Y : array or DataFrame
        Presence/absence, sites in rows and species in columns.
    g : sequence, optional
        Group of each species. Only pairs within groups enter the likelihood; by default
        each group ``q`` has coefficients ``beta_mu[p] + beta_sigma[p] * beta[p, q]`` with
        ``beta[p, q] ~ Normal(0, 1)``, ``beta_mu[p] ~ Normal(0, mu_sd)`` and
        ``beta_sigma[p] ~ Gamma(shape=sigma_params[0], scale=sigma_params[1])``; with
        ``pooled=True`` the groups share one set of coefficients.
    n_iter, n_chains : int
        Draws per chain after adaptation, and number of chains (run one after another).
    alpha_sd, beta_sd, lambda_sd : float
        Prior standard deviations (default 4, 1, 1).
    mu_sd, sigma_params, pooled, group_range :
        Options of the group model (default 1, (2, 0.1), False, False). With
        ``group_range=True`` co-occurrence within a group is judged only at the sites where
        at least one of its species occurs.
    strata : sequence, optional
        Type of each site. Co-occurrences are counted within each type; the types share the
        coefficients and each has its own ``alpha[l]`` and ``lambda[i, l]``. Not with ``g``.
    site_weights : bool or array, optional
        Site-weighted likelihood (see ``pairwise_affinity``). Not with ``g`` or ``strata``.
    metric : {"affinity", "jaccard"}
    n_adapts, delta : int, float
        Adaptation steps and target acceptance rate (default 100, 0.65).
    adtype : {"forwarddiff", "mooncake"}, optional
        Automatic differentiation backend. By default ForwardDiff for at most 20
        parameters, Mooncake otherwise.
    drop_lambda : bool
        Leave the species effects out of the result.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    DataFrame with columns ``chain`` and ``draw`` (0-based) and one column per parameter:
    ``alpha``, ``beta[p]`` (1-based, in the order of ``X``), ``lambda[i]`` (in the order of
    the columns of ``Y``; by group then column when ``g`` is given) and, with ``g``,
    ``beta_mu[p]``, ``beta_sigma[p]`` and ``beta[p,q]`` (groups in sorted order). Use
    ``to_inference_data`` for ArviZ.
    """
    P = _presence(Y)
    S = P.shape[1]
    Xs = list(X) if isinstance(X, (list, tuple)) else [X]
    Xs = [np.asarray(x, dtype=float) for x in Xs]
    for p, x in enumerate(Xs):
        if x.shape != (S, S):
            raise ValueError(f"explanatory variable {p} has shape {x.shape}; expected ({S}, {S}) "
                             "to match the columns of Y")
    X3 = np.stack(Xs, axis=-1)

    if g is None:
        if any(v is not None for v in (mu_sd, sigma_params, pooled, group_range)):
            raise ValueError("mu_sd, sigma_params, pooled and group_range apply only when g is given")
    else:
        if len(g) != S:
            raise ValueError("g must have one entry per species (column of Y)")
        if strata is not None or site_weights is not None:
            raise ValueError("strata and site_weights cannot be used with g")
    if strata is not None and len(strata) != P.shape[0]:
        raise ValueError("strata must have one entry per site (row of Y)")
    if site_weights is not None and not isinstance(site_weights, bool):
        site_weights = np.asarray(site_weights, dtype=float)

    kw = dict(g=_labels(g), seed=None if seed is None else int(seed), drop_lambda=bool(drop_lambda),
              alpha_sd=alpha_sd, beta_sd=beta_sd, lambda_sd=lambda_sd, mu_sd=mu_sd,
              sigma_params=None if sigma_params is None else [float(s) for s in sigma_params],
              pooled=pooled, group_range=group_range, strata=_labels(strata),
              site_weights=site_weights, metric=str(metric),
              n_adapts=None if n_adapts is None else int(n_adapts), delta=delta, adtype=adtype)
    kw = {k: v for k, v in kw.items() if v is not None}
    names, draws, chain, iteration, divergent = setup().regression(X3, P, int(n_iter), int(n_chains), **kw)

    out = pd.DataFrame(_np(draws), columns=list(names))
    out.insert(0, "draw", _np(iteration) - 1)
    out.insert(0, "chain", _np(chain) - 1)
    if divergent > 0:
        warnings.warn(f"{divergent} divergent transitions")
    return out


def to_inference_data(draws):
    """Convert the result of ``cooccurrence_regression`` to an ArviZ InferenceData."""
    import arviz as az

    draws = draws.sort_values(["chain", "draw"])
    n_chains = draws["chain"].nunique()
    params = [c for c in draws.columns if c not in ("chain", "draw")]
    return az.from_dict(posterior={p: draws[p].to_numpy().reshape(n_chains, -1) for p in params})
