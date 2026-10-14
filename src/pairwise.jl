# Affinity of a single pair of species: maximum likelihood, maximum a posteriori and full
# posterior summaries. Depends only on Base and fnch.jl.

"""
    cooccurrence_counts(Y; occupied_only=false) -> (K, m, N)

Summarise an `N × S` presence/absence matrix `Y` (rows are sites, columns are species; any
value greater than zero counts as a presence).

Returns the `S × S` matrix `K` of co-occurrence counts (`K[i, j]` is the number of sites at
which species `i` and `j` were both present), the vector `m` of prevalences (the number of
sites at which each species was present) and the number of sites `N`.

With `occupied_only=true`, sites at which none of the species was present are left out, so
that `N` is the number of sites occupied by at least one of them.
"""
function cooccurrence_counts(Y::AbstractMatrix; occupied_only::Bool=false)
    P = Int.(Y .> 0)
    if occupied_only
        P = P[vec(any(P .> 0, dims=2)), :]
    end
    N = size(P, 1)
    K = transpose(P) * P
    m = vec(sum(P, dims=1))
    return K, m, N
end

# Root of the decreasing function h(α) = k - E[K | α] - τ α, by bisection.
function _affinity_root(k, mi, mj, N, τ)
    h(α) = fnch_logpdf_and_gradient(k, mi, mj, N, α)[2] - τ * α
    lo, hi = -1.0, 1.0
    while h(lo) < 0
        lo *= 2
        lo < -1e4 && return -Inf
    end
    while h(hi) > 0
        hi *= 2
        hi > 1e4 && return Inf
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        if h(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

function _check_k(k, mi, mj, N)
    _check_fnch_args(mi, mj, N)
    lo, hi = _fnch_lo(mi, mj, N), _fnch_hi(mi, mj)
    lo <= k <= hi || throw(ArgumentError("k = $k is outside the possible range $lo:$hi for m_i = $mi, m_j = $mj, N = $N"))
    return lo, hi
end

"""
    affinity_mle(k, m_i, m_j, N)

Maximum likelihood estimate of the affinity (log odds ratio) `α` of two species that occupy
`m_i` and `m_j` of `N` sites and co-occur at `k` of them: the value at which the expected
number of co-occurrences equals `k` (Mainali et al. 2022).

The estimate is `Inf` when `k` is the largest value possible given the prevalences and `-Inf`
when it is the smallest. It is `NaN` when only one value of `k` is possible, in which case
the data say nothing about `α`. See [`affinity_map`](@ref) for an estimate that is always
finite.
"""
function affinity_mle(k::Integer, mi::Integer, mj::Integer, N::Integer)
    lo, hi = _check_k(k, mi, mj, N)
    lo == hi && return NaN
    k == hi && return Inf
    k == lo && return -Inf
    return _affinity_root(k, mi, mj, N, 0.0)
end

"""
    affinity_map(k, m_i, m_j, N; prior_sd=3.0)

Maximum a posteriori estimate of the affinity `α` of two species that occupy `m_i` and `m_j`
of `N` sites and co-occur at `k` of them, under the prior `α ~ Normal(0, prior_sd)`. Always
finite.
"""
function affinity_map(k::Integer, mi::Integer, mj::Integer, N::Integer; prior_sd::Real=3.0)
    _check_k(k, mi, mj, N)
    prior_sd > 0 || throw(ArgumentError("prior_sd must be positive"))
    return _affinity_root(k, mi, mj, N, 1 / prior_sd^2)
end

# Distance from the mode at which the log posterior has fallen by `drop`, in direction `dir`.
function _posterior_reach(logpost, a0, p0, dir, drop)
    step = 1.0
    while logpost(a0 + dir * step) > p0 - drop
        step *= 2
    end
    inner, outer = step / 2, step
    for _ in 1:30
        mid = (inner + outer) / 2
        if logpost(a0 + dir * mid) > p0 - drop
            inner = mid
        else
            outer = mid
        end
    end
    return outer
end

"""
    affinity_posterior(k, m_i, m_j, N; prior_sd=4.0, level=0.95, npoints=1001)

Posterior summary of the affinity `α` of two species that occupy `m_i` and `m_j` of `N` sites
and co-occur at `k` of them, under the prior `α ~ Normal(0, prior_sd)`.

Returns a named tuple with the posterior `median` and `mean`, the limits `lower` and `upper`
of the central credible interval of probability `level`, and `pd`, the probability of
direction: the posterior probability that `α` has the same sign as its median; and `p_positive`,
the posterior probability that `α > 0`.

The posterior is one-dimensional and is integrated numerically on a grid of `npoints` values,
so no sampling is involved and the result is deterministic.
"""
function affinity_posterior(k::Integer, mi::Integer, mj::Integer, N::Integer;
                            prior_sd::Real=4.0, level::Real=0.95, npoints::Integer=1001)
    _check_k(k, mi, mj, N)
    prior_sd > 0 || throw(ArgumentError("prior_sd must be positive"))
    0 < level < 1 || throw(ArgumentError("level must lie between 0 and 1"))
    npoints >= 3 || throw(ArgumentError("npoints must be at least 3"))
    τ = 1 / prior_sd^2
    logpost(α) = fnch_logpdf(k, mi, mj, N, α) - τ * α^2 / 2

    a0 = _affinity_root(k, mi, mj, N, τ)
    p0 = logpost(a0)
    left = a0 - _posterior_reach(logpost, a0, p0, -1, 40.0)
    right = a0 + _posterior_reach(logpost, a0, p0, 1, 40.0)

    grid = range(left, right; length=npoints)
    dens = [exp(logpost(α) - p0) for α in grid]
    cdf = zeros(npoints)
    for i in 2:npoints
        cdf[i] = cdf[i-1] + (dens[i-1] + dens[i]) / 2
    end
    total = cdf[end]
    cdf ./= total
    avg = sum(grid[i] * dens[i] for i in 1:npoints) / sum(dens)

    quant(p) = _interpolate(cdf, grid, p)
    below0 = left >= 0 ? 0.0 : right <= 0 ? 1.0 : _interpolate(grid, cdf, 0.0)
    return (median=quant(0.5), mean=avg, lower=quant((1 - level) / 2), upper=quant((1 + level) / 2),
            pd=max(below0, 1 - below0), p_positive=1 - below0)
end

# Linear interpolation of y at x, for increasing xs.
function _interpolate(xs, ys, x)
    i = searchsortedfirst(xs, x)
    i <= 1 && return float(ys[1])
    i > length(xs) && return float(ys[end])
    x0, x1 = xs[i-1], xs[i]
    x1 == x0 && return float(ys[i])
    return ys[i-1] + (ys[i] - ys[i-1]) * (x - x0) / (x1 - x0)
end

"""
    pairwise_affinity(Y; prior_sd=4.0, level=0.95)

Posterior summaries of the affinity of every pair of species in the `N × S` presence/absence
matrix `Y` (rows are sites, columns are species), each pair analysed on its own with the
prior `α ~ Normal(0, prior_sd)`.

Returns a named tuple of symmetric `S × S` matrices: `median`, `lower`, `upper` (credible
interval of probability `level`), `pd` (probability of direction) and `p_positive`
(probability that the affinity is positive). Entries are `NaN` on
the diagonal and for species present at none or all of the sites.

See [`affinity_posterior`](@ref) for a single pair.
"""
function pairwise_affinity(Y::AbstractMatrix; prior_sd::Real=4.0, level::Real=0.95)
    K, m, N = cooccurrence_counts(Y)
    S = length(m)
    med, lower, upper, pd, ppos = (fill(NaN, S, S) for _ in 1:5)
    for j in 1:S-1, i in j+1:S
        (0 < m[i] < N && 0 < m[j] < N) || continue
        post = affinity_posterior(K[i, j], m[i], m[j], N; prior_sd, level)
        med[i, j] = med[j, i] = post.median
        lower[i, j] = lower[j, i] = post.lower
        upper[i, j] = upper[j, i] = post.upper
        pd[i, j] = pd[j, i] = post.pd
        ppos[i, j] = ppos[j, i] = post.p_positive
    end
    return (median=med, lower=lower, upper=upper, pd=pd, p_positive=ppos)
end
