# Site-weighted co-occurrence likelihood.
#
# With logit P(species s at site n) = a_s + b_n and a pairwise log odds ratio α, the number of
# co-occurrences k of species i and j, given the sites at which i occurs and the prevalence m_j
# of j, has
#
#     P(K = k | α) = c(k) exp(α k) / Σᵤ c(u) exp(α u),    c(u) = e_u(w⁺) e_{m_j - u}(w⁻),
#
# where w_n = exp(b_n), w⁺ and w⁻ hold the weights of the sites with and without species i, and
# e_r is the elementary symmetric polynomial of order r. The species effects cancel. Equal
# weights give FNCH. Conditioning on i treats the two species differently, so the pair's log
# likelihood is the average of the two directions.
#
# This file depends only on Base and fnch.jl.

# log e_r(w) for r = 0:rmax. F[r+1] is the mean, over subsets of size r, of the product of the
# weights scaled to mean 1. Each update is a convex combination, so F stays in range unless
# the weights are very uneven, in which case the recursion is redone in log space.
function _log_esp(w::AbstractVector{<:Real}, rmax::Integer)
    n = length(w)
    0 <= rmax <= n || throw(ArgumentError("order $rmax is outside 0:$n"))
    s = n == 0 ? 1.0 : sum(w) / n
    if s > 0
        F = zeros(rmax + 1)
        F[1] = 1.0
        for (t, x) in enumerate(w)
            x /= s
            for r in min(t, rmax):-1:1
                F[r+1] = ((t - r) * F[r+1] + r * x * F[r]) / t
            end
        end
        if all(f -> 0 < f < Inf, F)
            out = similar(F)
            lb = 0.0                     # log binomial(n, r)
            for r in 0:rmax
                r > 0 && (lb += log((n - r + 1) / r))
                out[r+1] = lb + log(F[r+1]) + r * log(s)
            end
            return out
        end
    end
    L = fill(-Inf, rmax + 1)
    L[1] = 0.0
    for (t, x) in enumerate(w)
        lx = log(x)
        for r in min(t, rmax):-1:1
            L[r+1] = _logaddexp(L[r+1], lx + L[r])
        end
    end
    return L
end

_logaddexp(a, b) = a == -Inf ? b : b == -Inf ? a : max(a, b) + log1p(exp(-abs(a - b)))

"""
    weighted_coefficients(y_i, m_j, w) -> (lo, logc)

Log coefficients of the site-weighted co-occurrence distribution of two species, conditional
on the sites at which species `i` occurs (`y_i`, a `Bool` vector over sites) and on the
prevalence `m_j` of species `j`, with site weights `w`:

    log c(u) = log e_u(w at sites with i) + log e_{m_j - u}(w at sites without i),

for `u = lo, lo + 1, …`, up to `min(m_i, m_j)`, shifted so that the largest is 0. With equal
weights these are the FNCH coefficients `binomial(m_i, u) * binomial(N - m_i, m_j - u)` up to
a constant. Use with [`weighted_logpdf`](@ref).
"""
function weighted_coefficients(yi::AbstractVector{Bool}, mj::Integer, w::AbstractVector{<:Real})
    N = length(yi)
    length(w) == N || throw(DimensionMismatch("w has length $(length(w)); expected $N, one weight per site"))
    all(x -> 0 <= x < Inf, w) || throw(ArgumentError("site weights must be finite and non-negative"))
    mi = count(yi)
    _check_fnch_args(mi, mj, N)
    lo, hi = _fnch_lo(mi, mj, N), _fnch_hi(mi, mj)
    Lin = _log_esp(w[yi], hi)
    Lout = _log_esp(w[.!yi], mj - lo)
    logc = [Lin[u+1] + Lout[mj-u+1] for u in lo:hi]
    logc .-= maximum(logc)
    return lo, logc
end

"""
    weighted_logpdf_and_gradient(k, lo, logc, α) -> (logp, dlogp_dα)

Log-probability of `k` co-occurrences under the exponential family
`P(K = u) ∝ exp(logc[u - lo + 1] + α u)`, and its derivative `k - E[K | α]` with respect to
`α`. With coefficients from [`weighted_coefficients`](@ref) this is the site-weighted
co-occurrence distribution. Never throws: `k` outside the support and `α = NaN` give
`(-Inf, 0)`.
"""
function weighted_logpdf_and_gradient(k::Integer, lo::Integer, logc::AbstractVector{<:Real}, α::Real)
    T = float(promote_type(typeof(α), eltype(logc)))
    hi = lo + length(logc) - 1
    (lo <= k <= hi) || return T(-Inf), zero(T)
    isnan(α) && return T(-Inf), zero(T)
    if abs(α) > _FNCH_ALPHA_MAX
        edge = α > 0 ? hi : lo
        return (k == edge ? zero(T) : T(-Inf)), T(k - edge)
    end
    a = T(α)
    mx = T(-Inf)
    for (q, c) in enumerate(logc)
        mx = max(mx, c + a * (q - 1))
    end
    s0 = zero(T)
    s1 = zero(T)
    for (q, c) in enumerate(logc)
        e = exp(c + a * (q - 1) - mx)
        s0 += e
        s1 += e * (q - 1)
    end
    d = k - lo
    return T(logc[d+1]) + a * d - mx - log(s0), d - s1 / s0
end

"""
    weighted_logpdf(k, lo, logc, α)

Log-probability of `k` co-occurrences under the site-weighted distribution with coefficients
`(lo, logc)` from [`weighted_coefficients`](@ref) and log odds ratio `α`. Has closed-form
derivative rules, like [`fnch_logpdf`](@ref).
"""
weighted_logpdf(k::Integer, lo::Integer, logc::AbstractVector{<:Real}, α::Real) =
    first(weighted_logpdf_and_gradient(k, lo, logc, α))

# Species and site effects

_logistic(x) = 1 / (1 + exp(-x))

# Root of Σₙ σ(x + c[n]) + τ x = target, increasing in x: Newton steps, kept inside the
# bracket found so far and at most 10 long.
function _solve_effect(c, target, x, τ)
    lo, hi = -Inf, Inf
    for _ in 1:200
        v = τ * x
        d = τ
        for cn in c
            p = _logistic(x + cn)
            v += p
            d += p * (1 - p)
        end
        abs(v - target) < 1e-10 && break
        v < target ? (lo = x) : (hi = x)
        x2 = x - (v - target) / d
        if !(lo < x2 < hi)
            x2 = isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 : x2
        end
        x = x + clamp(x2 - x, -10.0, 10.0)
    end
    return x
end

# Rows and columns that carry information: drop species present at none or all of the
# remaining sites and sites holding none or all of the remaining species, until none are left
# to drop.
function _informative(Y::AbstractMatrix{Bool})
    rows, cols = collect(axes(Y, 1)), collect(axes(Y, 2))
    while true
        m = vec(sum(view(Y, rows, cols), dims=1))
        keepc = 0 .< m .< length(rows)
        cols = cols[keepc]
        r = vec(sum(view(Y, rows, cols), dims=2))
        keepr = 0 .< r .< length(cols)
        rows = rows[keepr]
        (all(keepc) && all(keepr)) && return rows, cols
        (isempty(rows) || isempty(cols)) && return rows, cols
    end
end

"""
    site_effects(Y; prior_sd=10.0, tol=1e-8, maxiter=1000)

Fit `logit P(Y[n, s] = 1) = a[s] + b[n]`, a logistic model with a species effect `a[s]` and a
site effect `b[n]`, to the `N × S` presence/absence matrix `Y` (rows are sites, columns are
species), with weak `Normal(0, prior_sd)` priors on all effects so that the estimates exist
for perfectly nested matrices.

Sites at which none or all of the species occur, and species present at none or all of the
sites, carry no information about the effects and are removed first (repeatedly, until none
remain). Returns a named tuple with the indices of the remaining `sites` and `species`, their
effects `b` and `a` (posterior modes), and `D`, the curvature `Σₛ p(1 - p) + 1/prior_sd²` of
the log posterior for each site effect, which [`site_weighted_pairs`](@ref) uses to remove a
pair from the fit.
"""
function site_effects(Y::AbstractMatrix; prior_sd::Real=10.0, tol::Real=1e-8, maxiter::Integer=1000)
    P = Y .> 0
    sites, species = _informative(P)
    Z = P[sites, species]
    N, S = size(Z)
    τ = 1 / prior_sd^2
    m = vec(sum(Z, dims=1))
    r = vec(sum(Z, dims=2))
    a = [log(m[s] / (N - m[s])) for s in 1:S]
    b = zeros(N)
    for _ in 1:maxiter
        for s in 1:S
            a[s] = _solve_effect(b, m[s], a[s], τ)
        end
        change = 0.0
        for n in 1:N
            bn = _solve_effect(a, r[n], b[n], τ)
            change = max(change, abs(bn - b[n]))
            b[n] = bn
        end
        change < tol && break
    end
    D = [τ + sum(p -> p * (1 - p), _logistic.(a .+ b[n])) for n in 1:N]
    return (sites=sites, species=species, b=b, a=a, D=D)
end

"""
    WeightedPair

Data for one pair of species under the site-weighted model: the species `i > j` (columns of
`Y`), the number of co-occurrences `k`, and the coefficients `(lo_i, logc_i)` conditional on
the sites of species `i` and `(lo_j, logc_j)` conditional on those of species `j`.
"""
struct WeightedPair
    i::Int
    j::Int
    k::Int
    lo_i::Int
    logc_i::Vector{Float64}
    lo_j::Int
    logc_j::Vector{Float64}
end

"""
    weighted_logpdf(p::WeightedPair, α)

Log likelihood of the affinity `α` of a pair under the site-weighted model: the average of the
log-probabilities of its co-occurrence count conditional on the sites of each species in turn.
"""
weighted_logpdf(p::WeightedPair, α::Real) =
    (weighted_logpdf(p.k, p.lo_i, p.logc_i, α) + weighted_logpdf(p.k, p.lo_j, p.logc_j, α)) / 2

function _weighted_logpdf_and_gradient(p::WeightedPair, α::Real)
    l1, g1 = weighted_logpdf_and_gradient(p.k, p.lo_i, p.logc_i, α)
    l2, g2 = weighted_logpdf_and_gradient(p.k, p.lo_j, p.logc_j, α)
    return (l1 + l2) / 2, (g1 + g2) / 2
end

"""
    site_weighted_pairs(Y; b=nothing, prior_sd=10.0) -> Vector{WeightedPair}

Data for the site-weighted co-occurrence model of every informative pair of species in the
`N × S` presence/absence matrix `Y` (rows are sites, columns are species).

By default the site effects are estimated with [`site_effects`](@ref). The richness of a site
includes the two species whose co-occurrence is being assessed, which biases affinity
downwards, so for each pair the site effects are re-estimated with that pair removed, by one
Newton step from the fit to all species. Sites holding none or all of the species, and species
present at none or all of the sites, are removed first: the first have weight zero and the
second weight infinity in the limit, so they carry no information about any pair.

Alternatively, known site effects can be supplied as `b`, one per row of `Y`; all sites are
then used with weights `exp.(b)`, and no pair is left out.
"""
function site_weighted_pairs(Y::AbstractMatrix; b=nothing, prior_sd::Real=10.0)
    P = Y .> 0
    pairs = WeightedPair[]
    if b !== nothing
        length(b) == size(P, 1) || throw(DimensionMismatch(
            "b has length $(length(b)); expected one site effect for each of the $(size(P, 1)) sites (rows) of Y"))
        w = exp.(b .- maximum(b))
        N = size(P, 1)
        m = vec(sum(P, dims=1))
        for j in 1:size(P, 2)-1, i in j+1:size(P, 2)
            (0 < m[i] < N && 0 < m[j] < N) || continue
            push!(pairs, _weighted_pair(P, i, j, m, w))
        end
        return pairs
    end
    fit = site_effects(P; prior_sd)
    Z = P[fit.sites, :]
    m = vec(sum(Z, dims=1))
    p = _logistic.(fit.b .+ fit.a')                 # sites × informative species
    for y in 2:length(fit.species), x in 1:y-1
        i, j = fit.species[y], fit.species[x]
        pi_, pj = view(p, :, y), view(p, :, x)
        resid = (Z[:, i] .- pi_) .+ (Z[:, j] .- pj)
        D = fit.D .- pi_ .* (1 .- pi_) .- pj .* (1 .- pj)
        bl = fit.b .- resid ./ D
        w = exp.(bl .- maximum(bl))
        push!(pairs, _weighted_pair(Z, i, j, m, w))
    end
    return pairs
end

function _weighted_pair(Z, i, j, m, w)
    yi, yj = Z[:, i], Z[:, j]
    lo_i, logc_i = weighted_coefficients(yi, m[j], w)
    lo_j, logc_j = weighted_coefficients(yj, m[i], w)
    return WeightedPair(i, j, count(yi .& yj), lo_i, logc_i, lo_j, logc_j)
end
