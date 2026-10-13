# Fisher's noncentral hypergeometric (FNCH) co-occurrence likelihood on the log-odds scale.
#
# For two species with prevalences mᵢ and mⱼ across N sites, the number of sites k at which
# they co-occur has, conditional on both prevalences,
#
#     P(K = k | α) = c(k) exp(α k) / Z(α),    c(u) = binomial(mᵢ, u) * binomial(N - mᵢ, mⱼ - u),
#     Z(α) = Σᵤ c(u) exp(α u),                max(0, mᵢ + mⱼ - N) ≤ u ≤ min(mᵢ, mⱼ),
#
# where α is the log odds ratio (the "affinity" of Mainali et al. 2022). This is a
# one-parameter exponential family in α, so
#
#     d/dα log P(K = k | α) = k - E[K | α].
#
# Terms are accumulated outwards from the observed k using
# c(u+1)/c(u) = (mᵢ - u)(mⱼ - u) / ((u + 1)(N - mᵢ - mⱼ + u + 1)), rescaled to the largest term so
# that nothing overflows, with a fully logarithmic fallback when exp(α) is extreme. There is no
# mode formula to fail, so the functions are finite and exception-free for every real α.
#
# This file depends only on Base.

# |α| beyond this is treated as infinite, so that sums of the form u*α cannot overflow.
const _FNCH_ALPHA_MAX = 1e100

@inline _fnch_lo(mi, mj, N) = max(0, mi + mj - N)
@inline _fnch_hi(mi, mj) = min(mi, mj)

function _check_fnch_args(mi, mj, N)
    N >= 0 || throw(ArgumentError("number of sites N must be non-negative, got $N"))
    0 <= mi <= N || throw(ArgumentError("prevalence m_i must lie in 0:N, got $mi with N = $N"))
    0 <= mj <= N || throw(ArgumentError("prevalence m_j must lie in 0:N, got $mj with N = $N"))
    return nothing
end

# log(c(u+1)/c(u)); valid for lo ≤ u < hi
@inline function _fnch_logratio(u, mi, mj, N)
    return log((float(mi - u) * float(mj - u)) / (float(u + 1) * float(N - mi - mj + u + 1)))
end

# Add a term with log-weight `t` and offset `d = u - k` to running sums that are held
# relative to the largest log-weight seen so far (`mx`).
@inline function _fnch_accumulate(t, d, mx, s0, s1)
    if t > mx
        c = exp(mx - t)
        return t, s0 * c + 1, s1 * c + d, false
    else
        w = exp(t - mx)
        # the sequence is log-concave: once a term is negligible, all later ones are too
        return mx, s0 + w, s1 + w * d, w < eps(typeof(w)) * s0
    end
end

"""
    fnch_logpdf_and_gradient(k, m_i, m_j, N, α) -> (logp, dlogp_dα)

Log-probability of observing `k` co-occurrences between two species with prevalences `m_i`
and `m_j` across `N` sites under Fisher's noncentral hypergeometric distribution with log
odds ratio (affinity) `α`, together with its derivative with respect to `α`.

The derivative is `k - E[K | α]`, the observed minus the expected number of co-occurrences.
Both values come from a single pass over the support.

Never throws for valid data: `α = ±Inf` returns the limiting point-mass values, `α = NaN`
and `k` outside the support return `(-Inf, 0)`.

See also [`fnch_logpdf`](@ref), [`fnch_mean`](@ref).
"""
function fnch_logpdf_and_gradient(k::Integer, mi::Integer, mj::Integer, N::Integer, α::Real)
    _check_fnch_args(mi, mj, N)
    T = float(typeof(α))
    lo = _fnch_lo(mi, mj, N)
    hi = _fnch_hi(mi, mj)
    (lo <= k <= hi) || return T(-Inf), zero(T)
    isnan(α) && return T(-Inf), zero(T)
    if abs(α) > _FNCH_ALPHA_MAX
        # all mass on one end of the support
        edge = α > 0 ? hi : lo
        return (k == edge ? zero(T) : T(-Inf)), T(k - edge)
    end

    a = T(α)
    ω = exp(a)
    ω² = ω * ω
    if isfinite(ω²) && isfinite(inv(ω²))
        return _fnch_sum(k, mi, mj, N, lo, hi, ω)
    else
        return _fnch_sum_log(k, mi, mj, N, lo, hi, a)
    end
end

# The largest term has grown by a factor f > 1: fold it into P, moving P into L = log(scale)
# before it can overflow.
@inline function _fnch_rescale(f, P, L)
    f > 1e8 && return P, L + log(f)
    P *= f
    P > 1e15 && return one(P), L + log(P)
    return P, L
end

# Running sums are held relative to the largest term seen so far. `f` is the current term on
# that scale, `P` the factor by which the scale has grown since the term u = k (with `L`
# taking over as log(P) whenever P gets large). Returns (log P(K = k), k - E[K]).
function _fnch_sum(k, mi, mj, N, lo, hi, ω::T) where {T}
    c = N - mi - mj
    s0 = one(T)     # Σ term(u) / largest term
    s1 = zero(T)    # Σ (u - k) term(u) / largest term
    P = one(T)
    L = zero(T)

    f = one(T)
    for u in k:(hi - 1)             # term u + 1
        f *= ω * T(((mi - u) * (mj - u)) / ((u + 1) * (c + u + 1)))
        d = u + 1 - k
        if f > 1
            s0 = s0 / f + 1
            s1 = s1 / f + d
            P, L = _fnch_rescale(f, P, L)
            f = one(T)
        else
            s0 += f
            s1 += f * d
            # the sequence is log-concave: once a term is negligible, all later ones are too
            f < eps(T) * s0 && break
        end
    end

    f = exp(-L) / P                 # the term u = k on the current scale
    for u in k:-1:(lo + 1)          # term u - 1
        f *= T((u * (c + u)) / ((mi - u + 1) * (mj - u + 1))) / ω
        d = u - 1 - k
        if f > 1
            s0 = s0 / f + 1
            s1 = s1 / f + d
            P, L = _fnch_rescale(f, P, L)
            f = one(T)
        else
            s0 += f
            s1 += f * d
            f < eps(T) * s0 && break
        end
    end

    return -(L + log(P) + log(s0)), -s1 / s0
end

# Same sums with every term kept as a logarithm. Slower, but valid when exp(α) or its
# reciprocal is too large to multiply safely.
function _fnch_sum_log(k, mi, mj, N, lo, hi, a::T) where {T}
    mx = zero(T)    # largest log-weight so far, relative to the term u = k
    s0 = one(T)     # Σ exp(t_u - mx)
    s1 = zero(T)    # Σ (u - k) exp(t_u - mx)

    t = zero(T)
    for u in k:(hi - 1)             # term u + 1
        t += T(_fnch_logratio(u, mi, mj, N)) + a
        mx, s0, s1, done = _fnch_accumulate(t, u + 1 - k, mx, s0, s1)
        done && break
    end
    t = zero(T)
    for u in k:-1:(lo + 1)          # term u - 1
        t -= T(_fnch_logratio(u - 1, mi, mj, N)) + a
        mx, s0, s1, done = _fnch_accumulate(t, u - 1 - k, mx, s0, s1)
        done && break
    end

    return -(mx + log(s0)), -s1 / s0
end

"""
    fnch_logpdf(k, m_i, m_j, N, α)

Log-probability of `k` co-occurrences between two species with prevalences `m_i` and `m_j`
across `N` sites, given log odds ratio (affinity) `α`:

    log P(K = k | α) = log c(k) + α k - log Σᵤ c(u) exp(α u),
    c(u) = binomial(m_i, u) * binomial(N - m_i, m_j - u).

Equivalent to `logpdf(FisherNoncentralHypergeometric(m_i, N - m_i, m_j, exp(α)), k)` from
Distributions.jl, but evaluated in log space so that it stays finite and accurate for
extreme `α` and for `k` far in the tails, and never throws during sampling.

Derivative rules with respect to `α` are provided for ChainRules-compatible automatic
differentiation, ForwardDiff and Mooncake, using the closed form `k - E[K | α]`.
"""
fnch_logpdf(k::Integer, mi::Integer, mj::Integer, N::Integer, α::Real) =
    first(fnch_logpdf_and_gradient(k, mi, mj, N, α))

"""
    fnch_mean(m_i, m_j, N, α)

Expected number of co-occurrences `E[K | α]` between two species with prevalences `m_i` and
`m_j` across `N` sites, given log odds ratio (affinity) `α`. At `α = 0` this is the
hypergeometric expectation `m_i * m_j / N`.
"""
function fnch_mean(mi::Integer, mj::Integer, N::Integer, α::Real)
    _check_fnch_args(mi, mj, N)
    lo = _fnch_lo(mi, mj, N)
    hi = _fnch_hi(mi, mj)
    T = float(typeof(α))
    isnan(α) && return T(NaN)
    # start from the null expectation so that few terms are far from the anchor
    k0 = N == 0 ? 0 : clamp(round(Int, mi * (mj / N)), lo, hi)
    _, g = fnch_logpdf_and_gradient(k0, mi, mj, N, α)
    return k0 - g
end
