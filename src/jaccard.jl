# Binomial likelihood for the Jaccard index.
#
# Of the u = m_i + m_j - k sites at which at least one of two species occurs, both occur at k.
# With sites independent, k | u ~ Binomial(u, J), where J is the probability that both species
# occur at a site where at least one does (the Jaccard index of the pair). On the logit scale
# η = logit(J),
#
#     d/dη log P(k | u, η) = k - u J,
#
# observed minus expected co-occurrences, as for FNCH.
#
# This file depends only on Base and fnch.jl.

_log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function _logbinomial(n::Integer, k::Integer)
    k = min(k, n - k)
    s = 0.0
    for t in 1:k
        s += log((n - k + t) / t)
    end
    return s
end

"""
    jaccard_logpdf_and_gradient(k, m_i, m_j, N, η) -> (logp, dlogp_dη)

Log-probability of `k` co-occurrences between two species with prevalences `m_i` and `m_j`
across `N` sites, given `η = logit(J)`, where `J` is the Jaccard index: the probability that
both species occur at a site where at least one of them does. With `u = m_i + m_j - k` sites
holding at least one of the two, `k ~ Binomial(u, J)`. The derivative is `k - u * J`.

Never throws for valid data: `η = ±Inf` returns the limiting values, `η = NaN` and `k`
outside the possible range return `(-Inf, 0)`.
"""
function jaccard_logpdf_and_gradient(k::Integer, mi::Integer, mj::Integer, N::Integer, η::Real)
    _check_fnch_args(mi, mj, N)
    T = float(typeof(η))
    (_fnch_lo(mi, mj, N) <= k <= _fnch_hi(mi, mj)) || return T(-Inf), zero(T)
    isnan(η) && return T(-Inf), zero(T)
    u = mi + mj - k
    if abs(η) > _FNCH_ALPHA_MAX
        edge = η > 0 ? u : 0
        return (k == edge ? zero(T) : T(-Inf)), T(k - edge)
    end
    e = T(η)
    logJ = -_log1pexp(-e)
    log1mJ = -_log1pexp(e)
    lp = T(_logbinomial(u, k)) + k * logJ + (u - k) * log1mJ
    return lp, k - u * exp(logJ)
end

"""
    jaccard_logpdf(k, m_i, m_j, N, η)

Log-probability of `k` co-occurrences under the binomial model for the Jaccard index, with
`η = logit(J)`. See [`jaccard_logpdf_and_gradient`](@ref). Has closed-form derivative rules,
like [`fnch_logpdf`](@ref), and takes the same arguments, so either can be used as the
likelihood of the regression (`metric = :jaccard` in [`cooccurrence_regression`](@ref)).
"""
jaccard_logpdf(k::Integer, mi::Integer, mj::Integer, N::Integer, η::Real) =
    first(jaccard_logpdf_and_gradient(k, mi, mj, N, η))
