"""
    LogOddsFNCH(m_i, m_j, N, α)

Fisher's noncentral hypergeometric distribution of the number of co-occurrences between two
species with prevalences `m_i` and `m_j` across `N` sites, parameterised by the log odds
ratio (affinity) `α`.

Equivalent to `FisherNoncentralHypergeometric(m_i, N - m_i, m_j, exp(α))` from
Distributions.jl. `logpdf` is evaluated by [`fnch_logpdf`](@ref), so it is stable for extreme
`α` and has closed-form derivative rules. Can be used directly in a Turing model:

    k[i, j] ~ LogOddsFNCH(m[i], m[j], N, α)
"""
struct LogOddsFNCH{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    m_i::Int
    m_j::Int
    N::Int
    α::T
    function LogOddsFNCH(m_i::Integer, m_j::Integer, N::Integer, α::T) where {T<:Real}
        _check_fnch_args(m_i, m_j, N)
        return new{T}(m_i, m_j, N, α)
    end
end

Distributions.params(d::LogOddsFNCH) = (d.m_i, d.m_j, d.N, d.α)
Base.minimum(d::LogOddsFNCH) = _fnch_lo(d.m_i, d.m_j, d.N)
Base.maximum(d::LogOddsFNCH) = _fnch_hi(d.m_i, d.m_j)
Distributions.insupport(d::LogOddsFNCH, k::Real) = isinteger(k) && minimum(d) <= k <= maximum(d)

function Distributions.logpdf(d::LogOddsFNCH, k::Real)
    isinteger(k) || return float(typeof(d.α))(-Inf)
    return fnch_logpdf(Int(k), d.m_i, d.m_j, d.N, d.α)
end

Distributions.mean(d::LogOddsFNCH) = fnch_mean(d.m_i, d.m_j, d.N, d.α)

# inverse-CDF sampling, walking up the support in log space
function Base.rand(rng::Random.AbstractRNG, d::LogOddsFNCH)
    lo, hi = minimum(d), maximum(d)
    isnan(d.α) && throw(DomainError(d.α, "cannot sample from LogOddsFNCH with α = NaN"))
    abs(d.α) > _FNCH_ALPHA_MAX && return d.α > 0 ? hi : lo
    u = rand(rng)
    k = lo
    lp = fnch_logpdf(k, d.m_i, d.m_j, d.N, d.α)
    c = exp(lp)
    while c < u && k < hi
        lp += _fnch_logratio(k, d.m_i, d.m_j, d.N) + d.α
        k += 1
        c += exp(lp)
    end
    return k
end
