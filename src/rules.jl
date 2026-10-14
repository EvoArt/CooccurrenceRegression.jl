# Derivative rules for `fnch_logpdf` with respect to the log odds ratio α, using the closed
# form d/dα log P(K = k | α) = k - E[K | α]. With these, automatic differentiation never has
# to trace the summation over the support.

function ChainRulesCore.frule(
    (_, _, _, _, _, Δα), ::typeof(fnch_logpdf), k::Integer, mi::Integer, mj::Integer, N::Integer, α::Real
)
    lp, g = fnch_logpdf_and_gradient(k, mi, mj, N, α)
    return lp, g * Δα
end

function ChainRulesCore.rrule(
    ::typeof(fnch_logpdf), k::Integer, mi::Integer, mj::Integer, N::Integer, α::Real
)
    lp, g = fnch_logpdf_and_gradient(k, mi, mj, N, α)
    function fnch_logpdf_pullback(Δ)
        nt = ChainRulesCore.NoTangent()
        return nt, nt, nt, nt, nt, g * ChainRulesCore.unthunk(Δ)
    end
    return lp, fnch_logpdf_pullback
end

# Mooncake (reverse mode)
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(fnch_logpdf),Int,Int,Int,Int,Float64}

# ForwardDiff
function fnch_logpdf(
    k::Integer, mi::Integer, mj::Integer, N::Integer, α::ForwardDiff.Dual{Tg}
) where {Tg}
    lp, g = fnch_logpdf_and_gradient(k, mi, mj, N, ForwardDiff.value(α))
    return ForwardDiff.Dual{Tg}(lp, g * ForwardDiff.partials(α))
end

# The same for the site-weighted likelihood, d/dα = k - E[K | α].

function ChainRulesCore.frule(
    (_, _, _, _, Δα), ::typeof(weighted_logpdf), k::Integer, lo::Integer, logc::AbstractVector{<:Real}, α::Real
)
    lp, g = weighted_logpdf_and_gradient(k, lo, logc, α)
    return lp, g * Δα
end

function ChainRulesCore.rrule(
    ::typeof(weighted_logpdf), k::Integer, lo::Integer, logc::AbstractVector{<:Real}, α::Real
)
    lp, g = weighted_logpdf_and_gradient(k, lo, logc, α)
    function weighted_logpdf_pullback(Δ)
        nt = ChainRulesCore.NoTangent()
        return nt, nt, nt, nt, g * ChainRulesCore.unthunk(Δ)
    end
    return lp, weighted_logpdf_pullback
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(weighted_logpdf),Int,Int,Vector{Float64},Float64}

function weighted_logpdf(k::Integer, lo::Integer, logc::AbstractVector{<:Real}, α::ForwardDiff.Dual{Tg}) where {Tg}
    lp, g = weighted_logpdf_and_gradient(k, lo, logc, ForwardDiff.value(α))
    return ForwardDiff.Dual{Tg}(lp, g * ForwardDiff.partials(α))
end

# The same for the binomial (Jaccard) likelihood, d/dη = k - u J.

function ChainRulesCore.frule(
    (_, _, _, _, _, Δη), ::typeof(jaccard_logpdf), k::Integer, mi::Integer, mj::Integer, N::Integer, η::Real
)
    lp, g = jaccard_logpdf_and_gradient(k, mi, mj, N, η)
    return lp, g * Δη
end

function ChainRulesCore.rrule(
    ::typeof(jaccard_logpdf), k::Integer, mi::Integer, mj::Integer, N::Integer, η::Real
)
    lp, g = jaccard_logpdf_and_gradient(k, mi, mj, N, η)
    function jaccard_logpdf_pullback(Δ)
        nt = ChainRulesCore.NoTangent()
        return nt, nt, nt, nt, nt, g * ChainRulesCore.unthunk(Δ)
    end
    return lp, jaccard_logpdf_pullback
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(jaccard_logpdf),Int,Int,Int,Int,Float64}

function jaccard_logpdf(
    k::Integer, mi::Integer, mj::Integer, N::Integer, η::ForwardDiff.Dual{Tg}
) where {Tg}
    lp, g = jaccard_logpdf_and_gradient(k, mi, mj, N, ForwardDiff.value(η))
    return ForwardDiff.Dual{Tg}(lp, g * ForwardDiff.partials(η))
end
