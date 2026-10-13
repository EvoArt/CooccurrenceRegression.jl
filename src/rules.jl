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
