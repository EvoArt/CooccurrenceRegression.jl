# Entry points for the R and Python wrappers. Arguments arrive as R or Python objects
# (PyArray, PyList, Int32 vectors, ...) and are copied into plain Julia types; results are
# tuples of arrays, which both RCall and PythonCall convert without help.
module Interop

const CR = parentmodule(@__MODULE__)
import ..ADTypes, ..Random

_ints(x) = Int.(vec(collect(x)))
_floats(x) = Float64.(vec(collect(x)))
_labels(x) = string.(vec(collect(x)))
_presence(Y) = Matrix{Bool}(collect(Y) .> 0)
_trait(x) = (v = vec(collect(x)); all(e -> e isa Real, v) ? Float64.(v) : string.(v))
_site_weights(w::Bool) = w
_site_weights(w) = _floats(w)

function _adtype(x)
    s = lowercase(string(x))
    s == "forwarddiff" && return ADTypes.AutoForwardDiff()
    s == "mooncake" && return ADTypes.AutoMooncake(config=nothing)
    throw(ArgumentError("adtype must be \"forwarddiff\" or \"mooncake\", got $(repr(s))"))
end

counts(Y; occupied_only=false) = CR.cooccurrence_counts(_presence(Y); occupied_only=Bool(occupied_only))

mle(k, mi, mj, N) = CR.affinity_mle.(_ints(k), _ints(mi), _ints(mj), _ints(N))

map_estimate(k, mi, mj, N; prior_sd=3.0) =
    CR.affinity_map.(_ints(k), _ints(mi), _ints(mj), _ints(N); prior_sd=Float64(prior_sd))

function posterior(k, mi, mj, N; prior_sd=4.0, level=0.95, npoints=1001)
    res = CR.affinity_posterior.(_ints(k), _ints(mi), _ints(mj), _ints(N);
                                 prior_sd=Float64(prior_sd), level=Float64(level), npoints=Int(npoints))
    return Tuple([getfield(r, f) for r in res] for f in (:median, :mean, :lower, :upper, :pd, :p_positive))
end

fnch_logpdf(k, mi, mj, N, α) = CR.fnch_logpdf.(_ints(k), _ints(mi), _ints(mj), _ints(N), _floats(α))
fnch_mean(mi, mj, N, α) = CR.fnch_mean.(_ints(mi), _ints(mj), _ints(N), _floats(α))

function pairwise(Y; prior_sd=4.0, level=0.95, site_weights=false, metric="affinity")
    r = CR.pairwise_affinity(_presence(Y); prior_sd=Float64(prior_sd), level=Float64(level),
                             site_weights=_site_weights(site_weights), metric=Symbol(metric))
    return (r.median, r.lower, r.upper, r.pd, r.p_positive)
end

trait_distance(x) = CR.trait_distance(_trait(x))
gower_distance(traits...) = CR.gower_distance(map(_trait, traits)...)
standardise(X; g=nothing) =
    g === nothing ? CR.standardise(Float64.(collect(X))) : CR.standardise(Float64.(collect(X)), _labels(g))

function swap_null(Y; n_trades=nothing, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(Int(seed))
    P = _presence(Y)
    Z = n_trades === nothing ? CR.swap_null(P; rng) : CR.swap_null(P, Int(n_trades); rng)
    return Matrix{Bool}(Z)
end

function site_effects(Y; prior_sd=10.0)
    r = CR.site_effects(_presence(Y); prior_sd=Float64(prior_sd))
    return (r.sites, r.species, r.b, r.a, r.D)
end

_keyword(k, v) =
    k === :metric ? Symbol(string(v)) :
    k === :adtype ? _adtype(v) :
    k === :strata ? _labels(v) :
    k === :site_weights ? _site_weights(v) :
    k === :sigma_params ? Tuple(_floats(v)) :
    k === :n_adapts ? Int(v) : v

_ascii(name) = replace(string(name), "βμ" => "beta_mu", "βσ" => "beta_sigma", "α" => "alpha",
                       "β" => "beta", "λ" => "lambda", " " => "")

"""
    regression(X, Y, n_iter, n_chains; g=nothing, seed=nothing, drop_lambda=false, kwargs...)

`cooccurrence_regression` run `n_chains` times in turn. `X` is an `S × S × P` array holding
the explanatory variables. Returns `(names, draws, chain, iteration, n_divergent)` with ASCII
parameter names (`alpha`, `beta[p]`, `lambda[i]`, `beta_mu[p]`, `beta_sigma[p]`).
"""
function regression(X, Y, n_iter, n_chains; g=nothing, seed=nothing, drop_lambda=false, kwargs...)
    seed === nothing || Random.seed!(Int(seed))
    X3 = Float64.(collect(X))
    ndims(X3) == 2 && (X3 = reshape(X3, size(X3)..., 1))
    Xs = [X3[:, :, p] for p in axes(X3, 3)]
    P = _presence(Y)
    kw = (; (k => _keyword(k, v) for (k, v) in pairs(kwargs) if v !== nothing)...)
    args = g === nothing ? (Xs, P) : (Xs, P, _labels(g))
    chns = [CR.cooccurrence_regression(args..., Int(n_iter); kw...) for _ in 1:Int(n_chains)]

    params = names(first(chns), :parameters)
    Bool(drop_lambda) && (params = filter(p -> !startswith(string(p), "λ"), params))
    draws = reduce(vcat, [stack(vec(chn[p]) for p in params) for chn in chns])
    n = size(first(chns), 1)
    chain = repeat(1:length(chns), inner=n)
    iteration = repeat(1:n, outer=length(chns))
    divergent = sum(chns) do chn
        :numerical_error in names(chn) ? Int(sum(vec(chn[:numerical_error]))) : 0
    end
    return (_ascii.(params), Matrix{Float64}(draws), chain, iteration, divergent)
end

end
