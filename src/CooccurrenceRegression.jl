module CooccurrenceRegression

using Reexport

using Turing
# Re-export only the chain summaries, not all of Turing and Distributions, whose names
# (Categorical, scale!, nsamples, setprogress!, ...) clash with Makie and other packages.
@reexport using MCMCChains: describe, summarystats, Chains
import ADTypes, ChainRulesCore, Distributions, ForwardDiff, MCMCChains, Mooncake, Random, Statistics
include("fnch.jl")
include("site_weights.jl")
include("jaccard.jl")
include("rules.jl")
include("distribution.jl")
include("pairwise.jl")
include("distances.jl")
include("null.jl")
include("models.jl")
include("interop.jl")
export cooccurrence_regression, default_adtype
export fnch_logpdf, fnch_logpdf_and_gradient, fnch_mean, LogOddsFNCH
export cooccurrence_counts, affinity_mle, affinity_map, affinity_posterior, pairwise_affinity
export trait_distance, gower_distance, standardise
export site_effects, site_weighted_pairs, weighted_coefficients, weighted_logpdf, weighted_logpdf_and_gradient
export WeightedPair, swap_null
export jaccard_logpdf, jaccard_logpdf_and_gradient
end
