module CooccurrenceRegression

using Reexport

@reexport using Turing
import ADTypes, ChainRulesCore, Distributions, ForwardDiff, Mooncake, Random, Statistics
include("fnch.jl")
include("rules.jl")
include("distribution.jl")
include("pairwise.jl")
include("distances.jl")
include("models.jl")
export cooccurrence_regression, default_adtype
export fnch_logpdf, fnch_logpdf_and_gradient, fnch_mean, LogOddsFNCH
export cooccurrence_counts, affinity_mle, affinity_map, affinity_posterior, pairwise_affinity
export trait_distance, gower_distance, standardise
end
