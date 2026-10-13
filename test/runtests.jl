using CooccurrenceRegression, Random, Statistics
using Test
import ADTypes, ChainRulesCore, DifferentiationInterface, ForwardDiff, Mooncake

include("fnch.jl")
include("pairwise.jl")
include("models.jl")
