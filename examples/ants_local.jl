# Trait dissimilarity and co-occurrence of ants in two nature reserves near Hong Kong
# (data: Wong et al. 2021).
#
# Plots were either intact or invaded by Solenopsis invicta, giving four site types (two
# reserves × intact/invaded). Co-occurrences are counted within site type; the regression
# coefficients are shared.
#
#     julia --project examples/ants_local.jl

using CooccurrenceRegression, DelimitedFiles, Random
Random.seed!(321)

invader = "Solenopsis_invicta"

occ = readdlm(joinpath(@__DIR__, "data", "ants.csv"), ',')
species = string.(occ[1, 3:end-1])
keep = species .!= invader
Y = occ[2:end, 3:end-1][:, keep] .> 0
site_type = string.(occ[2:end, 2], "_", occ[2:end, end])

traits = readdlm(joinpath(@__DIR__, "data", "ant_trait.csv"), ',')
trait_names = string.(traits[1, 2:end])
# the two files punctuate three species names differently
tidy(name) = replace(name, r"[ .-]" => "_")
rows = [findfirst(==(tidy(sp)), tidy.(string.(traits[2:end, 1]))) for sp in species[keep]]
T = Float64.(traits[2:end, 2:end][rows, :])

Xs = [standardise(trait_distance(T[:, p])) for p in axes(T, 2)]

chn = cooccurrence_regression(Xs, Y; strata=site_type, drop_lambda=true)

println("β is in the order: ", join(trait_names, ", "))
display(chn)
