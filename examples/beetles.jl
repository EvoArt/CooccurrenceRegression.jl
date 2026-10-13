# Trait dissimilarity and co-occurrence of dung beetles along an altitudinal gradient at
# Serra do Cipó, Brazil (data: Nunes et al. 2016).
#
# 56 species in 7 sites (three transects pooled per altitude), with the functional guild and
# mean biomass of each species.
#
#     julia --project examples/beetles.jl

using CooccurrenceRegression, DelimitedFiles, Random
Random.seed!(321)

raw = readdlm(joinpath(@__DIR__, "data", "beetles.csv"), ',')
guild = string.(raw[3:end, 2])
biomass = Float64.(raw[3:end, 3])
transects = raw[3:end, 4:end] .> 0                 # species × 21 transects

# a species is present at an altitude if it was caught in any of its three transects
Y = permutedims(reduce(hcat, [any(transects[:, t:t+2], dims=2) for t in 1:3:size(transects, 2)]))

biomass_dist = trait_distance(biomass)
guild_dist = trait_distance(guild)

# 1. One explanatory variable: Gower's distance over both traits
gower_chn = cooccurrence_regression(standardise(gower_distance(guild, biomass)), Y; drop_lambda=true)

# 2. Both traits and their interaction
X1 = standardise(biomass_dist)
interaction_chn = cooccurrence_regression([X1, guild_dist, X1 .* guild_dist], Y; drop_lambda=true)

# 3. Biomass distance among species of the same guild
within_chn = cooccurrence_regression(X1, Y, guild; pooled=true, drop_lambda=true)

for (name, chn) in ("Gower" => gower_chn, "Interaction" => interaction_chn, "Within guild" => within_chn)
    println("\n", name)
    display(chn)
end
