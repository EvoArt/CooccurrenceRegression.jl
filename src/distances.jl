# Building pairwise explanatory variables from species traits.

_isnumeric(x) = all(v -> v isa Real, x)

"""
    trait_distance(x)

`S × S` matrix of pairwise differences between species in a single trait `x` (one value per
species). For a numeric trait this is `abs(x[i] - x[j])`. For any other trait, such as a
functional guild, it is 0 where two species share a value and 1 where they differ.
"""
function trait_distance(x::AbstractVector)
    if _isnumeric(x)
        return [float(abs(a - b)) for a in x, b in x]
    else
        return [a == b ? 0.0 : 1.0 for a in x, b in x]
    end
end

"""
    gower_distance(traits...)

Gower's (1971) distance between species from one or more traits, each a vector with one value
per species. Numeric traits contribute `abs(x[i] - x[j])` divided by the range of the trait,
other traits contribute 0 or 1, and the result is the average over traits.
"""
function gower_distance(traits::AbstractVector...)
    isempty(traits) && throw(ArgumentError("at least one trait is needed"))
    S = length(first(traits))
    all(t -> length(t) == S, traits) || throw(DimensionMismatch("traits must all have one value per species"))
    D = zeros(S, S)
    for x in traits
        d = trait_distance(x)
        if _isnumeric(x)
            r = maximum(x) - minimum(x)
            r > 0 && (d ./= r)
        end
        D .+= d
    end
    return D ./ length(traits)
end

"""
    standardise(X)

Centre and scale a square matrix of pairwise values so that its off-diagonal elements have
mean 0 and standard deviation 1. The diagonal, which plays no part in the regression, is set
to 0. Standardising explanatory variables puts their coefficients on a common scale, which
the default priors of [`cooccurrence_regression`](@ref) assume.
"""
function standardise(X::AbstractMatrix)
    S = size(X, 1)
    size(X, 2) == S || throw(DimensionMismatch("X must be square"))
    offdiag = [X[i, j] for j in 1:S for i in 1:S if i != j]
    μ, σ = Statistics.mean(offdiag), Statistics.std(offdiag)
    σ > 0 || throw(ArgumentError("the off-diagonal elements of X are all equal"))
    Z = (X .- μ) ./ σ
    for i in 1:S
        Z[i, i] = 0.0
    end
    return Z
end

"""
    standardise(X, g)

Standardise within groups: for each group in `g` (one entry per species), the off-diagonal
elements of `X` between species of that group are centred and scaled to mean 0 and standard
deviation 1. Elements between species of different groups, which are not used by the
within-group models of [`cooccurrence_regression`](@ref), are set to 0, as are the
elements of a group whose pairs all share one value (any group of two species).
"""
function standardise(X::AbstractMatrix, g::AbstractVector)
    S = size(X, 1)
    size(X, 2) == S || throw(DimensionMismatch("X must be square"))
    length(g) == S || throw(DimensionMismatch("g must have one entry per species"))
    Z = zeros(S, S)
    for group in unique(g)
        inds = findall(==(group), g)
        length(inds) > 1 || continue
        Xg = X[inds, inds]
        # a group whose pairs all share one value (e.g. any group of two species)
        # has nothing to scale: leave it centred at 0
        Statistics.std([Xg[i, j] for j in axes(Xg, 2) for i in axes(Xg, 1) if i != j]) > 0 || continue
        Z[inds, inds] .= standardise(Xg)
    end
    return Z
end
