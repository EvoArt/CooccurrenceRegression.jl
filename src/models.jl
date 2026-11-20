
@model function regression(Xs,k, n_obs,N,m, n_X = size(Xs,1);priors =[4.0,1.0,1.0],lik=fnch_logpdf)
   
    # Priors
    α ~ Normal(0.0,priors[1])
    β ~ filldist(Normal(0.0,priors[2]),n_X)
    λ ~ filldist(Normal(0.0,priors[3]),n_obs)
    lp = 0.0
    for j in 1:n_obs-1
        mB = m[j]
        for i in j+1:n_obs
            mA = m[i]
            if (0<mA<N) & (0<mB<N)
                μ = α+λ[i] +λ[j]
                for pred in 1:n_X
                    μ += β[pred]*Xs[pred,i,j] 
                end
                lp += lik(k[i,j], mA, mB, N, μ)
            end
        end
    end
    Turing.@addlogprob! lp
end

function sub_regression(Xs,k,n_obs,N,m,n_X,β,λ,α,lik=fnch_logpdf)
    lp = 0.0
    for j in 1:n_obs-1
        mB = m[j]
        for i in j+1:n_obs
            mA = m[i]
            if (0<mA<N) & (0<mB<N)
                μ = α+λ[i] +λ[j]
                for pred in 1:n_X
                    μ += β[pred]*Xs[pred,i,j] 
                end
                lp += lik(k[i,j], mA, mB, N, μ)
            end
        end
    end
    return lp
end


@model function h_regression(P,N_obs,n_p,n_X;priors =[4.0,1.0,1.0,2.0,0.1],lik=fnch_logpdf)
   
    # Priors
    α ~ Normal(0.0,priors[1])
    λ ~ filldist(Normal(0.0,priors[2]),N_obs)
    βμ ~ filldist(Normal(0.0,priors[3]),n_X)
    βσ ~ filldist(Gamma(priors[4],priors[5]),n_X)
    β ~ filldist(Normal(0.0,1.0),n_X,n_p)
    λinds = 0:0
    for i in eachindex(P)
        p=P[i]
        λinds = (λinds[end]+1):(λinds[end]+p[3])
        Turing.@addlogprob!  sub_regression(p...,n_X,(β[:,i] .* βσ) .+ βμ,λ[λinds],α,lik)
    end
end

# Pairs within groups, with regression coefficients shared by all groups.
@model function within_regression(P,N_obs,n_X;priors =[4.0,1.0,1.0],lik=fnch_logpdf)
    α ~ Normal(0.0,priors[1])
    β ~ filldist(Normal(0.0,priors[2]),n_X)
    λ ~ filldist(Normal(0.0,priors[3]),N_obs)
    λinds = 0:0
    for i in eachindex(P)
        p=P[i]
        λinds = (λinds[end]+1):(λinds[end]+p[3])
        Turing.@addlogprob!  sub_regression(p...,n_X,β,λ[λinds],α,lik)
    end
end

# Site types (strata): co-occurrences are counted within each type. The types share the
# regression coefficients and each has its own intercept and species effects.
@model function strata_regression(Xs,P,n_obs,n_X;priors =[4.0,1.0,1.0],lik=fnch_logpdf)
    n_strata = length(P)
    α ~ filldist(Normal(0.0,priors[1]),n_strata)
    β ~ filldist(Normal(0.0,priors[2]),n_X)
    λ ~ filldist(Normal(0.0,priors[3]),n_obs,n_strata)
    for l in 1:n_strata
        k,N,m = P[l]
        Turing.@addlogprob!  sub_regression(Xs,k,n_obs,N,m,n_X,β,λ[:,l],α[l],lik)
    end
end

# Site-weighted likelihood: one precomputed `WeightedPair` per informative pair.
@model function weighted_regression(Xs,pairs,n_obs,n_X;priors =[4.0,1.0,1.0])
    α ~ Normal(0.0,priors[1])
    β ~ filldist(Normal(0.0,priors[2]),n_X)
    λ ~ filldist(Normal(0.0,priors[3]),n_obs)
    lp = 0.0
    for p in pairs
        μ = α+λ[p.i] +λ[p.j]
        for pred in 1:n_X
            μ += β[pred]*Xs[pred,p.i,p.j]
        end
        lp += weighted_logpdf(p, μ)
    end
    Turing.@addlogprob! lp
end

# Automatic differentiation backend

# Measured on Athena (athena/bench_gradient.jl): ForwardDiff and Mooncake are level at about 12
# parameters, Mooncake is twice as fast at 25 and ten times at 150.
const FORWARD_MODE_MAX_PARAMS = 20

"""
    default_adtype(n_params)

AD backend used when `adtype` is not given: ForwardDiff for models with at most
`FORWARD_MODE_MAX_PARAMS` parameters, Mooncake (reverse mode) otherwise. The dyadic model has
one parameter per species plus the intercept and regression coefficients, so communities of
more than a few dozen species use Mooncake.
"""
default_adtype(n_params::Integer) =
    n_params <= FORWARD_MODE_MAX_PARAMS ? ADTypes.AutoForwardDiff() : ADTypes.AutoMooncake(config=nothing)
# User facing methods

function _stack(Xs, S)
    X_3D = Array{Float64}(undef, length(Xs), S, S)
    for (p, X) in enumerate(Xs)
        size(X) == (S, S) || throw(DimensionMismatch(
            "explanatory variable $p has size $(size(X)); expected ($S, $S) to match the $S species (columns) of Y"))
        X_3D[p, :, :] .= X
    end
    return X_3D
end

_drop_lambda(chn) = chn[[n for n in names(chn, :parameters) if !startswith(string(n), "λ")]]

function _likelihood(metric)
    metric === :affinity && return fnch_logpdf
    metric === :jaccard && return jaccard_logpdf
    throw(ArgumentError("metric must be :affinity or :jaccard, got $(repr(metric))"))
end

"""
    cooccurrence_regression(X, Y, [n_iter=1000]; kwargs...)
    cooccurrence_regression(Xs::Vector, Y, [n_iter=1000]; kwargs...)

Bayesian dyadic regression of co-occurrence affinity on one or more pairwise explanatory
variables.

`Y` is an `N × S` presence/absence matrix with sites in rows and species in columns. `X` is an
`S × S` matrix whose element `X[i, j]` relates species `i` to species `j` (for example a trait
or phylogenetic distance); only the lower triangle is used. Several explanatory variables are
passed as a vector of such matrices, `Xs`.

For each pair of species `i > j`, with prevalences `m[i]`, `m[j]` and `k[i, j]` co-occurrences,

    k[i, j] ~ LogOddsFNCH(m[i], m[j], N, α + λ[i] + λ[j] + Σₚ β[p] * Xs[p][i, j])
    α    ~ Normal(0, alpha_sd)
    β[p] ~ Normal(0, beta_sd)
    λ[i] ~ Normal(0, lambda_sd)

where `LogOddsFNCH` is Fisher's noncentral hypergeometric distribution parameterised by the
log odds ratio (see [`LogOddsFNCH`](@ref)). The species effects `λ` account for the
non-independence of pairs that share a species. Species present at none or at all of the
sites carry no information and their pairs are left out of the likelihood.

Sampling uses NUTS and returns an `MCMCChains.Chains` object with parameters `α`, `β[p]` (in
the order the explanatory variables were given) and `λ[i]` (in the order of the columns of
`Y`).

# Keyword arguments
- `alpha_sd=4.0`, `beta_sd=1.0`, `lambda_sd=1.0`: prior standard deviations.
- `n_adapts=100`: number of adaptation steps.
- `delta=0.65`: target acceptance rate.
- `adtype`: automatic differentiation backend (an `ADTypes` type). By default chosen from
  the number of parameters, see [`default_adtype`](@ref).
- `strata`: a vector giving the type of each site (row of `Y`), for data collected from
  several kinds of site whose differences should not contribute to affinity. Prevalences
  and co-occurrences are then counted within each site type `l`, with `N[l]` sites:

      k[i, j, l] ~ LogOddsFNCH(m[i, l], m[j, l], N[l], α[l] + λ[i, l] + λ[j, l] + Σₚ β[p] * Xs[p][i, j])

  The site types share the coefficients `β` and each has its own intercept `α[l]` and
  species effects `λ[i, l]` (types in sorted order).
- `metric=:affinity`: the measure of co-occurrence being modelled. With `metric=:jaccard` the
  linear predictor is `logit(J[i, j])`, where `J` is the Jaccard index (the probability that
  both species occur at a site where at least one does), and the likelihood is
  `k[i, j] ~ Binomial(m[i] + m[j] - k[i, j], J[i, j])` (see [`jaccard_logpdf`](@ref)). Unlike
  affinity, `J` depends on how common the two species are.
- `site_weights=false`: if `true`, sites are not treated as equally available. The likelihood
  is the site-weighted one, with site effects from a logistic model with species and site
  effects, re-estimated for each pair with that pair left out (see
  [`site_weighted_pairs`](@ref)), so that differences in richness between sites do not
  appear as affinity. Known site effects can be given instead as a vector, one per row of `Y`.
  Cannot be combined with `strata`.
- `drop_lambda=false`: if `true`, the species effects are left out of the returned chain.

# Example
```julia
using CooccurrenceRegression
Y = rand(Bool, 30, 12)                 # 30 sites, 12 species
t = randn(12)                          # a trait
X = abs.(t .- t')                      # pairwise trait distance
chn = cooccurrence_regression(X, Y)
```
"""
function cooccurrence_regression(Xs::Vector,Y,n_iter::Real=1000;drop_lambda=false,alpha_sd =4.0,beta_sd = 1.0,lambda_sd =1.0,n_adapts=100,delta=0.65,adtype=nothing,strata=nothing,site_weights=false,metric=:affinity)
    lik = _likelihood(metric)
    if site_weights !== false
        metric === :affinity || throw(ArgumentError("site_weights is only available with metric = :affinity"))
        strata === nothing || throw(ArgumentError("site_weights and strata cannot be combined"))
        n_obs = size(Y, 2)
        n_X = length(Xs)
        X_3D = _stack(Xs, n_obs)
        pairs = site_weighted_pairs(Y; b = site_weights === true ? nothing : site_weights)
        inference_model = weighted_regression(X_3D, pairs, n_obs, n_X; priors =[alpha_sd,beta_sd,lambda_sd])
        adtype = something(adtype, default_adtype(1 + n_X + n_obs))
        chn = Turing.sample(inference_model, NUTS(n_adapts,delta; adtype=adtype),n_iter; chain_type=MCMCChains.Chains)
        return drop_lambda ? _drop_lambda(chn) : chn
    end
    if strata !== nothing
        length(strata) == size(Y, 1) || throw(DimensionMismatch(
            "strata has length $(length(strata)); expected one entry for each of the $(size(Y, 1)) sites (rows) of Y"))
        n_obs = size(Y, 2)
        n_X = length(Xs)
        X_3D = _stack(Xs, n_obs)
        P = map(unique(sort(strata))) do s
            k, m, n_sample = cooccurrence_counts(Y[strata .== s, :])
            (k, n_sample, m)
        end
        inference_model = strata_regression(X_3D, P, n_obs, n_X; priors =[alpha_sd,beta_sd,lambda_sd], lik)
        adtype = something(adtype, default_adtype(length(P) * (1 + n_obs) + n_X))
        chn = Turing.sample(inference_model, NUTS(n_adapts,delta; adtype=adtype),n_iter; chain_type=MCMCChains.Chains)
        return drop_lambda ? _drop_lambda(chn) : chn
    end
    k, m, n_sample = cooccurrence_counts(Y)
    n_obs = length(m)
    n_X = length(Xs)
    X_3D = _stack(Xs, n_obs)

    inference_model = regression(X_3D, k, n_obs, n_sample, m; priors =[alpha_sd,beta_sd,lambda_sd], lik)
    adtype = something(adtype, default_adtype(1 + n_X + n_obs))
    chn = Turing.sample(inference_model, NUTS(n_adapts,delta; adtype=adtype),n_iter; chain_type=MCMCChains.Chains)

    drop_lambda ? _drop_lambda(chn) : chn
end

function cooccurrence_regression(X,Y,n_iter::Real=1000;kwargs...)
    cooccurrence_regression([X],Y,n_iter;kwargs...)
end

"""
    cooccurrence_regression(X, Y, g::AbstractVector, [n_iter=1000]; kwargs...)
    cooccurrence_regression(Xs::Vector, Y, g::AbstractVector, [n_iter=1000]; kwargs...)

Hierarchical dyadic regression within groups. `g` assigns each species (column of `Y`) to a
group, such as a genus or a functional guild. Only pairs of species from the same group enter
the likelihood, and each group `q` has its own regression coefficients, drawn from a common
distribution:

    k[i, j] ~ LogOddsFNCH(m[i], m[j], N, α + λ[i] + λ[j] + Σₚ b[p, q] * Xs[p][i, j])   for g[i] == g[j] == q
    b[p, q] = βμ[p] + βσ[p] * β[p, q]
    β[p, q] ~ Normal(0, 1)
    βμ[p]   ~ Normal(0, mu_sd)
    βσ[p]   ~ Gamma(sigma_params[1], sigma_params[2])     (shape, scale)
    α       ~ Normal(0, alpha_sd)
    λ[i]    ~ Normal(0, lambda_sd)

The returned chain holds `α`, `βμ`, `βσ`, the standardised deviations `β[p, q]` (groups in
sorted order) and `λ`. The coefficient of group `q` is `βμ[p] + βσ[p] * β[p, q]`; `βμ` is the
average relationship across groups. The species effects `λ` are ordered by group and then by
column of `Y`.

With `pooled=true` the groups share a single set of coefficients `β[p] ~ Normal(0, beta_sd)`
instead, so that the model is the non-hierarchical one restricted to pairs within groups.

By default the prevalence of a species and the number of sites `N` refer to all sites, so
the model asks whether two species are found together more often than expected if each could
occur anywhere. With `group_range=true` the sites considered for a group are those at which
at least one of its species is present, so that co-occurrence is judged within the range of
the group. Use this when groups occupy different regions and absence from places a group
has never reached should not count as evidence.

Keyword arguments are as for the non-hierarchical method (including `metric`), with
`mu_sd=1.0` and `sigma_params=(2.0, 0.1)` in place of `beta_sd`.
"""
function cooccurrence_regression(Xs::Vector,Y,g::AbstractVector,n_iter::Real=1000;drop_lambda=false,alpha_sd =4.0,lambda_sd =1.0,
                                mu_sd =1.0,sigma_params=(2.0,0.1),pooled=false,beta_sd=1.0,group_range=false,n_adapts=100,delta=0.65,adtype=nothing,metric=:affinity)
    lik = _likelihood(metric)
    length(g) == size(Y, 2) || throw(DimensionMismatch(
        "g has length $(length(g)); expected one entry for each of the $(size(Y, 2)) species (columns) of Y"))
    n_X = length(Xs)
    X_3D = _stack(Xs, size(Y, 2))
    P= []
    for group in unique(sort(g))
        inds = g .== group
        k, m, n_sample = cooccurrence_counts(Y[:,inds]; occupied_only=group_range)
        push!(P, (X_3D[:,inds,inds], k, length(m), n_sample, m))
    end
    P = identity.(P)

    if pooled
        inference_model = within_regression(P,size(Y,2),n_X; priors =[alpha_sd,beta_sd,lambda_sd], lik)
        n_params = 1 + size(Y,2) + n_X
    else
        inference_model = h_regression(P,size(Y,2),length(P),n_X; priors =[alpha_sd,lambda_sd,mu_sd,sigma_params...], lik)
        n_params = 1 + size(Y,2) + 2n_X + n_X*length(P)
    end
    adtype = something(adtype, default_adtype(n_params))
    chn = Turing.sample(inference_model, NUTS(n_adapts,delta; adtype=adtype),n_iter; chain_type=MCMCChains.Chains)

    drop_lambda ? _drop_lambda(chn) : chn
end

function cooccurrence_regression(X::AbstractMatrix,Y,g::AbstractVector,n_iter::Real=1000;kwargs...)
    cooccurrence_regression([X],Y,g,n_iter;kwargs...)
end
