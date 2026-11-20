@testset "regression models" begin
    rng = Random.MersenneTwister(4)
    S, N = 10, 25
    Y = rand(rng, N, S) .< 0.5
    X = standardise(trait_distance(randn(rng, S)))
    X2 = standardise(trait_distance(randn(rng, S)))
    g = repeat(["a", "b"], inner=5)
    n_iter = 60

    Random.seed!(123)
    chn1 = cooccurrence_regression(X, Y, n_iter)
    Random.seed!(123)
    chn2 = cooccurrence_regression([X], Y, n_iter)
    @test chn1[:α] == chn2[:α]
    @test size(chn1, 1) == n_iter
    @test Symbol("β[1]") in names(chn1, :parameters)
    @test Symbol("λ[$S]") in names(chn1, :parameters)

    chn = cooccurrence_regression([X, X2], Y, n_iter; drop_lambda=true)
    @test sort(string.(names(chn, :parameters))) == ["α", "β[1]", "β[2]"]

    # backends agree on the model they are given
    for adtype in (ADTypes.AutoForwardDiff(), ADTypes.AutoMooncake(config=nothing))
        c = cooccurrence_regression(X, Y, n_iter; adtype)
        @test all(isfinite, c[:α])
    end

    hier = cooccurrence_regression(X, Y, g, n_iter)
    @test Symbol("βμ[1]") in names(hier, :parameters)
    @test Symbol("βσ[1]") in names(hier, :parameters)
    @test !any(startswith("λ"), string.(names(cooccurrence_regression(X, Y, g, n_iter; drop_lambda=true), :parameters)))

    pooled = cooccurrence_regression([X, X2], Y, g, n_iter; pooled=true)
    @test Symbol("β[2]") in names(pooled, :parameters)
    @test !(Symbol("βμ[1]") in names(pooled, :parameters))

    # a species present at every site is ignored rather than breaking the fit
    Yfull = copy(Y)
    Yfull[:, 1] .= true
    @test all(isfinite, cooccurrence_regression(X, Yfull, n_iter)[:α])

    @test_throws DimensionMismatch cooccurrence_regression(X[1:5, 1:5], Y, n_iter)
    @test_throws DimensionMismatch cooccurrence_regression(X, Y, g[1:4], n_iter)
end

@testset "recovers a known coefficient" begin
    rng = Random.MersenneTwister(11)
    S, N, β = 30, 60, -1.0
    X = standardise(trait_distance(randn(rng, S)))
    m = rand(rng, 15:45, S)
    # co-occurrence counts drawn directly from the model, given the prevalences
    K = zeros(Int, S, S)
    for j in 1:S-1, i in j+1:S
        K[i, j] = K[j, i] = rand(rng, LogOddsFNCH(m[i], m[j], N, β * X[i, j]))
    end
    model = CooccurrenceRegression.regression(reshape(X, 1, S, S), K, S, N, m)
    chn = sample(rng, model, NUTS(200, 0.65; adtype=ADTypes.AutoMooncake(config=nothing)), 300;
                 progress=false, chain_type=CooccurrenceRegression.MCMCChains.Chains)
    est = vec(chn[Symbol("β[1]")])
    @test Statistics.quantile(est, 0.005) < β < Statistics.quantile(est, 0.995)
    @test abs(Statistics.mean(est) - β) < 0.25
end

@testset "site types" begin
    rng = Random.MersenneTwister(8)
    S = 8
    Y = rand(rng, 40, S) .< 0.5
    Y[1:10, 2] .= false                       # a species missing from one site type
    strata = repeat(["a", "b", "c", "d"], inner=10)
    X = standardise(trait_distance(randn(rng, S)))
    chn = cooccurrence_regression(X, Y, 60; strata)
    pars = string.(names(chn, :parameters))
    @test count(startswith("α["), pars) == 4
    @test count(startswith("λ["), pars) == 4S
    @test "β[1]" in pars
    @test all(isfinite, chn[Symbol("β[1]")])
    @test !any(startswith("λ"), string.(names(cooccurrence_regression([X], Y, 60; strata, drop_lambda=true), :parameters)))
    @test_throws DimensionMismatch cooccurrence_regression(X, Y, 60; strata=strata[1:5])
end

@testset "group range" begin
    rng = Random.MersenneTwister(9)
    Y = falses(40, 10)
    Y[1:20, 1:5] .= rand(rng, 20, 5) .< 0.5        # one group in the first 20 sites
    Y[21:40, 6:10] .= rand(rng, 20, 5) .< 0.5      # the other in the last 20
    g = repeat([1, 2], inner=5)
    X = standardise(trait_distance(randn(rng, 10)), g)
    Random.seed!(1); a = cooccurrence_regression(X, Y, g, 60; pooled=true)
    Random.seed!(1); b = cooccurrence_regression(X, Y, g, 60; pooled=true, group_range=true)
    @test all(isfinite, b[Symbol("β[1]")])
    # sharing a region no longer counts as association
    @test Statistics.mean(b[:α]) < Statistics.mean(a[:α])
end
