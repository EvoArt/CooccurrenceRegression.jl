@testset "interop" begin
    I = CooccurrenceRegression.Interop
    rng = Random.MersenneTwister(7)
    S, N = 8, 20
    Y = rand(rng, N, S) .< 0.5
    K, m, n = cooccurrence_counts(Y)

    # R passes Int32 and Float64 where Julia expects Int
    @test I.counts(Int32.(Y)) == (K, m, n)
    @test I.mle(Float64(K[2, 1]), m[2], m[1], N) == [affinity_mle(K[2, 1], m[2], m[1], N)]
    @test I.map_estimate(Int32[K[2, 1], K[3, 1]], [m[2], m[3]], m[1], N; prior_sd=2) ==
          [affinity_map(K[i, 1], m[i], m[1], N; prior_sd=2.0) for i in 2:3]
    post = I.posterior(K[2, 1], m[2], m[1], N)
    @test length(post) == 6
    @test post[1] == [affinity_posterior(K[2, 1], m[2], m[1], N).median]
    @test I.fnch_logpdf(K[2, 1], m[2], m[1], N, [0.0, 1.0]) == fnch_logpdf.(K[2, 1], m[2], m[1], N, [0.0, 1.0])

    pw = I.pairwise(Float64.(Y); metric="affinity")
    @test I.fnch_mean(m[2], m[1], N, [0.0, 0.5]) == fnch_mean.(m[2], m[1], N, [0.0, 0.5])
    w = rand(rng, N) .+ 0.5
    @test isequal(I.pairwise(Y; site_weights=w)[1], pairwise_affinity(Y; site_weights=w).median)
    @test isequal(pw[1], pairwise_affinity(Y).median)
    @test isequal(pw[5], pairwise_affinity(Y).p_positive)

    t = randn(rng, S)
    guild = Any["a", "b", "a", "c", "b", "a", "c", "c"]
    @test I.trait_distance(t) == trait_distance(t)
    @test I.trait_distance(guild) == trait_distance(string.(guild))
    @test I.gower_distance(t, guild) == gower_distance(t, string.(guild))
    @test I.standardise(trait_distance(t); g=guild) == standardise(trait_distance(t), string.(guild))

    Z = I.swap_null(Y; seed=1)
    @test Z == I.swap_null(Y; seed=1)
    @test vec(sum(Z, dims=1)) == vec(sum(Y, dims=1))
    @test vec(sum(Z, dims=2)) == vec(sum(Y, dims=2))
    @test length(I.site_effects(Y)) == 5

    X = cat(standardise(trait_distance(t)), trait_distance(guild); dims=3)
    names_, draws, chain, iteration, divergent = I.regression(X, Y, 30, 2; seed=3, drop_lambda=true,
                                                             adtype="forwarddiff", metric="affinity")
    @test names_ == ["alpha", "beta[1]", "beta[2]"]
    @test size(draws) == (60, 3)
    @test chain == repeat([1, 2], inner=30)
    @test iteration == repeat(1:30, outer=2)
    @test divergent >= 0
    @test I.regression(X, Y, 30, 1; seed=3, drop_lambda=true, adtype="forwarddiff")[2] == draws[1:30, :]

    names_, draws = I.regression(X[:, :, 1], Y, 30, 1; g=guild, pooled=false, sigma_params=[2, 0.1])
    @test "beta_mu[1]" in names_
    @test "beta_sigma[1]" in names_
    @test "beta[1,3]" in names_
    @test "lambda[$S]" in names_

    strata = repeat(["x", "y"], N ÷ 2)
    @test "alpha[2]" in I.regression(X, Y, 30, 1; strata, drop_lambda=true)[1]
    @test_throws ArgumentError I.regression(X, Y, 30, 1; adtype="zygote")
end
