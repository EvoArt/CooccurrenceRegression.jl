@testset "Jaccard likelihood" begin
    @testset "agrees with the binomial" begin
        for (mi, mj, N) in ((5, 9, 12), (20, 31, 62), (1, 29, 30), (150, 300, 400)), η in (-3.0, 0.0, 1.2)
            for k in max(0, mi + mj - N):min(mi, mj)
                u = mi + mj - k
                J = 1 / (1 + exp(-η))
                @test jaccard_logpdf(k, mi, mj, N, η) ≈ logpdf(Binomial(u, J), k) atol = 1e-8
            end
        end
    end

    @testset "extreme and invalid inputs" begin
        @test jaccard_logpdf_and_gradient(10, 5, 9, 12, 0.0) == (-Inf, 0.0)
        @test jaccard_logpdf_and_gradient(3, 5, 9, 12, NaN) == (-Inf, 0.0)
        @test jaccard_logpdf_and_gradient(5, 5, 9, 12, Inf) == (-Inf, -4.0)     # u = 9
        @test jaccard_logpdf_and_gradient(0, 3, 4, 12, -Inf) == (0.0, 0.0)
        @test all(isfinite, (jaccard_logpdf(3, 5, 9, 12, η) for η in (-700.0, 700.0)))
        @test_throws ArgumentError jaccard_logpdf(3, 13, 9, 12, 0.0)
    end

    @testset "derivative rules" begin
        k, mi, mj, N, η = 4, 20, 31, 62, 0.3
        lp, g = jaccard_logpdf_and_gradient(k, mi, mj, N, η)
        f(x) = jaccard_logpdf(k, mi, mj, N, x)
        @test g ≈ (f(η + 1e-6) - f(η - 1e-6)) / 2e-6 atol = 1e-5
        y, pullback = ChainRulesCore.rrule(jaccard_logpdf, k, mi, mj, N, η)
        @test y == lp
        @test pullback(1.0)[end] ≈ g
        y2, ẏ = ChainRulesCore.frule((nothing, nothing, nothing, nothing, nothing, 3.0), jaccard_logpdf, k, mi, mj, N, η)
        @test y2 == lp && ẏ ≈ 3g
        @test ForwardDiff.derivative(f, η) ≈ g
        ks, ms, xs = [3, 0, 7, 5], [(5, 9), (2, 4), (10, 8), (6, 6)], [0.2, -1.0, 0.4, 1.5]
        F(θ) = sum(jaccard_logpdf(ks[i], ms[i][1], ms[i][2], 12, θ[1] + θ[2] * xs[i]) for i in eachindex(ks))
        θ = [0.3, -0.7]
        @test DifferentiationInterface.gradient(F, ADTypes.AutoMooncake(config=nothing), θ) ≈ ForwardDiff.gradient(F, θ)
    end

    @testset "pairs" begin
        Y = rand(Random.MersenneTwister(2), 40, 8) .< 0.5
        r = pairwise_affinity(Y; metric=:jaccard)
        K, m, N = cooccurrence_counts(Y)
        # with a vague prior the posterior median is close to the observed index
        @test r.median[2, 1] ≈ log(K[2, 1] / (m[1] + m[2] - 2K[2, 1])) atol = 0.15
        @test isequal(r.median, permutedims(r.median))
        @test_throws ArgumentError pairwise_affinity(Y; metric=:sorensen)
        @test_throws ArgumentError pairwise_affinity(Y; metric=:jaccard, site_weights=true)
    end

    @testset "regression" begin
        rng = Random.MersenneTwister(8)
        S = 8
        Y = rand(rng, 40, S) .< 0.5
        X = standardise(trait_distance(randn(rng, S)))
        chn = cooccurrence_regression(X, Y, 60; metric=:jaccard)
        @test all(isfinite, chn[Symbol("β[1]")])
        g = [1, 1, 1, 1, 2, 2, 2, 2]
        @test all(isfinite, cooccurrence_regression(X, Y, g, 60; metric=:jaccard, pooled=true)[:α])
        @test_throws ArgumentError cooccurrence_regression(X, Y, 60; metric=:sorensen)
        @test_throws ArgumentError cooccurrence_regression(X, Y, 60; metric=:jaccard, site_weights=true)
    end
end
