@testset "pairwise affinity" begin
    @testset "counts" begin
        Y = rand(Random.MersenneTwister(2), 40, 8) .< 0.5
        K, m, N = cooccurrence_counts(Y)
        @test N == 40
        @test m == vec(sum(Y, dims=1))
        @test K == K'
        @test all(K[i, i] == m[i] for i in 1:8)
        @test K[1, 2] == sum(Y[:, 1] .& Y[:, 2])
        @test cooccurrence_counts(2 .* Y)[1] == K          # counts are treated as presences
    end

    @testset "maximum likelihood" begin
        for (k, mi, mj, N) in ((3, 5, 9, 12), (4, 20, 31, 62), (10, 15, 15, 30), (1, 2, 29, 30))
            @test fnch_mean(mi, mj, N, affinity_mle(k, mi, mj, N)) ≈ k atol = 1e-8
        end
        @test affinity_mle(5, 5, 9, 12) == Inf
        @test affinity_mle(2, 5, 9, 12) == -Inf
        @test isnan(affinity_mle(0, 0, 9, 12))
        @test_throws ArgumentError affinity_mle(6, 5, 9, 12)
    end

    @testset "maximum a posteriori" begin
        for (k, mi, mj, N) in ((5, 5, 9, 12), (2, 5, 9, 12), (4, 20, 31, 62))
            a = affinity_map(k, mi, mj, N; prior_sd=3.0)
            logpost(x) = fnch_logpdf(k, mi, mj, N, x) - x^2 / 18
            @test isfinite(a)
            @test logpost(a) >= logpost(a + 1e-4)
            @test logpost(a) >= logpost(a - 1e-4)
        end
        @test abs(affinity_map(4, 20, 31, 62)) < abs(affinity_mle(4, 20, 31, 62))
        @test_throws ArgumentError affinity_map(3, 5, 9, 12; prior_sd=0)
    end

    @testset "posterior" begin
        # against direct summation on a fine, wide grid
        for (k, mi, mj, N) in ((3, 5, 9, 12), (5, 5, 9, 12), (4, 20, 31, 62), (0, 3, 4, 62))
            grid = range(-60, 60; length=240_001)
            w = [exp(fnch_logpdf(k, mi, mj, N, a) - a^2 / 32) for a in grid]
            c = cumsum(w) ./ sum(w)
            post = affinity_posterior(k, mi, mj, N; prior_sd=4.0)
            @test post.median ≈ grid[searchsortedfirst(c, 0.5)] atol = 5e-3
            @test post.lower ≈ grid[searchsortedfirst(c, 0.025)] atol = 5e-3
            @test post.upper ≈ grid[searchsortedfirst(c, 0.975)] atol = 5e-3
            @test post.mean ≈ sum(grid .* w) / sum(w) atol = 5e-3
            below = c[searchsortedfirst(grid, 0.0)]
            @test post.pd ≈ max(below, 1 - below) atol = 5e-3
            @test post.lower < post.median < post.upper
        end
        narrow = affinity_posterior(3, 5, 9, 12; level=0.5)
        wide = affinity_posterior(3, 5, 9, 12; level=0.95)
        @test wide.lower < narrow.lower < narrow.upper < wide.upper
    end

    @testset "all pairs" begin
        Y = rand(Random.MersenneTwister(2), 40, 8) .< 0.5
        Y[:, 3] .= true
        r = pairwise_affinity(Y)
        @test size(r.median) == (8, 8)
        @test isnan(r.median[1, 1])
        @test all(isnan, r.median[3, :])                    # species present everywhere
        @test r.median[2, 1] == r.median[1, 2]
        @test r.lower[2, 1] < r.median[2, 1] < r.upper[2, 1]
        @test 0.5 <= r.pd[2, 1] <= 1
        K, m, N = cooccurrence_counts(Y)
        @test r.median[2, 1] == affinity_posterior(K[2, 1], m[2], m[1], N).median
    end
end

@testset "trait distances" begin
    x = Any[1.0, 3.0, 7]
    g = ["a", "b", "a"]
    @test trait_distance(x) == [0 2 6; 2 0 4; 6 4 0]
    @test trait_distance(g) == [0 1 0; 1 0 1; 0 1 0]
    G = gower_distance(g, x)
    @test G[2, 1] ≈ (1 + 2 / 6) / 2
    @test G[3, 1] ≈ 1 / 2
    @test G == G'
    @test_throws DimensionMismatch gower_distance(g, [1, 2])
    Z = standardise([0 2 6; 2 0 4; 6 4 0.0])
    off = [Z[i, j] for i in 1:3, j in 1:3 if i != j]
    @test Statistics.mean(off) ≈ 0 atol = 1e-12
    @test Statistics.std(off) ≈ 1
    @test all(Z[i, i] == 0 for i in 1:3)
    @test_throws ArgumentError standardise(zeros(3, 3))
end

@testset "standardising within groups" begin
    rng = Random.MersenneTwister(5)
    X = trait_distance(randn(rng, 9))
    g = [1, 1, 1, 1, 2, 2, 2, 2, 3]
    Z = standardise(X, g)
    @test Z[1:4, 1:4] ≈ standardise(X[1:4, 1:4])
    @test Z[5:8, 5:8] ≈ standardise(X[5:8, 5:8])
    @test all(iszero, Z[1:4, 5:9])
    @test all(iszero, Z[9, :])               # a group of one has no pairs
    @test_throws DimensionMismatch standardise(X, g[1:3])
end
