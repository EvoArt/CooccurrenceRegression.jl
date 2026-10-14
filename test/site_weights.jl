function brute_esp(w, r)
    n = length(w)
    s = 0.0
    for mask in 0:(2^n - 1)
        count_ones(mask) == r || continue
        s += prod(w[q] for q in 1:n if (mask >> (q - 1)) & 1 == 1; init=1.0)
    end
    return s
end

logistic(x) = 1 / (1 + exp(-x))

# species and site effects, no associations
function simulate_sites(rng, N, S, site_sd)
    b = site_sd .* randn(rng, N)
    a = -1.0 .+ 1.2 .* randn(rng, S)
    return [rand(rng) < logistic(a[j] + b[n]) for n in 1:N, j in 1:S], b
end

@testset "site-weighted likelihood" begin
    rng = Random.MersenneTwister(1)

    @testset "elementary symmetric polynomials" begin
        esp = CooccurrenceRegression._log_esp
        w = exp.(randn(rng, 9))
        @test esp(w, 9) ≈ [log(brute_esp(w, r)) for r in 0:9]
        @test esp(w, 4) ≈ [log(brute_esp(w, r)) for r in 0:4]
        @test esp(Float64[], 0) == [0.0]
        big = exp.(40 .* randn(rng, 60))          # too uneven for the scaled recursion
        L = esp(big, 30)
        @test all(isfinite, L)
        @test L[2] ≈ log(sum(big))
    end

    @testset "equal weights give FNCH" begin
        for (mi, mj, N) in ((5, 9, 12), (20, 31, 62), (1, 29, 30), (150, 300, 400))
            yi = falses(N)
            yi[Random.randperm(rng, N)[1:mi]] .= true
            lo, logc = weighted_coefficients(yi, mj, fill(2.5, N))
            for k in lo:min(mi, mj), α in (-3.0, 0.0, 0.7, 4.0)
                a, g = weighted_logpdf_and_gradient(k, lo, logc, α)
                b, h = fnch_logpdf_and_gradient(k, mi, mj, N, α)
                @test a ≈ b atol = 1e-8
                @test g ≈ h atol = 1e-8
            end
        end
    end

    @testset "extreme and invalid inputs" begin
        yi = rand(rng, 40) .< 0.3
        lo, logc = weighted_coefficients(yi, 15, exp.(randn(rng, 40)))
        @test weighted_logpdf_and_gradient(lo - 1, lo, logc, 0.0) == (-Inf, 0.0)
        @test weighted_logpdf_and_gradient(lo, lo, logc, NaN) == (-Inf, 0.0)
        @test weighted_logpdf_and_gradient(lo, lo, logc, -Inf)[1] == 0.0
        @test all(isfinite, (weighted_logpdf(lo + 1, lo, logc, a) for a in (-500.0, 500.0)))
        @test_throws DimensionMismatch weighted_coefficients(yi, 15, ones(3))
        @test_throws ArgumentError weighted_coefficients(yi, 15, fill(-1.0, 40))
    end

    @testset "derivative rules" begin
        yi = rand(rng, 40) .< 0.3
        lo, logc = weighted_coefficients(yi, 15, exp.(randn(rng, 40)))
        k, α = lo + 2, 0.3
        lp, g = weighted_logpdf_and_gradient(k, lo, logc, α)
        f(x) = weighted_logpdf(k, lo, logc, x)
        @test g ≈ (f(α + 1e-6) - f(α - 1e-6)) / 2e-6 atol = 1e-5
        y, pullback = ChainRulesCore.rrule(weighted_logpdf, k, lo, logc, α)
        @test y == lp
        @test pullback(1.0)[end] ≈ g
        @test ForwardDiff.derivative(f, α) ≈ g

        Y, _ = simulate_sites(rng, 30, 6, 1.0)
        pairs = site_weighted_pairs(Y)
        xs = randn(rng, length(pairs))
        F(θ) = sum(weighted_logpdf(pairs[q], θ[1] + θ[2] * xs[q]) for q in eachindex(pairs))
        θ = [0.3, -0.7]
        g_fd = ForwardDiff.gradient(F, θ)
        g_mc = DifferentiationInterface.gradient(F, ADTypes.AutoMooncake(config=nothing), θ)
        @test g_mc ≈ g_fd
    end

    @testset "site effects" begin
        Y, b = simulate_sites(rng, 60, 40, 1.0)
        fit = site_effects(Y)
        @test Statistics.cor(fit.b, b[fit.sites]) > 0.85
        Y2 = copy(Y)
        Y2[1, :] .= false
        Y2[2, :] .= true
        Y2[:, 1] .= true
        f2 = site_effects(Y2)
        @test !(1 in f2.sites) && !(2 in f2.sites) && !(1 in f2.species)
        nested = [j <= i for i in 1:12, j in 1:10]
        @test all(isfinite, site_effects(nested).b)
    end

    @testset "pairs" begin
        Y, _ = simulate_sites(rng, 40, 12, 1.0)
        pairs = site_weighted_pairs(Y)
        @test all(p -> p.i > p.j, pairs)
        p = pairs[1]
        @test weighted_logpdf(p, 0.4) ≈
              (weighted_logpdf(p.k, p.lo_i, p.logc_i, 0.4) + weighted_logpdf(p.k, p.lo_j, p.logc_j, 0.4)) / 2
        K, m, N = cooccurrence_counts(Y)
        q = site_weighted_pairs(Y; b=zeros(N))
        @test all(weighted_logpdf(r, 0.3) ≈ fnch_logpdf(K[r.i, r.j], m[r.i], m[r.j], N, 0.3) for r in q)
        @test_throws DimensionMismatch site_weighted_pairs(Y; b=zeros(3))
    end

    @testset "no associations, unequal sites" begin
        Yp = rand(Random.MersenneTwister(2), 40, 8) .< 0.5
        @test pairwise_affinity(Yp; site_weights=zeros(40)).median ≈ pairwise_affinity(Yp).median nans = true atol = 1e-8
        for s in 1:2
            Y, _ = simulate_sites(Random.MersenneTwister(10 + s), 50, 30, 1.5)
            plain = pairwise_affinity(Y).p_positive
            weighted = pairwise_affinity(Y; site_weights=true).p_positive
            @test Statistics.mean(filter(!isnan, plain)) > 0.6
            @test abs(Statistics.mean(filter(!isnan, weighted)) - 0.5) < 0.05
        end
    end
end

@testset "swap null" begin
    Y, _ = simulate_sites(Random.MersenneTwister(4), 30, 20, 1.0)
    Z = swap_null(Y; rng=Random.MersenneTwister(3))
    @test vec(sum(Z, dims=1)) == vec(sum(Y, dims=1))
    @test vec(sum(Z, dims=2)) == vec(sum(Y, dims=2))
    @test Z != Y
    @test swap_null(Y, 0) == Y
end

@testset "error cases of Ulrich et al. (2022)" begin
    # their documented failure, and their random sampler of valid inputs
    @test affinity_mle(4, 7, 10, 20) ≈ 0.420 atol = 1e-3
    rng = Random.MersenneTwister(2022)
    worst = 0.0
    for _ in 1:2000
        N = rand(rng, 10:500)
        X = rand(rng, 1:N)
        mA = X == N ? X : rand(rng, X:N)
        mB = mA == N ? X : rand(rng, X:(X + (N - mA)))
        a = affinity_mle(X, mA, mB, N)
        p = affinity_posterior(X, mA, mB, N)
        isfinite(a) && (worst = max(worst, abs(fnch_mean(mA, mB, N, a) - X)))
        @test p.lower <= p.median <= p.upper
    end
    @test worst < 1e-8
end

@testset "regression with site weights" begin
    rng = Random.MersenneTwister(9)
    S = 8
    Y, b = simulate_sites(rng, 40, S, 1.0)
    X = standardise(trait_distance(randn(rng, S)))
    chn = cooccurrence_regression(X, Y, 60; site_weights=true)
    @test all(isfinite, chn[Symbol("β[1]")])
    @test count(startswith("λ["), string.(names(chn, :parameters))) == S
    chn_b = cooccurrence_regression(X, Y, 60; site_weights=b)
    @test all(isfinite, chn_b[:α])
    @test_throws ArgumentError cooccurrence_regression(X, Y, 60; site_weights=true, strata=repeat([1, 2], 20))
end
