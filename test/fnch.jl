# Tests of the FNCH log-odds likelihood, its derivative and the AD rules.

# exact values from the definition, in extended precision
function exact_fnch(mi, mj, N, α)
    lo = max(0, mi + mj - N)
    hi = min(mi, mj)
    w = setprecision(BigFloat, 256) do
        [binomial(big(mi), big(u)) * binomial(big(N - mi), big(mj - u)) * exp(big(α) * u) for u in lo:hi]
    end
    Z = sum(w)
    logp = [Float64(log(x / Z)) for x in w]
    EK = Float64(sum((lo + i - 1) * w[i] for i in eachindex(w)) / Z)
    return lo:hi, logp, EK
end

@testset "FNCH likelihood" begin
    grid = [(N, mi, mj) for N in (1, 2, 5, 12) for mi in 0:N for mj in 0:N]
    append!(grid, [(62, mi, mj) for mi in (1, 30, 61) for mj in (2, 31, 55)])
    append!(grid, [(300, 150, 150), (300, 3, 290)])
    αs = (-30.0, -8.5, -0.3, 0.0, 0.7, 3.0, 30.0)

    @testset "agrees with the definition" begin
        for (N, mi, mj) in grid, α in αs
            support, logp, EK = exact_fnch(mi, mj, N, α)
            total = 0.0
            for (idx, k) in enumerate(support)
                lp, g = fnch_logpdf_and_gradient(k, mi, mj, N, α)
                @test lp ≈ logp[idx] atol = 1e-12 rtol = 1e-12
                @test g ≈ k - EK atol = 1e-10 rtol = 1e-10
                @test fnch_logpdf(k, mj, mi, N, α) ≈ lp atol = 1e-12 rtol = 1e-12   # symmetric in species
                total += exp(lp)
            end
            @test total ≈ 1
            @test fnch_mean(mi, mj, N, α) ≈ EK atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "agrees with Distributions.jl" begin
        for (N, mi, mj) in [(12, 5, 9), (62, 30, 31), (30, 1, 29)], α in (-3.0, 0.0, 0.5, 4.0)
            d = FisherNoncentralHypergeometric(mi, N - mi, mj, exp(α))
            for k in max(0, mi + mj - N):min(mi, mj)
                @test fnch_logpdf(k, mi, mj, N, α) ≈ logpdf(d, k) atol = 1e-9 rtol = 1e-9
            end
            @test fnch_mean(mi, mj, N, α) ≈ mean(d) rtol = 1e-8
        end
        @test fnch_mean(5, 9, 12, 0.0) ≈ 5 * 9 / 12
    end

    @testset "extreme and invalid inputs" begin
        for α in (Inf, -Inf, NaN, 1e300, -1e300, 800.0, -800.0), k in 0:5
            lp, g = fnch_logpdf_and_gradient(k, 5, 9, 12, α)   # support is 2:5
            @test !isnan(lp) && !isnan(g)
        end
        @test fnch_logpdf_and_gradient(5, 5, 9, 12, Inf) == (0.0, 0.0)
        @test fnch_logpdf_and_gradient(2, 5, 9, 12, -Inf) == (0.0, 0.0)
        @test fnch_logpdf(3, 5, 9, 12, Inf) == -Inf
        @test fnch_logpdf(9, 5, 9, 12, 0.0) == -Inf            # outside the support
        @test fnch_logpdf(0, 0, 9, 12, 2.0) == 0.0             # species absent everywhere
        @test fnch_logpdf(9, 12, 9, 12, -2.0) == 0.0           # species present everywhere
        # finite in the tails, where a product on the odds-ratio scale underflows
        @test fnch_logpdf(0, 150, 150, 300, 30.0) ≈ exact_fnch(150, 150, 300, 30.0)[2][1]
        @test fnch_logpdf(3, 5, 9, 12, 800.0) ≈ exact_fnch(5, 9, 12, 800.0)[2][2]
        @test fnch_logpdf(3, 5, 9, 12, 0.5f0) isa Float32
        @test_throws ArgumentError fnch_logpdf(1, 13, 2, 12, 0.0)
        @test_throws ArgumentError fnch_logpdf(1, 2, -1, 12, 0.0)
    end

    @testset "derivative rules" begin
        k, mi, mj, N, α = 4, 20, 31, 62, 1.3
        lp, g = fnch_logpdf_and_gradient(k, mi, mj, N, α)
        h = 1e-5
        fd = (fnch_logpdf(k, mi, mj, N, α + h) - fnch_logpdf(k, mi, mj, N, α - h)) / 2h
        @test g ≈ fd rtol = 1e-7

        y, pullback = ChainRulesCore.rrule(fnch_logpdf, k, mi, mj, N, α)
        @test y == lp
        @test last(pullback(2.0)) ≈ 2g
        y, ẏ = ChainRulesCore.frule((nothing, nothing, nothing, nothing, nothing, 3.0), fnch_logpdf, k, mi, mj, N, α)
        @test y == lp && ẏ ≈ 3g

        @test ForwardDiff.derivative(a -> fnch_logpdf(k, mi, mj, N, a), α) ≈ g
        # second derivative is -Var[K]; goes through the generic (non-rule) code path
        var_fd = -(fnch_logpdf_and_gradient(k, mi, mj, N, α + h)[2] - fnch_logpdf_and_gradient(k, mi, mj, N, α - h)[2]) / 2h
        @test -ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> fnch_logpdf(k, mi, mj, N, b), a), α) ≈ var_fd rtol = 1e-6

        # composite function of several parameters: ForwardDiff vs Mooncake
        ks, ms, xs = [3, 0, 7, 5], [(5, 9), (2, 4), (10, 8), (6, 6)], [0.2, -1.0, 0.4, 1.5]
        f(θ) = sum(fnch_logpdf(ks[i], ms[i][1], ms[i][2], 12, θ[1] + θ[2] * xs[i]) for i in eachindex(ks))
        θ = [0.3, -0.7]
        g_fd = ForwardDiff.gradient(f, θ)
        backend = ADTypes.AutoMooncake(config=nothing)
        g_mc = DifferentiationInterface.gradient(f, backend, θ)
        @test g_mc ≈ g_fd
    end
end

@testset "LogOddsFNCH" begin
    d = LogOddsFNCH(5, 9, 12, 0.8)
    @test minimum(d) == 2 && maximum(d) == 5
    @test params(d) == (5, 9, 12, 0.8)
    @test insupport(d, 3) && !insupport(d, 6) && !insupport(d, 2.5)
    @test logpdf(d, 3) == fnch_logpdf(3, 5, 9, 12, 0.8)
    @test pdf(d, 3) ≈ exp(logpdf(d, 3))
    @test logpdf(d, 2.5) == -Inf
    @test mean(d) ≈ fnch_mean(5, 9, 12, 0.8)
    rng = Random.MersenneTwister(1)
    draws = [rand(rng, d) for _ in 1:20_000]
    @test all(insupport.(Ref(d), draws))
    @test mean(draws) ≈ mean(d) atol = 0.03
    @test rand(rng, LogOddsFNCH(5, 9, 12, Inf)) == 5
    @test rand(rng, LogOddsFNCH(5, 9, 12, -Inf)) == 2
    @test_throws ArgumentError LogOddsFNCH(13, 9, 12, 0.0)
end
