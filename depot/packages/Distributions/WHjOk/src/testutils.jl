import Test: @test
function _linspace(a::Float64, b::Float64, n::Int)
end
function test_distr(distr::DiscreteUnivariateDistribution, n::Int; testquan::Bool=true)
    @assert length(samples) == n
    for i = 1:n
        if rmin <= si <= rmax
        end
    end
    for i = 1:m
    end
end
function test_samples(s::Sampleable{Univariate, Continuous},    # the sampleable instance
                      verbose::Bool=false)                      # show intermediate info (for debugging)
    for i = 1:nbins
    end
    for i = 1:nbins
    end
end
function test_range(d::UnivariateDistribution)
    for (i, v) in enumerate(vs)
    end
    if isbounded(d)
        @test isapprox(mean(d), xmean, atol=1.0e-8)
    end
end
function test_params(d::Distribution)
end
