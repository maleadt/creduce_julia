""" """ struct Skellam{T<:Real} <: DiscreteUnivariateDistribution
    μ1::T
    μ2::T
    function Skellam{T}(μ1::T, μ2::T) where T
        @check_args(Skellam, μ1 > zero(μ1) && μ2 > zero(μ2))
        new{T}(μ1, μ2)
    end
end
Skellam(μ1::T, μ2::T) where {T<:Real} = Skellam{T}(μ1, μ2)
Skellam(μ1::Real, μ2::Real) = Skellam(promote(μ1, μ2)...)
Skellam(μ1::Integer, μ2::Integer) = Skellam(Float64(μ1), Float64(μ2))
Skellam(μ::Real) = Skellam(μ, μ)
Skellam() = Skellam(1.0, 1.0)
@distr_support Skellam -Inf Inf
convert(::Type{Skellam{T}}, μ1::S, μ2::S) where {T<:Real, S<:Real} = Skellam(T(μ1), T(μ2))
convert(::Type{Skellam{T}}, d::Skellam{S}) where {T<:Real, S<:Real} =  Skellam(T(d.μ1), T(d.μ2))
params(d::Skellam) = (d.μ1, d.μ2)
@inline partype(d::Skellam{T}) where {T<:Real} = T
mean(d::Skellam) = d.μ1 - d.μ2
var(d::Skellam) = d.μ1 + d.μ2
skewness(d::Skellam) = mean(d) / (var(d)^(3//2))
kurtosis(d::Skellam) = 1 / var(d)
function logpdf(d::Skellam, x::Int)
    μ1, μ2 = params(d)
    - (μ1 + μ2) + (x/2) * log(μ1/μ2) + log(besseli(x, 2*sqrt(μ1)*sqrt(μ2)))
end
pdf(d::Skellam, x::Int) = exp(logpdf(d, x))
function mgf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(μ1 * (exp(t) - 1) + μ2 * (exp(-t) - 1))
end
function cf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(μ1 * (cis(t) - 1) + μ2 * (cis(-t) - 1))
end
cdf(d::Skellam, x::Int) = throw(MethodError(cdf, (d, x)))
cdf(d::Skellam, x::Real) = throw(MethodError(cdf, (d, x)))
rand(d::Skellam) = rand(Poisson(d.μ1)) - rand(Poisson(d.μ2))
