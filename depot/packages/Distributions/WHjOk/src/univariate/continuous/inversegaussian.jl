""" """ struct InverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    λ::T
    function InverseGaussian{T}(μ::T, λ::T) where T
        @check_args(InverseGaussian, μ > zero(μ) && λ > zero(λ))
        new{T}(μ, λ)
    end
end
InverseGaussian(μ::T, λ::T) where {T<:Real} = InverseGaussian{T}(μ, λ)
InverseGaussian(μ::Real, λ::Real) = InverseGaussian(promote(μ, λ)...)
InverseGaussian(μ::Integer, λ::Integer) = InverseGaussian(Float64(μ), Float64(λ))
InverseGaussian(μ::Real) = InverseGaussian(μ, 1.0)
InverseGaussian() = InverseGaussian(1.0, 1.0)
@distr_support InverseGaussian 0.0 Inf
function convert(::Type{InverseGaussian{T}}, μ::S, λ::S) where {T <: Real, S <: Real}
    InverseGaussian(T(μ), T(λ))
end
function convert(::Type{InverseGaussian{T}}, d::InverseGaussian{S}) where {T <: Real, S <: Real}
    InverseGaussian(T(d.μ), T(d.λ))
end
shape(d::InverseGaussian) = d.λ
params(d::InverseGaussian) = (d.μ, d.λ)
@inline partype(d::InverseGaussian{T}) where {T<:Real} = T
mean(d::InverseGaussian) = d.μ
var(d::InverseGaussian) = d.μ^3 / d.λ
skewness(d::InverseGaussian) = 3sqrt(d.μ / d.λ)
kurtosis(d::InverseGaussian) = 15d.μ / d.λ
function mode(d::InverseGaussian)
    μ, λ = params(d)
    r = μ / λ
    μ * (sqrt(1 + (3r/2)^2) - (3r/2))
end
function pdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        return sqrt(λ / (twoπ * x^3)) * exp(-λ * (x - μ)^2 / (2μ^2 * x))
    else
        return zero(T)
    end
end
function logpdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        return (log(λ) - (log2π + 3log(x)) - λ * (x - μ)^2 / (μ^2 * x))/2
    else
        return -T(Inf)
    end
end
function cdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        return normcdf(u * (v - 1)) + exp(2λ / μ) * normcdf(-u * (v + 1))
    else
        return zero(T)
    end
end
function ccdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        normccdf(u * (v - 1)) - exp(2λ / μ) * normcdf(-u * (v + 1))
    else
        return one(T)
    end
end
function logcdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = normlogcdf(u * (v -1))
        b = 2λ / μ + normlogcdf(-u * (v + 1))
        a + log1pexp(b - a)
    else
        return -T(Inf)
    end
end
function logccdf(d::InverseGaussian{T}, x::Real) where T<:Real
    if x > 0
        μ, λ = params(d)
        u = sqrt(λ / x)
        v = x / μ
        a = normlogccdf(u * (v - 1))
        b = 2λ / μ + normlogcdf(-u * (v + 1))
        a + log1mexp(b - a)
    else
        return zero(T)
    end
end
@quantile_newton InverseGaussian
rand(d::InverseGaussian) = rand(GLOBAL_RNG, d)
function rand(rng::AbstractRNG, d::InverseGaussian)
    μ, λ = params(d)
    z = randn(rng)
    v = z * z
    w = μ * v
    x1 = μ + μ / (2λ) * (w - sqrt(w * (4λ + w)))
    p1 = μ / (μ + x1)
    u = rand(rng)
    u >= p1 ? μ^2 / x1 : x1
end
