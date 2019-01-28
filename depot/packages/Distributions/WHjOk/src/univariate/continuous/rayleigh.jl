""" """ struct Rayleigh{T<:Real} <: ContinuousUnivariateDistribution
    σ::T
    Rayleigh{T}(σ::T) where {T} = (@check_args(Rayleigh, σ > zero(σ)); new{T}(σ))
end
Rayleigh(σ::T) where {T<:Real} = Rayleigh{T}(σ)
Rayleigh(σ::Integer) = Rayleigh(Float64(σ))
Rayleigh() = Rayleigh(1.0)
@distr_support Rayleigh 0.0 Inf
convert(::Type{Rayleigh{T}}, σ::S) where {T <: Real, S <: Real} = Rayleigh(T(σ))
convert(::Type{Rayleigh{T}}, d::Rayleigh{S}) where {T <: Real, S <: Real} = Rayleigh(T(d.σ))
scale(d::Rayleigh) = d.σ
params(d::Rayleigh) = (d.σ,)
@inline partype(d::Rayleigh{T}) where {T<:Real} = T
mean(d::Rayleigh) = sqrthalfπ * d.σ
median(d::Rayleigh{T}) where {T<:Real} = sqrt2 * sqrt(T(logtwo)) * d.σ # sqrt(log(4))
mode(d::Rayleigh) = d.σ
var(d::Rayleigh{T}) where {T<:Real} = (2 - T(π)/2) * d.σ^2
std(d::Rayleigh{T}) where {T<:Real} = sqrt(2 - T(π)/2) * d.σ
skewness(d::Rayleigh{T}) where {T<:Real} = 2 * sqrtπ * (T(π) - 3)/(4 - T(π))^(3/2)
kurtosis(d::Rayleigh{T}) where {T<:Real} = -(6*T(π)^2 - 24*T(π) +16)/(4 - T(π))^2
entropy(d::Rayleigh{T}) where {T<:Real} = 1 - T(logtwo)/2 + T(MathConstants.γ)/2 + log(d.σ)
function pdf(d::Rayleigh{T}, x::Real) where T<:Real
    σ2 = d.σ^2
    x > 0 ? (x / σ2) * exp(- (x^2) / (2σ2)) : zero(T)
end
function logpdf(d::Rayleigh{T}, x::Real) where T<:Real
    σ2 = d.σ^2
    x > 0 ? log(x / σ2) - (x^2) / (2σ2) : -T(Inf)
end
logccdf(d::Rayleigh{T}, x::Real) where {T<:Real} = x > 0 ? - (x^2) / (2d.σ^2) : zero(T)
ccdf(d::Rayleigh, x::Real) = exp(logccdf(d, x))
cdf(d::Rayleigh, x::Real) = 1 - ccdf(d, x)
logcdf(d::Rayleigh, x::Real) = log1mexp(logccdf(d, x))
quantile(d::Rayleigh, p::Real) = sqrt(-2d.σ^2 * log1p(-p))
rand(d::Rayleigh) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Rayleigh) = d.σ * sqrt(2 * randexp(rng))
