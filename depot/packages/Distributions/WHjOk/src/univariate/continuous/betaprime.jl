""" """ struct BetaPrime{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    function BetaPrime{T}(α::T, β::T) where T
        @check_args(BetaPrime, α > zero(α) && β > zero(β))
        new{T}(α, β)
    end
end
BetaPrime(α::T, β::T) where {T<:Real} = BetaPrime{T}(α, β)
BetaPrime(α::Real, β::Real) = BetaPrime(promote(α, β)...)
BetaPrime(α::Integer, β::Integer) = BetaPrime(Float64(α), Float64(β))
BetaPrime(α::Real) = BetaPrime(α, α)
BetaPrime() = BetaPrime(1.0, 1.0)
@distr_support BetaPrime 0.0 Inf
function convert(::Type{BetaPrime{T}}, α::Real, β::Real) where T<:Real
    BetaPrime(T(α), T(β))
end
function convert(::Type{BetaPrime{T}}, d::BetaPrime{S}) where {T <: Real, S <: Real}
    BetaPrime(T(d.α), T(d.β))
end
params(d::BetaPrime) = (d.α, d.β)
@inline partype(d::BetaPrime{T}) where {T<:Real} = T
function mean(d::BetaPrime{T}) where T<:Real
    ((α, β) = params(d); β > 1 ? α / (β - 1) : T(NaN))
end
function mode(d::BetaPrime{T}) where T<:Real
    ((α, β) = params(d); α > 1 ? (α - 1) / (β + 1) : zero(T))
end
function var(d::BetaPrime{T}) where T<:Real
    (α, β) = params(d)
    β > 2 ? α * (α + β - 1) / ((β - 2) * (β - 1)^2) : T(NaN)
end
function skewness(d::BetaPrime{T}) where T<:Real
    (α, β) = params(d)
    if β > 3
        s = α + β - 1
        2(α + s) / (β - 3) * sqrt((β - 2) / (α * s))
    else
        return T(NaN)
    end
end
function logpdf(d::BetaPrime{T}, x::Real) where T<:Real
    (α, β) = params(d)
    if x < 0
        T(-Inf)
    else
        (α - 1) * log(x) - (α + β) * log1p(x) - lbeta(α, β)
    end
end
pdf(d::BetaPrime, x::Real) = exp(logpdf(d, x))
cdf(d::BetaPrime{T}, x::Real) where {T<:Real} = x <= 0 ? zero(T) : betacdf(d.α, d.β, x / (1 + x))
ccdf(d::BetaPrime{T}, x::Real) where {T<:Real} = x <= 0 ? one(T) : betaccdf(d.α, d.β, x / (1 + x))
logcdf(d::BetaPrime{T}, x::Real) where {T<:Real} =  x <= 0 ? T(-Inf) : betalogcdf(d.α, d.β, x / (1 + x))
logccdf(d::BetaPrime{T}, x::Real) where {T<:Real} =  x <= 0 ? zero(T) : betalogccdf(d.α, d.β, x / (1 + x))
quantile(d::BetaPrime, p::Real) = (x = betainvcdf(d.α, d.β, p); x / (1 - x))
cquantile(d::BetaPrime, p::Real) = (x = betainvccdf(d.α, d.β, p); x / (1 - x))
invlogcdf(d::BetaPrime, p::Real) = (x = betainvlogcdf(d.α, d.β, p); x / (1 - x))
invlogccdf(d::BetaPrime, p::Real) = (x = betainvlogccdf(d.α, d.β, p); x / (1 - x))
function rand(d::BetaPrime)
    (α, β) = params(d)
    rand(Gamma(α)) / rand(Gamma(β))
end
