""" """ struct NormalInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    α::T
    β::T
    δ::T
    function NormalInverseGaussian{T}(μ::T, α::T, β::T, δ::T) where T
        new{T}(μ, α, β, δ)
    end
end
NormalInverseGaussian(μ::T, α::T, β::T, δ::T) where {T<:Real} = NormalInverseGaussian{T}(μ, α, β, δ)
NormalInverseGaussian(μ::Real, α::Real, β::Real, δ::Real) = NormalInverseGaussian(promote(μ, α, β, δ)...)
function NormalInverseGaussian(μ::Integer, α::Integer, β::Integer, δ::Integer)
    NormalInverseGaussian(Float64(μ), Float64(α), Float64(β), Float64(δ))
end
@distr_support NormalInverseGaussian -Inf Inf
function convert(::Type{NormalInverseGaussian{T}}, μ::Real, α::Real, β::Real, δ::Real) where T<:Real
    NormalInverseGaussian(T(μ), T(α), T(β), T(δ))
end
function convert(::Type{NormalInverseGaussian{T}}, d::NormalInverseGaussian{S}) where {T <: Real, S <: Real}
    NormalInverseGaussian(T(d.μ), T(d.α), T(d.β), T(d.δ))
end
params(d::NormalInverseGaussian) = (d.μ, d.α, d.β, d.δ)
@inline partype(d::NormalInverseGaussian{T}) where {T<:Real} = T
mean(d::NormalInverseGaussian) = d.μ + d.δ * d.β / sqrt(d.α^2 - d.β^2)
var(d::NormalInverseGaussian) = d.δ * d.α^2 / sqrt(d.α^2 - d.β^2)^3
skewness(d::NormalInverseGaussian) = 3d.β / (d.α * sqrt(d.δ * sqrt(d.α^2 - d.β^2)))
kurtosis(d::NormalInverseGaussian) = 3 * (1 + 4*d.β^2/d.α^2) / (d.δ * sqrt(d.α^2 - d.β^2))
function pdf(d::NormalInverseGaussian, x::Real)
    μ, α, β, δ = params(d)
    α * δ * besselk(1, α*sqrt(δ^2+(x - μ)^2)) / (π*sqrt(δ^2 + (x - μ)^2)) * exp(δ*sqrt(α^2 - β^2) + β*(x - μ))
end
function logpdf(d::NormalInverseGaussian, x::Real)
    μ, α, β, δ = params(d)
    log(α*δ) + log(besselk(1, α*sqrt(δ^2+(x-μ)^2))) - log(π*sqrt(δ^2+(x-μ)^2)) + δ*sqrt(α^2-β^2) + β*(x-μ)
end
