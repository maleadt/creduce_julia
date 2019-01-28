""" """ struct Chisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    Chisq{T}(ν::T) where {T} = (@check_args(Chisq, ν > zero(ν)); new{T}(ν))
end
Chisq(ν::T) where {T<:Real} = Chisq{T}(ν)
Chisq(ν::Integer) = Chisq(Float64(ν))
@distr_support Chisq 0.0 Inf
dof(d::Chisq) = d.ν
params(d::Chisq) = (d.ν,)
@inline partype(d::Chisq{T}) where {T<:Real} = T
convert(::Type{Chisq{T}}, ν::Real) where {T<:Real} = Chisq(T(ν))
convert(::Type{Chisq{T}}, d::Chisq{S}) where {T <: Real, S <: Real} = Chisq(T(d.ν))
mean(d::Chisq) = d.ν
var(d::Chisq) = 2d.ν
skewness(d::Chisq) = sqrt(8 / d.ν)
kurtosis(d::Chisq) = 12 / d.ν
mode(d::Chisq{T}) where {T<:Real} = d.ν > 2 ? d.ν - 2 : zero(T)
function median(d::Chisq; approx::Bool=false)
    if approx
        return d.ν * (1 - 2 / (9 * d.ν))^3
    else
        return quantile(d, 1//2)
    end
end
function entropy(d::Chisq)
    hν = d.ν/2
    hν + logtwo + lgamma(hν) + (1 - hν) * digamma(hν)
end
@_delegate_statsfuns Chisq chisq ν
mgf(d::Chisq, t::Real) = (1 - 2 * t)^(-d.ν/2)
cf(d::Chisq, t::Real) = (1 - 2 * im * t)^(-d.ν/2)
gradlogpdf(d::Chisq{T}, x::Real) where {T<:Real} =  x > 0 ? (d.ν/2 - 1) / x - 1//2 : zero(T)
_chisq_rand(ν::Float64) = StatsFuns.RFunctions.chisqrand(ν)
rand(d::Chisq) = _chisq_rand(d.ν)
