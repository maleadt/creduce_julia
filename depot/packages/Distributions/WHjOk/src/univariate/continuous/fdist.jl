""" """ struct FDist{T<:Real} <: ContinuousUnivariateDistribution
    ν1::T
    ν2::T
    function FDist{T}(ν1::T, ν2::T) where T
        @check_args(FDist, ν1 > zero(ν1) && ν2 > zero(ν2))
        new{T}(ν1, ν2)
    end
end
FDist(ν1::T, ν2::T) where {T<:Real} = FDist{T}(ν1, ν2)
FDist(ν1::Integer, ν2::Integer) = FDist(Float64(ν1), Float64(ν2))
FDist(ν1::Real, ν2::Real) = FDist(promote(ν1, ν2)...)
@distr_support FDist 0.0 Inf
function convert(::Type{FDist{T}}, ν1::S, ν2::S) where {T <: Real, S <: Real}
    FDist(T(ν1), T(ν2))
end
function convert(::Type{FDist{T}}, d::FDist{S}) where {T <: Real, S <: Real}
    FDist(T(d.ν1), T(d.ν2))
end
params(d::FDist) = (d.ν1, d.ν2)
@inline partype(d::FDist{T}) where {T<:Real} = T
mean(d::FDist{T}) where {T<:Real} = (ν2 = d.ν2; ν2 > 2 ? ν2 / (ν2 - 2) : T(NaN))
function mode(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    ν1 > 2 ? ((ν1 - 2)/ν1) * (ν2 / (ν2 + 2)) : zero(T)
end
function var(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    ν2 > 4 ? 2ν2^2 * (ν1 + ν2 - 2) / (ν1 * (ν2 - 2)^2 * (ν2 - 4)) : T(NaN)
end
function skewness(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    if ν2 > 6
        return (2ν1 + ν2 - 2) * sqrt(8(ν2 - 4)) / ((ν2 - 6) * sqrt(ν1 * (ν1 + ν2 - 2)))
    else
        return T(NaN)
    end
end
function kurtosis(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    if ν2 > 8
        a = ν1 * (5ν2 - 22) * (ν1 + ν2 - 2) + (ν2 - 4) * (ν2 - 2)^2
        b = ν1 * (ν2 - 6) * (ν2 - 8) * (ν1 + ν2 - 2)
        return 12a / b
    else
        return T(NaN)
    end
end
function entropy(d::FDist)
    (ν1, ν2) = params(d)
    hν1 = ν1/2
    hν2 = ν2/2
    hs = (ν1 + ν2)/2
    return log(ν2 / ν1) + lgamma(hν1) + lgamma(hν2) - lgamma(hs) +
        (1 - hν1) * digamma(hν1) + (-1 - hν2) * digamma(hν2) +
        hs * digamma(hs)
end
@_delegate_statsfuns FDist fdist ν1 ν2
rand(d::FDist) = StatsFuns.RFunctions.fdistrand(d.ν1, d.ν2)