function suffstats(dt::Type{D}, xs...) where D<:Distribution
    argtypes = tuple(D, map(typeof, xs)...)
    error("suffstats is not implemented for $argtypes.")
end
""" """ fit_mle(D, x)
""" """ fit_mle(D, x, w)
fit_mle(dt::Type{D}, x::AbstractArray) where {D<:UnivariateDistribution} = fit_mle(D, suffstats(D, x))
fit_mle(dt::Type{D}, x::AbstractArray, w::AbstractArray) where {D<:UnivariateDistribution} = fit_mle(D, suffstats(D, x, w))
fit_mle(dt::Type{D}, x::AbstractMatrix) where {D<:MultivariateDistribution} = fit_mle(D, suffstats(D, x))
fit_mle(dt::Type{D}, x::AbstractMatrix, w::AbstractArray) where {D<:MultivariateDistribution} = fit_mle(D, suffstats(D, x, w))
fit(dt::Type{D}, x) where {D<:Distribution} = fit_mle(D, x)
fit(dt::Type{D}, args...) where {D<:Distribution} = fit_mle(D, args...)
