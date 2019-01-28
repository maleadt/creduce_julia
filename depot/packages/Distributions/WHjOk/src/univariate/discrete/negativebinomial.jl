""" """ struct NegativeBinomial{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T
    function NegativeBinomial{T}(r::T, p::T) where T
        @check_args(NegativeBinomial, r > zero(r))
        @check_args(NegativeBinomial, zero(p) < p <= one(p))
        new{T}(r, p)
    end
end
NegativeBinomial(r::T, p::T) where {T<:Real} = NegativeBinomial{T}(r, p)
NegativeBinomial(r::Real, p::Real) = NegativeBinomial(promote(r, p)...)
NegativeBinomial(r::Integer, p::Integer) = NegativeBinomial(Float64(r), Float64(p))
NegativeBinomial(r::Real) = NegativeBinomial(r, 0.5)
NegativeBinomial() = NegativeBinomial(1.0, 0.5)
@distr_support NegativeBinomial 0 Inf
function convert(::Type{NegativeBinomial{T}}, r::Real, p::Real) where T<:Real
    NegativeBinomial(T(r), T(p))
end
function convert(::Type{NegativeBinomial{T}}, d::NegativeBinomial{S}) where {T <: Real, S <: Real}
    NegativeBinomial(T(d.r), T(d.p))
end
params(d::NegativeBinomial) = (d.r, d.p)
@inline partype(d::NegativeBinomial{T}) where {T<:Real} = T
succprob(d::NegativeBinomial) = d.p
failprob(d::NegativeBinomial) = 1 - d.p
mean(d::NegativeBinomial) = (p = succprob(d); (1 - p) * d.r / p)
var(d::NegativeBinomial) = (p = succprob(d); (1 - p) * d.r / (p * p))
std(d::NegativeBinomial) = (p = succprob(d); sqrt((1 - p) * d.r) / p)
skewness(d::NegativeBinomial) = (p = succprob(d); (2 - p) / sqrt((1 - p) * d.r))
kurtosis(d::NegativeBinomial) = (p = succprob(d); 6 / d.r + (p * p) / ((1 - p) * d.r))
mode(d::NegativeBinomial) = (p = succprob(d); floor(Int,(1 - p) * (d.r - 1) / p))
@_delegate_statsfuns NegativeBinomial nbinom r p
rand(d::NegativeBinomial) = convert(Int, StatsFuns.RFunctions.nbinomrand(d.r, d.p))
struct RecursiveNegBinomProbEvaluator <: RecursiveProbabilityEvaluator
    r::Float64
    p0::Float64
end
RecursiveNegBinomProbEvaluator(d::NegativeBinomial) = RecursiveNegBinomProbEvaluator(d.r, failprob(d))
nextpdf(s::RecursiveNegBinomProbEvaluator, p::Float64, x::Integer) = ((x + s.r - 1) / x) * s.p0 * p
Base.broadcast!(::typeof(pdf), r::AbstractArray, d::NegativeBinomial, rgn::UnitRange) =
    _pdf!(r, d, rgn, RecursiveNegBinomProbEvaluator(d))
function Base.broadcast(::typeof(pdf), d::NegativeBinomial, X::UnitRange)
    r = similar(Array{promote_type(partype(d), eltype(X))}, axes(X))
    r .= pdf.(Ref(d),X)
end
function mgf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return ((1 - p) * exp(t))^r / (1 - p * exp(t))^r
end
function cf(d::NegativeBinomial, t::Real)
    r, p = params(d)
    return (((1 - p) * cis(t)) / (1 - p * cis(t)))^r
end
