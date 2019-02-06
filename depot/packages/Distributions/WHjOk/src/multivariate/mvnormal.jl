""" """ abstract type AbstractMvNormal <: ContinuousMultivariateDistribution end
function _logpdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix)
    sqmahal!(r, d, x)
    r
end
""" """ struct MvNormal{T<:Real,Cov<:AbstractPDMat,Mean<:Union{Vector, ZeroVector}} <: AbstractMvNormal
    μ::Mean
    Σ::Cov
end
const MultivariateNormal = MvNormal  # for the purpose of backward compatibility
const IsoNormal  = MvNormal{Float64,ScalMat{Float64},Vector{Float64}}
const DiagNormal = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const FullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},Vector{Float64}}
const ZeroMeanFullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},ZeroVector{Float64}}
function MvNormal(μ::Union{Vector{T}, ZeroVector{T}}, Σ::AbstractPDMat{T}) where T<:Real
    dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    MvNormal{T,typeof(Σ), typeof(μ)}(μ, Σ)
    MvNormal(convert(AbstractArray{R}, μ), R(σ))
end
MvNormal(Σ::Matrix{T}) where {T<:Real} = MvNormal(PDMat(Σ))
MvNormal(σ::Vector{T}) where {T<:Real} = MvNormal(PDiagMat(abs2.(σ)))
struct MvNormalKnownCov{Cov<:AbstractPDMat}
end
function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    d = length(g)
    for j = 1:n
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i])
        end
    end
    multiply!(va, inv(n))
    MvNormal(mu, PDiagMat(va))
end
function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))
    for j = 1:n
        for i = 1:m
        end
    end
end
function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64})
    for j = 1:n
        for i = 1:m
        end
    end
end
function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    for j = 1:n
    end
end
