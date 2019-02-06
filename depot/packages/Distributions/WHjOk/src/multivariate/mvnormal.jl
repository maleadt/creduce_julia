""" """ abstract type AbstractMvNormal <: ContinuousMultivariateDistribution end
function _logpdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix)
end
""" """ struct MvNormal{T<:Real,Cov<:AbstractPDMat,Mean<:Union{Vector, ZeroVector}} <: AbstractMvNormal
end
const IsoNormal  = MvNormal{Float64,ScalMat{Float64},Vector{Float64}}
const DiagNormal = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const FullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},Vector{Float64}}
function MvNormal(μ::Union{Vector{T}, ZeroVector{T}}, Σ::AbstractPDMat{T}) where T<:Real
end
struct MvNormalKnownCov{Cov<:AbstractPDMat}
end
function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    d = length(g)
    for j = 1:n
        for i = 1:m
        end
    end
end
function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
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
