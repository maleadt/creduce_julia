abstract type AbstractMvLogNormal <: ContinuousMultivariateDistribution end
function insupport(::Type{D},x::AbstractVector{T}) where {T<:Real,D<:AbstractMvLogNormal}
    for i=1:length(x)
      @inbounds 0.0<x[i]<Inf ? continue : (return false)
    end
    true
end
insupport(l::AbstractMvLogNormal,x::AbstractVector{T}) where {T<:Real} = insupport(typeof(l),x)
assertinsupport(::Type{D},m::AbstractVector) where {D<:AbstractMvLogNormal} = @assert insupport(D,m) "Mean of LogNormal distribution should be strictly positive"
function _location!(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]/sqrt(1+S[i,i]/mn[i]/mn[i]))
    end
    μ
end
function _scale!(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix) where D<:AbstractMvLogNormal
    for j=1:length(mn)
      @simd for i=j:length(mn)
        @inbounds Σ[i,j] = Σ[j,i] = log(1 + S[j,i]/mn[i]/mn[j])
      end
    end
    Σ
end
function _location!(::Type{D},::Type{Val{:mean}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]) - S[i,i]/2
    end
    μ
end
function _location!(::Type{D},::Type{Val{:median}},md::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(md)
      @inbounds μ[i] = log(md[i])
    end
    μ
end
function _location!(::Type{D},::Type{Val{:mode}},mo::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mo)
      @inbounds μ[i] = log(mo[i]) + S[i,i]
    end
    μ
end
""" """ function location!(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m)) && length(m) == length(μ)
    assertinsupport(D,m)
    _location!(D,Val{s},m,S,μ)
end
""" """ function location(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _location!(D,Val{s},m,S,similar(m))
end
""" """ function scale!(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == size(Σ) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,Σ)
end
""" """ function scale(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,similar(S))
end
""" """ params!(::Type{D},m::AbstractVector,S::AbstractMatrix,μ::AbstractVector,Σ::AbstractMatrix) where {D<:AbstractMvLogNormal} = location!(D,:meancov,m,S,μ),scale!(D,:meancov,m,S,Σ)
""" """ params(::Type{D},m::AbstractVector,S::AbstractMatrix) where {D<:AbstractMvLogNormal} = params!(D,m,S,similar(m),similar(S))
""" """ struct MvLogNormal{T<:Real,Cov<:AbstractPDMat,Mean<:Union{Vector, ZeroVector}} <: AbstractMvLogNormal
    normal::MvNormal{T,Cov,Mean}
end
MvLogNormal(μ::Union{Vector,ZeroVector},Σ::AbstractPDMat) = MvLogNormal(MvNormal(μ,Σ))
MvLogNormal(Σ::AbstractPDMat) = MvLogNormal(MvNormal(ZeroVector(eltype(Σ),dim(Σ)),Σ))
MvLogNormal(μ::Vector,Σ::Matrix) = MvLogNormal(MvNormal(μ,Σ))
MvLogNormal(μ::Vector,σ::Vector) = MvLogNormal(MvNormal(μ,σ))
MvLogNormal(μ::Vector,s::Real) = MvLogNormal(MvNormal(μ,s))
MvLogNormal(Σ::Matrix) = MvLogNormal(MvNormal(Σ))
MvLogNormal(σ::Vector) = MvLogNormal(MvNormal(σ))
MvLogNormal(d::Int,s::Real) = MvLogNormal(MvNormal(d,s))
function convert(::Type{MvLogNormal{T}}, d::MvLogNormal) where T<:Real
    MvLogNormal(convert(MvNormal{T}, d.normal))
end
function convert(::Type{MvLogNormal{T}}, pars...) where T<:Real
    MvLogNormal(convert(MvNormal{T}, MvNormal(pars...)))
end
length(d::MvLogNormal) = length(d.normal)
params(d::MvLogNormal) = params(d.normal)
@inline partype(d::MvLogNormal{T}) where {T<:Real} = T
""" """ location(d::MvLogNormal) = mean(d.normal)
""" """ scale(d::MvLogNormal) = cov(d.normal)
mean(d::MvLogNormal) = exp.(mean(d.normal) + var(d.normal)/2)
""" """ median(d::MvLogNormal) = exp.(mean(d.normal))
""" """ mode(d::MvLogNormal) = exp.(mean(d.normal) - var(d.normal))
function cov(d::MvLogNormal)
    m = mean(d)
    return m*m'.*(exp.(cov(d.normal)) .- 1)
end
var(d::MvLogNormal) = diag(cov(d))
entropy(d::MvLogNormal) = length(d)*(1+log2π)/2 + logdetcov(d.normal)/2 + sum(mean(d.normal))
_rand!(d::MvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real} = exp!(_rand!(d.normal, x))
_logpdf(d::MvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real} = insupport(d, x) ? (_logpdf(d.normal, log.(x)) - sum(log.(x))) : -Inf
_pdf(d::MvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real} = insupport(d,x) ? _pdf(d.normal, log.(x))/prod(x) : 0.0
Base.show(io::IO,d::MvLogNormal) = show_multline(io, d, [(:dim, length(d)), (:μ, mean(d.normal)), (:Σ, cov(d.normal))])
