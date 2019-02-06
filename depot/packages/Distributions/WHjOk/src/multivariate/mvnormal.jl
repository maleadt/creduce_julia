""" """ abstract type AbstractMvNormal <: ContinuousMultivariateDistribution end
insupport(d::AbstractMvNormal, x::AbstractVector) =
    length(d) == length(x) && allfinite(x)
mode(d::AbstractMvNormal) = mean(d)
modes(d::AbstractMvNormal) = [mean(d)]
function entropy(d::AbstractMvNormal)
    ldcd = logdetcov(d)
    (length(d) * (typeof(ldcd)(log2π) + 1) + ldcd)/2
end
mvnormal_c0(g::AbstractMvNormal) = -(length(g) * Float64(log2π) + logdetcov(g))/2
""" """ invcov(d::AbstractMvNormal)
""" """ logdetcov(d::AbstractMvNormal)
""" """ sqmahal(d::AbstractMvNormal, x::AbstractArray)
sqmahal(d::AbstractMvNormal, x::AbstractMatrix) = sqmahal!(Vector{promote_type(partype(d), eltype(x))}(undef, size(x, 2)), d, x)
_logpdf(d::AbstractMvNormal, x::AbstractVector) = mvnormal_c0(d) - sqmahal(d, x)/2
function _logpdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix)
    sqmahal!(r, d, x)
    c0 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        @inbounds r[i] = c0 - r[i]/2
    end
    r
end
_pdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix) = exp!(_logpdf!(r, d, x))
""" """ rand(rng::AbstractRNG, d::AbstractMvNormal) = _rand!(rng, d, Vector{eltype(d)}(undef, length(d)))
rand!(rng::AbstractRNG, d::AbstractMvNormal, x::VecOrMat) = _rand!(rng, d, x)
rand(rng::AbstractRNG, d::AbstractMvNormal, n::Integer) = _rand!(rng, d, Matrix{eltype(d)}(undef, length(d), n))
""" """ struct MvNormal{T<:Real,Cov<:AbstractPDMat,Mean<:Union{Vector, ZeroVector}} <: AbstractMvNormal
    μ::Mean
    Σ::Cov
end
const MultivariateNormal = MvNormal  # for the purpose of backward compatibility
const IsoNormal  = MvNormal{Float64,ScalMat{Float64},Vector{Float64}}
const DiagNormal = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const FullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},Vector{Float64}}
const ZeroMeanIsoNormal  = MvNormal{Float64,ScalMat{Float64},ZeroVector{Float64}}
const ZeroMeanDiagNormal = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},ZeroVector{Float64}}
const ZeroMeanFullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},ZeroVector{Float64}}
function MvNormal(μ::Union{Vector{T}, ZeroVector{T}}, Σ::AbstractPDMat{T}) where T<:Real
    dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    MvNormal{T,typeof(Σ), typeof(μ)}(μ, Σ)
end
function MvNormal(μ::Union{Vector{T}, ZeroVector{T}}, Σ::Cov) where {T<:Real, Cov<:AbstractPDMat}
    R = Base.promote_eltype(μ, Σ)
    MvNormal(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, Σ))
end
function MvNormal(Σ::Cov) where Cov<:AbstractPDMat
    T = eltype(Σ)
    MvNormal{T,Cov,ZeroVector{T}}(ZeroVector(T, dim(Σ)), Σ)
end
MvNormal(μ::Vector{T}, Σ::Matrix{T}) where {T<:Real} = MvNormal(μ, PDMat(Σ))
MvNormal(μ::Vector{T}, Σ::Union{Symmetric{T}, Hermitian{T}}) where {T<:Real} = MvNormal(μ, PDMat(Σ))
MvNormal(μ::Vector{T}, Σ::Diagonal{T}) where {T<:Real} = MvNormal(μ, PDiagMat(diag(Σ)))
MvNormal(μ::Vector{T}, σ::Vector{T}) where {T<:Real} = MvNormal(μ, PDiagMat(abs2.(σ)))
MvNormal(μ::Vector{T}, σ::T) where {T<:Real} = MvNormal(μ, ScalMat(length(μ), abs2(σ)))
function MvNormal(μ::Vector{T}, Σ::VecOrMat{S}) where {T<:Real,S<:Real}
    R = Base.promote_eltype(μ, Σ)
    MvNormal(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, Σ))
end
function MvNormal(μ::Vector{T}, σ::Real) where T<:Real
    R = Base.promote_eltype(μ, σ)
    MvNormal(convert(AbstractArray{R}, μ), R(σ))
end
MvNormal(Σ::Matrix{T}) where {T<:Real} = MvNormal(PDMat(Σ))
MvNormal(σ::Vector{T}) where {T<:Real} = MvNormal(PDiagMat(abs2.(σ)))
MvNormal(d::Int, σ::Real) = MvNormal(ScalMat(d, abs2(σ)))
function convert(::Type{MvNormal{T}}, d::MvNormal) where T<:Real
    MvNormal(convert(AbstractArray{T}, d.μ), convert(AbstractArray{T}, d.Σ))
end
function convert(::Type{MvNormal{T}}, μ::Union{Vector, ZeroVector}, Σ::AbstractPDMat) where T<:Real
    MvNormal(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end
distrname(d::IsoNormal) = "IsoNormal"    # Note: IsoNormal, etc are just alias names
distrname(d::DiagNormal) = "DiagNormal"
distrname(d::FullNormal) = "FullNormal"
distrname(d::ZeroMeanIsoNormal) = "ZeroMeanIsoNormal"
distrname(d::ZeroMeanDiagNormal) = "ZeroMeanDiagNormal"
distrname(d::ZeroMeanFullNormal) = "ZeroMeanFullNormal"
Base.show(io::IO, d::MvNormal) =
    show_multline(io, d, [(:dim, length(d)), (:μ, mean(d)), (:Σ, cov(d))])
length(d::MvNormal) = length(d.μ)
mean(d::MvNormal) = convert(Vector, d.μ)
params(d::MvNormal) = (d.μ, d.Σ)
@inline partype(d::MvNormal{T}) where {T<:Real} = T
var(d::MvNormal) = diag(d.Σ)
cov(d::MvNormal) = Matrix(d.Σ)
invcov(d::MvNormal) = Matrix(inv(d.Σ))
logdetcov(d::MvNormal) = logdet(d.Σ)
sqmahal(d::MvNormal, x::AbstractVector) = invquad(d.Σ, broadcast(-, x, d.μ))
sqmahal!(r::AbstractVector, d::MvNormal, x::AbstractMatrix) =
    invquad!(r, d.Σ, broadcast(-, x, d.μ))
gradlogpdf(d::MvNormal, x::Vector) = -(d.Σ \ broadcast(-, x, d.μ))
_rand!(d::MvNormal, x::VecOrMat) = _rand!(Random.GLOBAL_RNG, d, x)
_rand!(rng::AbstractRNG, d::MvNormal, x::VecOrMat) = add!(unwhiten!(d.Σ, randn!(rng, x)), d.μ)
function _rand_abstr!(rng::AbstractRNG, d::MvNormal, x::AbstractVecOrMat)
    for i = 1:length(x)
        @inbounds x[i] = randn()
    end
    add!(unwhiten!(d.Σ, x), d.μ)
end
_rand_abstr!(d::MvNormal, x::AbstractVecOrMat) = _rand_abstr!(Random.GLOBAL_RNG, d, x)
_rand!(rng::AbstractRNG, d::MvNormal, x::AbstractMatrix) = _rand_abstr!(rng, d, x)
_rand!(d::MvNormal, x::AbstractMatrix) = _rand!(Random.GLOBAL_RNG, d, x)
_rand!(rng::AbstractRNG, d::MvNormal, x::AbstractVector) = _rand_abstr!(rng, d, x)
_rand!(d::MvNormal, x::AbstractVector) = _rand!(Random.GLOBAL_RNG, d, x)
struct MvNormalKnownCov{Cov<:AbstractPDMat}
end
function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    d = length(g)
    (size(x,1) == d && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    μ = BLAS.gemv('N', inv(sum(w)), x, vec(w))
    MvNormal(μ, g.Σ)
end
struct MvNormalStats <: SufficientStats
    s::Vector{Float64}  # (weighted) sum of x
    m::Vector{Float64}  # (weighted) mean of x
    s2::Matrix{Float64} # (weighted) sum of (x-μ) * (x-μ)'
    tw::Float64         # total sample weight
end
function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64})
    d = size(x, 1)
    n = size(x, 2)
    s = vec(sum(x, dims=2))
    m = s * inv(n)
    z = x .- m
    s2 = z * z'
    MvNormalStats(s, m, s2, Float64(n))
end
function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64}, w::Array{Float64})
    d = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))
    tw = sum(w)
    s = x * vec(w)
    m = s * inv(tw)
    z = similar(x)
    for j = 1:n
        xj = view(x,:,j)
    end
    C = BLAS.syrk('U', 'N', inv_sw, z)
    LinearAlgebra.copytri!(C, 'U')
    MvNormal(mu, PDMat(C))
end
function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    mu = vec(mean(x, dims=2))
    va = zeros(Float64, m)
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
    inv_sw = 1.0 / sum(w)
    mu = BLAS.gemv('N', inv_sw, x, w)
    va = zeros(Float64, m)
    for j = 1:n
        @inbounds wj = w[j]
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i]) * wj
        end
    end
    multiply!(va, inv_sw)
    MvNormal(mu, PDiagMat(va))
end
function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64})
    m = size(x, 1)
    n = size(x, 2)
    mu = vec(mean(x, dims=2))
    va = 0.
    for j = 1:n
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i])
        end
        va += va_j
    end
    MvNormal(mu, ScalMat(m, va / (m * n)))
end
function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))
    sw = sum(w)
    inv_sw = 1.0 / sw
    mu = BLAS.gemv('N', inv_sw, x, w)
    va = 0.
    for j = 1:n
        @inbounds wj = w[j]
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i]) * wj
        end
        va += va_j
    end
    MvNormal(mu, ScalMat(m, va / (m * sw)))
end
