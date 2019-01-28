""" """ struct MvNormalCanon{T<:Real,P<:AbstractPDMat,V<:Union{Vector,ZeroVector}} <: AbstractMvNormal
    μ::V    # the mean vector
    h::V    # potential vector, i.e. inv(Σ) * μ
    J::P    # precision matrix, i.e. inv(Σ)
end
const FullNormalCanon = MvNormalCanon{Float64, PDMat{Float64,Matrix{Float64}},Vector{Float64}}
const DiagNormalCanon = MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const IsoNormalCanon  = MvNormalCanon{Float64,ScalMat{Float64},Vector{Float64}}
const ZeroMeanFullNormalCanon = MvNormalCanon{Float64,PDMat{Float64,Matrix{Float64}},ZeroVector{Float64}}
const ZeroMeanDiagNormalCanon = MvNormalCanon{Float64,PDiagMat{Float64,Vector{Float64}},ZeroVector{Float64}}
const ZeroMeanIsoNormalCanon  = MvNormalCanon{Float64,ScalMat{Float64},ZeroVector{Float64}}
function MvNormalCanon(μ::Vector{T}, h::Vector{T}, J::AbstractPDMat{T}) where T<:Real
    length(μ) == length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    MvNormalCanon{T,typeof(J),typeof(μ)}(μ, h, J)
end
function MvNormalCanon(μ::Vector{T}, h::Vector{T}, J::P) where {T<:Real, P<:AbstractPDMat}
    R = promote_type(T, eltype(J))
    MvNormalCanon(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, h), convert(AbstractArray{R}, J))
end
function MvNormalCanon(μ::Vector{T}, h::Vector{S}, J::P) where {T<:Real, S<:Real, P<:AbstractPDMat}
    R = Base.promote_eltype(μ, h, J)
    MvNormalCanon(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, h), convert(AbstractArray{R}, J))
end
function MvNormalCanon(J::P) where P<:AbstractPDMat
    z = ZeroVector(eltype(J), dim(J))
    MvNormalCanon{eltype(J),P,ZeroVector{eltype(J)}}(z, z, J)
end
function MvNormalCanon(h::Vector{T}, J::P) where {T<:Real, P<:AbstractPDMat}
    length(h) == dim(J) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    R = Base.promote_eltype(h, J)
    hh, JJ = convert(AbstractArray{R}, h), convert(AbstractArray{R}, J)
    MvNormalCanon{eltype(hh),typeof(JJ),typeof(hh)}(JJ \ hh, hh, JJ)
end
MvNormalCanon(h::Vector{T}, J::Matrix{T}) where {T<:Real} = MvNormalCanon(h, PDMat(J))
MvNormalCanon(h::Vector{T}, prec::Vector{T}) where {T<:Real} = MvNormalCanon(h, PDiagMat(prec))
MvNormalCanon(h::Vector{T}, prec::T) where {T<:Real} = MvNormalCanon(h, ScalMat(length(h), prec))
function MvNormalCanon(h::Vector{T}, J::VecOrMat{S}) where {T<:Real, S<:Real}
    R = Base.promote_eltype(h, J)
    MvNormalCanon(convert(AbstractArray{R}, h), convert(AbstractArray{R}, J))
end
function MvNormalCanon(h::Vector{T}, prec::S) where {T<:Real, S<:Real}
    R = Base.promote_eltype(h, prec)
    MvNormalCanon(convert(AbstractArray{R}, h), R(prec))
end
MvNormalCanon(J::Matrix) = MvNormalCanon(PDMat(J))
MvNormalCanon(prec::Vector) = MvNormalCanon(PDiagMat(prec))
MvNormalCanon(d::Int, prec) = MvNormalCanon(ScalMat(d, prec))
distrname(d::IsoNormalCanon) = "IsoNormalCanon"
distrname(d::DiagNormalCanon) = "DiagNormalCanon"
distrname(d::FullNormalCanon) = "FullNormalCanon"
distrname(d::ZeroMeanIsoNormalCanon) = "ZeroMeanIsoNormalCanon"
distrname(d::ZeroMeanDiagNormalCanon) = "ZeroMeanDiagormalCanon"
distrname(d::ZeroMeanFullNormalCanon) = "ZeroMeanFullNormalCanon"
function convert(::Type{MvNormalCanon{T}}, d::MvNormalCanon) where T<:Real
    MvNormalCanon(convert(AbstractArray{T}, d.μ), convert(AbstractArray{T}, d.h), convert(AbstractArray{T}, d.J))
end
function convert(::Type{MvNormalCanon{T}}, μ::V, h::V, J::AbstractPDMat) where {T<:Real,V<:Union{Vector, ZeroVector}}
    MvNormalCanon(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, h), convert(AbstractArray{T}, J))
end
meanform(d::MvNormalCanon) = MvNormal(d.μ, inv(d.J))
canonform(d::MvNormal{T,C,Vector{T}}) where {C, T<:Real} = (J = inv(d.Σ); MvNormalCanon(d.μ, J * d.μ, J))
canonform(d::MvNormal{T,C,ZeroVector{T}}) where {C, T<:Real} = MvNormalCanon(inv(d.Σ))
length(d::MvNormalCanon) = length(d.μ)
mean(d::MvNormalCanon) = convert(Vector{eltype(d.μ)}, d.μ)
params(d::MvNormalCanon) = (d.μ, d.h, d.J)
@inline partype(d::MvNormalCanon{T}) where {T<:Real} = T
var(d::MvNormalCanon) = diag(inv(d.J))
cov(d::MvNormalCanon) = Matrix(inv(d.J))
invcov(d::MvNormalCanon) = Matrix(d.J)
logdetcov(d::MvNormalCanon) = -logdet(d.J)
sqmahal(d::MvNormalCanon, x::AbstractVector) = quad(d.J, broadcast(-, x, d.μ))
sqmahal!(r::AbstractVector, d::MvNormalCanon, x::AbstractMatrix) = quad!(r, d.J, broadcast(-, x, d.μ))
unwhiten_winv!(J::AbstractPDMat, x::AbstractVecOrMat) = unwhiten!(inv(J), x)
unwhiten_winv!(J::PDiagMat, x::AbstractVecOrMat) = whiten!(J, x)
unwhiten_winv!(J::ScalMat, x::AbstractVecOrMat) = whiten!(J, x)
_rand!(rng::AbstractRNG, d::MvNormalCanon, x::AbstractMatrix) = add!(unwhiten_winv!(d.J, randn!(rng,x)), d.μ)
_rand!(d::MvNormalCanon, x::AbstractMatrix) = _rand!(Random.GLOBAL_RNG, d, x)
_rand!(rng::AbstractRNG, d::MvNormalCanon, x::AbstractVector) = add!(unwhiten_winv!(d.J, randn!(rng,x)), d.μ)
_rand!(d::MvNormalCanon, x::AbstractVector) = _rand!(Random.GLOBAL_RNG, d, x)
