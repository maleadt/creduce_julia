""" """ module Arpack
using Libdl
const depsfile = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(depsfile)
else
    throw(ErrorException("""
"""))
end
using LinearAlgebra: BlasFloat, BlasInt, Diagonal, I, SVD, UniformScaling,
                     rmul!, qr
import LinearAlgebra
function eigs(A::AbstractMatrix{T}, ::UniformScaling; kwargs...) where T
    if nev > nevmax
    end
    if isa(which,AbstractString)
    end
    if (which != :LM && which != :SM && which != :LR && which != :SR &&
        which != :LI && which != :SI && which != :BE)
        throw(ArgumentError("which must be :LM, :SM, :LR, :SR, :LI, :SI, or :BE, got $(repr(which))"))
    end
end
struct SVDAugmented{T,S} <: AbstractArray{T, 2}
end
struct AtA_or_AAt{T,S} <: AbstractArray{T, 2}
end
function AtA_or_AAt(A)
end
function LinearAlgebra.mul!(y::StridedVector{T}, A::AtA_or_AAt{T}, x::StridedVector{T}) where T
    if size(A.A, 1) >= size(A.A, 2)
    end
end
function svds(A::AbstractMatrix{T}; kwargs...) where T
    if nsv < 1
    end
    if ritzvec
        if size(X, 1) >= size(X, 2)
        end
        return (SVD(zeros(eltype(svals), n, 0),
                    zeros(eltype(svals), 0, m)),
                    ex[2], ex[3], ex[4], ex[5])
    end
end
end # module
