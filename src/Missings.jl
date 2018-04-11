__precompile__(true)
module Missings
using Compat
export allowmissing, disallowmissing, ismissing, missing, missings,
       Missing, MissingException, levels, coalesce
import Base: ==, !=, <, *, <=, !, +, -, ^, /, ~, &, |, xor
using Base: ismissing, missing, Missing, MissingException
@static if isdefined(Base, :adjoint) && !applicable(adjoint, missing)
    Base.adjoint(::Missing) = missing
end
if VERSION < v"0.7.0-DEV.3711"
    # Rounding and related functions from non-Missing type to Union{T, Missing}
    for f in (:(Base.ceil), :(Base.floor), :(Base.round), :(Base.trunc))
        @eval begin
            ($f)(::Type{T}, x::Any) where {T>:Missing} = $f(Missings.T(T), x)
            # to fix ambiguities
            ($f)(::Type{T}, x::Rational) where {T>:Missing} = $f(Missings.T(T), x)
            ($f)(::Type{T}, x::Rational{Bool}) where {T>:Missing} = $f(Missings.T(T), x)
        end
    end
end
if VERSION > v"0.7.0-DEV.3420"
    T(::Type{S}) where {S} = Core.Compiler.typesubtract(S, Missing)
else
    T(::Type{Union{T1, Missing}}) where {T1} = T1
    T(::Type{Missing}) = Union{}
    T(::Type{Any}) = Any
    T(::Type{S}) where {S} = Core.Inference.typesubtract(S, Missing)
end
missings(dims::Dims) = fill(missing, dims)
missings(::Type{T}, dims::Dims) where {T >: Missing} = fill!(Array{T}(undef, dims), missing)
missings(::Type{T}, dims::Dims) where {T} = fill!(Array{Union{T, Missing}}(undef, dims), missing)
missings(dims::Integer...) = missings(dims)
missings(::Type{T}, dims::Integer...) where {T} = missings(T, dims)
allowmissing(x::AbstractArray{T}) where {T} = convert(AbstractArray{Union{T, Missing}}, x)
disallowmissing(x::AbstractArray{T}) where {T} = convert(AbstractArray{Missings.T(T)}, x)
replace(itr, replacement) = EachReplaceMissing(itr, convert(eltype(itr), replacement))
struct EachReplaceMissing{T, U}
    x::T
    replacement::U
end
Compat.IteratorSize(::Type{<:EachReplaceMissing{T}}) where {T} =
    Compat.IteratorSize(T)
Compat.IteratorEltype(::Type{<:EachReplaceMissing{T}}) where {T} =
    Compat.IteratorEltype(T)
Base.length(itr::EachReplaceMissing) = length(itr.x)
Base.size(itr::EachReplaceMissing) = size(itr.x)
Base.start(itr::EachReplaceMissing) = start(itr.x)
Base.done(itr::EachReplaceMissing, state) = done(itr.x, state)
Base.eltype(itr::EachReplaceMissing) = Missings.T(eltype(itr.x))
@inline function Base.next(itr::EachReplaceMissing, state)
    v, s = next(itr.x, state)
    (v isa Missing ? itr.replacement : v, s)
end
@static if !isdefined(Base, :skipmissing)
export skipmissing
skipmissing(itr) = EachSkipMissing(itr)
struct EachSkipMissing{T}
    x::T
end
Compat.IteratorSize(::Type{<:EachSkipMissing}) =
    Base.SizeUnknown()
Compat.IteratorEltype(::Type{EachSkipMissing{T}}) where {T} =
    Compat.IteratorEltype(T)
Base.eltype(itr::EachSkipMissing) = Missings.T(eltype(itr.x))
@inline function Base.start(itr::EachSkipMissing)
    s = start(itr.x)
    v = missing
    @inbounds while !done(itr.x, s) && v isa Missing
        v, s = next(itr.x, s)
    end
    (v, s)
end
@inline Base.done(itr::EachSkipMissing, state) = ismissing(state[1]) && done(itr.x, state[2])
@inline function Base.next(itr::EachSkipMissing, state)
    v1, s = state
    v2 = missing
    @inbounds while !done(itr.x, s) && v2 isa Missing
        v2, s = next(itr.x, s)
    end
    (v1, (v2, s))
end
@inline function _next_nonmissing_ind(x::AbstractArray, s)
    idx = eachindex(x)
    @inbounds while !done(idx, s)
        i, new_s = next(idx, s)
        x[i] isa Missing || break
        s = new_s
    end
    s
end
@inline Base.start(itr::EachSkipMissing{<:AbstractArray}) =
    _next_nonmissing_ind(itr.x, start(eachindex(itr.x)))
@inline Base.done(itr::EachSkipMissing{<:AbstractArray}, state) =
    done(eachindex(itr.x), state)
@inline function Base.next(itr::EachSkipMissing{<:AbstractArray}, state)
    i, state = next(eachindex(itr.x), state)
    @inbounds v = itr.x[i]::eltype(itr)
    (v, _next_nonmissing_ind(itr.x, state))
end
end # isdefined
fail(itr) = EachFailMissing(itr)
struct EachFailMissing{T}
    x::T
end
Compat.IteratorSize(::Type{EachFailMissing{T}}) where {T} =
    Compat.IteratorSize(T)
Compat.IteratorEltype(::Type{EachFailMissing{T}}) where {T} =
    Compat.IteratorEltype(T)
Base.length(itr::EachFailMissing) = length(itr.x)
Base.size(itr::EachFailMissing) = size(itr.x)
Base.start(itr::EachFailMissing) = start(itr.x)
Base.done(itr::EachFailMissing, state) = done(itr.x, state)
Base.eltype(itr::EachFailMissing) = Missings.T(eltype(itr.x))
@inline function Base.next(itr::EachFailMissing, state)
    v, s = next(itr.x, state)
    # NOTE: v isa Missing currently gives incorrect code, cf. JuliaLang/julia#24177
    ismissing(v) && throw(MissingException("missing value encountered by Missings.fail"))
    (v::eltype(itr), s)
end
function levels(x)
    T = Missings.T(eltype(x))
    levs = convert(AbstractArray{T}, filter!(!ismissing, unique(x)))
    if hasmethod(isless, Tuple{T, T})
        try; sort!(levs); end
    end
    levs
end
@deprecate skip(itr) skipmissing(itr) false
end # module
