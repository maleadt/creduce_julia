__precompile__(true)
module Nullables
if !isdefined(Base, :NullSafeTypes)
using Compat
struct NullException <: Exception end
struct Nullable{T}
    hasvalue::Bool
    value::T
    Nullable{T}() where {T} = new(false)
    Nullable{T}(value::T, hasvalue::Bool=true) where {T} = new(hasvalue, value)
end
Nullable(value::T, hasvalue::Bool=true) where {T} = Nullable{T}(value, hasvalue)
Nullable() = Nullable{Union{}}()
Base.eltype(::Type{Nullable{T}}) where {T} = T
Base.convert(::Type{Nullable{T}}, x::Nullable{T}) where {T} = x
Base.convert(::Type{Nullable   }, x::Nullable   ) = x
Base.convert(t::Type{Nullable{T}}, x::Any) where {T} = convert(t, convert(T, x))
function Base.convert(::Type{Nullable{T}}, x::Nullable) where T
    return isnull(x) ? Nullable{T}() : Nullable{T}(convert(T, get(x)))
end
Base.convert(::Type{Nullable{T}}, x::T) where {T<:Nullable} = Nullable{T}(x)
Base.convert(::Type{Nullable{T}}, x::T) where {T} = Nullable{T}(x)
Base.convert(::Type{Nullable   }, x::T) where {T} = Nullable{T}(x)
Base.convert(::Type{Nullable{T}}, ::Nothing) where {T} = Nullable{T}()
Base.convert(::Type{Nullable   }, ::Nothing) = Nullable{Union{}}()
Base.promote_rule(::Type{Nullable{S}}, ::Type{T}) where {S,T} = Nullable{promote_type(S, T)}
Base.promote_rule(::Type{Nullable{S}}, ::Type{Nullable{T}}) where {S,T} = Nullable{promote_type(S, T)}
Base.promote_op(op::Any, ::Type{Nullable{S}}, ::Type{Nullable{T}}) where {S,T} = Nullable{Base.promote_op(op, S, T)}
Base.promote_op(op::Type, ::Type{Nullable{S}}, ::Type{Nullable{T}}) where {S,T} = Nullable{Base.promote_op(op, S, T)}
function Base.show(io::IO, x::Nullable)
    if get(io, :typeinfo, Any) == typeof(x)
        if isnull(x)
            print(io, "#NULL")
        else
            show(IOContext(io, :typeinfo => eltype(x)), x.value)
        end
    else
        print(io, "Nullable{")
        show(io, eltype(x))
        print(io, "}(")
        if !isnull(x)
            show(IOContext(io, :typeinfo => eltype(x)), x.value)
        end
        print(io, ')')
    end
end
@inline function Base.get(x::Nullable{T}, y) where T
    if isbits(T)
        ifelse(isnull(x), y, x.value)
    else
        isnull(x) ? y : x.value
    end
end
Base.get(x::Nullable) = isnull(x) ? throw(NullException()) : x.value
unsafe_get(x::Nullable) = x.value
unsafe_get(x) = x
isnull(x::Nullable) = !x.hasvalue
isnull(x) = false
null_safe_op(f::Any, ::Type, ::Type...) = false
const NullSafeSignedInts = Union{Type{Int128}, Type{Int16}, Type{Int32},
                                 Type{Int64}, Type{Int8}}
const NullSafeUnsignedInts = Union{Type{Bool}, Type{UInt128}, Type{UInt16},
                                   Type{UInt32}, Type{UInt64}, Type{UInt8}}
const NullSafeInts = Union{NullSafeSignedInts, NullSafeUnsignedInts}
const NullSafeFloats = Union{Type{Float16}, Type{Float32}, Type{Float64}}
const NullSafeTypes = Union{NullSafeInts, NullSafeFloats}
const EqualOrLess = Union{typeof(isequal), typeof(isless)}
null_safe_op(::typeof(identity), ::Type{T}) where {T} = isbits(T)
null_safe_op(f::EqualOrLess, ::NullSafeTypes, ::NullSafeTypes) = true
null_safe_op(f::EqualOrLess, ::Type{Rational{S}}, ::Type{T}) where {S,T} =
    null_safe_op(f, T, S)
null_safe_op(::typeof(isequal), ::Type{Complex{S}}, ::Type{T}) where {S,T} =
    null_safe_op(isequal, T, S)
@inline function Base.isequal(x::Nullable{S}, y::Nullable{T}) where {S,T}
    if null_safe_op(isequal, S, T)
        (isnull(x) & isnull(y)) | (!isnull(x) & !isnull(y) & isequal(x.value, y.value))
    else
        (isnull(x) & isnull(y)) || (!isnull(x) & !isnull(y) && isequal(x.value, y.value))
    end
end
Base.isequal(x::Nullable{Union{}}, y::Nullable{Union{}}) = true
Base.isequal(x::Nullable{Union{}}, y::Nullable) = isnull(y)
Base.isequal(x::Nullable, y::Nullable{Union{}}) = isnull(x)
@inline function Base.isless(x::Nullable{S}, y::Nullable{T}) where {S,T}
    # NULL values are sorted last
    if null_safe_op(isless, S, T)
        (!isnull(x) & isnull(y)) | (!isnull(x) & !isnull(y) & isless(x.value, y.value))
    else
        (!isnull(x) & isnull(y)) || (!isnull(x) & !isnull(y) && isless(x.value, y.value))
    end
end
Base.isless(x::Nullable{Union{}}, y::Nullable{Union{}}) = false
Base.isless(x::Nullable{Union{}}, y::Nullable) = false
Base.isless(x::Nullable, y::Nullable{Union{}}) = !isnull(x)
Base.:(==)(x::Nullable, y::Nullable) = throw(NullException())
const nullablehash_seed = UInt === UInt64 ? 0x932e0143e51d0171 : 0xe51d0171
function Base.hash(x::Nullable, h::UInt)
    if isnull(x)
        return h + nullablehash_seed
    else
        return hash(x.value, h + nullablehash_seed)
    end
end
function Base.filter(p, x::Nullable{T}) where T
    if isbits(T)
        val = unsafe_get(x)
        Nullable{T}(val, !isnull(x) && p(val))
    else
        isnull(x) || p(unsafe_get(x)) ? x : Nullable{T}()
    end
end
nullable_returntype(::Type{T}) where {T} = isconcretetype(T) ? T : Union{}
function Base.map(f, x::Nullable{T}) where T
    S = Base.promote_op(f, T)
    if isconcretetype(S) && null_safe_op(f, T)
        Nullable(f(unsafe_get(x)), !isnull(x))
    else
        if isnull(x)
            Nullable{nullable_returntype(S)}()
        else
            Nullable(f(unsafe_get(x)))
        end
    end
end
hasvalue(x::Nullable) = x.hasvalue
hasvalue(x) = true
Base.all(f::typeof(hasvalue), t::Tuple) = f(t[1]) & all(f, Base.tail(t))
Base.all(f::typeof(hasvalue), t::Tuple{}) = true
for op in (+, -, abs, abs2)
    global null_safe_op(::typeof(op), ::NullSafeTypes) = true
    global null_safe_op(::typeof(op), ::Type{Complex{S}}) where {S} = null_safe_op(op, S)
    global null_safe_op(::typeof(op), ::Type{Rational{S}}) where {S} = null_safe_op(op, S)
end
null_safe_op(::typeof(~), ::NullSafeInts) = true
null_safe_op(::typeof(!), ::Type{Bool}) = true
const BINARY_OPS = [+, -, *, /, &, |, <<, >>, >>>]
VERSION < v"0.7.0-DEV.3383" && append!(BINARY_OPS, [Base.scalarmin, Base.scalarmax])
for op in BINARY_OPS
    # to fix ambiguities
    global null_safe_op(::typeof(op), ::NullSafeFloats, ::NullSafeFloats) = true
    global null_safe_op(::typeof(op), ::NullSafeSignedInts, ::NullSafeSignedInts) = true
    global null_safe_op(::typeof(op), ::NullSafeUnsignedInts, ::NullSafeUnsignedInts) = true
end
for op in (+, -, *, /)
    global null_safe_op(::typeof(op), ::Type{Complex{S}}, ::Type{T}) where {S,T} =
        null_safe_op(op, T, S)
    global null_safe_op(::typeof(op), ::Type{Rational{S}}, ::Type{T}) where {S,T} =
        null_safe_op(op, T, S)
end
Base.BroadcastStyle(::Type{<:Nullable}) = Base.Broadcast.Style{Nullable}()
Base.BroadcastStyle(::Base.Broadcast.Style{Nullable}, ::Base.Broadcast.DefaultArrayStyle{0}) =
    Base.Broadcast.Style{Nullable}()
Base.broadcast_indices(::Base.Broadcast.Style{Nullable}, A) = ()
Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(::Base.Broadcast.Style{Nullable}, A, I) = A
Base.Broadcast._broadcast_getindex_eltype(::Base.Broadcast.Style{Nullable}, A) = typeof(A)
_unsafe_get_eltype(x::Nullable) = eltype(x)
_unsafe_get_eltype(::Type{T}) where T = Type{T}
_unsafe_get_eltype(x) = typeof(x)
_nullable_eltype(f, A, As...) =
    Base._return_type(f, Base.Broadcast.maptoTuple(_unsafe_get_eltype, A, As...))
@inline function Base.broadcast(f, ::Base.Broadcast.Style{Nullable}, ::Nothing, ::Nothing, a...)
    nonnull = all(hasvalue, a)
    S = _nullable_eltype(f, a...)
    if Base.isconcretetype(S) && null_safe_op(f, Base.Broadcast.maptoTuple(_unsafe_get_eltype,
                                                                           a...).types...)
        Nullable{S}(f(map(unsafe_get, a)...), nonnull)
    else
        if nonnull
            Nullable(f(map(unsafe_get, a)...))
        else
            Nullable{nullable_returntype(S)}()
        end
    end
end
else
    using Base: NullSafeTypes
end
export Nullable, NullException, isnull, unsafe_get
end # module
