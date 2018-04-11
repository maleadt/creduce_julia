__precompile__()
module CategoricalArrays
    using Compat
    using Missings
    using Printf
    using JSON
const DefaultRefType = UInt32
mutable struct CategoricalPool{T, R <: Integer, V}
    index::Vector{T}
    invindex::Dict{T, R}
    order::Vector{R}
    levels::Vector{T}
    valindex::Vector{V}
    ordered::Bool
    function CategoricalPool{T, R, V}(index::Vector{T},
                                      invindex::Dict{T, R},
                                      order::Vector{R},
                                      ordered::Bool) where {T, R, V}
        if iscatvalue(T)
            throw(ArgumentError("Level type $T cannot be a categorical value type"))
        end
        if !iscatvalue(V)
            throw(ArgumentError("Type $V is not a categorical value type"))
        end
        if leveltype(V) !== T
            throw(ArgumentError("Level type of the categorical value ($(leveltype(V))) and of the pool ($T) do not match"))
        end
        if reftype(V) !== R
            throw(ArgumentError("Reference type of the categorical value ($(reftype(V))) and of the pool ($R) do not match"))
        end
        levels = similar(index)
        levels[order] = index
        pool = new(index, invindex, order, levels, V[], ordered)
        buildvalues!(pool)
        return pool
    end
end
struct LevelsException{T, R} <: Exception
    levels::Vector{T}
end
struct OrderedLevelsException{T, S} <: Exception
    newlevel::S
    levels::Vector{T}
end
struct CategoricalValue{T, R <: Integer}
    level::R
    pool::CategoricalPool{T, R, CategoricalValue{T, R}}
end
struct CategoricalString{R <: Integer} <: AbstractString
    level::R
    pool::CategoricalPool{String, R, CategoricalString{R}}
end
abstract type AbstractCategoricalArray{T, N, R, V, C, U} <: AbstractArray{Union{C, U}, N} end
const AbstractCategoricalVector{T, R, V, C, U} = AbstractCategoricalArray{T, 1, R, V, C, U}
const AbstractCategoricalMatrix{T, R, V, C, U} = AbstractCategoricalArray{T, 2, R, V, C, U}
struct CategoricalArray{T, N, R <: Integer, V, C, U} <: AbstractCategoricalArray{T, N, R, V, C, U}
    refs::Array{R, N}
    pool::CategoricalPool{V, R, C}
    function CategoricalArray{T, N}(refs::Array{R, N},
                                    pool::CategoricalPool{V, R, C}) where
                                                 {T, N, R <: Integer, V, C}
        T === V || T == Union{V, Missing} || throw(ArgumentError("T ($T) must be equal to $V or Union{$V, Missing}"))
        U = T >: Missing ? Missing : Union{}
        new{T, N, R, V, C, U}(refs, pool)
    end
end
const CategoricalVector{T, R, V, C, U} = CategoricalArray{T, 1, V, C, U}
const CategoricalMatrix{T, R, V, C, U} = CategoricalArray{T, 2, V, C, U}
function buildindex(invindex::Dict{S, R}) where {S, R <: Integer}
    index = Vector{S}(undef, length(invindex))
    for (v, i) in invindex
        index[i] = v
    end
    return index
end
function buildinvindex(index::Vector{T}, ::Type{R}=DefaultRefType) where {T, R}
    if length(index) > typemax(R)
        throw(LevelsException{T, R}(index[typemax(R)+1:end]))
    end
    invindex = Dict{T, R}()
    for (i, v) in enumerate(index)
        invindex[v] = i
    end
    return invindex
end
function buildvalues!(pool::CategoricalPool)
    resize!(pool.valindex, length(levels(pool)))
    for i in eachindex(pool.valindex)
        v = catvalue(i, pool)
        @inbounds pool.valindex[i] = v
    end
    return pool.valindex
end
function buildorder!(order::Array{R},
                     invindex::Dict{S, R},
                     levels::Vector{S}) where {S, R <: Integer}
    for (i, v) in enumerate(levels)
        order[invindex[convert(S, v)]] = i
    end
    return order
end
function buildorder(invindex::Dict{S, R}, levels::Vector) where {S, R <: Integer}
    order = Vector{R}(undef, length(invindex))
    return buildorder!(order, invindex, levels)
end
function CategoricalPool(index::Vector{S},
                         invindex::Dict{S, T},
                         order::Vector{R},
                         ordered::Bool=false) where {S, T <: Integer, R <: Integer}
    invindex = convert(Dict{S, R}, invindex)
    C = catvaluetype(S, R)
    V = leveltype(C)
    CategoricalPool{V, R, C}(index, invindex, order, ordered)
end
CategoricalPool{T, R, C}(ordered::Bool=false) where {T, R, C} =
    CategoricalPool{T, R, C}(T[], Dict{T, R}(), R[], ordered)
CategoricalPool{T, R}(ordered::Bool=false) where {T, R} =
    CategoricalPool(T[], Dict{T, R}(), R[], ordered)
CategoricalPool{T}(ordered::Bool=false) where {T} =
    CategoricalPool{T, DefaultRefType}(ordered)
function CategoricalPool{T, R}(index::Vector,
                               ordered::Bool=false) where {T, R}
    invindex = buildinvindex(index, R)
    order = Vector{R}(1:length(index))
    CategoricalPool(index, invindex, order, ordered)
end
function CategoricalPool(index::Vector, ordered::Bool=false)
    invindex = buildinvindex(index)
    order = Vector{DefaultRefType}(1:length(index))
    return CategoricalPool(index, invindex, order, ordered)
end
function CategoricalPool(invindex::Dict{S, R},
                         ordered::Bool=false) where {S, R <: Integer}
    index = buildindex(invindex)
    order = Vector{DefaultRefType}(1:length(index))
    return CategoricalPool(index, invindex, order, ordered)
end
function CategoricalPool(index::Vector{S},
                         invindex::Dict{S, R},
                         ordered::Bool=false) where {S, R <: Integer}
    order = Vector{DefaultRefType}(1:length(index))
    return CategoricalPool(index, invindex, order, ordered)
end
function CategoricalPool(index::Vector{T},
                         levels::Vector{T},
                         ordered::Bool=false) where {T}
    invindex = buildinvindex(index)
    order = buildorder(invindex, levels)
    return CategoricalPool(index, invindex, order, ordered)
end
function CategoricalPool(invindex::Dict{S, R},
                         levels::Vector{S},
                         ordered::Bool=false) where {S, R <: Integer}
    index = buildindex(invindex)
    order = buildorder(invindex, levels)
    return CategoricalPool(index, invindex, order, ordered)
end
Base.convert(::Type{T}, pool::T) where {T <: CategoricalPool} = pool
Base.convert(::Type{CategoricalPool{S}}, pool::CategoricalPool{T, R}) where {S, T, R <: Integer} =
    convert(CategoricalPool{S, R}, pool)
function Base.convert(::Type{CategoricalPool{S, R}}, pool::CategoricalPool) where {S, R <: Integer}
    if length(levels(pool)) > typemax(R)
        throw(LevelsException{S, R}(levels(pool)[typemax(R)+1:end]))
    end
    indexS = convert(Vector{S}, pool.index)
    invindexS = convert(Dict{S, R}, pool.invindex)
    order = convert(Vector{R}, pool.order)
    return CategoricalPool(indexS, invindexS, order, pool.ordered)
end
function Base.show(io::IO, pool::CategoricalPool{T, R}) where {T, R}
    @printf(io, "%s{%s,%s}([%s])", typeof(pool).name, T, R,
                join(map(repr, levels(pool)), ","))
    pool.ordered && print(io, " with ordered levels")
end
Base.length(pool::CategoricalPool) = length(pool.index)
Base.getindex(pool::CategoricalPool, i::Integer) = pool.valindex[i]
Base.get(pool::CategoricalPool, level::Any) = pool.invindex[level]
Base.get(pool::CategoricalPool, level::Any, default::Any) = get(pool.invindex, level, default)
@inline function push_level!(pool::CategoricalPool{T, R}, level) where {T, R}
    x = convert(T, level)
    n = length(pool)
    if n >= typemax(R)
        throw(LevelsException{T, R}([level]))
    end
    i = R(n + 1)
    push!(pool.index, x)
    push!(pool.order, i)
    push!(pool.levels, x)
    push!(pool.valindex, catvalue(i, pool))
    i
end
@inline function Base.get!(pool::CategoricalPool, level::Any)
    get!(pool.invindex, level) do
        if isordered(pool)
            throw(OrderedLevelsException(level, pool.levels))
        end
        push_level!(pool, level)
    end
end
@inline function Base.push!(pool::CategoricalPool, level)
    get!(pool.invindex, level) do
        push_level!(pool, level)
    end
    return pool
end
function Base.append!(pool::CategoricalPool, levels)
    for level in levels
        push!(pool, level)
    end
    return pool
end
function Base.delete!(pool::CategoricalPool{S}, levels...) where S
    for level in levels
        levelS = convert(S, level)
        if haskey(pool.invindex, levelS)
            ind = pool.invindex[levelS]
            delete!(pool.invindex, levelS)
            splice!(pool.index, ind)
            ord = splice!(pool.order, ind)
            splice!(pool.levels, ord)
            splice!(pool.valindex, ind)
            for i in ind:length(pool)
                pool.invindex[pool.index[i]] -= 1
                pool.valindex[i] = catvalue(i, pool)
            end
            for i in 1:length(pool)
                pool.order[i] > ord && (pool.order[i] -= 1)
            end
        end
    end
    return pool
end
function levels!(pool::CategoricalPool{S, R}, newlevels::Vector) where {S, R}
    levs = convert(Vector{S}, newlevels)
    if !allunique(levs)
        throw(ArgumentError(string("duplicated levels found in levs: ",
                                   join(unique(filter(x->sum(levs.==x)>1, levs)), ", "))))
    end
    n = length(levs)
    if n > typemax(R)
        throw(LevelsException{S, R}(setdiff(levs, levels(pool))[typemax(R)-length(levels(pool))+1:end]))
    end
    if isempty(setdiff(pool.index, levs))
        append!(pool, setdiff(levs, pool.index))
    else
        empty!(pool.invindex)
        resize!(pool.index, n)
        resize!(pool.valindex, n)
        resize!(pool.order, n)
        resize!(pool.levels, n)
        for i in 1:n
            v = levs[i]
            pool.index[i] = v
            pool.invindex[v] = i
            pool.valindex[i] = catvalue(i, pool)
        end
    end
    buildorder!(pool.order, pool.invindex, levs)
    for (i, x) in enumerate(pool.order)
        pool.levels[x] = pool.index[i]
    end
    return pool
end
index(pool::CategoricalPool) = pool.index
Missings.levels(pool::CategoricalPool) = pool.levels
order(pool::CategoricalPool) = pool.order
isordered(pool::CategoricalPool) = pool.ordered
ordered!(pool::CategoricalPool, ordered) = (pool.ordered = ordered; pool)
function Base.showerror(io::IO, err::LevelsException{T, R}) where {T, R}
    levs = join(repr.(err.levels), ", ", " and ")
    print(io, "cannot store level(s) $levs since reference type $R can only hold $(typemax(R)) levels. Use the decompress function to make room for more levels.")
end
function Base.showerror(io::IO, err::OrderedLevelsException)
    print(io, "cannot add new level $(err.newlevel) since ordered pools cannot be extended implicitly. Use the levels! function to set new levels, or the ordered! function to mark the pool as unordered.")
end
const CatValue{R} = Union{CategoricalValue{T, R} where T,
                          CategoricalString{R}}
iscatvalue(::Type) = false
iscatvalue(::Type{Union{}}) = false
iscatvalue(::Type{<:CatValue}) = true
iscatvalue(x::Any) = iscatvalue(typeof(x))
leveltype(::Type{<:CategoricalValue{T}}) where {T} = T
leveltype(::Type{<:CategoricalString}) = String
leveltype(::Type) = throw(ArgumentError("Not a categorical value type"))
leveltype(x::Any) = leveltype(typeof(x))
reftype(::Type{<:CatValue{R}}) where {R} = R
reftype(x::Any) = reftype(typeof(x))
pool(x::CatValue) = x.pool
level(x::CatValue) = x.level
unwrap_catvaluetype(::Type{T}) where {T} = T
unwrap_catvaluetype(::Type{T}) where {T >: Missing} =
    Union{unwrap_catvaluetype(Missings.T(T)), Missing}
unwrap_catvaluetype(::Type{Union{}}) = Union{}
unwrap_catvaluetype(::Type{Any}) = Any
unwrap_catvaluetype(::Type{T}) where {T <: CatValue} = leveltype(T)
catvaluetype(::Type{T}, ::Type{R}) where {T >: Missing, R} =
    catvaluetype(Missings.T(T), R)
catvaluetype(::Type{T}, ::Type{R}) where {T <: CatValue, R} =
    catvaluetype(leveltype(T), R)
catvaluetype(::Type{Any}, ::Type{R}) where {R} =
    CategoricalValue{Any, R}
catvaluetype(::Type{T}, ::Type{R}) where {T, R} =
    CategoricalValue{T, R}
catvaluetype(::Type{<:AbstractString}, ::Type{R}) where {R} =
    CategoricalString{R}
catvaluetype(::Type{Union{}}, ::Type{R}) where {R} = CategoricalValue{Union{}, R}
Base.get(x::CatValue) = index(pool(x))[level(x)]
order(x::CatValue) = order(pool(x))[level(x)]
catvalue(level::Integer, pool::CategoricalPool{T, R, C}) where {T, R, C} =
    C(convert(R, level), pool)
Base.promote_rule(::Type{C}, ::Type{T}) where {C <: CatValue, T} = promote_type(leveltype(C), T)
Base.promote_rule(::Type{C}, ::Type{T}) where {C <: CategoricalString, T <: AbstractString} =
    promote_type(leveltype(C), T)
Base.promote_rule(::Type{C}, ::Type{Missing}) where {C <: CatValue} = Union{C, Missing}
Base.convert(::Type{Ref}, x::CatValue) = RefValue{leveltype(x)}(x)
Base.convert(::Type{String}, x::CatValue) = convert(String, get(x))
Base.convert(::Type{Any}, x::CatValue) = x
Base.convert(::Type{T}, x::T) where {T <: CatValue} = x
Base.convert(::Type{Union{T, Missing}}, x::T) where {T <: CatValue} = x
Base.convert(::Type{S}, x::CatValue) where {S} = convert(S, get(x))
    function Base.show(io::IO, x::CatValue)
        if get(io, :typeinfo, Any) === typeof(x)
            print(io, repr(x))
        elseif isordered(pool(x))
            @printf(io, "%s %s (%i/%i)",
                    typeof(x), repr(x),
                    order(x), length(pool(x)))
        else
            @printf(io, "%s %s", typeof(x), repr(x))
        end
    end
Base.print(io::IO, x::CatValue) = print(io, get(x))
Base.repr(x::CatValue) = repr(get(x))
@inline function Base.:(==)(x::CatValue, y::CatValue)
    if pool(x) === pool(y)
        return level(x) == level(y)
    else
        return get(x) == get(y)
    end
end
Base.:(==)(::CatValue, ::Missing) = missing
Base.:(==)(::Missing, ::CatValue) = missing
Base.:(==)(x::CatValue, y::WeakRef) = get(x) == y
Base.:(==)(x::WeakRef, y::CatValue) = y == x
Base.:(==)(x::CatValue, y::AbstractString) = get(x) == y
Base.:(==)(x::AbstractString, y::CatValue) = y == x
Base.:(==)(x::CatValue, y::Any) = get(x) == y
Base.:(==)(x::Any, y::CatValue) = y == x
@inline function Base.isequal(x::CatValue, y::CatValue)
    if pool(x) === pool(y)
        return level(x) == level(y)
    else
        return isequal(get(x), get(y))
    end
end
Base.isequal(x::CatValue, y::Any) = isequal(get(x), y)
Base.isequal(x::Any, y::CatValue) = isequal(y, x)
Base.isequal(::CatValue, ::Missing) = false
Base.isequal(::Missing, ::CatValue) = false
Base.in(x::CatValue, y::AbstractRange{T}) where {T<:Integer} = get(x) in y
Base.hash(x::CatValue, h::UInt) = hash(get(x), h)
function Base.isless(x::CatValue, y::CatValue)
    if pool(x) !== pool(y)
        throw(ArgumentError("CategoricalValue objects with different pools cannot be tested for order"))
    else
        return order(x) < order(y)
    end
end
function Base.:<(x::CatValue, y::CatValue)
    if pool(x) !== pool(y)
        throw(ArgumentError("CategoricalValue objects with different pools cannot be tested for order"))
    elseif !isordered(pool(x))
        throw(ArgumentError("Unordered CategoricalValue objects cannot be tested for order using <. Use isless instead, or call the ordered! function on the parent array to change this"))
    else
        return order(x) < order(y)
    end
end
Base.string(x::CategoricalString) = get(x)
Base.eltype(x::CategoricalString) = Char
Base.length(x::CategoricalString) = length(get(x))
Compat.lastindex(x::CategoricalString) = lastindex(get(x))
Base.sizeof(x::CategoricalString) = sizeof(get(x))
Base.nextind(x::CategoricalString, i::Int) = nextind(get(x), i)
Base.prevind(x::CategoricalString, i::Int) = prevind(get(x), i)
Base.next(x::CategoricalString, i::Int) = next(get(x), i)
Base.getindex(x::CategoricalString, i::Int) = getindex(get(x), i)
Base.codeunit(x::CategoricalString, i::Integer) = codeunit(get(x), i)
Base.ascii(x::CategoricalString) = ascii(get(x))
Base.isvalid(x::CategoricalString) = isvalid(get(x))
Base.isvalid(x::CategoricalString, i::Integer) = isvalid(get(x), i)
Base.match(r::Regex, s::CategoricalString,
           idx::Integer=start(s), add_opts::UInt32=UInt32(0)) =
    match(r, get(s), idx, add_opts)
    Base.matchall(r::Regex, s::CategoricalString; overlap::Bool=false) =
        matchall(r, get(s); overlap=overlap)
Base.collect(x::CategoricalString) = collect(get(x))
Base.reverse(x::CategoricalString) = reverse(get(x))
Compat.ncodeunits(x::CategoricalString) = ncodeunits(get(x))
JSON.lower(x::CatValue) = get(x)
end
