const ≅ = isequal
""" """ function recode! end
recode!(dest::AbstractArray, src::AbstractArray, pairs::Pair...) =
    recode!(dest, src, nothing, pairs...)
recode!(dest::CategoricalArray, src::AbstractArray, pairs::Pair...) =
    recode!(dest, src, nothing, pairs...)
recode!(dest::CategoricalArray, src::CategoricalArray, pairs::Pair...) =
    recode!(dest, src, nothing, pairs...)
function recode!(dest::AbstractArray{T}, src::AbstractArray, default::Any, pairs::Pair...) where {T}
    if length(dest) != length(src)
        throw(DimensionMismatch("dest and src must be of the same length (got $(length(dest)) and $(length(src)))"))
    end
    @inbounds for i in eachindex(dest, src)
        x = src[i]
        for j in 1:length(pairs)
            p = pairs[j]
            if ((isa(p.first, Union{AbstractArray, Tuple}) && any(x ≅ y for y in p.first)) ||
                x ≅ p.first)
                dest[i] = p.second
                @goto nextitem
            end
        end
        if ismissing(x)
            eltype(dest) >: Missing ||
                throw(MissingException("missing value found, but dest does not support them: " *
                                       "recode them to a supported value"))
            dest[i] = missing
        elseif default isa Nothing
            try
                dest[i] = x
            catch err
                isa(err, MethodError) || rethrow(err)
                throw(ArgumentError("cannot `convert` value $(repr(x)) (of type $(typeof(x))) to type of recoded levels ($T). " *
                                    "This will happen with recode() when not all original levels are recoded " *
                                    "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
            end
        else
            dest[i] = default
        end
        @label nextitem
    end
    dest
end
function recode!(dest::CategoricalArray{T}, src::AbstractArray, default::Any, pairs::Pair...) where {T}
    if length(dest) != length(src)
        throw(DimensionMismatch("dest and src must be of the same length (got $(length(dest)) and $(length(src)))"))
    end
    vals = T[p.second for p in pairs]
    default !== nothing && push!(vals, default)
    levels!(dest.pool, filter!(!ismissing, unique(vals)))
    dupvals = length(vals) != length(levels(dest.pool))
    drefs = dest.refs
    pairmap = [ismissing(v) ? 0 : get(dest.pool, v) for v in vals]
    defaultref = default === nothing || ismissing(default) ? 0 : get(dest.pool, default)
    @inbounds for i in eachindex(drefs, src)
        x = src[i]
        for j in 1:length(pairs)
            p = pairs[j]
            if ((isa(p.first, Union{AbstractArray, Tuple}) && any(x ≅ y for y in p.first)) ||
                x ≅ p.first)
                drefs[i] = dupvals ? pairmap[j] : j
                @goto nextitem
            end
        end
        if ismissing(x)
            eltype(dest) >: Missing ||
                throw(MissingException("missing value found, but dest does not support them: " *
                                       "recode them to a supported value"))
            drefs[i] = 0
        elseif default === nothing
            try
                dest[i] = x # Need a dictionary lookup, and potentially adding a new level
            catch err
                isa(err, MethodError) || rethrow(err)
                throw(ArgumentError("cannot `convert` value $(repr(x)) (of type $(typeof(x))) to type of recoded levels ($T). " *
                                    "This will happen with recode() when not all original levels are recoded "*
                                    "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
            end
        else
            drefs[i] = defaultref
        end
        @label nextitem
    end
    oldlevels = setdiff(levels(dest), vals)
    filter!(!ismissing, oldlevels)
    if hasmethod(isless, (eltype(oldlevels), eltype(oldlevels)))
        sort!(oldlevels)
    end
    levels!(dest, union(oldlevels, levels(dest)))
    dest
end
function recode!(dest::CategoricalArray{T}, src::CategoricalArray, default::Any, pairs::Pair...) where {T}
    if length(dest) != length(src)
        throw(DimensionMismatch("dest and src must be of the same length (got $(length(dest)) and $(length(src)))"))
    end
    vals = T[p.second for p in pairs]
    if default === nothing
        srclevels = levels(src)
        firsts = (p.first for p in pairs)
        keptlevels = Vector{T}(undef, 0)
        sizehint!(keptlevels, length(srclevels))
        for l in srclevels
            if !(any(x -> x ≅ l, firsts) ||
                 any(f -> isa(f, Union{AbstractArray, Tuple}) && any(l ≅ y for y in f), firsts))
                try
                    push!(keptlevels, l)
                catch err
                    isa(err, MethodError) || rethrow(err)
                    throw(ArgumentError("cannot `convert` value $(repr(l)) (of type $(typeof(l))) to type of recoded levels ($T). " *
                                        "This will happen with recode() when not all original levels are recoded " *
                                        "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
                end
            end
        end
        levs, ordered = mergelevels(isordered(src), keptlevels, filter!(!ismissing, unique(vals)))
    else
        push!(vals, default)
        levs = filter!(!ismissing, unique(vals))
        ordered = false
    end
    srcindex = src.pool === dest.pool ? copy(index(src.pool)) : index(src.pool)
    levels!(dest.pool, levs)
    drefs = dest.refs
    srefs = src.refs
    origmap = [get(dest.pool, v, 0) for v in srcindex]
    indexmap = Vector{DefaultRefType}(undef, length(srcindex)+1)
    indexmap[1] = 0
    for p in pairs
        if ((isa(p.first, Union{AbstractArray, Tuple}) && any(ismissing, p.first)) ||
            ismissing(p.first))
            indexmap[1] = get(dest.pool, p.second)
            break
        end
    end
    pairmap = [ismissing(p.second) ? 0 : get(dest.pool, p.second) for p in pairs]
    ordered && (ordered = issorted(pairmap))
    ordered!(dest, ordered)
    defaultref = default === nothing || ismissing(default) ? 0 : get(dest.pool, default)
    @inbounds for (i, l) in enumerate(srcindex)
        for j in 1:length(pairs)
            p = pairs[j]
            if ((isa(p.first, Union{AbstractArray, Tuple}) && any(l ≅ y for y in p.first)) ||
                l ≅ p.first)
                indexmap[i+1] = pairmap[j]
                @goto nextitem
            end
        end
        if default === nothing
            indexmap[i+1] = origmap[i]
        else
            indexmap[i+1] = defaultref
        end
        @label nextitem
    end
    @inbounds for i in eachindex(drefs)
        v = indexmap[srefs[i]+1]
        if !(eltype(dest) >: Missing)
            v > 0 || throw(MissingException("missing value found, but dest does not support them: " *
                                            "recode them to a supported value"))
        end
        drefs[i] = v
    end
    dest
end
""" """ recode!(a::AbstractArray, default::Any, pairs::Pair...) =
    recode!(a, a, default, pairs...)
recode!(a::AbstractArray, pairs::Pair...) = recode!(a, a, nothing, pairs...)
promote_valuetype(x::Pair{K, V}) where {K, V} = V
promote_valuetype(x::Pair{K, V}, y::Pair...) where {K, V} = promote_type(V, promote_valuetype(y...))
keytype_hasmissing(x::Pair{K}) where {K} = K === Missing
keytype_hasmissing(x::Pair{K}, y::Pair...) where {K} = K === Missing || keytype_hasmissing(y...)
""" """ function recode end
recode(a::AbstractArray, pairs::Pair...) = recode(a, nothing, pairs...)
recode(a::CategoricalArray, pairs::Pair...) = recode(a, nothing, pairs...)
function recode(a::AbstractArray, default::Any, pairs::Pair...)
    V = promote_valuetype(pairs...)
    T = default isa Nothing ? V : promote_type(typeof(default), V)
    if T === Missing && !isa(default, Missing)
        dest = Array{Union{eltype(a), Missing}}(undef, size(a))
    elseif T >: Missing || default isa Missing || (eltype(a) >: Missing && !keytype_hasmissing(pairs...))
        dest = Array{Union{T, Missing}}(undef, size(a))
    else
        dest = Array{Missings.T(T)}(undef, size(a))
    end
    recode!(dest, a, default, pairs...)
end
function recode(a::CategoricalArray{S, N, R}, default::Any, pairs::Pair...) where {S, N, R}
    V = promote_valuetype(pairs...)
    T = default isa Nothing ? V : promote_type(typeof(default), V)
    if T === Missing && !isa(default, Missing)
        dest = CategoricalArray{Union{S, Missing}, N, R}(undef, size(a))
    elseif T >: Missing || default isa Missing || (eltype(a) >: Missing && !keytype_hasmissing(pairs...))
        dest = CategoricalArray{Union{T, Missing}, N, R}(undef, size(a))
    else
        dest = CategoricalArray{Missings.T(T), N, R}(undef, size(a))
    end
    recode!(dest, a, default, pairs...)
end
function Base.replace(a::CategoricalArray{S, N, R}, pairs::Pair...) where {S, N, R}
    T = promote_valuetype(pairs...)
    if keytype_hasmissing(pairs...)
        dest = CategoricalArray{promote_type(Missings.T(S), T), N, R}(undef, size(a))
    else
        dest = CategoricalArray{promote_type(S, T), N, R}(undef, size(a))
    end
    recode!(dest, a, nothing, pairs...)
end
if VERSION >= v"0.7.0-"
    Base.replace!(a::CategoricalArray, pairs::Pair...) = recode!(a, pairs...)
else
    replace!(a::CategoricalArray, pairs::Pair...) = recode!(a, pairs...)
end
