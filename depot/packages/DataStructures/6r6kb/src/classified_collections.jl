mutable struct ClassifiedCollections{K, Collection}
    map::Dict{K, Collection}
end
ClassifiedCollections(K::Type, C::Type) = ClassifiedCollections{K, C}(Dict{K,C}())
classified_lists(K::Type, V::Type) = ClassifiedCollections(K, Vector{V})
classified_sets(K::Type, V::Type) = ClassifiedCollections(K, Set{V})
classified_counters(K::Type, T::Type) = ClassifiedCollections(K, Accumulator{T, Int})
_create_empty(::Type{Vector{T}}) where {T} = Vector{T}()
_create_empty(::Type{Set{T}}) where {T} = Set{T}()
_create_empty(::Type{Accumulator{T,V}}) where {T,V} = Accumulator{T, V}()
copy(cc::ClassifiedCollections{K, C}) where {K, C} = ClassifiedCollections{K, C}(copy(cc.map))
length(cc::ClassifiedCollections) = length(cc.map)
getindex(cc::ClassifiedCollections{T,C}, x::T) where {T,C} = cc.map[x]
haskey(cc::ClassifiedCollections{T,C}, x::T) where {T,C} = haskey(cc.map, x)
keys(cc::ClassifiedCollections) = keys(cc.map)
iterate(cc::ClassifiedCollections, s...) = iterate(cc.map, s...)
function push!(cc::ClassifiedCollections{K, C}, key::K, e) where {K, C}
    c = get(cc.map, key, nothing)
    if c === nothing
        c = _create_empty(C)
        cc.map[key] = c
    end
    push!(c, e)
end
pop!(cc::ClassifiedCollections{K}, key::K) where {K} = pop!(cc.map, key)
