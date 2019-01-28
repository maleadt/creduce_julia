struct Accumulator{T, V<:Number} <: AbstractDict{T,V}
    map::Dict{T,V}
end
Accumulator{T, V}() where {T,V<:Number} = Accumulator{T,V}(Dict{T,V}())
@deprecate Accumulator(::Type{T}, ::Type{V}) where {T,V<:Number} Accumulator{T, V}()
counter(T::Type) = Accumulator{T,Int}()
counter(dct::Dict{T,V}) where {T,V<:Integer} = Accumulator{T,V}(copy(dct))
""" """ function counter(seq)
    ct = counter(eltype_for_accumulator(seq))
    for x in seq
        inc!(ct, x)
    end
    return ct
end
eltype_for_accumulator(seq::T) where T = eltype(T)
function eltype_for_accumulator(seq::T) where {T<:Base.Generator}
    Base.@default_eltype(seq)
end
copy(ct::Accumulator) = Accumulator(copy(ct.map))
length(a::Accumulator) = length(a.map)
get(ct::Accumulator, x, default) = get(ct.map, x, default)
getindex(ct::Accumulator{T,V}, x) where {T,V} = get(ct.map, x, zero(V))
setindex!(ct::Accumulator, x, v) = setindex!(ct.map, x, v)
haskey(ct::Accumulator, x) = haskey(ct.map, x)
keys(ct::Accumulator) = keys(ct.map)
values(ct::Accumulator) = values(ct.map)
sum(ct::Accumulator) = sum(values(ct.map))
iterate(ct::Accumulator, s...) = iterate(ct.map, s...)
""" """ inc!(ct::Accumulator, x, a::Number) = (ct[x] += a)
inc!(ct::Accumulator{T,V}, x) where {T,V} = inc!(ct, x, one(V))
push!(ct::Accumulator, x) = inc!(ct, x)
push!(ct::Accumulator, x, a::Number) = inc!(ct, x, a)
push!(ct::Accumulator, x::Pair)  = inc!(ct, x)
""" """ dec!(ct::Accumulator, x, a::Number) = (ct[x] -= a)
dec!(ct::Accumulator{T,V}, x) where {T,V} = dec!(ct, x, one(V))
""" """ function merge!(ct::Accumulator, other::Accumulator)
    for (x, v) in other
        inc!(ct, x, v)
    end
    ct
end
function merge!(ct1::Accumulator, others::Accumulator...)
    for ct in others
        merge!(ct1,ct)
    end
    return ct1
end
""" """ function merge(ct1::Accumulator, others::Accumulator...)
    ct = copy(ct1)
    merge!(ct,others...)
end
""" """ reset!(ct::Accumulator, x) = pop!(ct.map, x)
""" """ nlargest(acc::Accumulator) = sort!(collect(acc), by=last, rev=true)
nlargest(acc::Accumulator, n) = partialsort!(collect(acc), 1:n, by=last, rev=true)
""" """ nsmallest(acc::Accumulator) = sort!(collect(acc), by=last, rev=false)
nsmallest(acc::Accumulator, n) = partialsort!(collect(acc), 1:n, by=last, rev=false)
@deprecate pop!(ct::Accumulator, x) reset!(ct, x)
@deprecate push!(ct1::Accumulator, ct2::Accumulator) merge!(ct1,ct2)
