mutable struct SortedDict{K, D, Ord <: Ordering} <: AbstractDict{K,D}
    bt::BalancedTree23{K,D,Ord}
    """
        SortedDict{K,V}(o=Forward)
    Construct an empty `SortedDict` with key type `K` and value type
    `V` with `o` ordering (default to forward ordering).
    """
    SortedDict{K,D,Ord}(o::Ord) where {K, D, Ord <: Ordering} =
        new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
    function SortedDict{K,D,Ord}(o::Ord, kv) where {K, D, Ord <: Ordering}
        s = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
        if eltype(kv) <: Pair
            for p in kv
                s[p.first] = p.second
            end
        else
            for (k,v) in kv
                s[k] = v
            end
        end
        return s
    end
end
""" """ SortedDict() = SortedDict{Any,Any,ForwardOrdering}(Forward)
""" """ SortedDict(o::Ord) where {Ord <: Ordering} = SortedDict{Any,Any,Ord}(o)
""" """ SortedDict(ps::Pair...) = SortedDict(Forward, ps)
SortedDict{K,D}(ps::Pair...) where {K,D} = SortedDict{K,D,ForwardOrdering}(Forward, ps)
""" """ SortedDict(o::Ordering, ps::Pair...) = SortedDict(o, ps)
""" """ SortedDict{K,D}(o::Ord, ps::Pair...) where {K,D,Ord<:Ordering} = SortedDict{K,D,Ord}(o, ps)
SortedDict(o::Ord, d::AbstractDict{K,D}) where {K,D,Ord<:Ordering} = SortedDict{K,D,Ord}(o, d)
""" """ SortedDict{K,D}(kv) where {K,D} = SortedDict{K,D}(Forward, kv)
function SortedDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
        SortedDict{K,D,Ord}(o, kv)
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
SortedDict(o1::Ordering, o2::Ordering) = throw(ArgumentError("SortedDict with two parameters must be called with an Ordering and an interable of pairs"))
""" """ SortedDict(kv, o::Ordering=Forward) = SortedDict(o, kv)
function SortedDict(o::Ordering, kv)
    try
        _sorted_dict_with_eltype(o, kv, eltype(kv))
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
_sorted_dict_with_eltype(o::Ord, ps, ::Type{Pair{K,D}}) where {K,D,Ord} = SortedDict{  K,  D,Ord}(o, ps)
_sorted_dict_with_eltype(o::Ord, kv, ::Type{Tuple{K,D}}) where {K,D,Ord} = SortedDict{  K,  D,Ord}(o, kv)
_sorted_dict_with_eltype(o::Ord, ps, ::Type{Pair{K}}  ) where {K,  Ord} = SortedDict{  K,Any,Ord}(o, ps)
_sorted_dict_with_eltype(o::Ord, kv, ::Type            ) where {    Ord} = SortedDict{Any,Any,Ord}(o, kv)
const SDSemiToken = IntSemiToken
const SDToken = Tuple{SortedDict,IntSemiToken}
""" """ @inline function getindex(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m),k_))
    !exactfound && throw(KeyError(k_))
    return m.bt.data[i].d
end
""" """ @inline function setindex!(m::SortedDict{K,D,Ord}, d_, k_) where {K, D, Ord <: Ordering}
    insert!(m.bt, convert(K,k_), convert(D,d_), false)
    m
end
""" """ @inline function push!(m::SortedDict{K,D}, pr::Pair) where {K,D}
    insert!(m.bt, convert(K, pr[1]), convert(D, pr[2]), false)
    m
end
""" """ @inline function findkey(m::SortedDict, k_)
    ll, exactfound = findkey(m.bt, convert(keytype(m),k_))
    IntSemiToken(exactfound ? ll : 2)
end
""" """ @inline function insert!(m::SortedDict{K,D,Ord}, k_, d_) where {K,D, Ord <: Ordering}
    b, i = insert!(m.bt, convert(K,k_), convert(D,d_), false)
    b, IntSemiToken(i)
end
""" """ @inline eltype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} =  Pair{K,D}
@inline eltype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =  Pair{K,D}
""" """ @inline function in(pr::Pair, m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    i, exactfound = findkey(m.bt,convert(K,pr[1]))
    return exactfound && isequal(m.bt.data[i].d,convert(D,pr[2]))
end
@inline in(::Tuple{Any,Any}, ::SortedDict) =
    throw(ArgumentError("'(k,v) in sorteddict' not supported in Julia 0.4 or 0.5.  See documentation"))
""" """ @inline keytype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = K
@inline keytype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = K
""" """ @inline valtype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = D
@inline valtype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = D
""" """ @inline ordtype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = Ord
@inline ordtype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = Ord
""" """ @inline orderobject(m::SortedDict) = m.bt.ord
""" """ @inline function first(m::SortedDict)
    i = beginloc(m.bt)
    i == 2 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
""" """ @inline function last(m::SortedDict)
    i = endloc(m.bt)
    i == 1 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
""" """ @inline function haskey(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    exactfound
end
""" """ function get(default_::Union{Function,Type}, m::SortedDict{K,D}, k_) where {K,D}
    i, exactfound = findkey(m.bt, convert(K, k_))
    return exactfound ? m.bt.data[i].d : convert(D, default_())
end
get(m::SortedDict, k_, default_) = get(()->default_, m, k_)
""" """ function get!(default_::Union{Function,Type}, m::SortedDict{K,D}, k_) where {K,D}
    k = convert(K,k_)
    i, exactfound = findkey(m.bt, k)
    if exactfound
        return m.bt.data[i].d
    else
        default = convert(D, default_())
        insert!(m.bt,k, default, false)
        return default
    end
end
get!(m::SortedDict, k_, default_) = get!(()->default_, m, k_)
""" """ function getkey(m::SortedDict{K,D,Ord}, k_, default_) where {K,D,Ord <: Ordering}
    i, exactfound = findkey(m.bt, convert(K, k_))
    exactfound ? m.bt.data[i].k : convert(K, default_)
end
""" """ @inline function delete!(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    !exactfound && throw(KeyError(k_))
    delete!(m.bt, i)
    m
end
""" """ @inline function pop!(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    !exactfound && throw(KeyError(k_))
    d = m.bt.data[i].d
    delete!(m.bt, i)
    d
end
""" """ function isequal(m1::SortedDict, m2::SortedDict)
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
        throw(ArgumentError("Cannot use isequal for two SortedDicts unless their element types and ordering objects are equal"))
    end
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1)
            return p2 == pastendsemitoken(m2)
        end
        if p2 == pastendsemitoken(m2)
            return false
        end
        k1,d1 = deref((m1,p1))
        k2,d2 = deref((m2,p2))
        if !eq(ord,k1,k2) || !isequal(d1,d2)
            return false
        end
        p1 = advance((m1,p1))
        p2 = advance((m2,p2))
    end
end
function mergetwo!(m::SortedDict{K,D,Ord},
                   m2::AbstractDict{K,D}) where {K,D,Ord <: Ordering}
    for (k,v) in m2
        m[convert(K,k)] = convert(D,v)
    end
end
""" """ function packcopy(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedDict(Dict{K,D}(), orderobject(m))
    mergetwo!(w,m)
    w
end
""" """ function packdeepcopy(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedDict(Dict{K,D}(),orderobject(m))
    for (k,v) in m
        newk = deepcopy(k)
        newv = deepcopy(v)
        w[newk] = newv
    end
    w
end
""" """ function merge!(m::SortedDict{K,D,Ord},
                others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    for o in others
        mergetwo!(m,o)
    end
end
""" """ function merge(m::SortedDict{K,D,Ord},
               others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    result = packcopy(m)
    merge!(result, others...)
    result
end
""" """ empty(m::SortedDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
    SortedDict{K,D,Ord}(orderobject(m))
@deprecate similar(m::SortedDict) empty(m)
isordered(::Type{T}) where {T<:SortedDict} = true
