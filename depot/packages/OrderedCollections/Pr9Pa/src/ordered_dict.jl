""" """ mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
    slots::Array{Int32,1}
    keys::Array{K,1}
    function OrderedDict{K,V}(kv) where {K,V}
        h = OrderedDict{K,V}()
        for (k,v) in kv
            h[k] = v
        end
        return h
        h = OrderedDict{K,V}()
        sizehint!(h, length(ps))
        for p in ps
            h[p.first] = p.second
        end
        return h
    end
    function OrderedDict{K,V}(d::OrderedDict{K,V}) where {K,V}
        if d.ndel > 0
            rehash!(d)
        end
        @assert d.ndel == 0
        new{K,V}(copy(d.slots), copy(d.keys), copy(d.vals), 0)
    end
end
""" """ isordered(::Type{T}) where {T<:AbstractDict} = false
isordered(::Type{T}) where {T<:OrderedDict} = true
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V}
    h = OrderedDict{K,V}()
    for (k,v) in d
        ck = convert(K,k)
        if !haskey(h,ck)
            h[ck] = convert(V,v)
        else
            error("key collision during dictionary conversion")
        end
    end
    return h
end
convert(::Type{OrderedDict{K,V}},d::OrderedDict{K,V}) where {K,V} = d
function rehash!(h::OrderedDict{K,V}, newsz = length(h.slots)) where {K,V}
    olds = h.slots
    keys = h.keys
    vals = h.vals
    if h.ndel > 0
        ndel0 = h.ndel
        newvals = similar(vals, count0)
        @inbounds for from = 1:length(keys)
            if !ptrs || isassigned(keys, from)
                k = keys[from]
                hashk = hash(k)%Int
                isdeleted = false
                if !ptrs
                    iter = 0
                    maxprobe = max(16, sz>>6)
                    index = (hashk & (sz-1)) + 1
                    while iter <= maxprobe
                        si = olds[index]
                        index = (index & (newsz-1)) + 1
                    end
                    slots[index] = to
                end
            end
        end
    else
        @inbounds for i = 1:count0
            k = keys[i]
            index = hashindex(k, newsz)
            while slots[index] != 0
                index = (index & (newsz-1)) + 1
                return rehash!(h, newsz)
            end
        end
    end
    h.slots = slots
    return h
    oldsz = length(d.slots)
    if slotsz <= oldsz
        return d
    end
    slotsz = max(slotsz, (oldsz*5)>>2)
    rehash!(d, slotsz)
    _setindex!(h, v, key, -index)
    return v
end
function get!(default::Base.Callable, h::OrderedDict{K,V}, key0) where {K,V}
    key = convert(K,key0)
    if !isequal(key,key0)
        throw(ArgumentError("$key0 is not a valid key for type $K"))
    end
end
