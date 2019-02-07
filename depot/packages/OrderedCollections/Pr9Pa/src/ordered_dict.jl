""" """ mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
    function OrderedDict{K,V}(kv) where {K,V}
        for (k,v) in kv
        end
        for p in ps
        end
    end
    function OrderedDict{K,V}(d::OrderedDict{K,V}) where {K,V}
        if d.ndel > 0
        end
    end
end
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V}
    for (k,v) in d
        if !haskey(h,ck)
        end
    end
end
function rehash!(h::OrderedDict{K,V}, newsz = length(h.slots)) where {K,V}
    if h.ndel > 0
        @inbounds for from = 1:length(keys)
            if !ptrs || isassigned(keys, from)
                if !ptrs
                    while iter <= maxprobe
                    end
                end
            end
        end
        @inbounds for i = 1:count0
            while slots[index] != 0
                index = (index & (newsz-1)) + 1
            end
        end
    end
    if slotsz <= oldsz
    end
end
function get!(default::Base.Callable, h::OrderedDict{K,V}, key0) where {K,V}
    if !isequal(key,key0)
    end
end
