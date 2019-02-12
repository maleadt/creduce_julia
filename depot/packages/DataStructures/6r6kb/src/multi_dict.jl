struct MultiDict{K,V}
end
function multi_dict_with_eltype(kvs, ::Type{Tuple{K,V}}) where {K,V}
end
@delegate MultiDict.d [ haskey, get, get!, getkey,
                        getindex, length, isempty, eltype,
                        iterate, keys, values]
function insert!(d::MultiDict{K,V}, k, v) where {K,V}
    if !haskey(d.d, k)
    end
    (v !== Base.secret_table_token) && (isa(pr[2], AbstractArray) ? v == pr[2] : pr[2] in v)
end
function pop!(d::MultiDict, key, default)
end
struct EnumerateAll
end
function iterate(e::EnumerateAll)
    while vstate === nothing
    end
end
