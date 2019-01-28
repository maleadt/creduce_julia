struct DefaultDictBase{K,V,F,D} <: AbstractDict{K,V}
    default::F
    d::D
    passkey::Bool
    check_D(D,K,V) = (D <: AbstractDict{K,V}) ||
        throw(ArgumentError("Default dict must be <: AbstractDict{$K,$V}"))
    DefaultDictBase{K,V,F,D}(x::F, kv::AbstractArray{Tuple{K,V}}; passkey=false) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, D(kv), passkey))
    DefaultDictBase{K,V,F,D}(x::F, ps::Pair{K,V}...; passkey=false) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, D(ps...), passkey))
    DefaultDictBase{K,V,F,D}(x::F, d::D; passkey=d.passkey) where {K,V,F,D<:DefaultDictBase} =
        (check_D(D,K,V); DefaultDictBase(x, d.d; passkey=passkey))
    DefaultDictBase{K,V,F,D}(x::F, d::D = D(); passkey=false) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, d, passkey))
end
DefaultDictBase(; kwargs...) = throw(ArgumentError("no default specified"))
DefaultDictBase(k, v; kwargs...) = throw(ArgumentError("no default specified"))
DefaultDictBase(default::F; kwargs...) where {F} = DefaultDictBase{Any,Any,F,Dict{Any,Any}}(default; kwargs...)
DefaultDictBase(default::F, kv::AbstractArray{Tuple{K,V}}; kwargs...) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default, kv; kwargs...)
DefaultDictBase(default::F, ps::Pair{K,V}...; kwargs...) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default, ps...; kwargs...)
DefaultDictBase(default::F, d::D; kwargs...) where {F,D<:AbstractDict} = (K=keytype(d); V=valtype(d); DefaultDictBase{K,V,F,D}(default, d; kwargs...))
DefaultDictBase{K,V}(default::F; kwargs...) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default; kwargs...)
@delegate DefaultDictBase.d [ get, haskey, getkey, pop!,
                              iterate, isempty, length ]
@delegate_return_parent DefaultDictBase.d [ delete!, empty!, setindex!, sizehint! ]
empty(d::DefaultDictBase{K,V,F}) where {K,V,F} = DefaultDictBase{K,V,F}(d.default; passkey=d.passkey)
@deprecate similar(d::DefaultDictBase) empty(d)
function iterate(v::Base.ValueIterator{T}, i::Int) where {T <: DefaultDictBase}
    i > length(v.dict.d.vals) && return nothing
    return (v.dict.d.vals[i], Base.skip_deleted(v.dict.d, i+1))
end
getindex(d::DefaultDictBase, key) = get!(d.d, key, d.default)
function getindex(d::DefaultDictBase{K,V,F}, key) where {K,V,F<:Base.Callable}
    if d.passkey
        return get!(d.d, key) do
            d.default(key)
        end
    else
        return get!(d.d, key) do
            d.default()
        end
    end
end
for _Dict in [:Dict, :OrderedDict]
    DefaultDict = Symbol("Default"*string(_Dict))
    @eval begin
        struct $DefaultDict{K,V,F} <: AbstractDict{K,V}
            d::DefaultDictBase{K,V,F,$_Dict{K,V}}
            $DefaultDict{K,V,F}(x, ps::Pair{K,V}...; kwargs...) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, ps...; kwargs...))
            $DefaultDict{K,V,F}(x, kv::AbstractArray{Tuple{K,V}}; kwargs...) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, kv; kwargs...))
            $DefaultDict{K,V,F}(x, d::$DefaultDict) where {K,V,F} = $DefaultDict(x, d.d)
            $DefaultDict{K,V,F}(x, d::$_Dict; kwargs...) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, d; kwargs...))
            $DefaultDict{K,V,F}(x; kwargs...) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x; kwargs...))
        end
        $DefaultDict() = throw(ArgumentError("$DefaultDict: no default specified"))
        $DefaultDict(k,v) = throw(ArgumentError("$DefaultDict: no default specified"))
        $DefaultDict(default::F; kwargs...) where {F} = $DefaultDict{Any,Any,F}(default; kwargs...)
        $DefaultDict(default::F, kv::AbstractArray{Tuple{K,V}}; kwargs...) where {K,V,F} = $DefaultDict{K,V,F}(default, kv; kwargs...)
        $DefaultDict(default::F, ps::Pair{K,V}...; kwargs...) where {K,V,F} = $DefaultDict{K,V,F}(default, ps...; kwargs...)
        $DefaultDict(default::F, d::AbstractDict; kwargs...) where {F} = ((K,V)= (Base.keytype(d), Base.valtype(d)); $DefaultDict{K,V,F}(default, $_Dict(d); kwargs...))
        $DefaultDict{K,V}(; kwargs...) where {K,V} = throw(ArgumentError("$DefaultDict: no default specified"))
        $DefaultDict{K,V}(default::F; kwargs...) where {K,V,F} = $DefaultDict{K,V,F}(default; kwargs...)
        @delegate $DefaultDict.d [ getindex, get, get!, haskey,
                                   getkey, pop!, iterate,
                                   isempty, length ]
        @delegate_return_parent $DefaultDict.d [ delete!, empty!, setindex!, sizehint! ]
        push!(d::$DefaultDict, p::Pair) = (setindex!(d.d, p.second, p.first); d)
        push!(d::$DefaultDict, p::Pair, q::Pair) = push!(push!(d, p), q)
        push!(d::$DefaultDict, p::Pair, q::Pair, r::Pair...) = push!(push!(push!(d, p), q), r...)
        push!(d::$DefaultDict, p) = (setindex!(d.d, p[2], p[1]); d)
        push!(d::$DefaultDict, p, q) = push!(push!(d, p), q)
        push!(d::$DefaultDict, p, q, r...) = push!(push!(push!(d, p), q), r...)
        empty(d::$DefaultDict{K,V,F}) where {K,V,F} = $DefaultDict{K,V,F}(d.d.default)
        in(key, v::Base.KeySet{K,T}) where {K,T<:$DefaultDict{K}} = key in keys(v.dict.d.d)
        @deprecate similar(d::$DefaultDict) empty(d)
    end
end
isordered(::Type{T}) where {T<:DefaultOrderedDict} = true
