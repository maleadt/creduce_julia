module Cassette

using Core: CodeInfo, SlotNumber, NewvarNode, GotoNode, SSAValue

function replace_match!(replace, ismatch, x)
    if ismatch(x)
        return replace(x)
    elseif isa(x, Array) || isa(x, SubArray)
        for i in eachindex(x)
            x[i] = replace_match!(replace, ismatch, x[i])
        end
    elseif isa(x, Expr)
        replace_match!(replace, ismatch, x.args)
    end
    return x
end

mutable struct Reflection
    signature::DataType
    method::Method
    static_params::Vector{Any}
    code_info::CodeInfo
end
function reflect(@nospecialize(sigtypes::Tuple), world::UInt = typemax(UInt))
    S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
    (S.parameters[1]::DataType).name.module === Core.Compiler && return nothing
    _methods = Base._methods_by_ftype(S, -1, world)
    length(_methods) == 1 || return nothing
    type_signature, raw_static_params, method = first(_methods)
    method_instance = Core.Compiler.code_for_method(method, type_signature, raw_static_params, world, false)
    method_instance === nothing && return nothing
    method_signature = method.sig
    static_params = Any[raw_static_params...]
    code_info = Core.Compiler.retrieve_code_info(method_instance)
    isa(code_info, CodeInfo) || return nothing
    code_info = Core.Compiler.copy_code_info(code_info)
    return Reflection(S, method, static_params, code_info)
end

function insert_statements!(code, codelocs, stmtcount, newstmts)
    ssachangemap = fill(0, length(code))
    labelchangemap = fill(0, length(code))
    worklist = Tuple{Int,Int}[]
    for i in 1:length(code)
        stmt = code[i]
        nstmts = stmtcount(stmt, i)
        if nstmts !== nothing
            addedstmts = nstmts - 1
            push!(worklist, (i, addedstmts))
            ssachangemap[i] = addedstmts
            if i < length(code)
                labelchangemap[i + 1] = addedstmts
            end
        end
    end
    Core.Compiler.renumber_ir_elements!(code, ssachangemap, labelchangemap)
    for (i, addedstmts) in worklist
        i += ssachangemap[i] - addedstmts # correct the index for accumulated offsets
        stmts = newstmts(code[i], i)
        @assert length(stmts) == (addedstmts + 1)
        code[i] = stmts[end]
        for j in 1:(length(stmts) - 1) # insert in reverse to maintain the provided ordering
            insert!(code, i, stmts[end - j])
            insert!(codelocs, i, codelocs[i])
        end
    end
end

overdub_typed(args...; optimize=false) = code_typed(overdub, map(Core.Typeof, args); optimize=optimize)

abstract type AbstractPass end

struct NoPass <: AbstractPass end

(::Type{NoPass})(::Any, ::Any, code_info) = code_info
Base.@pure _pure_objectid(x) = objectid(x)

abstract type AbstractContextName end

struct Tag{N<:AbstractContextName,X,E#=<:Union{Nothing,Tag}=#} end

Tag(::Type{N}, ::Type{X}) where {N,X} = Tag(N, X, Nothing)

Tag(::Type{N}, ::Type{X}, ::Type{E}) where {N,X,E} = Tag{N,_pure_objectid(X),E}()

mutable struct BindingMeta
    data::Any
    BindingMeta() = new()
end

const BindingMetaDict = Dict{Symbol,BindingMeta}
const BindingMetaDictCache = IdDict{Module,BindingMetaDict}

struct Context{N<:AbstractContextName,
               M<:Any,
               P<:AbstractPass,
               T<:Union{Nothing,Tag},
               B<:Union{Nothing,BindingMetaDictCache}}
    name::N
    metadata::M
    pass::P
    tag::T
    bindingscache::B
    function Context(name::N, metadata::M, pass::P, ::Nothing, ::Nothing) where {N,M,P}
        return new{N,M,P,Nothing,Nothing}(name, metadata, pass, nothing, nothing)
    end
    function Context(name::N, metadata::M, pass::P, tag::Tag{N}, bindingscache::BindingMetaDictCache) where {N,M,P}
        return new{N,M,P,typeof(tag),BindingMetaDictCache}(name, metadata, pass, tag, bindingscache)
    end
end

const ContextWithTag{T} = Context{<:AbstractContextName,<:Any,<:AbstractPass,T}
const ContextWithPass{P} = Context{<:AbstractContextName,<:Any,P}

function Context(name::AbstractContextName; metadata = nothing, pass::AbstractPass = NoPass())
    return Context(name, metadata, pass, nothing, nothing)
end

function similarcontext(context::Context;
                        metadata = context.metadata,
                        pass = context.pass,
                        tag = context.tag,
                        bindingscache = context.bindingscache)
    return Context(context.name, metadata, pass, tag, bindingscache)
end

function enabletagging(context::Context, f)
    return similarcontext(context;
                          tag = Tag(typeof(context.name), typeof(f)),
                          bindingscache = BindingMetaDictCache())
end

hastagging(::Type{<:ContextWithTag{<:Tag}}) = true
hastagging(::Type{<:ContextWithTag{Nothing}}) = false

tagtype(::C) where {C<:Context} = tagtype(C)
tagtype(::Type{<:ContextWithTag{T}}) where {T} = T

nametype(::Type{<:Context{N}}) where {N} = N

abstract type FieldStorage{D} end

mutable struct Mutable{D} <: FieldStorage{D}
    data::D
    Mutable{D}() where D = new{D}()
    Mutable{D}(data) where D = new{D}(data)
end

load(x::Mutable) = x.data
store!(x::Mutable, y) = (x.data = y)

struct Immutable{D} <: FieldStorage{D}
    data::D
    Immutable{D}(data) where D = new{D}(data)
end

load(x::Immutable) = x.data
store!(x::Immutable, y) = error("cannot mutate immutable field")

struct NoMetaData end
struct NoMetaMeta end

struct Meta{D,M#=<:Union{Tuple,NamedTuple,Array,ModuleMeta}=#}
    data::Union{D,NoMetaData}
    meta::Union{M,NoMetaMeta}
    Meta(data::D, meta::M) where {D,M} = Meta{D,M}(data, meta)
    Meta{D,M}(data, meta) where {D,M} = new{D,M}(data, meta)
end

const NOMETA = Meta(NoMetaData(), NoMetaMeta())
Base.convert(::Type{M}, meta::M) where {M<:Meta} = meta

function Base.convert(::Type{Meta{D,M}}, meta::Meta) where {D,M}
    metadata = _metadataconvert(D, meta.data)
    metameta = _metametaconvert(M, meta.meta)
    return Meta{D,M}(metadata, metameta)
end

_metadataconvert(T, x::NoMetaData) = x
_metadataconvert(T, x) = convert(T, x)

_metametaconvert(T, x::NoMetaMeta) = x
_metametaconvert(T, x) = convert(T, x)

struct ModuleMeta{D,M}
    name::Meta{D,M}
    bindings::BindingMetaDict
end

Base.convert(::Type{M}, meta::M) where {M<:ModuleMeta} = meta

function Base.convert(::Type{ModuleMeta{D,M}}, meta::ModuleMeta) where {D,M}
    return ModuleMeta(convert(Meta{D,M}, meta.name), meta.bindings)
end
Base.@pure @noinline function fetch_tagged_module(context::Context, m::Module)
    return Tagged(context, m, Meta(NoMetaData(), fetch_modulemeta(context, m)))
end

Base.@pure @noinline function fetch_modulemeta(context::Context, m::Module)
    if haskey(context.bindingscache, m)
        bindings = context.bindingscache[m]
    else
        bindings = Cassette.BindingMetaDict()
        context.bindingscache[m] = bindings
    end
    return ModuleMeta(NOMETA, bindings::BindingMetaDict)
end

Base.@pure @noinline function _fetch_bindingmeta!(context::Context,
                                                  m::Module,
                                                  bindings::BindingMetaDict,
                                                  name::Symbol)
    return get!(bindings, name) do
        bindingmeta = BindingMeta()
        if isdefined(m, name)
            bindingmeta.data = initmeta(context, getfield(m, name), NoMetaData())
        end
        return bindingmeta
    end
end

function fetch_bindingmeta!(context::Context,
                             m::Module,
                             bindings::BindingMetaDict,
                             name::Symbol,
                             primal)
    M = metatype(typeof(context), typeof(primal))
    return convert(M, _fetch_bindingmeta!(context, m, bindings, name).data)::M
end

function metatype(::Type{C}, ::Type{T}) where {C<:Context,T}
    if isconcretetype(T) || T <: Type
        return Meta{metadatatype(C, T),metametatype(C, T)}
    end
    return Meta
end

function _fieldtypes_for_metatype(T::Type)
    ftypes = Any[]
    for i in 1:fieldcount(T)
        ftype = fieldtype(T, i)
        ftype = ftype === T ? Any : ftype
        push!(ftypes, ftype)
    end
    return ftypes
end

doesnotneedmetatype(::Type{T}) where {T} = isbitstype(T)
doesnotneedmetatype(::Type{Symbol}) = true
doesnotneedmetatype(::Type{<:Type}) = true

doesnotneedmeta(x) = isbits(x)
doesnotneedmeta(::Symbol) = true
doesnotneedmeta(::Type) = true

metadatatype(::Type{<:Context}, ::DataType) = NoMetaData

@generated function metametatype(::Type{C}, ::Type{T}) where {C<:Context,T}
    if T <: Type || fieldcount(T) == 0
        body = :(NoMetaMeta)
    elseif !(isconcretetype(T))
        body = :(error("cannot call metametatype on non-concrete type ", $T))
    else
        F = T.mutable ? :Mutable : :Immutable
        ftypes = [:($F{metatype(C, $S)}) for S in _fieldtypes_for_metatype(T)]
        tuplemetatype = :(Tuple{$(ftypes...)})
        if T <: Tuple
            body = tuplemetatype
        else
            fnames = Expr(:tuple, map(Base.Meta.quot, fieldnames(T))...)
            body = :(NamedTuple{$fnames,$tuplemetatype})
        end
    end
    return quote
        $body
    end
end

@generated function metametatype(::Type{C}, ::Type{T}) where {C<:Context,T<:Array}
    return :(Array{metatype(C, $(eltype(T))),$(ndims(T))})
end

function metametatype(::Type{C}, ::Type{Module}) where {C<:Context}
    return ModuleMeta{metadatatype(C, Symbol), metametatype(C, Symbol)}
end

function _metametaexpr(::Type{C}, ::Type{V}, metaexprs::Vector) where {C,V}
    if V <: Type || fieldcount(V) == 0 || (all(x == :NOMETA for x in metaexprs) && doesnotneedmetatype(V))
        return :(NoMetaMeta())
    else
        F = V.mutable ? :Mutable : :Immutable
        metatypes = [:(metatype(C, $S)) for S in _fieldtypes_for_metatype(V)]
        metaconverts = [:(convert($(metatypes[i]), $(metaexprs[i]))) for i in 1:fieldcount(V)]
        metametafields = [:($F{$(metatypes[i])}($(metaconverts[i]))) for i in 1:fieldcount(V)]
        if !(V <: Tuple)
            fnames = fieldnames(V)
            for i in 1:fieldcount(V)
                metametafields[i] = :($(fnames[i]) = $(metametafields[i]))
            end
        end
        return Expr(:tuple, metametafields...)
    end
end

initmetameta(context::Context, value::Module) = fetch_modulemeta(context, value)

function initmetameta(context::C, value::Array{V}) where {C<:Context,V}
    M = metatype(C, V)
    if M <: typeof(NOMETA)
        return NoMetaMeta()
    else
        return fill!(similar(value, M), NOMETA)
    end
end

@generated function initmetameta(context::C, value::V) where {C<:Context,V}
    return quote
        $(_metametaexpr(C, V, [:NOMETA for i in 1:fieldcount(V)]))
    end
end

function initmeta(context::C, value::V, metadata::D) where {C<:Context,V,D}
    return Meta{metadatatype(C, V),metametatype(C, V)}(metadata, initmetameta(context, value))
end

struct Tagged{T<:Tag,V,D,M}
    tag::T
    value::V
    meta::Meta{D,M}
    function Tagged(context::C, value::V, meta::Meta) where {T<:Tag,V,C<:ContextWithTag{T}}
        D = metadatatype(C, V)
        M = metametatype(C, V)
        return new{T,V,D,M}(context.tag, value, convert(Meta{D,M}, meta))
    end
end

function tag(value, context::Context, metadata = NoMetaData())
    return Tagged(context, value, initmeta(context, value, metadata))
end

function tag(value, context::ContextWithTag{Nothing}, metadata = NoMetaData())
    error("cannot `tag` a value w.r.t. a `context` if `!hastagging(typeof(context))`")
end

untag(x, context::Context) = untag(x, context.tag)
untag(x::Tagged{T}, tag::T) where {T<:Tag} = x.value
untag(x, ::Union{Tag,Nothing}) = x

untagtype(X::Type, ::Type{C}) where {C<:Context} = untagtype(X, tagtype(C))
untagtype(::Type{<:Tagged{T,V}}, ::Type{T}) where {T<:Tag,V} = V
untagtype(X::Type, ::Type{<:Union{Tag,Nothing}}) = X

metadata(x, context::Context) = metadata(x, context.tag)
metadata(x::Tagged{T}, tag::T) where {T<:Tag} = x.meta.data
metadata(::Any, ::Union{Tag,Nothing}) = NoMetaData()

metameta(x, context::Context) = metameta(x, context.tag)
metameta(x::Tagged{T}, tag::T) where {T<:Tag} = x.meta.meta
metameta(::Any, ::Union{Tag,Nothing}) = NoMetaMeta()

istagged(x, context::Context) = istagged(x, context.tag)
istagged(x::Tagged{T}, tag::T) where {T<:Tag} = true
istagged(::Any, ::Union{Tag,Nothing}) = false

istaggedtype(X::Type, ::Type{C}) where {C<:Context} = istaggedtype(X, tagtype(C))
istaggedtype(::Type{<:Tagged{T}}, ::Type{T}) where {T<:Tag} = true
istaggedtype(::DataType, ::Type{<:Union{Tag,Nothing}}) = false

hasmetadata(x, context::Context) = hasmetadata(x, context.tag)
hasmetadata(x, tag::Union{Tag,Nothing}) = !isa(metadata(x, tag), NoMetaData)

hasmetameta(x, context::Context) = hasmetameta(x, context.tag)
hasmetameta(x, tag::Union{Tag,Nothing}) = !isa(metameta(x, tag), NoMetaMeta)

struct TaggedApplyIterable{C<:Context,T<:Tagged}
    context::C
    tagged::T
end

destructstate(ctx::ContextWithTag{T}, state::Tagged{T,<:Tuple}) where {T} = (tagged_getfield(ctx, state, 1), tagged_getfield(ctx, state, 2))
destructstate(ctx, state) = untag(state, ctx)

Base.iterate(iter::TaggedApplyIterable) = destructstate(iter.context, overdub(iter.context, iterate, iter.tagged))
Base.iterate(iter::TaggedApplyIterable, state) = destructstate(iter.context, overdub(iter.context, iterate, iter.tagged, state))

@generated function tagged_apply_args(context::ContextWithTag{T}, args...) where {T}
    newargs = Any[]
    for i in 1:nfields(args)
        x = args[i]
        newarg = istaggedtype(x, context) ? :(TaggedApplyIterable(context, args[$i])) : :(args[$i])
        push!(newargs, newarg)
    end
    return quote
        Core._apply(tuple, $(newargs...))
    end
end

@generated function tagged_new(context::C, ::Type{T}, args...) where {C<:Context,T}
    argmetaexprs = Any[]
    for i in 1:fieldcount(T)
        if i <= nfields(args) && istaggedtype(args[i], C)
            push!(argmetaexprs, :(args[$i].meta))
        else
            push!(argmetaexprs, :NOMETA)
        end
    end
    untagged_args = [:(untag(args[$i], context)) for i in 1:nfields(args)]
    newexpr = (T <: Tuple) ? Expr(:tuple, untagged_args...) : Expr(:new, T, untagged_args...)
    onlytypeargs = true
    for arg in args
        if !(arg <: Type)
            onlytypeargs = false
            break
        end
    end
    if (all(x == :NOMETA for x in argmetaexprs) && doesnotneedmetatype(T)) || onlytypeargs
        return newexpr
    else
        metametaexpr = _metametaexpr(C, T, argmetaexprs)
        return quote
            M = metatype(C, T)
            return Tagged(context, $newexpr, Meta(NoMetaData(), $metametaexpr))
        end
    end
end

@generated function tagged_new_array(context::C, ::Type{T}, args...) where {C<:Context,T<:Array}
    untagged_args = [:(untag(args[$i], context)) for i in 1:nfields(args)]
    return quote
        return tag($(T)($(untagged_args...)), context)
    end
end

@generated function tagged_new_module(context::C, args...) where {C<:Context}
    if istaggedtype(args[1], C)
        return_expr = quote
            Tagged(context, tagged_module.value,
                   Meta(NoMetaData(),
                        ModuleMeta(args[1].meta, tagged_module.meta.meta.bindings)))
        end
    else
        return_expr = :(tagged_module)
    end
    return quote
        new_module = Module(args...)
        tagged_module = fetch_tagged_module(context, new_module)
        return $return_expr
    end
end

@generated function tagged_new_tuple(context::C, args...) where {C<:Context}
    T = Tuple{[untagtype(args[i], C) for i in 1:nfields(args)]...}
    return quote
        tagged_new(context, $T, args...)
    end
end
@generated function _tagged_new_tuple_unsafe(context::C, args...) where {C<:Context}
    if all(!istaggedtype(arg, C) for arg in args)
        return quote
            Core.tuple(args...)
        end
    else
        return quote
            tagged_new_tuple(context, args...)
        end
    end
end

tagged_nameof(context::Context, x) = nameof(untag(x, context))

function tagged_nameof(context::ContextWithTag{T}, x::Tagged{T,Module}) where {T}
    name_value = nameof(x.value)
    name_meta = hasmetameta(x, context) ? x.meta.meta.name : NOMETA
    return Tagged(context, name_value, name_meta)
end

@inline function tagged_globalref(context::ContextWithTag{T},
                                  m::Tagged{T},
                                  name,
                                  primal) where {T}
    if hasmetameta(m, context) && !istagged(primal, context)
        return _tagged_globalref(context, m, name, primal)
    else
        return primal
    end
end
@inline function tagged_globalref(context::ContextWithTag{T},
                                  m::Tagged{T},
                                  name,
                                  primal::ContextWithTag{T}) where {T}
    return primal
end

@inline function _tagged_globalref(context::ContextWithTag{T},
                                   m::Tagged{T},
                                   name,
                                   primal) where {T}
    untagged_name = untag(name, context)
    if isconst(m.value, untagged_name) && doesnotneedmeta(primal)
        return primal
    else
        meta = fetch_bindingmeta!(context, m.value, m.meta.meta.bindings, untagged_name, primal)
        return Tagged(context, primal, meta)
    end
end

@inline function tagged_globalref_set_meta!(context::ContextWithTag{T}, m::Tagged{T}, name::Symbol, primal) where {T}
    bindingmeta = _fetch_bindingmeta!(context, m.value, m.meta.meta.bindings, name)
    bindingmeta.data = istagged(primal, context) ? primal.meta : NOMETA
    return nothing
end

tagged_getfield(context::ContextWithTag{T}, x, name, boundscheck) where {T} = getfield(x, untag(name, context), untag(boundscheck, context))

tagged_getfield(context::ContextWithTag{T}, x, name) where {T} = getfield(x, untag(name, context))

function tagged_getfield(context::ContextWithTag{T}, x::Tagged{T}, name, boundscheck) where {T}
    untagged_boundscheck = untag(boundscheck, context)
    untagged_name = untag(name, context)
    y_value = getfield(untag(x, context), untagged_name, untagged_boundscheck)
    x_value = untag(x, context)
    if isa(x_value, Module)
        return tagged_globalref(context, x, untagged_name, getfield(x_value, untagged_name))
    elseif hasmetameta(x, context)
        y_meta = load(getfield(x.meta.meta, untagged_name, untagged_boundscheck))
    else
        y_meta = NOMETA
    end
    return Tagged(context, y_value, y_meta)
end

function tagged_getfield(context::ContextWithTag{T}, x::Tagged{T}, name) where {T}
    untagged_name = untag(name, context)
    y_value = getfield(untag(x, context), untagged_name)
    x_value = untag(x, context)
    if isa(x_value, Module)
        return tagged_globalref(context, x, untagged_name, getfield(x_value, untagged_name))
    elseif hasmetameta(x, context)
        y_meta = load(getfield(x.meta.meta, untagged_name))
    else
        y_meta = NOMETA
    end
    return Tagged(context, y_value, y_meta)
end

tagged_setfield!(context::ContextWithTag{T}, x, name, y, boundscheck) where {T} = setfield!(x, untag(name, context), y, untag(boundscheck, context))

tagged_setfield!(context::ContextWithTag{T}, x, name, y) where {T} = setfield!(x, untag(name, context), y)

function tagged_setfield!(context::ContextWithTag{T}, x::Tagged{T}, name, y, boundscheck) where {T}
    untagged_boundscheck = untag(boundscheck, context)
    untagged_name = untag(name, context)
    y_value = untag(y, context)
    y_meta = istagged(y, context) ? y.meta : NOMETA
    setfield!(x.value, untagged_name, y_value, untagged_boundscheck)
    if hasmetameta(x, context)
        store!(getfield(x.meta.meta, untagged_name, untagged_boundscheck), y_meta)
    end
    return y
end

function tagged_setfield!(context::ContextWithTag{T}, x::Tagged{T}, name, y) where {T}
    untagged_name = untag(name, context)
    y_value = untag(y, context)
    y_meta = istagged(y, context) ? y.meta : NOMETA
    setfield!(x.value, untagged_name, y_value)
    if hasmetameta(x, context)
        store!(getfield(x.meta.meta, untagged_name), y_meta)
    end
    return y
end

function tagged_arrayref(context::ContextWithTag{T}, boundscheck, x, i) where {T}
    return Core.arrayref(untag(boundscheck, context), x, untag(i, context))
end

function tagged_arrayref(context::ContextWithTag{T}, boundscheck, x::Tagged{T}, i) where {T}
    untagged_boundscheck = untag(boundscheck, context)
    untagged_i = untag(i, context)
    y_value = Core.arrayref(untagged_boundscheck, untag(x, context), untagged_i)
    if hasmetameta(x, context)
        y_meta = Core.arrayref(untagged_boundscheck, x.meta.meta, untagged_i)
    else
        y_meta = NOMETA
    end
    return Tagged(context, y_value, y_meta)
end

function tagged_arrayset(context::ContextWithTag{T}, boundscheck, x, y, i) where {T}
    return Core.arrayset(untag(boundscheck, context), x, y, untag(i, context))
end

function tagged_arrayset(context::ContextWithTag{T}, boundscheck, x::Tagged{T}, y, i) where {T}
    untagged_boundscheck = untag(boundscheck, context)
    untagged_i = untag(i, context)
    y_value = untag(y, context)
    y_meta = istagged(y, context) ? y.meta : NOMETA
    Core.arrayset(untagged_boundscheck, untag(x, context), y_value, untagged_i)
    if hasmetameta(x, context)
        Core.arrayset(untagged_boundscheck, x.meta.meta, convert(eltype(x.meta.meta), y_meta), untagged_i)
    end
    return x
end

tagged_growbeg!(context::ContextWithTag{T}, x, delta) where {T} = Base._growbeg!(x, untag(delta, context))

function tagged_growbeg!(context::ContextWithTag{T}, x::Tagged{T}, delta) where {T}
    delta_untagged = untag(delta, context)
    Base._growbeg!(x.value, delta_untagged)
    if hasmetameta(x, context)
        Base._growbeg!(x.meta.meta, delta_untagged)
        x.meta.meta[1:delta_untagged] .= Ref(NOMETA)
    end
    return nothing
end

tagged_growend!(context::ContextWithTag{T}, x, delta) where {T} = Base._growend!(x, untag(delta, context))

function tagged_growend!(context::ContextWithTag{T}, x::Tagged{T}, delta) where {T}
    delta_untagged = untag(delta, context)
    Base._growend!(x.value, delta_untagged)
    if hasmetameta(x, context)
        old_length = length(x.meta.meta)
        Base._growend!(x.meta.meta, delta_untagged)
        x.meta.meta[(old_length + 1):(old_length + delta_untagged)] .= Ref(NOMETA)
    end
    return nothing
end

function tagged_growat!(context::ContextWithTag{T}, x, i, delta) where {T}
    return Base._growat!(x, untag(i, context), untag(delta, context))
end

function tagged_growat!(context::ContextWithTag{T}, x::Tagged{T}, i, delta) where {T}
    i_untagged = untag(i, context)
    delta_untagged = untag(delta, context)
    Base._growat!(x.value, i_untagged, delta_untagged)
    if hasmetameta(x, context)
        Base._growat!(x.meta.meta, i_untagged, delta_untagged)
        x.meta.meta[i_untagged:(i_untagged + delta_untagged - 1)] .= Ref(NOMETA)
    end
    return nothing
end

tagged_deletebeg!(context::ContextWithTag{T}, x, delta) where {T} = Base._deletebeg!(x, untag(delta, context))

function tagged_deletebeg!(context::ContextWithTag{T}, x::Tagged{T}, delta) where {T}
    delta_untagged = untag(delta, context)
    Base._deletebeg!(x.value, delta_untagged)
    hasmetameta(x, context) && Base._deletebeg!(x.meta.meta, delta_untagged)
    return nothing
end

tagged_deleteend!(context::ContextWithTag{T}, x, delta) where {T} = Base._deleteend!(x, untag(delta, context))

function tagged_deleteend!(context::ContextWithTag{T}, x::Tagged{T}, delta) where {T}
    delta_untagged = untag(delta, context)
    Base._deleteend!(x.value, delta_untagged)
    hasmetameta(x, context) && Base._deleteend!(x.meta.meta, delta_untagged)
    return nothing
end

function tagged_deleteat!(context::ContextWithTag{T}, x, i, delta) where {T}
    return Base._deleteat!(x, untag(i, context), untag(delta, context))
end

function tagged_deleteat!(context::ContextWithTag{T}, x::Tagged{T}, i, delta) where {T}
    i_untagged = untag(i, context)
    delta_untagged = untag(delta, context)
    Base._deleteat!(x.value, i_untagged, delta_untagged)
    hasmetameta(x, context) && Base._deleteat!(x.meta.meta, i_untagged, delta_untagged)
    return nothing
end

function tagged_typeassert(context::ContextWithTag{T}, x, typ) where {T}
    return Core.typeassert(x, untag(typ, context))
end

function tagged_typeassert(context::ContextWithTag{T}, x::Tagged{T}, typ) where {T}
    untagged_result = Core.typeassert(untag(x, context), untag(typ, context))
    return Tagged(context, untagged_result, x.meta)
end

function tagged_sitofp(context::ContextWithTag{T}, F, x) where {T}
    return Base.sitofp(untag(F, context), x)
end

function tagged_sitofp(context::ContextWithTag{T}, F, x::Tagged{T}) where {T}
    return Tagged(context, Base.sitofp(untag(F, context), x.value), x.meta)
end

tagged_sle_int(context::Context, x, y) = Base.sle_int(untag(x, context), untag(y, context))

Base.show(io::IO, meta::Union{NoMetaMeta,NoMetaData}) = print(io, "_")

function Base.show(io::IO, meta::Meta)
    if isa(meta.data, NoMetaData) && isa(meta.meta, NoMetaMeta)
        print(io, "_")
    else
        if isa(meta.meta, NamedTuple)
            tmp = IOBuffer()
            write(tmp, "(")
            i = 1
            for (k, v) in pairs(meta.meta)
                print(tmp, k, " = ", load(v))
                if i == length(meta.meta)
                    print(tmp, ")")
                else
                    print(tmp, ", ")
                    i += 1
                end
            end
            metametastr = String(take!(tmp))
        elseif isa(meta.meta, Tuple)
            tmp = IOBuffer()
            write(tmp, "(")
            i = 1
            for v in meta.meta
                print(tmp, load(v))
                if i == length(meta.meta)
                    print(tmp, ")")
                else
                    print(tmp, ", ")
                    i += 1
                end
            end
            metametastr = String(take!(tmp))
        else
            metametastr = sprint(show, meta.meta)
        end
        print(io, "Meta(", meta.data, ", ", metametastr, ")")
    end
end

Base.show(io::IO, x::Tagged) = print(io, "Tagged(", x.tag, ", ", x.value, ", ", x.meta, ")")

Base.show(io::IO, ::Tag{N,X,E}) where {N,X,E} = print(io, "Tag{", N, ",", X, ",", E, "}()")


struct OverdubInstead end

@inline prehook(::Context, ::Vararg{Any}) = nothing

@inline posthook(::Context, ::Vararg{Any}) = nothing

@inline execute(::Context, ::Vararg{Any}) = OverdubInstead()

@inline fallback(ctx::Context, args...) = call(ctx, args...)

@inline call(::ContextWithTag{Nothing}, f, args...) = f(args...)
@inline call(context::Context, f, args...) = untag(f, context)(ntuple(i -> untag(args[i], context), Val(nfields(args)))...)
@inline call(::ContextWithTag{Nothing}, f::typeof(Core.apply_type), ::Type{A}, ::Type{B}) where {A,B} = f(A, B)
@inline call(::Context, f::typeof(Core.apply_type), ::Type{A}, ::Type{B}) where {A,B} = f(A, B)

@inline canoverdub(ctx::Context, f, args...) = !isa(untag(f, ctx), Core.Builtin)

const OVERDUB_CTX_SYMBOL = gensym("overdub_context")
const OVERDUB_ARGS_SYMBOL = gensym("overdub_arguments")
const OVERDUB_TMP_SYMBOL = gensym("overdub_tmp")
function overdub_pass!(reflection::Reflection,
                       context_type::DataType,
                       pass_type::DataType = NoPass)
    signature = reflection.signature
    method = reflection.method
    static_params = reflection.static_params
    code_info = reflection.code_info
    iskwfunc = startswith(String(signature.parameters[1].name.name), "#kw##")
    istaggingenabled = hastagging(context_type)

    if !iskwfunc
        code_info = pass_type(context_type, signature, code_info)
    end
    code_info.slotnames = Any[:overdub, OVERDUB_CTX_SYMBOL, OVERDUB_ARGS_SYMBOL, code_info.slotnames..., OVERDUB_TMP_SYMBOL]
    code_info.slotflags = UInt8[0x00, 0x00, 0x00, code_info.slotflags..., 0x00]
    n_prepended_slots = 3
    overdub_ctx_slot = SlotNumber(2)
    overdub_args_slot = SlotNumber(3)
    overdub_tmp_slot = SlotNumber(length(code_info.slotnames))
    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]
    n_actual_args = fieldcount(signature)
    n_method_args = Int(method.nargs)
    Base.Meta.partially_inline!(code_info.code, Any[], method.sig, static_params,
                                n_prepended_slots, length(overdubbed_code), :propagate)

    original_code_start_index = length(overdubbed_code) + 1


    original_code_region = view(overdubbed_code, original_code_start_index:length(overdubbed_code))
    replace_match!(x -> isa(x, GlobalRef), original_code_region) do x end

    code_info.code = overdubbed_code
    code_info.codelocs = overdubbed_codelocs
    code_info.ssavaluetypes = length(overdubbed_code)
    code_info.method_for_inference_limit_heuristics = method
    reflection.code_info = code_info

    return reflection
end

function overdub_generator(pass_type, self, context_type, args::Tuple)
    untagged_args = ()
    reflection = reflect(untagged_args)
    overdub_pass!(reflection, context_type, pass_type)
    return reflection.code_info
end

@eval begin
    function overdub(whatever::ContextWithPass{pass}, ex...) where {pass<:NoPass}
        $(Expr(:meta,
               :generated,
               Expr(:new,
                    Core.GeneratedFunctionStub,
                    :overdub_generator,
                    Any[:whatever],
                    Any[:whatever],
                    0,
                    QuoteNode(:whatever),
                    true)))
    end
end

macro context(Ctx)
    @assert isa(Ctx, Symbol) "context name must be a Symbol"
    CtxName = gensym(string(Ctx, "Name"))
    TaggedCtx = gensym(string(Ctx, "Tagged"))
    Typ = :(Core.Typeof)
    return esc(quote
        struct $CtxName <: $Cassette.AbstractContextName end

        Base.show(io::IO, ::Type{$CtxName}) = print(io, "nametype(", $(string(Ctx)), ")")

        const $Ctx{M,T<:Union{Nothing,$Cassette.Tag},P<:$Cassette.AbstractPass} = $Cassette.Context{$CtxName,M,P,T}
        const $TaggedCtx = $Ctx{<:Any,<:$Cassette.Tag}

        $Ctx(; kwargs...) = $Cassette.Context($CtxName(); kwargs...)

        @doc (@doc $Cassette.Context) $Ctx

        @inline $Cassette.execute(::C, ::$Typ($Cassette.Tag), ::Type{N}, ::Type{X}) where {C<:$Ctx,N,X} = $Cassette.Tag(N, X, $Cassette.tagtype(C))
        @inline $Cassette.execute(ctx::$Ctx, f::$Typ(Base.isdispatchtuple), T::Type) = $Cassette.fallback(ctx, f, T)
        @inline $Cassette.execute(ctx::$Ctx, f::$Typ(Base.eltype), T::Type) = $Cassette.fallback(ctx, f, T)
        @inline $Cassette.execute(ctx::$Ctx, f::$Typ(Base.convert), T::Type, t::Tuple) = $Cassette.fallback(ctx, f, T, t)
        @inline $Cassette.execute(ctx::$Ctx{<:Any,Nothing}, f::$Typ(Base.getproperty), x::Any, s::Symbol) = $Cassette.fallback(ctx, f, x, s)

        @inline $Cassette.execute(ctx::C, f::$Typ($Cassette.tag), value, ::C, metadata) where {C<:$TaggedCtx} = $Cassette.fallback(ctx, f, value, ctx, metadata)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Array{T,N}), undef::UndefInitializer, args...) where {T,N} = $Cassette.tagged_new_array(ctx, Array{T,N}, undef, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.Module), args...) = $Cassette.tagged_new_module(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.tuple), args...) = $Cassette.tagged_new_tuple(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base.nameof), args...) = $Cassette.tagged_nameof(ctx, m)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.getfield), args...) = $Cassette.tagged_getfield(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.setfield!), args...) = $Cassette.tagged_setfield!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.arrayref), args...) = $Cassette.tagged_arrayref(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.arrayset), args...) = $Cassette.tagged_arrayset(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._growbeg!), args...) = $Cassette.tagged_growbeg!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._growend!), args...) = $Cassette.tagged_growend!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._growat!), args...) = $Cassette.tagged_growat!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._deletebeg!), args...) = $Cassette.tagged_deletebeg!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._deleteend!), args...) = $Cassette.tagged_deleteend!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Base._deleteat!), args...) = $Cassette.tagged_deleteat!(ctx, args...)
        @inline $Cassette.execute(ctx::$TaggedCtx, ::$Typ(Core.typeassert), args...) = $Cassette.tagged_typeassert(ctx, args...)

        @inline function $Cassette.execute(ctx::$TaggedCtx, f::Core.IntrinsicFunction, args...)
            if f === Base.sitofp
                return $Cassette.tagged_sitofp(ctx, args...)
            elseif f === Base.sle_int
                return $Cassette.tagged_sle_int(ctx, args...)
            else # TODO: add more cases
                return $Cassette.fallback(ctx, f, args...)
            end
        end

        $Ctx
    end)
end

macro overdub(ctx, expr)
    return :($Cassette.overdub($(esc(ctx)), () -> $(esc(expr))))
end

macro pass(transform)
    Pass = gensym("PassType")
    name = Expr(:quote, :($__module__.$Pass))
    line = Expr(:quote, __source__.line)
    file = Expr(:quote, __source__.file)
    return esc(quote
        struct $Pass <: $Cassette.AbstractPass end
        (::Type{$Pass})(ctxtype, signature, codeinfo) = $transform(ctxtype, signature, codeinfo)
        Core.eval($Cassette, $Cassette.overdub_definition($name, $line, $file))
        $Pass()
    end)
end


end # module

Cassette.@context Ctx
Cassette.overdub(Ctx(), /, 1, 2)
