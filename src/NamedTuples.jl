__precompile__()
module NamedTuples
export @NT, NamedTuple, setindex, delete
abstract type NamedTuple end
Base.keys( t::NamedTuple ) = fieldnames( t )
Base.values( t::NamedTuple ) = [ getfield( t, i ) for i in 1:nfields( t ) ]
Base.haskey( t::NamedTuple, k ) = k in keys(t)
Base.length( t::NamedTuple ) = nfields( t )
Base.start( t::NamedTuple ) = 1
Base.done( t::NamedTuple, iter ) = iter > nfields( t )
Base.next( t::NamedTuple, iter ) = ( getfield( t, iter ), iter + 1 )
Base.endof( t::NamedTuple ) = length( t )
Base.last( t::NamedTuple ) = t[end]
function Base.show( io::IO, t::NamedTuple )
    print(io, "(")
    first = true
    for (k,v) in zip(keys(t),values(t))
        !first && print(io, ", ")
        print(io, k, " = "); show(io, v)
        first = false
    end
    print(io, ")")
end
Base.getindex( t::NamedTuple, i::Int ) = getfield( t, i )
Base.getindex( t::NamedTuple, i::Symbol ) = getfield( t, i )
Base.getindex( t::NamedTuple, i::Symbol, default ) = get( t, i, default )
Base.get( t::NamedTuple, i::Symbol, default ) = i in keys(t) ? t[i] : default
import Base: ==
@generated function ==( lhs::NamedTuple, rhs::NamedTuple)
    if !isequal(fieldnames(lhs), fieldnames(rhs))
        return false
    end
    quote
        ( lhs === rhs ) && return true
        for i in 1:length( lhs )
            if ( lhs[i] != rhs[i])
                return false
            end
        end
        return true
    end
end
function Base.hash( nt::NamedTuple, hs::UInt64)
    h = 17
    for v in values(nt)
        h = h * 23 + hash( v, hs )
    end
    return h
end
struct ParseNode{T} end
function trans( ::Type{ParseNode{:Symbol}}, expr::Expr)
    (expr.args[1],nothing,nothing)
end
escape( e::Expr ) = esc( e )
escape( e::Symbol ) = esc( e )
escape( e ) = e
function trans( ::Union{Type{ParseNode{:(=)}},Type{ParseNode{:kw}}}, expr::Expr)
    (sym, typ ) = trans( expr.args[1])
    return (sym, typ, escape( expr.args[2] ))
end
function trans( ::Type{ParseNode{:call}}, expr::Expr)
    if expr.args[1] == :(=>)
        Base.depwarn("\"=>\" syntax for NamedTuple construction is deprecated, use \"=\" instead.", Symbol("@NT"))
        (sym, typ ) = trans( expr.args[1])
        return (sym, typ, escape( expr.args[2] ))
    end
    return (nothing, nothing, escape(expr) )
end
function trans( ::Type{ParseNode{:(::)}}, expr::Expr)
    if( length( expr.args ) > 1 )
        return ( expr.args[1], expr.args[2], nothing)
    else
        return ( nothing, expr.args[1], nothing)
    end
end
function trans( expr::Expr )
    trans( ParseNode{expr.head}, expr)
end
function trans( sym::Symbol )
    return (sym, nothing, nothing)
end
function trans( ::Type{ParseNode{:quote}}, expr::Expr )
    return trans( expr.args[1] )
end
function trans{T}( lit::T )
    return (nothing, nothing, escape(lit) )
end
function trans{T}( ::Type{ParseNode{T}}, expr::Expr)
    return (nothing, nothing, escape(expr) )
end
function gen_namedtuple_ctor_body(n::Int, args)
    types = [ :(typeof($x)) for x in args ]
    cnvt = [ :(convert(fieldtype(TT,$n),$(args[n]))) for n = 1:n ]
    if n == 0
        texpr = :T
    else
        texpr = :(NT{$(types...)})
    end
    if isless(Base.VERSION, v"0.6.0-")
        tcond = :(NT === NT.name.primary)
    else
        tcond = :(isa(NT,UnionAll))
    end
    quote
        if $tcond
            TT = $texpr
        else
            TT = NT
        end
        if nfields(TT) !== $n
            throw(ArgumentError("wrong number of arguments to named tuple constructor"))
        end
        $(Expr(:new, :TT, cnvt...))
    end
end
@generated function (::Type{NT}){NT<:NamedTuple}(args...)
    n = length(args)
    aexprs = [ :(args[$i]) for i = 1:n ]
    return gen_namedtuple_ctor_body(n, aexprs)
end
for n = 0:5
    args = [ Symbol("x$n") for n = 1:n ]
    @eval function (::Type{NT}){NT<:NamedTuple}($(args...))
        $(gen_namedtuple_ctor_body(n, args))
    end
end
function create_namedtuple_type(fields::Vector{Symbol}, mod::Module = NamedTuples)
    escaped_fieldnames = [replace(string(i), "_", "__") for i in fields]
    name = Symbol( string( "_NT_", join( escaped_fieldnames, "_")) )
    if !isdefined(mod, name)
        len = length( fields )
        types = [Symbol("T$n") for n in 1:len]
        tfields = [ Expr(:(::), Symbol( fields[n] ), Symbol( "T$n") ) for n in 1:len ]
        def = Expr(:type, false, Expr( :(<:), Expr( :curly, name, types... ), GlobalRef(NamedTuples, :NamedTuple) ),
                   Expr(:block, tfields...,
                        Expr(:tuple)))  # suppress default constructors
        eval(mod, def)
    end
    return getfield(mod, name)
end
@doc doc"Given a symbol vector create the `NamedTuple`" ->
function make_tuple( syms::Vector{Symbol} )
    return create_namedtuple_type( syms )
end
@doc doc"Given an expression vector create the `NamedTuple`" ->
function make_tuple( exprs::Vector)
    len    = length( exprs )
    fields = Array{Symbol}(len)
    values = Array{Any}(len)
    typs   = Array{Any}(len)
    # Are we directly constructing the type, if so all values must be
    # supplied by the caller, we use this state to ensure this
    construct = false
    # handle the case where this is defining a datatype
    for i in 1:len
        expr = exprs[i]
        ( sym, typ, val ) = trans( expr )
        if( construct == true && val == nothing || ( i > 1 && construct == false && val != nothing ))
            error( "Invalid tuple, all values must be specified during construction @ ($expr)")
        end
        construct  = val != nothing
        fields[i]  = sym != nothing?sym:Symbol( "_$(i)_")
        typs[i] = typ
        # On construction ensure that the types are consitent with the declared types, if applicable
        values[i]  = ( typ != nothing && construct)? Expr( :call, :convert, typ, val ) : val
    end
    ty = create_namedtuple_type( fields )
    # Either call the constructor with the supplied values or return the type
    if( !construct )
        if len == 0
            return ty
        end
        return Expr( :curly, ty, typs... )
    else
        return Expr( :call, ty, values ... )
    end
end
function delete( t::NamedTuple, key::Symbol )
    nms = filter( x->x!=key, fieldnames( t ) )
    ty = create_namedtuple_type( nms )
    vals = [ getindex( t, nm ) for nm in nms ]
    return ty(vals...)
end
Base.Broadcast._containertype(::Type{<:NamedTuple}) = NamedTuple
Base.Broadcast.promote_containertype(::Type{NamedTuple}, ::Type{NamedTuple}) = NamedTuple
Base.Broadcast.promote_containertype(::Type{NamedTuple}, _) = error()
Base.Broadcast.promote_containertype(_, ::Type{NamedTuple}) = error()
@inline function Base.Broadcast.broadcast_c(f, ::Type{NamedTuple}, nts...)
    _map(f, nts...)
end
moduleof(t::DataType) = t.name.module
if VERSION < v"0.6.0-dev"
    uniontypes(u) = u.types
else
    const uniontypes = Base.uniontypes
    moduleof(t::UnionAll) = moduleof(Base.unwrap_unionall(t))
end
struct NTType end
struct NTVal end
function Base.serialize{NT<:NamedTuple}(io::AbstractSerializer, ::Type{NT})
    if NT === Union{}
        Base.Serializer.write_as_tag(io, Base.Serializer.BOTTOM_TAG)
    elseif isa(NT, Union)
        Base.serialize_type(io, NTType)
        serialize(io, Union)
        serialize(io, [uniontypes(NT)...])
    elseif isleaftype(NT)
        Base.serialize_type(io, NTType)
        serialize(io, fieldnames(NT))
        serialize(io, moduleof(NT))
        write(io.io, UInt8(0))
        serialize(io, NT.parameters)
    else
        u = Base.unwrap_unionall(NT)
        if isa(u, DataType) && NT === u.name.wrapper
            Base.serialize_type(io, NTType)
            serialize(io, fieldnames(NT))
            serialize(io, moduleof(NT))
            write(io.io, UInt8(1))
        else
            error("cannot serialize type $NT")
        end
    end
end
function Base.deserialize(io::AbstractSerializer, ::Type{NTType})
    fnames = deserialize(io)
    if fnames == Union
        types = deserialize(io)
        return Union{types...}
    else
        mod = deserialize(io)
        NT = create_namedtuple_type(fnames, mod)
        if read(io.io, UInt8) == 0
            params = deserialize(io)
            return NT{params...}
        else
            return NT
        end
    end
end
function Base.serialize(io::AbstractSerializer, x::NamedTuple)
    Base.serialize_type(io, NTVal)
    serialize(io, typeof(x))
    for i in 1:nfields(x)
        serialize(io, getfield(x, i))
    end
end
function Base.deserialize(io::AbstractSerializer, ::Type{NTVal})
    NT = deserialize(io)
    nf = nfields(NT)
    if nf == 0
        return NT()
    elseif nf == 1
        return NT(deserialize(io))
    elseif nf == 2
        return NT(deserialize(io), deserialize(io))
    elseif nf == 3
        return NT(deserialize(io), deserialize(io), deserialize(io))
    else
        return NT(Any[ deserialize(io) for i = 1:nf ]...)
    end
end
end # module