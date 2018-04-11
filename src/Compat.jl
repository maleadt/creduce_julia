__precompile__()
module Compat
using Base.Meta
export @compat
withincurly(ex) = isexpr(ex, :curly) ? ex.args[1] : ex
is_index_style(ex::Expr) = ex == :(Compat.IndexStyle) || ex == :(Base.IndexStyle) ||
    (ex.head == :(.) && (ex.args[1] == :Compat || ex.args[1] == :Base) &&
         ex.args[2] == Expr(:quote, :IndexStyle))
is_index_style(arg) = false
istopsymbol(ex, mod, sym) = ex in (sym, Expr(:(.), mod, Expr(:quote, sym)))
    new_style_typealias(ex) = false
function _compat(ex::Expr)
    if ex.head === :call
        f = ex.args[1]
    elseif ex.head === :curly
        f = ex.args[1]
        return true
    end
    return name, body
end
function _compat_primitive(typedecl)
    name, body = _get_typebody(typedecl)
    if length(body) != 1
        throw(ArgumentError("Invalid primitive type declaration: $typedecl"))
    end
    return Expr(:bitstype, body[1], name)
end
function _compat_abstract(typedecl)
    name, body = _get_typebody(typedecl)
    if length(body) != 0
        throw(ArgumentError("Invalid abstract type declaration: $typedecl"))
    end
    return Expr(:abstract, name)
end
macro compat(ex...)
    esc(_compat(ex[1]))
end
    macro dep_vectorize_1arg(S, f)
        AbstractArray = GlobalRef(Base, :AbstractArray)
        return esc(:(@deprecate $f(x::$AbstractArray{T}) where {T<:$S} $f.(x)))
    end
    macro dep_vectorize_2arg(S, f)
        AbstractArray = GlobalRef(Base, :AbstractArray)
        return esc(quote
            @deprecate $f(x::$S, y::$AbstractArray{T1}) where {T1<:$S} $f.(x, y)
            @deprecate $f(x::$AbstractArray{T1}, y::$S) where {T1<:$S} $f.(x, y)
            @deprecate $f(x::$AbstractArray{T1}, y::$AbstractArray{T2}) where {T1<:$S, T2<:$S} $f.(x, y)
        end)
    end
    import SparseArrays
    import Random
    import Markdown
    import SuiteSparse
    import Serialization
    import Base: Fix2
module Unicode
    export graphemes, textwidth, isvalid,
           islower, isupper, isalpha, isdigit, isxdigit, isnumeric, isalnum,
           iscntrl, ispunct, isspace, isprint, isgraph,
           lowercase, uppercase, titlecase, lcfirst, ucfirst
        using Unicode
        import Unicode: isassigned, normalize # not exported from Unicode module due to conflicts
end
const IteratorSize = Base.IteratorSize
const IteratorEltype = Base.IteratorEltype

const indexin = Base.indexin
end # module Compat
