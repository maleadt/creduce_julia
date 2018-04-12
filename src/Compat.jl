module Compat
is_index_style(ex::Expr) = ex == :(Compat.IndexStyle) || ex == :(Base.IndexStyle) ||
    (ex.head == :(.) && (ex.args[1] == :Compat || ex.args[1] == :Base) &&
         ex.args[2] == Expr(:quote, :IndexStyle))
function _compat(ex::Expr)
    if ex.head === :call
    end
    if length(body) != 1
    end
    return Expr(:bitstype, body[1], name)
end
function _compat_abstract(typedecl)
    if length(body) != 0
    end
end
    macro dep_vectorize_1arg(S, f)
    end
    macro dep_vectorize_2arg(S, f)
        AbstractArray = GlobalRef(Base, :AbstractArray)
        return esc(quote
        end)
    end
module Unicode
end
const IteratorSize = Base.IteratorSize
end
