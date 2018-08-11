module a
function b!(c, d, e)
    if isa(e, SubArray)
        for f in g
            e[] = !
        end
    end

    end
function ag!(h,
                       : )
    b!(v, w) do e end
end
function ah(j, ai, k, l)
    ag!(aj, j)
end
@eval begin
    function ak(m, al...) where i
        $(Expr(:meta,
               :generated,
               Expr(:new,
                    Core.GeneratedFunctionStub,
                    :ah,
                    [],
                    [],
                    0,
                    QuoteNode(:am),
                    true)))
    end
end
macro an(ao)
    esc(quote
        ao = a
        end)
end
end  
a.@an ao
a.ak(ao)
