function bug(something::SubArray)
    something[] = !
end

function generator(unused...)
    bug(whatever)
end

@eval begin
    function stub(unused...) where T
        $(Expr(:meta, :generated,
               Expr(:new, Core.GeneratedFunctionStub,
                    :generator, [], [], 0, QuoteNode(:am), true)))
    end
end

macro unused() end

stub(nothing)
