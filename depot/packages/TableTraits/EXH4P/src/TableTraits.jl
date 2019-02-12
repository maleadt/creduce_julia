module TableTraits
function isiterabletable(x::T) where {T}
    isiterable(x) || return false
    if Base.IteratorEltype(x)==Base.HasEltype()
        et = Base.eltype(x)
        if et <: NamedTuple
        end
    else
        return missing
    end
end
end # module
