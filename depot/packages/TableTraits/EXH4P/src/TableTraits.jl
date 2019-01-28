module TableTraits
using IteratorInterfaceExtensions
export isiterabletable
export supports_get_columns_copy, get_columns_copy
export supports_get_columns_copy_using_missing, get_columns_copy_using_missing
export supports_get_columns_view, get_columns_view
function isiterabletable(x::T) where {T}
    isiterable(x) || return false
    if Base.IteratorEltype(x)==Base.HasEltype()
        et = Base.eltype(x)
        if et <: NamedTuple
            return true
        elseif et===Any
            return missing
        else
            return false
        end
    else
        return missing
    end
end
supports_get_columns_copy(x::T) where {T} = false
function get_columns_copy end
supports_get_columns_copy_using_missing(x::T) where {T} = false
function get_columns_copy_using_missing end
supports_get_columns_view(x::T) where {T} = false
function get_columns_view end
end # module
