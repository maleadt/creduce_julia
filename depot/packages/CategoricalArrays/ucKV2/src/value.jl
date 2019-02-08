const CatValue{R} = Union{CategoricalValue{T, R} where T,
                          CategoricalString{R}}
if VERSION >= v"0.7.0-DEV.2797"
    function Base.show(io::IO, x::CatValue)
        if Missings.T(get(io, :typeinfo, Any)) === Missings.T(typeof(x))
        end
        if get(io, :compact, false)
        end
    end
end
