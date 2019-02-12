module Missings
export allowmissing, disallowmissing, ismissing, missing, missings,
       Missing, MissingException, levels, coalesce, passmissing
struct EachReplaceMissing{T, U}
end
@inline function Base.iterate(itr::EachReplaceMissing)
end
struct EachFailMissing{T}
end
Base.IteratorSize(::Type{EachFailMissing{T}}) where {T} =
    Base.IteratorEltype(T)
Base.length(itr::EachFailMissing) = length(itr.x)
@inline function Base.iterate(itr::EachFailMissing)
end
""" """ function levels(x)
    if hasmethod(isless, Tuple{T, T})
        try
        catch
        end
    end
end
struct PassMissing{F} <: Function
end
function (f::PassMissing{F})(x) where {F}
    if @generated
        return x === Missing ? missing : :(f.f(x))
    else
        return x === missing ? missing : f.f(x)
    end
end
function (f::PassMissing{F})(xs...) where {F}
    if @generated
        for T in xs
        end
        return :(f.f(xs...))
    else
    end
end
""" """ passmissing(f::Base.Callable) = PassMissing{typeof(f)}(f)
end # module
