Base.Broadcast._containertype(::Type{<:DataValue}) = DataValue

Base.Broadcast.promote_containertype(::Type{Any}, ::Type{DataValue}) = DataValue
Base.Broadcast.promote_containertype(::Type{DataValue}, ::Type{Any}) = DataValue
Base.Broadcast.promote_containertype(::Type{Tuple}, ::Type{DataValue}) = Tuple
Base.Broadcast.promote_containertype(::Type{DataValue}, ::Type{Tuple}) = Tuple

Base.Broadcast.broadcast_indices(::Type{DataValue}, A) = ()

Base.Broadcast._unsafe_get_eltype(x::DataValue) = eltype(x)

Base.Broadcast._broadcast_getindex_eltype(::Type{DataValue}, T::Type) = Type{T}
Base.Broadcast._broadcast_getindex_eltype(::Type{DataValue}, A) = typeof(A)

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(::Type{DataValue}, A, I) = A

@inline function Base.Broadcast.broadcast_c(f, ::Type{DataValue}, a...)
    nonnull = all(hasvalue, a)
    S = _nullable_eltype(f, a...)
    if isleaftype(S) && Base.null_safe_op(f, maptoTuple(_unsafe_get_eltype,a...).types...)
        DataValue{S}(f(map(unsafe_get, a)...), nonnull)
    else
        if nonnull
            DataValue(f(map(unsafe_get, a)...))
        else
            DataValue{nullable_returntype(S)}()
        end
    end
end
