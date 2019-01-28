Base.Broadcast.promote_containertype(::Type{DataValueArray}, ::Type{DataValueArray}) = DataValueArray
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{DataValueArray}) = DataValueArray
Base.Broadcast.promote_containertype(::Type{DataValueArray}, ::Type{Array}) = DataValueArray
Base.Broadcast.promote_containertype(::Type{DataValueArray}, _) = DataValueArray
Base.Broadcast.promote_containertype(_, ::Type{DataValueArray}) = DataValueArray

Base.Broadcast._containertype(::Type{<:DataValueArray}) = DataValueArray

Base.Broadcast.broadcast_indices(::Type{DataValueArray}, A) = indices(A)

@inline function Base.Broadcast.broadcast_t(f, ::Type{DataValue{T}}, shape, iter, A, Bs::Vararg{Any,N}) where {N,T}
    C = similar(DataValueArray{T}, shape)
    keeps, Idefaults = Base.Broadcast.map_newindexer(shape, A, Bs)
    Base.Broadcast._broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N}, iter)
    return C
end

# broadcast methods that dispatch on the type of the final container
@inline function Base.Broadcast.broadcast_c(f, ::Type{DataValueArray}, A, Bs...)
    T = Base.Broadcast._broadcast_eltype(f, A, Bs...)
    shape = Base.Broadcast.broadcast_indices(A, Bs...)
    iter = CartesianRange(shape)
    if isleaftype(T)
        return Base.Broadcast.broadcast_t(f, T, shape, iter, A, Bs...)
    end
    if isempty(iter)
        return similar(Array{T}, shape)
    end
    return Base.Broadcast.broadcast_t(f, Any, shape, iter, A, Bs...)
end

# This one is much faster than normal broadcasting but the method won't get called
# in fusing operations like (!).(isna.(x))
Base.broadcast(::typeof(isna), data::DataValueArray) = copy(data.isna)
