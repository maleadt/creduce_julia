# # interface for skipping null entries

# function skipna_init(f, op, X::DataValueArray,
#                        ifirst::Int, ilast::Int)
#     # Get first non-null element
#     ifirst = Base.findnext(x -> x == false, X.isna, ifirst)
#     @inbounds v1 = X.values[ifirst]

#     # Get next non-null element
#     ifirst = Base.findnext(x -> x == false, X.isna, ifirst + 1)
#     @inbounds v2 = X.values[ifirst]

#     # Reduce first two elements
#     return op(f(v1), f(v2)), ifirst
# end

# # sequential map-reduce
# function mapreduce_seq_impl_skipna(f, op, X::DataValueArray,
#                                      ifirst::Int, ilast::Int)
#     # initialize first reduction
#     v, i = skipna_init(f, op, X, ifirst, ilast)

#     while i < ilast
#         i += 1
#         @inbounds isna = X.isna[i]
#         isna && continue
#         @inbounds entry = X.values[i]
#         v = op(v, f(entry))
#     end
#     return DataValue(v)
# end

# # pairwise map-reduce
# function mapreduce_pairwise_impl_skipna{T}(f, op, X::DataValueArray{T},
#                                              ifirst::Int, ilast::Int,
#                                             #  n_notnull::Int, blksize::Int)
#                                             blksize::Int)
#     if ifirst + blksize > ilast
#         # fall back to Base implementation if no nulls in block
#         # if any(isna, slice(X, ifirst:ilast))
#             return mapreduce_seq_impl_skipna(f, op, X, ifirst, ilast)
#         # else
#             # DataValue(Base.mapreduce_seq_impl(f, op, X.values, ifirst, ilast))
#         # end
#     else
#         imid = (ifirst + ilast) >>> 1
#         # n_notnull1 = imid - ifirst + 1 - countnz(X.isna[ifirst:imid])
#         # n_notnull2 = ilast - imid - countnz(X.isna[imid+1:ilast])
#         v1 = mapreduce_pairwise_impl_skipna(f, op, X, ifirst, imid,
#                                               blksize)
#         v2 = mapreduce_pairwise_impl_skipna(f, op, X, imid+1, ilast,
#                                               blksize)
#         return op(v1, v2)
#     end
# end

# # from comment: https://github.com/JuliaLang/julia/pull/16217#issuecomment-223768129
# sum_pairwise_blocksize(T) = Base.pairwise_blocksize(T, +)

# mapreduce_impl_skipna{T}(f, op, X::DataValueArray{T}) =
#     mapreduce_seq_impl_skipna(f, op, X, 1, length(X.values))
# mapreduce_impl_skipna(f, op::typeof(+), X::DataValueArray) =
#     mapreduce_pairwise_impl_skipna(f, op, X, 1, length(X.values),
#                                    max(128, sum_pairwise_blocksize(f)))

# # general mapreduce interface

# function _mapreduce_skipna{T}(f, op, X::DataValueArray{T}, missingdata::Bool)
#     n = length(X)
#     !missingdata && return DataValue(Base.mapreduce_impl(f, op, X.values, 1, n))

#     nnull = countnz(X.isna)
#     nnull == n && return Base.mr_empty(f, op, T)
#     nnull == n - 1 && return Base.r_promote(op, f(X.values[findnext(x -> x == false), X, 1]))
#     # nnull == 0 && return Base.mapreduce_impl(f, op, X, 1, n)

#     return mapreduce_impl_skipna(f, op, X)
# end

# function Base._mapreduce(f, op, X::DataValueArray, missingdata)
#     missingdata && return Base._mapreduce(f, op, X)
#     DataValue(Base._mapreduce(f, op, X.values))
# end

# # to fix ambiguity warnings
# function Base.mapreduce(f, op::Union{typeof(&), typeof(|)},
#                         X::DataValueArray, skipna::Bool = false)
#     missingdata = any(isna, X)
#     if skipna
#         return _mapreduce_skipna(f, op, X, missingdata)
#     else
#         return Base._mapreduce(f, op, X, missingdata)
#     end
# end


# const specialized_binary = identity

"""
    mapreduce(f, op::Function, X::DataValueArray; [skipna::Bool=false])

Map a function `f` over the elements of `X` and reduce the result under the
operation `op`. One can set the behavior of this method to skip the null entries
of `X` by setting the keyword argument `skipna` equal to true. If `skipna`
behavior is enabled, `f` will be automatically lifted over the elements of `X`.
Note that, in general, mapreducing over a `DataValueArray` will return a
`DataValue` object regardless of whether `skipna` is set to `true` or `false`.
"""
function Base.mapreduce(f, op::Function, X::T; skipna::Bool = false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
    if skipna
        return DataValue(mapreduce(f, op, dropna(X)))
    else
        return Base._mapreduce(f, op, IndexStyle(X), X)
    end
end

"""
    reduce(f, op::Function, X::DataValueArray; [skipna::Bool=false])

Reduce `X`under the operation `op`. One can set the behavior of this method to
skip the null entries of `X` by setting the keyword argument `skipna` equal
to true. If `skipna` behavior is enabled, `f` will be automatically lifted
over the elements of `X`. Note that, in general, mapreducing over a
`DataValueArray` will return a `DataValue` object regardless of whether `skipna`
is set to `true` or `false`.
"""
function Base.reduce(op, X::T; skipna::Bool = false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
    return mapreduce(identity, op, X; skipna = skipna)
end

# standard reductions

for (fn, op) in ((:(Base.sum), +),
                 (:(Base.prod), *),
                 (:(Base.minimum), min),
                 (:(Base.maximum), max))
    @eval begin
        # supertype(typeof(@functorize(abs))) returns Func{1} on Julia 0.4,
        # and Function on 0.5
        function $fn(f::Union{Function,Type},X::T; skipna::Bool=false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
            return mapreduce(f, $op, X; skipna = skipna)
        end

        function $fn(X::T; skipna::Bool=false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
            return mapreduce(identity, $op, X; skipna = skipna)
        end
    end
end

for op in (Base.min, Base.max)
    @eval begin
        function Base._mapreduce(::typeof(identity), ::$(typeof(op)),
                                    X::DataValueArray{T}, missingdata) where {T}
            missingdata && return DataValue{T}()
            DataValue(Base._mapreduce(identity, $op, X.values))
        end
    end
end

# function Base.mapreduce_impl{T}(f, op::typeof(Base.scalarmin), X::DataValueArray{T},
#                                 first::Int, last::Int)
#     i = first
#     v = f(X[i])
#     i += 1
#     while i <= last
#         @inbounds x = f(X[i])
#         if isna(x) | isna(v)
#             return DataValue{eltype(x)}()
#         elseif x.value < v.value
#             v = x
#         end
#         i += 1
#     end
#     return v
# end

# function Base.mapreduce_impl{T}(f, op::typeof(Base.scalarmax), X::DataValueArray{T},
#                                 first::Int, last::Int)
#     i = first
#     v = f(X[i])
#     i += 1
#     while i <= last
#         @inbounds x = f(X[i])
#         if isna(x) | isna(v)
#             return DataValue{eltype(x)}()
#         elseif x.value > v.value
#             v = x
#         end
#         i += 1
#     end
#     return v
# end

function Base.extrema(X::T; skipna::Bool = false) where {N,T2,T<:DataValueArray{T2,N}}
    length(X) > 0 || throw(ArgumentError("collection must be non-empty"))
    vmin = DataValue{T2}()
    vmax = DataValue{T2}()
    @inbounds for i in 1:length(X)
        x = X.values[i]
        missing = X.isna[i]
        if skipna && missing
            continue
        elseif missing
            return (DataValue{T2}(), DataValue{T2}())
        elseif isna(vmax) # Equivalent to isna(vmin)
            vmax = vmin = DataValue(x)
        elseif x > vmax.value
            vmax = DataValue(x)
        elseif x < vmin.value
            vmin = DataValue(x)
        end
    end
    return (vmin, vmax)
end

function Base.extrema(X::T; skipna::Bool = false) where {N,T2,S<:DataValue{T2},T<:AbstractArray{S,N}}
    length(X) > 0 || throw(ArgumentError("collection must be non-empty"))
    vmin = DataValue{T2}()
    vmax = DataValue{T2}()
    @inbounds for i in 1:length(X)
        missing = isna(X[i])
        
        if skipna && missing
            continue
        elseif missing
            return (DataValue{T2}(), DataValue{T2}())
        else
            x = get(X[i])
            if isna(vmax) # Equivalent to isna(vmin)
                vmax = vmin = DataValue(x)
            elseif x > vmax.value
                vmax = DataValue(x)
            elseif x < vmin.value
                vmin = DataValue(x)
            end
        end
    end
    return (vmin, vmax)
end