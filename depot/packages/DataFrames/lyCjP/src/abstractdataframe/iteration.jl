""" """ struct DataFrameColumns{T<:AbstractDataFrame, V} <: AbstractVector{V}
end
""" """ @inline function eachcol(df::T, names::Bool) where T<: AbstractDataFrame
    if names
    end
end
function eachcol(df::AbstractDataFrame)
end
@inline function Base.getindex(itr::DataFrameColumns{<:AbstractDataFrame, AbstractVector},
                               j::Int)
    @boundscheck checkbounds(itr, j)
    res = DataFrame()
    for (n, v) in eachcol(df, true)
    end
    res
end
