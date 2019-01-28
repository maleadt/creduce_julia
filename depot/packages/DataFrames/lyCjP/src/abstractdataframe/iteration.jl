""" """ struct DataFrameRows{D<:AbstractDataFrame,S<:AbstractIndex} <: AbstractVector{DataFrameRow{D,S}}
    df::D
    index::S
end
Base.summary(dfrs::DataFrameRows) = "$(length(dfrs))-element DataFrameRows"
Base.summary(io::IO, dfrs::DataFrameRows) = print(io, summary(dfrs))
""" """ eachrow(df::AbstractDataFrame) = DataFrameRows(df, index(df))
Base.IndexStyle(::Type{<:DataFrameRows}) = Base.IndexLinear()
Base.size(itr::DataFrameRows) = (size(itr.df, 1), )
Base.@propagate_inbounds Base.getindex(itr::DataFrameRows, i::Int) =
    DataFrameRow(itr.df, itr.index, i)
Base.@propagate_inbounds Base.getindex(itr::DataFrameRows{<:SubDataFrame}, i::Int) =
    DataFrameRow(parent(itr.df), itr.index, rows(itr.df)[i])
""" """ struct DataFrameColumns{T<:AbstractDataFrame, V} <: AbstractVector{V}
    df::T
end
""" """ @inline function eachcol(df::T, names::Bool) where T<: AbstractDataFrame
    if names
        DataFrameColumns{T, Pair{Symbol, AbstractVector}}(df)
    else
        DataFrameColumns{T, AbstractVector}(df)
    end
end
function eachcol(df::AbstractDataFrame)
    Base.depwarn("In the future eachcol will have names argument set to false by default", :eachcol)
    eachcol(df, true)
end
columns(df::AbstractDataFrame) = eachcol(df, false)
Base.size(itr::DataFrameColumns) = (size(itr.df, 2),)
Base.IndexStyle(::Type{<:DataFrameColumns}) = Base.IndexLinear()
@inline function Base.getindex(itr::DataFrameColumns{<:AbstractDataFrame,
                                                     Pair{Symbol, AbstractVector}},
                               j::Int)
    @boundscheck checkbounds(itr, j)
    _names(itr.df)[j] => itr.df[j]
end
@inline function Base.getindex(itr::DataFrameColumns{<:AbstractDataFrame, AbstractVector},
                               j::Int)
    @boundscheck checkbounds(itr, j)
    itr.df[j]
end
""" """ function mapcols(f::Union{Function,Type}, df::AbstractDataFrame)
    res = DataFrame()
    for (n, v) in eachcol(df, true)
        res[n] = f(v)
    end
    res
end
