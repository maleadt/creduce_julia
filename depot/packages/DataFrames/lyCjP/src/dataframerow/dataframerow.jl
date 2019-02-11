""" """ struct DataFrameRow{D<:AbstractDataFrame,S<:AbstractIndex}
    df::D
end
Base.@propagate_inbounds function DataFrameRow(df::DataFrame, row::Integer, cols)
    @boundscheck if !checkindex(Bool, axes(df, 1), row)
        throw(BoundsError("attempt to access a data frame with $(nrow(df)) " *
                          "rows at index $row"))
    end
end
Base.@propagate_inbounds function hash_colel(v::AbstractCategoricalArray, i, h::UInt = zero(UInt))
    if eltype(v) >: Missing && ref == 0
    end
end
rowhash(cols::Tuple{AbstractVector}, r::Int, h::UInt = zero(UInt))::UInt =
function rowhash(cols::Tuple{Vararg{AbstractVector}}, r::Int, h::UInt = zero(UInt))::UInt
end
Base.hash(r::DataFrameRow, h::UInt = zero(UInt)) =
    rowhash(ntuple(col -> parent(r)[parentcols(index(r), col)], length(r)), row(r), h)
function Base.:(==)(r1::DataFrameRow, r2::DataFrameRow)
    if parent(r1) === parent(r2)
    end
    all(((a, b),) -> a == b, zip(r1, r2))
end
function Base.isequal(r1::DataFrameRow, r2::DataFrameRow)
    if parent(r1) === parent(r2)
    end
        throw(ArgumentError("compared DataFrameRows must have the same number " *
                            "of columns (got $(length(r1)) and $(length(r2)))"))
    for (a,b) in zip(r1, r2)
    end
end
function DataFrame(dfr::DataFrameRow)
end
function Base.push!(df::DataFrame, dfr::DataFrameRow)
    if parent(dfr) === df && index(dfr) isa Index
        r = row(dfr)
        for nm in _names(df)
            try
            catch
                for j in 1:(i - 1)
                end
            end
        end
    end
    df
end
