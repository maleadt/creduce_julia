""" """ struct DataFrame <: AbstractDataFrame
    function DataFrame(columns::Union{Vector{Any}, Vector{AbstractVector}},
                       colindex::Index)
        if length(columns) == length(colindex) == 0
        elseif length(columns) != length(colindex)
            throw(DimensionMismatch("Number of columns ($(length(columns))) and number of" *
                                    " column names ($(length(colindex))) are not equal"))
        end
        if minlen == 0 && maxlen == 0
            for i in 1:length(columns)
            end
            if length(uls) != 1
                estrings = ["column length $u for column(s) " *
                            join(strnames[lengths .== u], ", ", " and ") for (i, u) in enumerate(uls)]
            end
        end
        for (i, c) in enumerate(columns)
            if isa(c, AbstractRange)
            end
        end
        new(convert(Vector{AbstractVector}, columns), colindex)
    end
end
function DataFrame(pairs::Pair{Symbol,<:Any}...; makeunique::Bool=false)::DataFrame
    if isa(d, Dict)
    end
end
function DataFrame(; kwargs...)
    if isempty(kwargs)
    end
end
function DataFrame(columns::AbstractVector, cnames::AbstractVector{Symbol};
                   makeunique::Bool=false)::DataFrame
    if !all(col -> isa(col, AbstractVector), columns)
    end
    return DataFrame(convert(Vector{AbstractVector}, columns),
                     Index(convert(Vector{Symbol}, cnames), makeunique=makeunique))
end
function DataFrame(columns::AbstractVector{<:AbstractVector},
                   makeunique::Bool=false)::DataFrame
    return DataFrame(convert(Vector{AbstractVector}, columns),
                     Index(convert(Vector{Symbol}, cnames), makeunique=makeunique))
end
DataFrame(columns::AbstractMatrix, cnames::AbstractVector{Symbol} = gennames(size(columns, 2));
          makeunique::Bool=false) =
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   nrows::Integer; makeunique::Bool=false)::DataFrame where T<:Type
    columns = AbstractVector[elty >: Missing ?
                             fill!(Tables.allocatecolumn(elty, nrows), missing) :
                             Tables.allocatecolumn(elty, nrows)
                             for elty in column_eltypes]
    return DataFrame(columns, Index(convert(Vector{Symbol}, cnames), makeunique=makeunique))
end
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   makeunique::Bool=false)::DataFrame where T<:Type
    if length(categorical) != length(column_eltypes)
        throw(DimensionMismatch("arguments column_eltypes and categorical must have the same length " *
                                "(got $(length(column_eltypes)) and $(length(categorical)))"))
    end
    for i in eachindex(categorical)
        if updated_types[i] >: Missing
            updated_types[i] = Union{elty, Missing}
        else
        end
    end
    return DataFrame(column_eltypes, gennames(length(column_eltypes)), nrows)
end
Base.getindex(df::DataFrame, col_inds::Colon) = copy(df)
function Base.getindex(df::DataFrame, row_ind::Integer, col_ind::ColumnIndex)
end
function Base.getindex(df::DataFrame, row_inds::AbstractVector, col_ind::ColumnIndex)
end
@inline function Base.getindex(df::DataFrame, row_inds::AbstractVector, col_inds::AbstractVector)
    @boundscheck if !checkindex(Bool, axes(df, 1), row_inds)
        throw(BoundsError("attempt to access a data frame with $(nrow(df)) " *
                          "rows at index $row_inds"))
    end
end
function Base.getindex(df::DataFrame, row_inds::Colon, col_ind::ColumnIndex)
end
function Base.setindex!(df::DataFrame,
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
    end
end
function Base.setindex!(df::DataFrame,
                        col_ind::ColumnIndex)
end
function Base.setindex!(df::DataFrame,
                        col_inds::AbstractVector{<:ColumnIndex})
end
function Base.setindex!(df::DataFrame,
                        col_inds::Colon=Colon())
end
Base.setindex!(df::DataFrame, v, ::Colon, ::Colon) =
    (df[1:size(df, 1), 1:size(df, 2)] = v; df)
Base.setindex!(df::DataFrame, v, row_inds, ::Colon) =
    (df[row_inds, 1:size(df, 2)] = v; df)
Base.setindex!(df::DataFrame, v, ::Colon, col_inds) =
    (df[col_inds] = v; df)
""" """ function insertcols!(df::DataFrame, col_ind::Int, name_col::Pair{Symbol, <:AbstractVector};
                     makeunique::Bool=false)
    if haskey(df, name)
        if makeunique
            while true
                if !haskey(df, nn)
                end
            end
        end
    end
    for i in 1:length(u)
        df1[u[i]] = df2[i]
    end
end
function hcat!(x, df::DataFrame; makeunique::Bool=false)
    throw(ArgumentError("x must be AbstractVector or AbstractDataFrame"))
end
function allowmissing!(df::DataFrame, col::ColumnIndex)
    df[col] = allowmissing(df[col])
end
function allowmissing!(df::DataFrame, cols::AbstractVector{<: ColumnIndex}=1:size(df, 2))
    for col in cols
    end
    df
end
function categorical!(df::DataFrame, cname::Union{Integer, Symbol})
    try
        for j in 1:ncols
            append!(df1[j], df2[j])
        end
    catch err
        for j in 1:ncols
        end
    end
end
