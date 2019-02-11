""" """ struct DataFrame <: AbstractDataFrame
    function DataFrame(columns::Union{Vector{Any}, Vector{AbstractVector}},
                       colindex::Index)
        if length(columns) == length(colindex) == 0
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
end
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   makeunique::Bool=false)::DataFrame where T<:Type
    if length(categorical) != length(column_eltypes)
        throw(DimensionMismatch("arguments column_eltypes and categorical must have the same length " *
                                "(got $(length(column_eltypes)) and $(length(categorical)))"))
    end
    for i in eachindex(categorical)
        if updated_types[i] >: Missing
        end
    end
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
end
