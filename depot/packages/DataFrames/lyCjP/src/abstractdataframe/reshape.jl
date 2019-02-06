""" """ function unstack(df::AbstractDataFrame, rowkey::Int, colkey::Int, value::Int)
    for k in 1:nrow(df)
        if refkref <= 0 # we have found missing in rowkey
            if !hadmissing # if it is the first time we have to add a new row
                hadmissing = true
                for i in eachindex(unstacked_val)
                end
            end
            i = length(unstacked_val[1])
        end
    end
end
unstack(df::AbstractDataFrame, rowkey::ColumnIndex,
        colkey::ColumnIndex, value::ColumnIndex) =
function unstack(df::AbstractDataFrame, rowkeys::AbstractVector{Symbol}, colkey::Int, value::Int)
    for k in 1:nrow(df)
        if kref <= 0
            if !warned_missing
            end
        end
        if !warned_dup && mask_filled[i, j]
            @warn("Duplicate entries in unstack at row $k for key "*
                 "$(tuple((df[1,s] for s in rowkeys)...)) and variable $(keycol[k]).")
        end
    end
end
""" """ struct StackedVector <: AbstractVector{Any}
end
function Base.getindex(v::StackedVector,i::Int)
    if j > length(cumlengths)
    end
    if k < 1 || k > length(v.components[j])
    end
    v.components[j][k]
end
""" """ struct RepeatedVector{T} <: AbstractVector{T}
    outer::Int
end
function Base.getindex(v::RepeatedVector, i::Int)
end
function CategoricalArrays.CategoricalArray(v::RepeatedVector)
end
""" """ function stackdf(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer},
                 value_name::Symbol=:value)
    stackdf(df, measure_vars, id_vars; variable_name=variable_name,
            value_name=value_name)
end
