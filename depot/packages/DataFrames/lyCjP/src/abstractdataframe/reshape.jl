""" """ function stack(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer},
               id_vars::AbstractVector{<:Integer}; variable_name::Symbol=:variable,
               value_name::Symbol=:value)
    N = length(measure_vars)
    cnames = names(df)[id_vars]
    insert!(cnames, 1, value_name)
    insert!(cnames, 1, variable_name)
    DataFrame(AbstractVector[repeat(_names(df)[measure_vars], inner=nrow(df)),   # variable
                  vcat([df[c] for c in measure_vars]...),             # value
                  [repeat(df[c], outer=N) for c in id_vars]...],      # id_var columns
              cnames)
end
function stack(df::AbstractDataFrame, measure_var::Int, id_var::Int;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, [measure_var], [id_var];
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer}, id_var::Int;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, measure_vars, [id_var];
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_var::Int, id_vars::AbstractVector{<:Integer};
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, [measure_var], id_vars;
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_vars, id_vars;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, index(df)[measure_vars], index(df)[id_vars];
          variable_name=variable_name, value_name=value_name)
end
numeric_vars(df::AbstractDataFrame) =
    [T <: AbstractFloat || (T >: Missing && Missings.T(T) <: AbstractFloat)
     for T in eltypes(df)]
function stack(df::AbstractDataFrame, measure_vars = numeric_vars(df);
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    mv_inds = index(df)[measure_vars]
    stack(df, mv_inds, setdiff(1:ncol(df), mv_inds);
          variable_name=variable_name, value_name=value_name)
end
""" """ function melt(df::AbstractDataFrame, id_vars::Union{Int,Symbol};
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    melt(df, [id_vars]; variable_name=variable_name, value_name=value_name)
end
function melt(df::AbstractDataFrame, id_vars;
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    id_inds = index(df)[id_vars]
    stack(df, setdiff(1:ncol(df), id_inds), id_inds;
          variable_name=variable_name, value_name=value_name)
end
function melt(df::AbstractDataFrame, id_vars, measure_vars;
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, measure_vars, id_vars; variable_name=variable_name,
          value_name=value_name)
end
melt(df::AbstractDataFrame; variable_name::Symbol=:variable, value_name::Symbol=:value) =
    stack(df; variable_name=variable_name, value_name=value_name)
""" """ function unstack(df::AbstractDataFrame, rowkey::Int, colkey::Int, value::Int)
    refkeycol = categorical(df[rowkey])
    droplevels!(refkeycol)
    keycol = categorical(df[colkey])
    droplevels!(keycol)
    valuecol = df[value]
    _unstack(df, rowkey, colkey, value, keycol, valuecol, refkeycol)
end
function _unstack(df::AbstractDataFrame, rowkey::Int,
                  colkey::Int, value::Int, keycol, valuecol, refkeycol)
    Nrow = length(refkeycol.pool)
    Ncol = length(keycol.pool)
    unstacked_val = [similar_missing(valuecol, Nrow) for i in 1:Ncol]
    hadmissing = false # have we encountered missing in refkeycol
    mask_filled = falses(Nrow+1, Ncol) # has a given [row,col] entry been filled?
    warned_dup = false # have we already printed duplicate entries warning?
    warned_missing = false # have we already printed missing in keycol warning?
    keycol_order = Vector{Int}(CategoricalArrays.order(keycol.pool))
    refkeycol_order = Vector{Int}(CategoricalArrays.order(refkeycol.pool))
    for k in 1:nrow(df)
        kref = keycol.refs[k]
        if kref <= 0 # we have found missing in colkey
            if !warned_missing
                @warn("Missing value in variable $(_names(df)[colkey]) at row $k. Skipping.")
                warned_missing = true
            end
            continue # skip processing it
        end
        j = keycol_order[kref]
        refkref = refkeycol.refs[k]
        if refkref <= 0 # we have found missing in rowkey
            if !hadmissing # if it is the first time we have to add a new row
                hadmissing = true
                for i in eachindex(unstacked_val)
                    push!(unstacked_val[i], missing)
                end
            end
            i = length(unstacked_val[1])
        else
            i = refkeycol_order[refkref]
        end
        if !warned_dup && mask_filled[i, j]
            @warn("Duplicate entries in unstack at row $k for key "*
                  "$(refkeycol[k]) and variable $(keycol[k]).")
            warned_dup = true
        end
        unstacked_val[j][i] = valuecol[k]
        mask_filled[i, j] = true
    end
    levs = levels(refkeycol)
    col = similar(df[rowkey], length(levs) + hadmissing)
    copyto!(col, levs)
    hadmissing && (col[end] = missing)
    df2 = DataFrame(unstacked_val, map(Symbol, levels(keycol)))
    insertcols!(df2, 1, _names(df)[rowkey] => col)
end
unstack(df::AbstractDataFrame, rowkey::ColumnIndex,
        colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, index(df)[rowkey], index(df)[colkey], index(df)[value])
unstack(df::AbstractDataFrame, colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, index(df)[colkey], index(df)[value])
unstack(df::AbstractDataFrame, colkey::Int, value::Int) =
    unstack(df, setdiff(_names(df), _names(df)[[colkey, value]]), colkey, value)
unstack(df::AbstractDataFrame, rowkeys, colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, rowkeys, index(df)[colkey], index(df)[value])
unstack(df::AbstractDataFrame, rowkeys::AbstractVector{<:Real}, colkey::Int, value::Int) =
    unstack(df, names(df)[rowkeys], colkey, value)
function unstack(df::AbstractDataFrame, rowkeys::AbstractVector{Symbol}, colkey::Int, value::Int)
    length(rowkeys) == 0 && throw(ArgumentError("No key column found"))
    length(rowkeys) == 1 && return unstack(df, rowkeys[1], colkey, value)
    g = groupby(df, rowkeys, sort=true)
    keycol = categorical(df[colkey])
    droplevels!(keycol)
    valuecol = df[value]
    _unstack(df, rowkeys, colkey, value, keycol, valuecol, g)
end
function _unstack(df::AbstractDataFrame, rowkeys::AbstractVector{Symbol},
                  colkey::Int, value::Int, keycol, valuecol, g)
    groupidxs = [g.idx[g.starts[i]:g.ends[i]] for i in 1:length(g.starts)]
    rowkey = zeros(Int, size(df, 1))
    for i in 1:length(groupidxs)
        rowkey[groupidxs[i]] .= i
    end
    df1 = df[g.idx[g.starts], g.cols]
    Nrow = length(g)
    Ncol = length(levels(keycol))
    unstacked_val = [similar_missing(valuecol, Nrow) for i in 1:Ncol]
    mask_filled = falses(Nrow, Ncol)
    warned_dup = false
    warned_missing = false
    keycol_order = Vector{Int}(CategoricalArrays.order(keycol.pool))
    for k in 1:nrow(df)
        kref = keycol.refs[k]
        if kref <= 0
            if !warned_missing
                @warn("Missing value in variable $(_names(df)[colkey]) at row $k. Skipping.")
                warned_missing = true
            end
            continue
        end
        j = keycol_order[kref]
        i = rowkey[k]
        if !warned_dup && mask_filled[i, j]
            @warn("Duplicate entries in unstack at row $k for key "*
                 "$(tuple((df[1,s] for s in rowkeys)...)) and variable $(keycol[k]).")
            warned_dup = true
        end
        unstacked_val[j][i] = valuecol[k]
        mask_filled[i, j] = true
    end
    df2 = DataFrame(unstacked_val, map(Symbol, levels(keycol)))
    hcat(df1, df2)
end
unstack(df::AbstractDataFrame) = unstack(df, :variable, :value)
""" """ struct StackedVector <: AbstractVector{Any}
    components::Vector{Any}
end
function Base.getindex(v::StackedVector,i::Int)
    lengths = [length(x)::Int for x in v.components]
    cumlengths = [0; cumsum(lengths)]
    j = searchsortedlast(cumlengths .+ 1, i)
    if j > length(cumlengths)
        error("indexing bounds error")
    end
    k = i - cumlengths[j]
    if k < 1 || k > length(v.components[j])
        error("indexing bounds error")
    end
    v.components[j][k]
end
Base.IndexStyle(::Type{StackedVector}) = Base.IndexLinear()
Base.size(v::StackedVector) = (length(v),)
Base.length(v::StackedVector) = sum(map(length, v.components))
Base.eltype(v::StackedVector) = promote_type(map(eltype, v.components)...)
Base.similar(v::StackedVector, T::Type, dims::Union{Integer, AbstractUnitRange}...) =
    similar(v.components[1], T, dims...)
CategoricalArrays.CategoricalArray(v::StackedVector) = CategoricalArray(v[:]) # could be more efficient
""" """ struct RepeatedVector{T} <: AbstractVector{T}
    parent::AbstractVector{T}
    inner::Int
    outer::Int
end
function Base.getindex(v::RepeatedVector, i::Int)
    N = length(v.parent)
    idx = Base.fld1(mod1(i,v.inner*N),v.inner)
    v.parent[idx]
end
Base.IndexStyle(::Type{<:RepeatedVector}) = Base.IndexLinear()
Base.size(v::RepeatedVector) = (length(v),)
Base.length(v::RepeatedVector) = v.inner * v.outer * length(v.parent)
Base.eltype(v::RepeatedVector{T}) where {T} = T
Base.reverse(v::RepeatedVector) = RepeatedVector(reverse(v.parent), v.inner, v.outer)
Base.similar(v::RepeatedVector, T::Type, dims::Dims) = similar(v.parent, T, dims)
Base.unique(v::RepeatedVector) = unique(v.parent)
function CategoricalArrays.CategoricalArray(v::RepeatedVector)
    res = CategoricalArrays.CategoricalArray(v.parent)
    res.refs = repeat(res.refs, inner = [v.inner], outer = [v.outer])
    res
end
""" """ function stackdf(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer},
                 id_vars::AbstractVector{<:Integer}; variable_name::Symbol=:variable,
                 value_name::Symbol=:value)
    N = length(measure_vars)
    cnames = names(df)[id_vars]
    insert!(cnames, 1, value_name)
    insert!(cnames, 1, variable_name)
    DataFrame(AbstractVector[RepeatedVector(_names(df)[measure_vars], nrow(df), 1), # variable
                             StackedVector(Any[df[c] for c in measure_vars]),       # value
                             [RepeatedVector(df[c], 1, N) for c in id_vars]...],    # id_var columns
              cnames)
end
function stackdf(df::AbstractDataFrame, measure_var::Int, id_var::Int;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, [measure_var], [id_var]; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars, id_var::Int;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, measure_vars, [id_var]; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_var::Int, id_vars;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, [measure_var], id_vars; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars, id_vars;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, index(df)[measure_vars], index(df)[id_vars];
            variable_name=variable_name, value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars = numeric_vars(df);
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    m_inds = index(df)[measure_vars]
    stackdf(df, m_inds, setdiff(1:ncol(df), m_inds);
            variable_name=variable_name, value_name=value_name)
end
""" """ function meltdf(df::AbstractDataFrame, id_vars; variable_name::Symbol=:variable,
                value_name::Symbol=:value)
    id_inds = index(df)[id_vars]
    stackdf(df, setdiff(1:ncol(df), id_inds), id_inds;
            variable_name=variable_name, value_name=value_name)
end
function meltdf(df::AbstractDataFrame, id_vars, measure_vars;
                variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, measure_vars, id_vars; variable_name=variable_name,
            value_name=value_name)
end
meltdf(df::AbstractDataFrame; variable_name::Symbol=:variable, value_name::Symbol=:value) =
    stackdf(df; variable_name=variable_name, value_name=value_name)
