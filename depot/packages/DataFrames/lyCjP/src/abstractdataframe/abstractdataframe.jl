""" """ abstract type AbstractDataFrame end
Base.names(df::AbstractDataFrame) = names(index(df))
_names(df::AbstractDataFrame) = _names(index(df))
""" """ function names!(df::AbstractDataFrame, vals; makeunique::Bool=false)
    names!(index(df), vals, makeunique=makeunique)
    return df
end
function rename!(df::AbstractDataFrame, args...)
    rename!(index(df), args...)
    return df
end
function rename!(f::Function, df::AbstractDataFrame)
    rename!(f, index(df))
    return df
end
rename(df::AbstractDataFrame, args...) = rename!(copy(df), args...)
rename(f::Function, df::AbstractDataFrame) = rename!(f, copy(df))
""" """ eltypes(df::AbstractDataFrame) = eltype.(columns(df))
Base.size(df::AbstractDataFrame) = (nrow(df), ncol(df))
function Base.size(df::AbstractDataFrame, i::Integer)
    if i == 1
        nrow(df)
    elseif i == 2
        ncol(df)
    else
        throw(ArgumentError("DataFrames only have two dimensions"))
    end
end
Base.lastindex(df::AbstractDataFrame) = ncol(df)
Base.lastindex(df::AbstractDataFrame, i::Integer) = last(axes(df, i))
Base.axes(df::AbstractDataFrame, i::Integer) = Base.OneTo(size(df, i))
Base.ndims(::AbstractDataFrame) = 2
Base.ndims(::Type{<:AbstractDataFrame}) = 2
Base.getproperty(df::AbstractDataFrame, col_ind::Symbol) = getindex(df, col_ind)
Base.setproperty!(df::AbstractDataFrame, col_ind::Symbol, x) = setindex!(df, x, col_ind)
Base.propertynames(df::AbstractDataFrame, private::Bool=false) = names(df)
""" """ function Base.similar(df::AbstractDataFrame, rows::Integer = size(df, 1))
    rows < 0 && throw(ArgumentError("the number of rows must be positive"))
    DataFrame(AbstractVector[similar(x, rows) for x in columns(df)], copy(index(df)))
end
function Base.:(==)(df1::AbstractDataFrame, df2::AbstractDataFrame)
    size(df1, 2) == size(df2, 2) || return false
    isequal(index(df1), index(df2)) || return false
    eq = true
    for idx in 1:size(df1, 2)
        coleq = df1[idx] == df2[idx]
        !isequal(coleq, false) || return false
        eq &= coleq
    end
    return eq
end
function Base.isequal(df1::AbstractDataFrame, df2::AbstractDataFrame)
    size(df1, 2) == size(df2, 2) || return false
    isequal(index(df1), index(df2)) || return false
    for idx in 1:size(df1, 2)
        isequal(df1[idx], df2[idx]) || return false
    end
    return true
end
Base.haskey(df::AbstractDataFrame, key::Any) = haskey(index(df), key)
Base.get(df::AbstractDataFrame, key::Any, default::Any) = haskey(df, key) ? df[key] : default
Base.isempty(df::AbstractDataFrame) = size(df, 1) == 0 || size(df, 2) == 0
""" """ Base.first(df::AbstractDataFrame) = df[1, :]
""" """ Base.first(df::AbstractDataFrame, n::Integer) = df[1:min(n,nrow(df)), :]
""" """ Base.last(df::AbstractDataFrame) = df[nrow(df), :]
""" """ Base.last(df::AbstractDataFrame, n::Integer) = df[max(1,nrow(df)-n+1):nrow(df), :]
function Base.dump(io::IOContext, df::AbstractDataFrame, n::Int, indent)
    println(io, typeof(df), "  $(nrow(df)) observations of $(ncol(df)) variables")
    if n > 0
        for (name, col) in eachcol(df, true)
            println(io, indent, "  ", name, ": ", col)
        end
    end
end
""" """ function StatsBase.describe(df::AbstractDataFrame; stats::Union{Symbol,AbstractVector{Symbol}} =
                            [:mean, :min, :median, :max, :nunique, :nmissing, :eltype])
    allowed_fields = [:mean, :std, :min, :q25, :median, :q75,
                      :max, :nunique, :nmissing, :first, :last, :eltype]
    if stats == :all
        stats = allowed_fields
    end
    if stats isa Symbol
        if !(stats in allowed_fields)
            allowed_msg = "\nAllowed fields are: :" * join(allowed_fields, ", :")
            throw(ArgumentError(":$stats not allowed." * allowed_msg))
        else
            stats = [stats]
        end
    end
    if !issubset(stats, allowed_fields)
        disallowed_fields = setdiff(stats, allowed_fields)
        allowed_msg = "\nAllowed fields are: :" * join(allowed_fields, ", :")
        not_allowed = "Field(s) not allowed: :" * join(disallowed_fields, ", :") * "."
        throw(ArgumentError(not_allowed * allowed_msg))
    end
    data = DataFrame()
    data[:variable] = names(df)
    column_stats_dicts = map(columns(df)) do col
        if eltype(col) >: Missing
            d = get_stats(collect(skipmissing(col)), stats)
        else
            d = get_stats(col, stats)
        end
        if :nmissing in stats
            d[:nmissing] = eltype(col) >: Missing ? count(ismissing, col) : nothing
        end
        if :first in stats
            d[:first] = isempty(col) ? nothing : first(col)
        end
        if :last in stats
            d[:last] = isempty(col) ? nothing : last(col)
        end
        return d
    end
    for stat in stats
        data[stat] = [column_stats_dict[stat] for column_stats_dict in column_stats_dicts]
    end
    return data
end
function get_stats(col::AbstractVector, stats::AbstractVector{Symbol})
    d = Dict{Symbol, Any}()
    if :q25 in stats || :median in stats || :q75 in stats
        q = try quantile(col, [.25, .5, .75]) catch; (nothing, nothing, nothing) end
        d[:q25] = q[1]
        d[:median] = q[2]
        d[:q75] = q[3]
    end
    if :min in stats || :max in stats
        ex = try extrema(col) catch; (nothing, nothing) end
        d[:min] = ex[1]
        d[:max] = ex[2]
    end
    if :mean in stats || :std in stats
        m = try mean(col) catch end
        d[:mean] = m
    end
    if :std in stats
        d[:std] = try std(col, mean = m) catch end
    end
    if :nunique in stats
        if eltype(col) <: Real
            d[:nunique] = nothing
        else
            d[:nunique] = try length(unique(col)) catch end
        end
    end
    if :eltype in stats
        d[:eltype] = eltype(col)
    end
    return d
end
function _nonmissing!(res, col)
    @inbounds for (i, el) in enumerate(col)
        res[i] &= !ismissing(el)
    end
    return nothing
end
function _nonmissing!(res, col::CategoricalArray{>: Missing})
    for (i, el) in enumerate(col.refs)
        res[i] &= el > 0
    end
    return nothing
end
""" """ function completecases(df::AbstractDataFrame)
    res = trues(size(df, 1))
    for i in 1:size(df, 2)
        _nonmissing!(res, df[i])
    end
    res
end
function completecases(df::AbstractDataFrame, col::Union{Integer, Symbol})
    res = trues(size(df, 1))
    _nonmissing!(res, df[col])
    res
end
completecases(df::AbstractDataFrame, cols::AbstractVector) =
    completecases(df[cols])
""" """ function dropmissing(df::AbstractDataFrame,
                     cols::Union{Integer, Symbol, AbstractVector}=1:size(df, 2);
                     disallowmissing::Bool=false)
    newdf = df[completecases(df, cols), :]
    if disallowmissing
        disallowmissing!(newdf, cols)
    else
        Base.depwarn("dropmissing will change eltype of cols to disallow missing by default. " *
                     "Use dropmissing(df, cols, disallowmissing=false) to allow for missing values.", :dropmissing)
    end
    newdf
end
""" """ function dropmissing!(df::AbstractDataFrame,
                      cols::Union{Integer, Symbol, AbstractVector}=1:size(df, 2);
                      disallowmissing::Bool=false)
    deleterows!(df, (!).(completecases(df, cols)))
    if disallowmissing
        disallowmissing!(df, cols)
    else
        Base.depwarn("dropmissing! will change eltype of cols to disallow missing by default. " *
                     "Use dropmissing!(df, cols, disallowmissing=false) to retain missing.", :dropmissing!)
    end
    df
end
""" """ Base.filter(f, df::AbstractDataFrame) = df[collect(f(r)::Bool for r in eachrow(df)), :]
""" """ Base.filter!(f, df::AbstractDataFrame) =
    deleterows!(df, findall(collect(!f(r)::Bool for r in eachrow(df))))
function Base.convert(::Type{Matrix}, df::AbstractDataFrame)
    T = reduce(promote_type, eltypes(df))
    convert(Matrix{T}, df)
end
function Base.convert(::Type{Matrix{T}}, df::AbstractDataFrame) where T
    n, p = size(df)
    res = Matrix{T}(undef, n, p)
    idx = 1
    for (name, col) in zip(names(df), columns(df))
        try
            copyto!(res, idx, col)
        catch err
            if err isa MethodError && err.f == convert &&
               !(T >: Missing) && any(ismissing, col)
                error("cannot convert a DataFrame containing missing values to Matrix{$T} (found for column $name)")
            else
                rethrow(err)
            end
        end
        idx += n
    end
    return res
end
Base.Matrix(df::AbstractDataFrame) = Base.convert(Matrix, df)
Base.Matrix{T}(df::AbstractDataFrame) where {T} = Base.convert(Matrix{T}, df)
""" """ function nonunique(df::AbstractDataFrame)
    gslots = row_group_slots(ntuple(i -> df[i], ncol(df)), Val(true))[3]
    res = fill(true, nrow(df))
    @inbounds for g_row in gslots
        (g_row > 0) && (res[g_row] = false)
    end
    return res
end
nonunique(df::AbstractDataFrame, cols::Union{Integer, Symbol}) = nonunique(df[[cols]])
nonunique(df::AbstractDataFrame, cols::Any) = nonunique(df[cols])
Base.unique!(df::AbstractDataFrame) = deleterows!(df, findall(nonunique(df)))
Base.unique!(df::AbstractDataFrame, cols::AbstractVector) =
    deleterows!(df, findall(nonunique(df, cols)))
Base.unique!(df::AbstractDataFrame, cols::Union{Integer, Symbol, Colon}) =
    deleterows!(df, findall(nonunique(df, cols)))
Base.unique(df::AbstractDataFrame) = df[(!).(nonunique(df)), :]
Base.unique(df::AbstractDataFrame, cols::AbstractVector) =
    df[(!).(nonunique(df, cols)), :]
Base.unique(df::AbstractDataFrame, cols::Union{Integer, Symbol, Colon}) =
    df[(!).(nonunique(df, cols)), :]
function without(df::AbstractDataFrame, icols::Vector{<:Integer})
    newcols = setdiff(1:ncol(df), icols)
    df[newcols]
end
without(df::AbstractDataFrame, i::Int) = without(df, [i])
without(df::AbstractDataFrame, c::Any) = without(df, index(df)[c])
Base.hcat(df::AbstractDataFrame, x; makeunique::Bool=false) =
    hcat!(copy(df), x, makeunique=makeunique)
Base.hcat(x, df::AbstractDataFrame; makeunique::Bool=false) =
    hcat!(x, df, makeunique=makeunique)
Base.hcat(df1::AbstractDataFrame, df2::AbstractDataFrame; makeunique::Bool=false) =
    hcat!(copy(df1), df2, makeunique=makeunique)
Base.hcat(df::AbstractDataFrame, x, y...; makeunique::Bool=false) =
    hcat!(hcat(df, x, makeunique=makeunique), y..., makeunique=makeunique)
Base.hcat(df1::AbstractDataFrame, df2::AbstractDataFrame, dfn::AbstractDataFrame...;
          makeunique::Bool=false) =
    hcat!(hcat(df1, df2, makeunique=makeunique), dfn..., makeunique=makeunique)
""" """ Base.vcat(df::AbstractDataFrame) = df
Base.vcat(dfs::AbstractDataFrame...) = _vcat(collect(dfs))
function _vcat(dfs::AbstractVector{<:AbstractDataFrame})
    isempty(dfs) && return DataFrame()
    allheaders = map(names, dfs)
    uniqueheaders = unique(allheaders)
    unionunique = union(uniqueheaders...)
    intersectunique = intersect(uniqueheaders...)
    coldiff = setdiff(unionunique, intersectunique)
    if !isempty(coldiff)
        filter!(u -> Set(u) != Set(unionunique), uniqueheaders)
        estrings = Vector{String}(undef, length(uniqueheaders))
        for (i, u) in enumerate(uniqueheaders)
            matching = findall(h -> u == h, allheaders)
            headerdiff = setdiff(coldiff, u)
            cols = join(headerdiff, ", ", " and ")
            args = join(matching, ", ", " and ")
            estrings[i] = "column(s) $cols are missing from argument(s) $args"
        end
        throw(ArgumentError(join(estrings, ", ", ", and ")))
    end
    header = allheaders[1]
    length(header) == 0 && return DataFrame()
    cols = Vector{AbstractVector}(undef, length(header))
    for (i, name) in enumerate(header)
        data = [df[name] for df in dfs]
        lens = map(length, data)
        T = mapreduce(eltype, promote_type, data)
        cols[i] = Tables.allocatecolumn(T, sum(lens))
        offset = 1
        for j in 1:length(data)
            copyto!(cols[i], offset, data[j])
            offset += lens[j]
        end
    end
    return DataFrame(cols, header)
end
""" """ Base.repeat(df::AbstractDataFrame; inner::Integer = 1, outer::Integer = 1) =
    mapcols(x -> repeat(x, inner = inner, outer = outer), df)
""" """ Base.repeat(df::AbstractDataFrame, count::Integer) =
    mapcols(x -> repeat(x, count), df)
const hashdf_seed = UInt == UInt32 ? 0xfd8bb02e : 0x6215bada8c8c46de
function Base.hash(df::AbstractDataFrame, h::UInt)
    h += hashdf_seed
    h += hash(size(df))
    for i in 1:size(df, 2)
        h = hash(df[i], h)
    end
    return h
end
Base.parent(adf::AbstractDataFrame) = adf
Base.parentindices(adf::AbstractDataFrame) = axes(adf)
""" """ 
