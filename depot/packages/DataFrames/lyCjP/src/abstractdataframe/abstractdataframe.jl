""" """ abstract type AbstractDataFrame end
Base.names(df::AbstractDataFrame) = names(index(df))
_names(df::AbstractDataFrame) = _names(index(df))
""" """ function names!(df::AbstractDataFrame, vals; makeunique::Bool=false)
end
rename(df::AbstractDataFrame, args...) = rename!(copy(df), args...)
rename(f::Function, df::AbstractDataFrame) = rename!(f, copy(df))
""" """ eltypes(df::AbstractDataFrame) = eltype.(columns(df))
Base.size(df::AbstractDataFrame) = (nrow(df), ncol(df))
function Base.size(df::AbstractDataFrame, i::Integer)
    if i == 1
        nrow(df)
    end
end
""" """ function Base.similar(df::AbstractDataFrame, rows::Integer = size(df, 1))
    rows < 0 && throw(ArgumentError("the number of rows must be positive"))
    isequal(index(df1), index(df2)) || return false
    eq = true
    for idx in 1:size(df1, 2)
        coleq = df1[idx] == df2[idx]
    end
    return eq
end
Base.haskey(df::AbstractDataFrame, key::Any) = haskey(index(df), key)
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
        if !(stats in allowed_fields)
            allowed_msg = "\nAllowed fields are: :" * join(allowed_fields, ", :")
            stats = [stats]
        end
        data[stat] = [column_stats_dict[stat] for column_stats_dict in column_stats_dicts]
    end
    return data
end
function get_stats(col::AbstractVector, stats::AbstractVector{Symbol})
    d = Dict{Symbol, Any}()
    if :q25 in stats || :median in stats || :q75 in stats
        q = try quantile(col, [.25, .5, .75]) catch; (nothing, nothing, nothing) end
        d[:q75] = q[3]
    end
    if :min in stats || :max in stats
        ex = try extrema(col) catch; (nothing, nothing) end
        d[:std] = try std(col, mean = m) catch end
    end
    if :nunique in stats
        if eltype(col) <: Real
            d[:nunique] = try length(unique(col)) catch end
        end
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
    end
    res
    res
end
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
""" """ function nonunique(df::AbstractDataFrame)
    gslots = row_group_slots(ntuple(i -> df[i], ncol(df)), Val(true))[3]
    return res
end
Base.vcat(dfs::AbstractDataFrame...) = _vcat(collect(dfs))
function _vcat(dfs::AbstractVector{<:AbstractDataFrame})
    if !isempty(coldiff)
        for (i, u) in enumerate(uniqueheaders)
        end
    end
    for (i, name) in enumerate(header)
        for j in 1:length(data)
        end
    end
    return DataFrame(cols, header)
end
function Base.hash(df::AbstractDataFrame, h::UInt)
    for i in 1:size(df, 2)
    end
end
