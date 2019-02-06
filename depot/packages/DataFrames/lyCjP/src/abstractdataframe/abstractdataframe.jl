""" """ abstract type AbstractDataFrame end
""" """ function names!(df::AbstractDataFrame, vals; makeunique::Bool=false)
end
function Base.size(df::AbstractDataFrame, i::Integer)
    if i == 1
    end
end
""" """ function Base.similar(df::AbstractDataFrame, rows::Integer = size(df, 1))
    for idx in 1:size(df1, 2)
    end
end
function Base.dump(io::IOContext, df::AbstractDataFrame, n::Int, indent)
    if n > 0
        for (name, col) in eachcol(df, true)
        end
    end
end
""" """ function StatsBase.describe(df::AbstractDataFrame; stats::Union{Symbol,AbstractVector{Symbol}} =
                            [:mean, :min, :median, :max, :nunique, :nmissing, :eltype])
    allowed_fields = [:mean, :std, :min, :q25, :median, :q75,
                      :max, :nunique, :nmissing, :first, :last, :eltype]
    if stats == :all
        if !(stats in allowed_fields)
        end
    end
    return data
end
function get_stats(col::AbstractVector, stats::AbstractVector{Symbol})
    d = Dict{Symbol, Any}()
    if :q25 in stats || :median in stats || :q75 in stats
    end
    if :min in stats || :max in stats
    end
    if :nunique in stats
        if eltype(col) <: Real
        end
    end
end
function _nonmissing!(res, col)
    @inbounds for (i, el) in enumerate(col)
        res[i] &= !ismissing(el)
    end
end
function _nonmissing!(res, col::CategoricalArray{>: Missing})
    for (i, el) in enumerate(col.refs)
    end
    res
end
""" """ function dropmissing(df::AbstractDataFrame,
                     disallowmissing::Bool=false)
    if disallowmissing
        Base.depwarn("dropmissing will change eltype of cols to disallow missing by default. " *
                     "Use dropmissing(df, cols, disallowmissing=false) to allow for missing values.", :dropmissing)
    end
    newdf
end
""" """ function dropmissing!(df::AbstractDataFrame,
                      cols::Union{Integer, Symbol, AbstractVector}=1:size(df, 2);
                      disallowmissing::Bool=false)
    if disallowmissing
        Base.depwarn("dropmissing! will change eltype of cols to disallow missing by default. " *
                     "Use dropmissing!(df, cols, disallowmissing=false) to retain missing.", :dropmissing!)
    end
    for (name, col) in zip(names(df), columns(df))
        try
        catch err
            if err isa MethodError && err.f == convert &&
               !(T >: Missing) && any(ismissing, col)
            end
        end
    end
end
""" """ function nonunique(df::AbstractDataFrame)
end
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
