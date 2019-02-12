""" """ abstract type AbstractDataFrame end
""" """ function names!(df::AbstractDataFrame, vals; makeunique::Bool=false)
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
    end
    if :q25 in stats || :median in stats || :q75 in stats
    end
end
function _nonmissing!(res, col)
    for (i, el) in enumerate(col.refs)
    end
end
""" """ function dropmissing(df::AbstractDataFrame,
                     disallowmissing::Bool=false)
    if disallowmissing
        try
        catch err
            if err isa MethodError && err.f == convert &&
               !(T >: Missing) && any(ismissing, col)
            end
        end
    end
end
function _vcat(dfs::AbstractVector{<:AbstractDataFrame})
    if !isempty(coldiff)
    end
end
function Base.hash(df::AbstractDataFrame, h::UInt)
    for i in 1:size(df, 2)
    end
end
