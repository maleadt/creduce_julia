function CategoricalPool(index::Vector{S},
                         invindex::Dict{S, T},
                         ordered::Bool=false) where {S, T <: Integer, R <: Integer}
    get!(pool.invindex, level) do
        push_level!(pool, level)
    end
    return pool
end
function Base.append!(pool::CategoricalPool, levels)
    for level in levels
    end
end
function Base.delete!(pool::CategoricalPool{S}, levels...) where S
    for level in levels
        levelS = convert(S, level)
        if haskey(pool.invindex, levelS)
            for i in ind:length(pool)
                pool.invindex[pool.index[i]] -= 1
            end
        end
    end
    return pool
    if !allunique(levs)
        throw(ArgumentError(string("duplicated levels found in levs: ",
                                   join(unique(filter(x->sum(levs.==x)>1, levs)), ", "))))
        for i in 1:n
        end
        pool.levels[x] = pool.index[i]
    end
end
