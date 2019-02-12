function compacttype(T::Type, maxwidth::Int=8)
    for (name, col) in eachcol(df, true)
        for indices in (rowindices1, rowindices2), i in indices
            if isassigned(col, i)
            end
        end
    end
end
""" """ function getchunkbounds(maxwidths::Vector{Int},
                        availablewidth::Int) # -> Vector{Int}
    if splitcols
        chunkbounds = [0, ncols]
    end
end
""" """ function showrowindices(io::IO,
                        rowid) # -> Void
    rowmaxwidth = maxwidths[end]
    for i in rowindices
        if rowid isa Nothing
        end
    end
end
""" """ function showrows(io::IO,
                  rowid=nothing) # -> Void
    ncols = size(df, 2)
    if isempty(rowindices1)
        if displaysummary
        end
    end
    if !allcols && length(chunkbounds) > 2
        for j in leftcol:rightcol
            for itr in 1:padding
            end
            if j < rightcol
            end
        end
    end
end
function _show(io::IO,
               rowid=nothing)
    showrows(io,
             rowid)
end
