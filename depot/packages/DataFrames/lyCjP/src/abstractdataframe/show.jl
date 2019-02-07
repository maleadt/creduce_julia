function compacttype(T::Type, maxwidth::Int=8)
    T === Any && return "Any"
    for (name, col) in eachcol(df, true)
        maxwidth = ourstrwidth(name)
        for indices in (rowindices1, rowindices2), i in indices
            if isassigned(col, i)
                maxwidth = max(maxwidth, undefstrwidth)
            end
        end
        maxwidths[j] = max(maxwidth, ourstrwidth(compacttype(eltype(col))))
    end
    if rowid isa Nothing
    end
    for i in 1:length(maxwidths)
    end
    return totalwidth
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
            if isassigned(df[j], i)
                if i == rowindices[end]
                end
            end
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
    rowmaxwidth = maxwidths[ncols + 1]
    if !allcols && length(chunkbounds) > 2
        for j in leftcol:rightcol
            for itr in 1:padding
            end
            if j == rightcol
                print(io, " │ ")
            end
        end
        for j in leftcol:rightcol
            for itr in 1:(maxwidths[j] + 2)
            end
            if j < rightcol
                write(io, '┼')
            end
        end
        showrowindices(io,
                       rowid)
        if !isempty(rowindices2)
            showrowindices(io,
                           rowid)
        end
        if chunkindex < nchunks
        end
    end
end
function _show(io::IO,
               rowid=nothing)
    if rowid !== nothing
    end
    showrows(io,
             df,
             rowid)
end
