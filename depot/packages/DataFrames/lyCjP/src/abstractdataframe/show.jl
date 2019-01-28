Base.summary(df::AbstractDataFrame) = # -> String
    @sprintf("%d×%d %s", size(df)..., typeof(df).name)
let
    local io = IOBuffer(Vector{UInt8}(undef, 80), read=true, write=true)
    global ourstrwidth
    """
        DataFrames.ourstrwidth(x::Any)
    Determine the number of characters that would be used to print a value.
    """
    function ourstrwidth(x::Any) # -> Int
        truncate(io, 0)
        ourshowcompact(io, x)
        textwidth(String(take!(io)))
    end
end
""" """ ourshowcompact(io::IO, x::Any) =
    show(IOContext(io, :compact=>true, :typeinfo=>typeof(x)), x) # -> Void
ourshowcompact(io::IO, x::AbstractString) = escape_string(io, x, "") # -> Void
ourshowcompact(io::IO, x::Symbol) = ourshowcompact(io, string(x)) # -> Void
ourshowcompact(io::IO, x::Nothing) = nothing
"""Return compact string representation of type T"""
function compacttype(T::Type, maxwidth::Int=8)
    T === Any && return "Any"
    T === Missing && return "Missing"
    sT = string(T)
    length(sT) ≤ maxwidth && return sT
    if T >: Missing
        T = Base.nonmissingtype(T)
        sT = string(T)
        suffix = "⍰"
        length(sT) ≤ 8 && return sT * suffix
    else
        suffix = ""
    end
    T <: Union{CategoricalString, CategoricalValue} && return "Categorical…"*suffix
    match(Regex("^.\\w{0,$(7-length(suffix))}"), sT).match * "…"*suffix
end
""" """ function getmaxwidths(df::AbstractDataFrame,
                      rowindices1::AbstractVector{Int},
                      rowindices2::AbstractVector{Int},
                      rowlabel::Symbol,
                      rowid=nothing) # -> Vector{Int}
    maxwidths = Vector{Int}(undef, size(df, 2) + 1)
    undefstrwidth = ourstrwidth(Base.undef_ref_str)
    j = 1
    for (name, col) in eachcol(df, true)
        maxwidth = ourstrwidth(name)
        for indices in (rowindices1, rowindices2), i in indices
            if isassigned(col, i)
                maxwidth = max(maxwidth, ourstrwidth(col[i]))
            else
                maxwidth = max(maxwidth, undefstrwidth)
            end
        end
        maxwidths[j] = max(maxwidth, ourstrwidth(compacttype(eltype(col))))
        j += 1
    end
    if rowid isa Nothing
        rowmaxwidth1 = isempty(rowindices1) ? 0 : ndigits(maximum(rowindices1))
        rowmaxwidth2 = isempty(rowindices2) ? 0 : ndigits(maximum(rowindices2))
        maxwidths[j] = max(max(rowmaxwidth1, rowmaxwidth2), ourstrwidth(rowlabel))
    else
        maxwidths[j] = max(ndigits(rowid), ourstrwidth(rowlabel))
    end
    return maxwidths
end
""" """ function getprintedwidth(maxwidths::Vector{Int}) # -> Int
    totalwidth = 1
    for i in 1:length(maxwidths)
        totalwidth += maxwidths[i] + 3
    end
    return totalwidth
end
""" """ function getchunkbounds(maxwidths::Vector{Int},
                        splitcols::Bool,
                        availablewidth::Int) # -> Vector{Int}
    ncols = length(maxwidths) - 1
    rowmaxwidth = maxwidths[ncols + 1]
    if splitcols
        chunkbounds = [0]
        totalwidth = rowmaxwidth + 4
        for j in 1:ncols
            totalwidth += maxwidths[j] + 3
            if totalwidth > availablewidth
                push!(chunkbounds, j - 1)
                totalwidth = rowmaxwidth + 4 + maxwidths[j] + 3
            end
        end
        push!(chunkbounds, ncols)
    else
        chunkbounds = [0, ncols]
    end
    return chunkbounds
end
""" """ function showrowindices(io::IO,
                        df::AbstractDataFrame,
                        rowindices::AbstractVector{Int},
                        maxwidths::Vector{Int},
                        leftcol::Int,
                        rightcol::Int,
                        rowid) # -> Void
    rowmaxwidth = maxwidths[end]
    for i in rowindices
        if rowid isa Nothing
            @printf io "│ %d" i
        else
            @printf io "│ %d" rowid
        end
        padding = rowmaxwidth - ndigits(rowid isa Nothing ? i : rowid)
        for _ in 1:padding
            write(io, ' ')
        end
        print(io, " │ ")
        for j in leftcol:rightcol
            strlen = 0
            if isassigned(df[j], i)
                s = df[i, j]
                strlen = ourstrwidth(s)
                if ismissing(s)
                    printstyled(io, s, color=:light_black)
                elseif s === nothing
                    strlen = 0
                else
                    ourshowcompact(io, s)
                end
            else
                strlen = ourstrwidth(Base.undef_ref_str)
                ourshowcompact(io, Base.undef_ref_str)
            end
            padding = maxwidths[j] - strlen
            for _ in 1:padding
                write(io, ' ')
            end
            if j == rightcol
                if i == rowindices[end]
                    print(io, " │")
                else
                    print(io, " │\n")
                end
            else
                print(io, " │ ")
            end
        end
    end
    return
end
""" """ function showrows(io::IO,
                  df::AbstractDataFrame,
                  rowindices1::AbstractVector{Int},
                  rowindices2::AbstractVector{Int},
                  maxwidths::Vector{Int},
                  splitcols::Bool = false,
                  allcols::Bool = false,
                  rowlabel::Symbol = :Row,
                  displaysummary::Bool = true,
                  rowid=nothing) # -> Void
    ncols = size(df, 2)
    if isempty(rowindices1)
        if displaysummary
            println(io, summary(df))
        end
        return
    end
    rowmaxwidth = maxwidths[ncols + 1]
    chunkbounds = getchunkbounds(maxwidths, splitcols, displaysize(io)[2])
    nchunks = allcols ? length(chunkbounds) - 1 : min(length(chunkbounds) - 1, 1)
    header = displaysummary ? summary(df) : ""
    if !allcols && length(chunkbounds) > 2
        header *= ". Omitted printing of $(chunkbounds[end] - chunkbounds[2]) columns"
    end
    println(io, header)
    for chunkindex in 1:nchunks
        leftcol = chunkbounds[chunkindex] + 1
        rightcol = chunkbounds[chunkindex + 1]
        @printf io "│ %s" rowlabel
        padding = rowmaxwidth - ourstrwidth(rowlabel)
        for itr in 1:padding
            write(io, ' ')
        end
        print(io, " │ ")
        for j in leftcol:rightcol
            s = _names(df)[j]
            ourshowcompact(io, s)
            padding = maxwidths[j] - ourstrwidth(s)
            for itr in 1:padding
                write(io, ' ')
            end
            if j == rightcol
                print(io, " │\n")
            else
                print(io, " │ ")
            end
        end
        print(io, "│ ")
        padding = rowmaxwidth
        for itr in 1:padding
            write(io, ' ')
        end
        print(io, " │ ")
        for j in leftcol:rightcol
            s = compacttype(eltype(df[j]), maxwidths[j])
            printstyled(io, s, color=:light_black)
            padding = maxwidths[j] - ourstrwidth(s)
            for itr in 1:padding
                write(io, ' ')
            end
            if j == rightcol
                print(io, " │\n")
            else
                print(io, " │ ")
            end
        end
        write(io, '├')
        for itr in 1:(rowmaxwidth + 2)
            write(io, '─')
        end
        write(io, '┼')
        for j in leftcol:rightcol
            for itr in 1:(maxwidths[j] + 2)
                write(io, '─')
            end
            if j < rightcol
                write(io, '┼')
            else
                write(io, '┤')
            end
        end
        write(io, '\n')
        showrowindices(io,
                       df,
                       rowindices1,
                       maxwidths,
                       leftcol,
                       rightcol,
                       rowid)
        if !isempty(rowindices2)
            print(io, "\n⋮\n")
            showrowindices(io,
                           df,
                           rowindices2,
                           maxwidths,
                           leftcol,
                           rightcol,
                           rowid)
        end
        if chunkindex < nchunks
            print(io, "\n\n")
        end
    end
    return
end
function _show(io::IO,
               df::AbstractDataFrame;
               allrows::Bool = !get(io, :limit, false),
               allcols::Bool = !get(io, :limit, false),
               splitcols = get(io, :limit, false),
               rowlabel::Symbol = :Row,
               summary::Bool = true,
               rowid=nothing)
    nrows = size(df, 1)
    if rowid !== nothing
        nrows == 1 || throw(ArgumentError("rowid may be passed only with a single row data frame"))
    end
    dsize = displaysize(io)
    availableheight = dsize[1] - 7
    nrowssubset = fld(availableheight, 2)
    bound = min(nrowssubset - 1, nrows)
    if allrows || nrows <= availableheight
        rowindices1 = 1:nrows
        rowindices2 = 1:0
    else
        rowindices1 = 1:bound
        rowindices2 = max(bound + 1, nrows - nrowssubset + 1):nrows
    end
    maxwidths = getmaxwidths(df, rowindices1, rowindices2, rowlabel, rowid)
    width = getprintedwidth(maxwidths)
    showrows(io,
             df,
             rowindices1,
             rowindices2,
             maxwidths,
             splitcols,
             allcols,
             rowlabel,
             summary,
             rowid)
    return
end
""" """ Base.show(io::IO,
          df::AbstractDataFrame;
          allrows::Bool = !get(io, :limit, false),
          allcols::Bool = !get(io, :limit, false),
          splitcols = get(io, :limit, false),
          rowlabel::Symbol = :Row,
          summary::Bool = true) =
    _show(io, df, allrows=allrows, allcols=allcols, splitcols=splitcols,
          rowlabel=rowlabel, summary=summary)
Base.show(df::AbstractDataFrame;
          allrows::Bool = !get(stdout, :limit, true),
          allcols::Bool = !get(stdout, :limit, true),
          splitcols = get(stdout, :limit, true),
          rowlabel::Symbol = :Row,
          summary::Bool = true) =
    show(stdout, df,
         allrows=allrows, allcols=allcols, splitcols=splitcols,
         rowlabel=rowlabel, summary=summary)
