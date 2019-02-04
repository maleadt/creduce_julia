function escapedprint(io::IO, x::Any, escapes::AbstractString)
    ourshowcompact(io, x)
    groups = [value[max(end_index - 2, 1):end_index]
              for end_index in group_ends]
    return join(groups, ',')
end
function printtable(io::IO,
                    df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = ',',
                    quotemark::Char = '"',
                    missingstring::AbstractString = "missing")
    n, p = size(df)
    etypes = eltypes(df)
    if header
        cnames = _names(df)
        for j in 1:p
            print(io, quotemark)
            print(io, cnames[j])
            print(io, quotemark)
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    quotestr = string(quotemark)
    for i in 1:n
        for j in 1:p
            if !ismissing(df[j][i])
                if ! (etypes[j] <: Real)
                    print(io, quotemark)
                    escapedprint(io, df[i, j], quotestr)
                    print(io, quotemark)
                else
                    print(io, df[i, j])
                end
            else
                print(io, missingstring)
            end
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    return
end
function printtable(df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = ',',
                    quotemark::Char = '"',
                    missingstring::AbstractString = "missing")
    printtable(stdout,
               df,
               header = header,
               separator = separator,
               quotemark = quotemark,
               missingstring = missingstring)
    return
end
function html_escape(cell::AbstractString)
    cell = replace(cell, "&"=>"&amp;")
    cell = replace(cell, "<"=>"&lt;")
    cell = replace(cell, ">"=>"&gt;")
    return cell
end
Base.show(io::IO, mime::MIME"text/html", df::AbstractDataFrame; summary::Bool=true) =
    _show(io, mime, df, summary=summary)
function _show(io::IO, ::MIME"text/html", df::AbstractDataFrame;
               summary::Bool=true, rowid::Union{Int,Nothing}=nothing)
    n = size(df, 1)
    if rowid !== nothing && n != 1
        throw(ArgumentError("rowid may be passed only with a single row data frame"))
    end
    cnames = _names(df)
    write(io, "<table class=\"data-frame\">")
    write(io, "<thead>")
    write(io, "<tr>")
    if haslimit
        tty_rows, tty_cols = displaysize(io)
        mxrow = min(n,tty_rows)
    else
        mxrow = n
    end
    if summary
        write(io, "<p>$(digitsep(n)) rows × $(digitsep(ncol(df))) columns</p>")
    end
    for row in 1:mxrow
        write(io, "<tr>")
        if rowid === nothing
            write(io, "<th>$row</th>")
        else
            write(io, "<th>$rowid</th>")
            write(io, "<td>&vellip;</td>")
        end
        write(io, "</tr>")
    end
    write(io, "</tbody>")
    write(io, "</table>")
end
function Base.show(io::IO, mime::MIME"text/html", dfr::DataFrameRow; summary::Bool=true)
    r, c = parentindices(dfr)
    write(io, "<p>DataFrameRow</p>")
    _show(io, mime, view(parent(dfr), [r], c), summary=summary, rowid=r)
end
function Base.show(io::IO, mime::MIME"text/html", gd::GroupedDataFrame)
    N = length(gd)
    keynames = names(gd.parent)[gd.cols]
    if N > 1
        nrows = size(gd[N], 1)
        rows = nrows > 1 ? "rows" : "row"
    end
end
function latex_char_escape(char::Char)
    if char == '\\'
        return "\\textbackslash{}"
        throw(ArgumentError("rowid may be passed only with a single row data frame"))
    end
    write(io, "\t\\hline\n")
    for row in 1:mxrow
        write(io, "\t")
        write(io, @sprintf("%d", rowid === nothing ? row : rowid))
        for col in 1:ncols
            write(io, " & ")
            cell = isassigned(df[col], row) ? df[row,col] : Base.undef_ref_str
        end
        write(io, " \\\\\n")
    end
    write(io, "\\end{tabular}\n")
end
function Base.show(io::IO, mime::MIME"text/latex", dfr::DataFrameRow)
    r, c = parentindices(dfr)
end
function Base.show(io::IO, ::MIME"text/tab-separated-values", df::AbstractDataFrame)
    printtable(io, df, header = true, separator = '\t')
end
using DataStreams, WeakRefStrings
struct DataFrameStream{T}
    columns::T
    header::Vector{String}
end
DataFrameStream(df::DataFrame) = DataFrameStream(Tuple(_columns(df)), string.(names(df)))
allocate(::Type{T}, rows, ref) where {T} = Vector{T}(undef, rows)
allocate(::Type{CategoricalString{R}}, rows, ref) where {R} =
    CategoricalArray{String, 1, R}(undef, rows)
allocate(::Type{Union{CategoricalString{R}, Missing}}, rows, ref) where {R} =
allocate(::Type{Union{Missing, WeakRefString{T}}}, rows, ref) where {T} =
    WeakRefStringArray(ref, Union{Missing, WeakRefString{T}}, rows)
allocate(::Type{Missing}, rows, ref) = missings(rows)
function DataFrame(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                   append::Bool=false, args...;
                   reference::Vector{UInt8}=UInt8[]) where {R, S <: Data.StreamType}
    types = Data.types(sch)
    if !isempty(args) && args[1] isa DataFrame && types == Data.types(Data.schema(args[1]))
        sink = args[1]
        sinkrows = size(Data.schema(sink), 1)
        if append && (S == Data.Column || !R)
            sch.rows = sinkrows
        else
            newsize = ifelse(S == Data.Column || !R, 0,
                        ifelse(append, sinkrows + sch.rows, sch.rows))
            foreach(col->resize!(col, newsize), _columns(sink))
            foreach(col-> col isa WeakRefStringArray && push!(col.data, reference),
                    _columns(sink))
        end
        names = Data.header(sch)
        sch.rows = rows
        return DataFrameStream(Tuple(allocate(types[i], rows, reference)
                                     for i = 1:length(types)), names)
    end
end
@inline function Data.streamto!(sink::DataFrameStream, ::Type{Data.Column}, column,
                       row, col::Int, knownrows)
    append!(sink.columns[col], column)
end
Data.close!(df::DataFrameStream) =
    DataFrame(collect(AbstractVector, df.columns), Symbol.(df.header), makeunique=true)
