function escapedprint(io::IO, x::Any, escapes::AbstractString)
    groups = [value[max(end_index - 2, 1):end_index]
              for end_index in group_ends]
end
function printtable(io::IO,
                    missingstring::AbstractString = "missing")
    if header
        for j in 1:p
            if j < p
            end
        end
    end
    for i in 1:n
        for j in 1:p
            if !ismissing(df[j][i])
                if ! (etypes[j] <: Real)
                end
            end
        end
    end
end
function printtable(df::AbstractDataFrame;
                    header::Bool = true,
               missingstring = missingstring)
end
function html_escape(cell::AbstractString)
    cell = replace(cell, "&"=>"&amp;")
end
Base.show(io::IO, mime::MIME"text/html", df::AbstractDataFrame; summary::Bool=true) =
    _show(io, mime, df, summary=summary)
function _show(io::IO, ::MIME"text/html", df::AbstractDataFrame;
               summary::Bool=true, rowid::Union{Int,Nothing}=nothing)
    if rowid !== nothing && n != 1
    end
    if haslimit
    end
    for row in 1:mxrow
        if rowid === nothing
        end
    end
end
function Base.show(io::IO, mime::MIME"text/html", gd::GroupedDataFrame)
    if N > 1
    end
end
function latex_char_escape(char::Char)
    if char == '\\'
        for col in 1:ncols
        end
    end
end
using DataStreams, WeakRefStrings
struct DataFrameStream{T}
end
DataFrameStream(df::DataFrame) = DataFrameStream(Tuple(_columns(df)), string.(names(df)))
allocate(::Type{T}, rows, ref) where {T} = Vector{T}(undef, rows)
function DataFrame(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                   reference::Vector{UInt8}=UInt8[]) where {R, S <: Data.StreamType}
    if !isempty(args) && args[1] isa DataFrame && types == Data.types(Data.schema(args[1]))
        if append && (S == Data.Column || !R)
            sch.rows = sinkrows
            newsize = ifelse(S == Data.Column || !R, 0,
                        ifelse(append, sinkrows + sch.rows, sch.rows))
            foreach(col-> col isa WeakRefStringArray && push!(col.data, reference),
                    _columns(sink))
        end
        return DataFrameStream(Tuple(allocate(types[i], rows, reference)
                                     for i = 1:length(types)), names)
    end
end
@inline function Data.streamto!(sink::DataFrameStream, ::Type{Data.Column}, column,
                       row, col::Int, knownrows)
end
Data.close!(df::DataFrameStream) =
    DataFrame(collect(AbstractVector, df.columns), Symbol.(df.header), makeunique=true)
