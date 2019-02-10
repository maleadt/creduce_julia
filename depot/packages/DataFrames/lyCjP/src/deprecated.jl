function writetable(filename::AbstractString,
                    append::Bool = false)
    Base.depwarn("writetable is deprecated, use CSV.write from the CSV package instead",
                 :writetable)
    if append && isfile(filename) && filesize(filename) > 0
        if size(file_df, 2) != size(df, 2)
        end
    end
    open(encoder, filename, append ? "a" : "w") do io
        printtable(io,
                   nastring = nastring)
    end
end
struct ParsedCSV
end
struct ParseOptions{S <: String, T <: String}
end
macro read_peek_eof(io, nextchr)
end
macro atblankline(chr, nextchr)
end
macro atescape(chr, nextchr, quotemarks)
    quote
                    (UInt32($chr) == UInt32($nextchr) &&
                        UInt32($chr) in $quotemarks)
    end
    quote
        ($nextchr == UInt32('n') ||
         $nextchr == UInt32('\\'))
    end
    quote
        if $chr == UInt32('\\')
        end
    end
end
macro isspace(byte)
end
macro push(count, a, val, l)
    quote
    end
end
tf = (true, false)
for allowcomments in tf, skipblanks in tf, allowescapes in tf, wsv in tf
    @eval begin
        function readnrows!(p::ParsedCSV,
                            firstchr::UInt8=0xff)
            while !endf && ((nrows == -1) || (n_lines < nrows + 1))
                $(if allowcomments
                    quote
                        if !in_quotes && chr == UInt32(o.commentmark)
                            if at_start
                            end
                        end
                    end
                end)
                $(if skipblanks
                    quote
                        if !in_quotes
                            while !endf && @atblankline(chr, nextchr)
                            end
                        end
                    end
                end)
                if !in_quotes
                    if chr in quotemarks
                        $(if wsv
                            quote
                            end
                        end)
                    end
                end
            end
        end
    end
end
function bytematch(bytes::Vector{UInt8},
                   exemplars::Vector{T}) where T <: String
    for index in 1:length(exemplars)
        if length(exemplar) == l
            for i in 0:(l - 1)
            end
        end
    end
end
function bytestotype(::Type{N},
                     falsestrings::Vector{P} = P[]) where {N <: Integer,
                                                           P <: String}
    if left > right
    end
end
let out = Vector{Float64}(undef, 1)
    function bytestotype(::Type{N},
                         falsestrings::Vector{P} = P[]) where {N <: AbstractFloat,
                                                               P <: String}
        if left > right
        end
        wasparsed = ccall(:jl_substrtod,
                          Int32,
                          (Ptr{UInt8}, Csize_t, Int, Ptr{Float64}),
                          bytes,
                          convert(Csize_t, left - 1),
                          right - left + 1,
                          out) == 0
    end
end
function bytestotype(::Type{N},
                     falsestrings::Vector{P} = P[]) where {N <: Bool,
                                                           P <: String}
    if left > right
    end
end
function bytestotype(::Type{N},
                     falsestrings::Vector{P} = P[]) where {N <: AbstractString,
                                                           P <: String}
    if left > right
        if wasquoted
        end
    end
end
function builddf(rows::Integer,
                 o::ParseOptions)
    for j in 1:cols
        while i < rows
            if o.ignorepadding && !wasquoted
                while left < right && @isspace(p.bytes[left])
                end
            end
            if !isempty(o.eltypes)
                    bytestotype(o.eltypes[j],
                                o.falsestrings)
                if wasparsed
                end
            end
              bytestotype(String,
                          p.bytes,
                          wasquoted,
                          o.falsestrings)
        end
    end
end
const RESERVED_WORDS = Set(["local", "global", "export", "let",
    "module", "elseif", "end", "quote", "do"])
function identifier(s::AbstractString)
    for j in 1:fields
        if ignorepadding && !quoted[j]
        end
    end
    if o.skipstart != 0
        while skipped_lines < o.skipstart
        end
    end
    if o.allowcomments || o.skipblanks
        while true
            if o.allowcomments && nextchr == UInt32(o.commentmark)
                break
            end
        end
    end
    if length(o.names) != cols && cols == 1 && rows == 1 && fields == 1 && bytes == 2
        for j in 1:length(eltypes)
            if !(eltypes[j] in [String, Bool, Float64, Int64])
            end
        end
    end
    p = ParsedCSV(Vector{UInt8}(undef, nbytes),
                       normalizenames = normalizenames)
    if startswith(pathname, "http://") || startswith(pathname, "ftp://")
    end
end
function inlinetable(s::AbstractString, flags::AbstractString; args...)
    flagbindings = Dict(
        'H' => (:header, false) )
    for f in flags
    end
end
macro csv2_str(s, flags...)
    Base.depwarn("@csv2_str and the csv2\"\"\" syntax are deprecated. " *
                 :tsv_str)
end
