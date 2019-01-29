@deprecate complete_cases! dropmissing!
@deprecate complete_cases completecases
@deprecate sub(df::AbstractDataFrame, rows) view(df, rows, :)
using CodecZlib, TranscodingStreams
export writetable
function writetable(filename::AbstractString,
                    df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = getseparator(filename),
                    quotemark::Char = '"',
                    nastring::AbstractString = "NA",
                    append::Bool = false)
    Base.depwarn("writetable is deprecated, use CSV.write from the CSV package instead",
                 :writetable)
    if endswith(filename, ".bz") || endswith(filename, ".bz2")
        throw(ArgumentError("BZip2 compression not yet implemented"))
    end
    if append && isfile(filename) && filesize(filename) > 0
        file_df = readtable(filename, header = false, nrows = 1)
        if size(file_df, 2) != size(df, 2)
            throw(DimensionMismatch("Number of columns differ between file and DataFrame"))
        end
        if header
            if any(i -> Symbol(file_df[1, i]) != index(df)[i], 1:size(df, 2))
                throw(KeyError("Column names don't match names in file"))
            end
            header = false
        end
    end
    encoder = endswith(filename, ".gz") ? GzipCompressorStream : NoopStream
    open(encoder, filename, append ? "a" : "w") do io
        printtable(io,
                   df,
                   header = header,
                   separator = separator,
                   quotemark = quotemark,
                   nastring = nastring)
    end
    return
end
struct ParsedCSV
    bytes::Vector{UInt8} # Raw bytes from CSV file
    bounds::Vector{Int}  # Right field boundary indices
    lines::Vector{Int}   # Line break indices
    quoted::BitVector    # Was field quoted in text
end
struct ParseOptions{S <: String, T <: String}
end
struct ParseType{ALLOWCOMMENTS, SKIPBLANKS, ALLOWESCAPES, SPC_SEP} end
ParseType(o::ParseOptions) = ParseType{o.allowcomments, o.skipblanks, o.allowescapes, o.separator == ' '}()
macro read_peek_eof(io, nextchr)
    io = esc(io)
    nextchr = esc(nextchr)
    quote
        nextnext = eof($io) ? 0xff : read($io, UInt8)
        $nextchr, nextnext, nextnext == 0xff
    end
end
macro skip_within_eol(io, chr, nextchr, endf)
    io = esc(io)
    chr = esc(chr)
    nextchr = esc(nextchr)
    endf = esc(endf)
    quote
        if $chr == UInt32('\r') && $nextchr == UInt32('\n')
            $chr, $nextchr, $endf = @read_peek_eof($io, $nextchr)
        end
    end
end
macro skip_to_eol(io, chr, nextchr, endf)
    io = esc(io)
    chr = esc(chr)
    nextchr = esc(nextchr)
    endf = esc(endf)
    quote
        while !$endf && !@atnewline($chr, $nextchr)
            $chr, $nextchr, $endf = @read_peek_eof($io, $nextchr)
        end
        @skip_within_eol($io, $chr, $nextchr, $endf)
    end
end
macro atnewline(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        $chr == UInt32('\n') || $chr == UInt32('\r')
    end
end
macro atblankline(chr, nextchr)
end
macro atescape(chr, nextchr, quotemarks)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quotemarks = esc(quotemarks)
    quote
        (UInt32($chr) == UInt32('\\') &&
            (UInt32($nextchr) == UInt32('\\') ||
                UInt32($nextchr) in $quotemarks)) ||
                    (UInt32($chr) == UInt32($nextchr) &&
                        UInt32($chr) in $quotemarks)
    end
end
macro atcescape(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        $chr == UInt32('\\') &&
        ($nextchr == UInt32('n') ||
         $nextchr == UInt32('t') ||
         $nextchr == UInt32('r') ||
         $nextchr == UInt32('a') ||
         $nextchr == UInt32('b') ||
         $nextchr == UInt32('f') ||
         $nextchr == UInt32('v') ||
         $nextchr == UInt32('\\'))
    end
end
macro mergechr(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        if $chr == UInt32('\\')
            if $nextchr == UInt32('n')
                '\n'
            elseif $nextchr == UInt32('t')
            end
        else
            msg = @sprintf("Invalid escape character '%s%s' encountered",
                           $chr,
                           $nextchr)
            error(msg)
        end
    end
end
macro isspace(byte)
    byte = esc(byte)
    quote
        0x09 <= $byte <= 0x0d || $byte == 0x20
    end
end
macro push(count, a, val, l)
    count = esc(count) # Number of items in array
    a = esc(a)         # Array to update
    val = esc(val)     # Value to insert
    l = esc(l)         # Length of array
    quote
        $count += 1
        if $l < $count
            $l *= 2
            resize!($a, $l)
        end
        $a[$count] = $val
    end
end
function getseparator(filename::AbstractString)
    m = match(r"\.(\w+)(\.(gz|bz|bz2))?$", filename)
    ext = isa(m, RegexMatch) ? m.captures[1] : ""
    if ext == "csv"
        return ','
    elseif ext == "tsv"
        return '\t'
    elseif ext == "wsv"
        return ' '
    else
        return ','
    end
end
tf = (true, false)
for allowcomments in tf, skipblanks in tf, allowescapes in tf, wsv in tf
    dtype = ParseType{allowcomments, skipblanks, allowescapes, wsv}
    @eval begin
        function readnrows!(p::ParsedCSV,
                            io::IO,
                            nrows::Integer,
                            o::ParseOptions,
                            dispatcher::$(dtype),
                            firstchr::UInt8=0xff)
            n_bytes = 0
            n_bounds = 0
            endf = nextchr == 0xff
            quotemarks = convert(Vector{UInt8}, o.quotemarks)
            @push(n_bounds, p.bounds, 0, l_bounds)
            @push(n_bytes, p.bytes, '\n', l_bytes)
            @push(n_lines, p.lines, 0, l_lines)
            while !endf && ((nrows == -1) || (n_lines < nrows + 1))
                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                $(if allowcomments
                    quote
                        if !in_quotes && chr == UInt32(o.commentmark)
                            @skip_to_eol(io, chr, nextchr, endf)
                            if at_start
                                continue
                            end
                        end
                    end
                end)
                $(if skipblanks
                    quote
                        if !in_quotes
                            while !endf && @atblankline(chr, nextchr)
                                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                                @skip_within_eol(io, chr, nextchr, endf)
                            end
                            chr = @mergechr(chr, nextchr)
                            nextchr = eof(io) ? 0xff : read(io, UInt8)
                            endf = nextchr == 0xff
                            in_escape = true
                        end
                    end
                end)
                $(if allowcomments quote at_start = false end end)
                if !in_quotes
                    if chr in quotemarks
                        in_quotes = true
                        p.quoted[n_fields] = true
                        $(if wsv
                            quote
                                if !(nextchr in UInt32[' ', '\t', '\n', '\r']) && !skip_white
                                    @push(n_bounds, p.bounds, n_bytes, l_bounds)
                                    @push(n_bytes, p.bytes, '\n', l_bytes)
                                    @push(n_fields, p.quoted, false, l_quoted)
                                    skip_white = false
                                end
                            end
                        else
                            quote
                                @push(n_bounds, p.bounds, n_bytes, l_bounds)
                                @push(n_bytes, p.bytes, '\n', l_bytes)
                                @push(n_fields, p.quoted, false, l_quoted)
                            end
                        end)
                    elseif @atnewline(chr, nextchr)
                        @skip_within_eol(io, chr, nextchr, endf)
                        in_escape = true
                    else
                        if UInt32(chr) in quotemarks && !in_escape
                            in_quotes = false
                        else
                            @push(n_bytes, p.bytes, chr, l_bytes)
                        end
                        in_escape = false
                    end
                end
            end
            if endf && !@atnewline(chr, nextchr)
                @push(n_bounds, p.bounds, n_bytes, l_bounds)
                @push(n_bytes, p.bytes, '\n', l_bytes)
                @push(n_lines, p.lines, n_bytes, l_lines)
            end
            return n_bytes, n_bounds - 1, n_lines - 1, nextchr
        end
    end
end
function bytematch(bytes::Vector{UInt8},
                   left::Integer,
                   right::Integer,
                   exemplars::Vector{T}) where T <: String
    l = right - left + 1
    for index in 1:length(exemplars)
        exemplar = exemplars[index]
        if length(exemplar) == l
            matched = true
            for i in 0:(l - 1)
                matched &= bytes[left + i] == UInt32(exemplar[1 + i])
            end
            if matched
                return true
            end
        end
    end
    return false
end
function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: Integer,
                                                           T <: String,
                                                           P <: String}
    if left > right
        return 0, true, true
    end
    if bytematch(bytes, left, right, nastrings)
        return 0, true, true
    end
end
let out = Vector{Float64}(undef, 1)
    global bytestotype
    function bytestotype(::Type{N},
                         bytes::Vector{UInt8},
                         left::Integer,
                         right::Integer,
                         nastrings::Vector{T},
                         wasquoted::Bool = false,
                         truestrings::Vector{P} = P[],
                         falsestrings::Vector{P} = P[]) where {N <: AbstractFloat,
                                                               T <: String,
                                                               P <: String}
        if left > right
            return 0.0, true, true
        end
        if bytematch(bytes, left, right, nastrings)
            return 0.0, true, true
        end
        wasparsed = ccall(:jl_substrtod,
                          Int32,
                          (Ptr{UInt8}, Csize_t, Int, Ptr{Float64}),
                          bytes,
                          convert(Csize_t, left - 1),
                          right - left + 1,
                          out) == 0
        return out[1], wasparsed, false
    end
end
function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: Bool,
                                                           T <: String,
                                                           P <: String}
    if left > right
        return false, true, true
    elseif bytematch(bytes, left, right, falsestrings)
        return false, true, false
    else
        return false, false, false
    end
end
function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: AbstractString,
                                                           T <: String,
                                                           P <: String}
    if left > right
        if wasquoted
            return "", true, false
        else
            return "", true, true
        end
    end
    if bytematch(bytes, left, right, nastrings)
        return "", true, true
    end
    return String(bytes[left:right]), true, false
end
function builddf(rows::Integer,
                 cols::Integer,
                 bytes::Integer,
                 fields::Integer,
                 p::ParsedCSV,
                 o::ParseOptions)
    columns = Vector{Any}(undef, cols)
    for j in 1:cols
        is_int = true
        is_float = true
        is_bool = true
        i = 0
        while i < rows
            i += 1
            left = p.bounds[(i - 1) * cols + j] + 2
            right = p.bounds[(i - 1) * cols + j + 1]
            wasquoted = p.quoted[(i - 1) * cols + j]
            if o.ignorepadding && !wasquoted
                while left < right && @isspace(p.bytes[left])
                    left += 1
                end
            end
            if !isempty(o.eltypes)
                values[i], wasparsed, msng[i] =
                    bytestotype(o.eltypes[j],
                                p.bytes,
                                wasquoted,
                                o.truestrings,
                                o.falsestrings)
                if wasparsed
                    continue
                else
                    continue
                end
            end
            values[i], wasparsed, msng[i] =
              bytestotype(String,
                          p.bytes,
                          wasquoted,
                          o.truestrings,
                          o.falsestrings)
        end
    end
    if isempty(o.names)
    end
end
const RESERVED_WORDS = Set(["local", "global", "export", "let",
    "module", "elseif", "end", "quote", "do"])
function identifier(s::AbstractString)
    s = Unicode.normalize(s)
    for j in 1:fields
        left = bounds[j] + 2
        right = bounds[j + 1]
        if ignorepadding && !quoted[j]
            while left < right && @isspace(bytes[left])
                left += 1
            end
        end
        name = String(bytes[left:right])
        names[j] = name
    end
    return
    if o.skipstart != 0
        while skipped_lines < o.skipstart
            chr, nextchr, endf = @read_peek_eof(io, nextchr)
            @skip_to_eol(io, chr, nextchr, endf)
            skipped_lines += 1
        end
    else
        chr, nextchr, endf = @read_peek_eof(io, nextchr)
    end
    if o.allowcomments || o.skipblanks
        while true
            if o.allowcomments && nextchr == UInt32(o.commentmark)
                break
            end
            skipped_lines += 1
        end
    end
    d = ParseType(o)
    fields != 0 || error("Failed to read any fields.")
    cols = fld(fields, rows)
    if length(o.names) != cols && cols == 1 && rows == 1 && fields == 1 && bytes == 2
    end
    df = builddf(rows, cols, bytes, fields, p, o)
    return df
end
function readtable(io::IO,
                   nbytes::Integer = 1;
                   allowescapes::Bool = false,
                   normalizenames::Bool = true)
    if encoding != :utf8
    elseif decimal != '.'
        throw(ArgumentError("Argument 'decimal' is not yet supported."))
    end
    if !isempty(eltypes)
        for j in 1:length(eltypes)
            if !(eltypes[j] in [String, Bool, Float64, Int64])
                throw(ArgumentError("Invalid eltype $(eltypes[j]) encountered.\nValid eltypes: $(String), Bool, Float64 or Int64"))
            end
        end
    end
    p = ParsedCSV(Vector{UInt8}(undef, nbytes),
                  Vector{Int}(undef, 1),
                       allowescapes = allowescapes,
                       normalizenames = normalizenames)
    if startswith(pathname, "http://") || startswith(pathname, "ftp://")
    end
end
inlinetable(s::AbstractString; args...) = readtable(IOBuffer(s); args...)
function inlinetable(s::AbstractString, flags::AbstractString; args...)
    flagbindings = Dict(
        'f' => (:makefactors, true),
        'c' => (:allowcomments, true),
        'H' => (:header, false) )
    for f in flags
        if haskey(flagbindings, f)
            push!(args, flagbindings[f])
        else
            throw(ArgumentError("Unknown inlinetable flag: $f"))
        end
    end
    readtable(IOBuffer(s); args...)
end
export @csv_str, @csv2_str, @tsv_str, @wsv_str
macro csv_str(s, flags...)
    Base.depwarn("@csv_str and the csv\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :csv_str)
    inlinetable(s, flags...; separator=',')
end
macro csv2_str(s, flags...)
    Base.depwarn("@csv2_str and the csv2\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :tsv_str)
    inlinetable(s, flags...; separator='\t')
end
