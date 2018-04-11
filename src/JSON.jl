__precompile__()
module JSON
using Compat
export json
export JSONText
module Common
using Compat
using Unicode
const BACKSPACE      = UInt8('\b')
const TAB            = UInt8('\t')
const NEWLINE        = UInt8('\n')
const FORM_FEED      = UInt8('\f')
const RETURN         = UInt8('\r')
const SPACE          = UInt8(' ')
const STRING_DELIM   = UInt8('"')
const PLUS_SIGN      = UInt8('+')
const DELIMITER      = UInt8(',')
const MINUS_SIGN     = UInt8('-')
const DECIMAL_POINT  = UInt8('.')
const SOLIDUS        = UInt8('/')
const DIGIT_ZERO     = UInt8('0')
const DIGIT_NINE     = UInt8('9')
const SEPARATOR      = UInt8(':')
const LATIN_UPPER_A  = UInt8('A')
const LATIN_UPPER_E  = UInt8('E')
const LATIN_UPPER_F  = UInt8('F')
const ARRAY_BEGIN    = UInt8('[')
const BACKSLASH      = UInt8('\\')
const ARRAY_END      = UInt8(']')
const LATIN_A        = UInt8('a')
const LATIN_B        = UInt8('b')
const LATIN_E        = UInt8('e')
const LATIN_F        = UInt8('f')
const LATIN_L        = UInt8('l')
const LATIN_N        = UInt8('n')
const LATIN_R        = UInt8('r')
const LATIN_S        = UInt8('s')
const LATIN_T        = UInt8('t')
const LATIN_U        = UInt8('u')
const OBJECT_BEGIN   = UInt8('{')
const OBJECT_END     = UInt8('}')
const ESCAPES = Dict(
    STRING_DELIM => STRING_DELIM,
    BACKSLASH    => BACKSLASH,
    SOLIDUS      => SOLIDUS,
    LATIN_B      => BACKSPACE,
    LATIN_F      => FORM_FEED,
    LATIN_N      => NEWLINE,
    LATIN_R      => RETURN,
    LATIN_T      => TAB)
const REVERSE_ESCAPES = Dict(reverse(p) for p in ESCAPES)
const ESCAPED_ARRAY = Vector{Vector{UInt8}}(undef, 256)
for c in 0x00:0xFF
    ESCAPED_ARRAY[c + 1] = if c == SOLIDUS
        [SOLIDUS]
    elseif c ≥ 0x80
        [c]
    elseif haskey(REVERSE_ESCAPES, c)
        [BACKSLASH, REVERSE_ESCAPES[c]]
    elseif iscntrl(Char(c)) || !isprint(Char(c))
            UInt8[BACKSLASH, LATIN_U, string(c, base=16, pad=4)...]
    else
        [c]
    end
end
export BACKSPACE, TAB, NEWLINE, FORM_FEED, RETURN, SPACE, STRING_DELIM,
       PLUS_SIGN, DELIMITER, MINUS_SIGN, DECIMAL_POINT, SOLIDUS, DIGIT_ZERO,
       DIGIT_NINE, SEPARATOR, LATIN_UPPER_A, LATIN_UPPER_E, LATIN_UPPER_F,
       ARRAY_BEGIN, BACKSLASH, ARRAY_END, LATIN_A, LATIN_B, LATIN_E, LATIN_F,
       LATIN_L, LATIN_N, LATIN_R, LATIN_S, LATIN_T, LATIN_U, OBJECT_BEGIN,
       OBJECT_END, ESCAPES, REVERSE_ESCAPES, ESCAPED_ARRAY
const E_EXPECTED_EOF    = "Expected end of input"
const E_UNEXPECTED_EOF  = "Unexpected end of input"
const E_UNEXPECTED_CHAR = "Unexpected character"
const E_BAD_KEY         = "Invalid object key"
const E_BAD_ESCAPE      = "Invalid escape sequence"
const E_BAD_CONTROL     = "ASCII control character in string"
const E_LEADING_ZERO    = "Invalid leading zero in number"
const E_BAD_NUMBER      = "Invalid number"
export E_EXPECTED_EOF, E_UNEXPECTED_EOF, E_UNEXPECTED_CHAR, E_BAD_KEY,
       E_BAD_ESCAPE, E_BAD_CONTROL, E_LEADING_ZERO, E_BAD_NUMBER
end
module Parser
using Compat
using Compat.Mmap
using ..Common
isjsonspace(b::UInt8) = b == SPACE || b == TAB || b == NEWLINE || b == RETURN
isjsondigit(b::UInt8) = DIGIT_ZERO ≤ b ≤ DIGIT_NINE
abstract type ParserState end
mutable struct MemoryParserState <: ParserState
    utf8::String
    s::Int
end
Base.@propagate_inbounds Base.getindex(state::MemoryParserState, i::Int) = codeunit(state.utf8, i)
Base.length(state::MemoryParserState) = sizeof(state.utf8)
Base.unsafe_convert(::Type{Ptr{UInt8}}, state::MemoryParserState) = unsafe_convert(Ptr{UInt8}, state.utf8)
mutable struct StreamingParserState{T <: IO} <: ParserState
    io::T
    cur::UInt8
    used::Bool
end
StreamingParserState(io::IO) = StreamingParserState(io, 0x00, true)
struct ParserContext{DictType, IntType} end
@inline function byteat(ps::MemoryParserState)
    @inbounds if hasmore(ps)
        return ps[ps.s]
    else
        _error(E_UNEXPECTED_EOF, ps)
    end
end
@inline function byteat(ps::StreamingParserState)
    if ps.used
        ps.used = false
        if eof(ps.io)
            _error(E_UNEXPECTED_EOF, ps)
        else
            ps.cur = read(ps.io, UInt8)
        end
    end
    ps.cur
end
@inline current(ps::MemoryParserState) = ps[ps.s]
@inline current(ps::StreamingParserState) = byteat(ps)
@inline function skip!(ps::ParserState, c::UInt8)
    if byteat(ps) == c
        incr!(ps)
    else
        _error("Expected '$(Char(c))' here", ps)
    end
end
function skip!(ps::ParserState, cs::UInt8...)
    for c in cs
        skip!(ps, c)
    end
end
@inline incr!(ps::MemoryParserState) = (ps.s += 1)
@inline incr!(ps::StreamingParserState) = (ps.used = true)
@inline advance!(ps::ParserState) = (b = byteat(ps); incr!(ps); b)
@inline hasmore(ps::MemoryParserState) = ps.s ≤ length(ps)
@inline hasmore(ps::StreamingParserState) = true
@inline function chomp_space!(ps::ParserState)
    @inbounds while hasmore(ps) && isjsonspace(current(ps))
        incr!(ps)
    end
end
function _count_before(haystack::AbstractString, needle::Char, _end::Int)
    count = 0
    for (i,c) in enumerate(haystack)
        i >= _end && return count
        count += c == needle
    end
    return count
end
function _error(message::AbstractString, ps::MemoryParserState)
    orig = ps.utf8
    lines = _count_before(orig, '\n', ps.s)
    strnl = replace(orig, r"[\b\f\n\r\t\s]" => " ")
    li = (ps.s > 20) ? ps.s - 9 : 1
    ri = min(lastindex(orig), ps.s + 20)
    error(message *
      "\nLine: " * string(lines) *
      "\nAround: ..." * strnl[li:ri] * "..." *
      "\n           " * (" " ^ (ps.s - li)) * "^\n"
    )
end
function _error(message::AbstractString, ps::StreamingParserState)
    error("$message\n ...when parsing byte with value '$(current(ps))'")
end
function parse_value(pc::ParserContext, ps::ParserState)
    chomp_space!(ps)
    @inbounds byte = byteat(ps)
    if byte == STRING_DELIM
        parse_string(ps)
    elseif isjsondigit(byte) || byte == MINUS_SIGN
        parse_number(pc, ps)
    elseif byte == OBJECT_BEGIN
        parse_object(pc, ps)
    elseif byte == ARRAY_BEGIN
        parse_array(pc, ps)
    else
        parse_jsconstant(ps::ParserState)
    end
end
function parse_jsconstant(ps::ParserState)
    c = advance!(ps)
    if c == LATIN_T
        skip!(ps, LATIN_R, LATIN_U, LATIN_E)
        true
    elseif c == LATIN_F
        skip!(ps, LATIN_A, LATIN_L, LATIN_S, LATIN_E)
        false
    elseif c == LATIN_N
        skip!(ps, LATIN_U, LATIN_L, LATIN_L)
        nothing
    else
        _error(E_UNEXPECTED_CHAR, ps)
    end
end
function parse_array(pc::ParserContext, ps::ParserState)
    result = Any[]
    @inbounds incr!(ps)
    chomp_space!(ps)
    if byteat(ps) ≠ ARRAY_END
        @inbounds while true
            push!(result, parse_value(pc, ps))
            chomp_space!(ps)
            byteat(ps) == ARRAY_END && break
            skip!(ps, DELIMITER)
        end
    end
    @inbounds incr!(ps)
    result
end
function parse_object(pc::ParserContext{DictType, <:Real}, ps::ParserState) where DictType
    obj = DictType()
    keyT = keytype(DictType)
    incr!(ps)
    chomp_space!(ps)
    if byteat(ps) ≠ OBJECT_END
        @inbounds while true
            chomp_space!(ps)
            byteat(ps) == STRING_DELIM || _error(E_BAD_KEY, ps)
            key = parse_string(ps)
            chomp_space!(ps)
            skip!(ps, SEPARATOR)
            value = parse_value(pc, ps)
            chomp_space!(ps)
            obj[keyT === Symbol ? Symbol(key) : convert(keyT, key)] = value
            byteat(ps) == OBJECT_END && break
            skip!(ps, DELIMITER)
        end
    end
    incr!(ps)
    obj
end
utf16_is_surrogate(c::UInt16) = (c & 0xf800) == 0xd800
utf16_get_supplementary(lead::UInt16, trail::UInt16) = Char(UInt32(lead-0xd7f7)<<10 + trail)
function read_four_hex_digits!(ps::ParserState)
    local n::UInt16 = 0
    for _ in 1:4
        b = advance!(ps)
        n = n << 4 + if isjsondigit(b)
            b - DIGIT_ZERO
        elseif LATIN_A ≤ b ≤ LATIN_F
            b - (LATIN_A - UInt8(10))
        elseif LATIN_UPPER_A ≤ b ≤ LATIN_UPPER_F
            b - (LATIN_UPPER_A - UInt8(10))
        else
            _error(E_BAD_ESCAPE, ps)
        end
    end
    n
end
function read_unicode_escape!(ps)
    u1 = read_four_hex_digits!(ps)
    if utf16_is_surrogate(u1)
        skip!(ps, BACKSLASH)
        skip!(ps, LATIN_U)
        u2 = read_four_hex_digits!(ps)
        utf16_get_supplementary(u1, u2)
    else
        Char(u1)
    end
end
function parse_string(ps::ParserState)
    b = IOBuffer()
    incr!(ps)
    while true
        c = advance!(ps)
        if c == BACKSLASH
            c = advance!(ps)
            if c == LATIN_U
                write(b, read_unicode_escape!(ps))
            else
                c = get(ESCAPES, c, 0x00)
                c == 0x00 && _error(E_BAD_ESCAPE, ps)
                write(b, c)
            end
            continue
        elseif c < SPACE
            _error(E_BAD_CONTROL, ps)
        elseif c == STRING_DELIM
            return String(take!(b))
        end
        write(b, c)
    end
end
function hasleadingzero(bytes, from::Int, to::Int)
    c = bytes[from]
    from + 1 < to && c == UInt8('-') &&
            bytes[from + 1] == DIGIT_ZERO && isjsondigit(bytes[from + 2]) ||
    from < to && to > from + 1 && c == DIGIT_ZERO &&
            isjsondigit(bytes[from + 1])
end
function int_from_bytes(pc::ParserContext{<:AbstractDict,IntType},
                        ps::ParserState,
                        bytes,
                        from::Int,
                        to::Int) where IntType <: Real
    @inbounds isnegative = bytes[from] == MINUS_SIGN ? (from += 1; true) : false
    num = IntType(0)
    @inbounds for i in from:to
        num = IntType(10) * num + IntType(bytes[i] - DIGIT_ZERO)
    end
    ifelse(isnegative, -num, num)
end
function number_from_bytes(pc::ParserContext,
                           ps::ParserState,
                           isint::Bool,
                           bytes,
                           from::Int,
                           to::Int)
    @inbounds if hasleadingzero(bytes, from, to)
        _error(E_LEADING_ZERO, ps)
    end
    if isint
        @inbounds if to == from && bytes[from] == MINUS_SIGN
            _error(E_BAD_NUMBER, ps)
        end
        int_from_bytes(pc, ps, bytes, from, to)
    else
        res = float_from_bytes(bytes, from, to)
        isnull(res) ? _error(E_BAD_NUMBER, ps) : get(res)
    end
end
function parse_number(pc::ParserContext, ps::ParserState)
    number = UInt8[]
    isint = true
    @inbounds while hasmore(ps)
        c = current(ps)
        if isjsondigit(c) || c == MINUS_SIGN
            push!(number, UInt8(c))
        elseif c in (PLUS_SIGN, LATIN_E, LATIN_UPPER_E, DECIMAL_POINT)
            push!(number, UInt8(c))
            isint = false
        else
            break
        end
        incr!(ps)
    end
    number_from_bytes(pc, ps, isint, number, 1, length(number))
end
function unparameterize_type(T::Type)
    candidate = typeintersect(T, AbstractDict{String, Any})
    candidate <: Union{} ? T : candidate
end
function parse(str::AbstractString;
               dicttype::Type{<:AbstractDict}=Dict{String,Any},
               inttype::Type{<:Real}=Int64)
    pc = ParserContext{unparameterize_type(dicttype), inttype}()
    ps = MemoryParserState(str, 1)
    v = parse_value(pc, ps)
    chomp_space!(ps)
    if hasmore(ps)
        _error(E_EXPECTED_EOF, ps)
    end
    v
end
function parse(io::IO;
               dicttype::Type{<:AbstractDict}=Dict{String,Any},
               inttype::Type{<:Real}=Int64)
    pc = ParserContext{unparameterize_type(dicttype), inttype}()
    ps = StreamingParserState(io)
    parse_value(pc, ps)
end
function parsefile(filename::AbstractString;
                   dicttype::Type{<:AbstractDict}=Dict{String, Any},
                   inttype::Type{<:Real}=Int64,
                   use_mmap=true)
    sz = filesize(filename)
    open(filename) do io
        s = use_mmap ? String(Mmap.mmap(io, Vector{UInt8}, sz)) : read(io, String)
        parse(s; dicttype=dicttype, inttype=inttype)
    end
end
function maxsize_buffer(maxsize::Int)
        IOBuffer(maxsize=maxsize)
end
function parse_string(ps::MemoryParserState)
    fastpath, len = predict_string(ps)
    if fastpath
        s = ps.s
        ps.s = s + len + 2
        return unsafe_string(pointer(ps.utf8)+s, len)
    else
        String(take!(parse_string(ps, maxsize_buffer(len))))
    end
end
function predict_string(ps::MemoryParserState)
    e = length(ps)
    fastpath = true
    len = 0
    s = ps.s + 1
    @inbounds while s <= e
        c = ps[s]
        if c == BACKSLASH
            fastpath = false
            (s += 1) > e && break
            if ps[s] == LATIN_U
                t = ps.s
                ps.s = s + 1
                len += write(devnull, read_unicode_escape!(ps))
                s = ps.s
                ps.s = t
                continue
            end
        elseif c == STRING_DELIM
            return fastpath, len
        elseif c < SPACE
            ps.s = s
            _error(E_BAD_CONTROL, ps)
        end
        len += 1
        s += 1
    end
    ps.s = s
    _error(E_UNEXPECTED_EOF, ps)
end
function parse_string(ps::MemoryParserState, b::IOBuffer)
    s = ps.s
    e = length(ps)
    s += 1
    len = b.maxsize
    @inbounds while b.size < len
        c = ps[s]
        if c == BACKSLASH
            s += 1
            s > e && break
            c = ps[s]
            if c == LATIN_U
                ps.s = s + 1
                write(b, read_unicode_escape!(ps))
                s = ps.s
                continue
            else
                c = get(ESCAPES, c, 0x00)
                if c == 0x00
                    ps.s = s
                    _error(E_BAD_ESCAPE, ps)
                end
            end
        end
        write(b, c)
        s += 1
    end
    ps.s = s + 1
    b
end
function parse_number(ps::MemoryParserState)
    s = p = ps.s
    e = length(ps)
    isint = true
    while p ≤ e
        @inbounds c = ps[p]
        if isjsondigit(c) || MINUS_SIGN == c
        elseif PLUS_SIGN == c || LATIN_E == c || LATIN_UPPER_E == c ||
                DECIMAL_POINT == c
            isint = false
        else
            break
        end
        p += 1
    end
    ps.s = p
    number_from_bytes(ps, isint, ps, s, p - 1)
end
end
module Serializations
using ..Common
abstract type Serialization end
abstract type CommonSerialization <: Serialization end
struct StandardSerialization <: CommonSerialization end
end
module Writer
using Compat
using Compat.Dates
using ..Common
using ..Serializations: Serialization, StandardSerialization,
                        CommonSerialization
using Unicode
struct CompositeTypeWrapper{T}
    wrapped::T
    fns::Vector{Symbol}
end
CompositeTypeWrapper(x, syms) = CompositeTypeWrapper(x, collect(syms))
CompositeTypeWrapper(x) = CompositeTypeWrapper(x, fieldnames(typeof(x)))
function lower(a)
    if nfields(typeof(a)) > 0
        CompositeTypeWrapper(a)
    else
        error("Cannot serialize type $(typeof(a))")
    end
end
const IsPrintedAsString = Union{
    Dates.TimeType, Char, Type, AbstractString, Enum, Symbol}
lower(x::IsPrintedAsString) = x
lower(m::Module) = throw(ArgumentError("cannot serialize Module $m as JSON"))
lower(x::Real) = convert(Float64, x)
lower(x::Base.AbstractSet) = collect(x)
abstract type StructuralContext <: IO end
abstract type JSONContext <: StructuralContext end
mutable struct PrettyContext{T<:IO} <: JSONContext
    io::T
    step::Int
    state::Int
    first::Bool
end
PrettyContext(io::IO, step) = PrettyContext(io, step, 0, false)
mutable struct CompactContext{T<:IO} <: JSONContext
    io::T
    first::Bool
end
CompactContext(io::IO) = CompactContext(io, false)
struct StringContext{T<:IO} <: IO
    io::T
end
const CS = CommonSerialization
const SC = StructuralContext
Base.write(io::JSONContext, byte::UInt8) = write(io.io, byte)
Base.write(io::StringContext, byte::UInt8) =
    write(io.io, ESCAPED_ARRAY[byte + 0x01])
@inline function indent(io::PrettyContext)
    write(io, NEWLINE)
    for _ in 1:io.state
        write(io, SPACE)
    end
end
@inline indent(io::CompactContext) = nothing
@inline separate(io::PrettyContext) = write(io, SEPARATOR, SPACE)
@inline separate(io::CompactContext) = write(io, SEPARATOR)
@inline function delimit(io::JSONContext)
    if !io.first
        write(io, DELIMITER)
    end
    io.first = false
end
for kind in ("object", "array")
    beginfn = Symbol("begin_", kind)
    beginsym = Symbol(uppercase(kind), "_BEGIN")
    endfn = Symbol("end_", kind)
    endsym = Symbol(uppercase(kind), "_END")
    @eval function $beginfn(io::PrettyContext)
        write(io, $beginsym)
        io.state += io.step
        io.first = true
    end
    @eval $beginfn(io::CompactContext) = (write(io, $beginsym); io.first = true)
    @eval function $endfn(io::PrettyContext)
        io.state -= io.step
        if !io.first
            indent(io)
        end
        write(io, $endsym)
        io.first = false
    end
    @eval $endfn(io::CompactContext) = (write(io, $endsym); io.first = false)
end
function show_string(io::IO, x)
    write(io, STRING_DELIM)
    Base.print(StringContext(io), x)
    write(io, STRING_DELIM)
end
show_null(io::IO) = Base.print(io, "null")
function show_element(io::JSONContext, s, x)
    delimit(io)
    indent(io)
    show_json(io, s, x)
end
function show_key(io::JSONContext, k)
    delimit(io)
    indent(io)
    show_string(io, k)
    separate(io)
end
function show_pair(io::JSONContext, s, k, v)
    show_key(io, k)
    show_json(io, s, v)
end
show_pair(io::JSONContext, s, kv) = show_pair(io, s, first(kv), last(kv))
function show_json(io::SC, s::CS, x::IsPrintedAsString)
    lx = lower(x)
    if x === lx
        show_string(io, x)
    else
        show_json(io, s, lx)
    end
end
function show_json(io::SC, s::CS, x::Union{Integer, AbstractFloat})
    if isfinite(x)
        Base.print(io, x)
    else
        show_null(io)
    end
end
show_json(io::SC, ::CS, ::Nothing) = show_null(io)
function show_json(io::SC, s::CS, a::AbstractDict)
    begin_object(io)
    for kv in a
        show_pair(io, s, kv)
    end
    end_object(io)
end
function show_json(io::SC, s::CS, kv::Pair)
    begin_object(io)
    show_pair(io, s, kv)
    end_object(io)
end
function show_json(io::SC, s::CS, x::CompositeTypeWrapper)
    begin_object(io)
    for fn in x.fns
        show_pair(io, s, fn, getfield(x.wrapped, fn))
    end
    end_object(io)
end
function show_json(io::SC, s::CS, x::Union{AbstractVector, Tuple})
    begin_array(io)
    for elt in x
        show_element(io, s, elt)
    end
    end_array(io)
end
function show_json(io::SC, s::CS, A::AbstractArray{<:Any,n}) where n
    begin_array(io)
    newdims = ntuple(_ -> :, n - 1)
    for j in Compat.axes(A, n)
        show_element(io, s, view(A, newdims..., j))
    end
    end_array(io)
end
show_json(io::SC, s::CS, A::AbstractArray{<:Any,0}) = show_json(io, s, A[])
show_json(io::SC, s::CS, a) = show_json(io, s, lower(a))
function show_json(io::IO, s::Serialization, obj; indent=nothing)
    ctx = indent === nothing ? CompactContext(io) : PrettyContext(io, indent)
    show_json(ctx, s, obj)
    if indent !== nothing
        println(io)
    end
end
struct JSONText
    s::String
end
show_json(io::CompactContext, s::CS, json::JSONText) = write(io, json.s)
print(io::IO, obj, indent) =
    show_json(io, StandardSerialization(), obj; indent=indent)
print(io::IO, obj) = show_json(io, StandardSerialization(), obj)
print(a, indent) = print(stdout, a, indent)
print(a) = print(stdout, a)
json(a) = sprint(print, a)
json(a, indent) = sprint(print, a, indent)
end
using .Parser: parse, parsefile
using .Writer: show_json, json, lower, print, StructuralContext, show_element,
               show_string, show_key, show_pair, show_null, begin_array,
               end_array, begin_object, end_object, indent, delimit, separate,
               JSONText
using .Serializations: Serialization, CommonSerialization,
                       StandardSerialization
Writer.lower(json::JSONText) = parse(json.s)
end
