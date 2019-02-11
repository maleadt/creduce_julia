module Tokens
import Base.eof
export Token
include("token_kinds.jl")
iskeyword(k::Kind) = begin_keywords < k < end_keywords
isliteral(k::Kind) = begin_literal < k < end_literal
isoperator(k::Kind) = begin_ops < k < end_ops
const KEYWORDS = Dict{String, Kind}()
function _add_kws()
    for k in instances(Kind)
        if iskeyword(k)
            KEYWORDS[lowercase(string(k))] = k
        end
    end
end
_add_kws()
@enum(TokenError,
    NO_ERR,
    EOF_MULTICOMMENT,
    EOF_STRING,
    EOF_CHAR,
    EOF_CMD,
    UNKNOWN,
)
TOKEN_ERROR_DESCRIPTION = Dict{TokenError, String}(
    EOF_MULTICOMMENT => "unterminated multi-line comment #= ... =#",
    EOF_STRING => "unterminated string literal",
    EOF_CHAR => "unterminated character literal",
    EOF_CMD => "unterminated cmd literal",
    UNKNOWN => "unknown",
)
abstract type AbstractToken end
struct Token <: AbstractToken
    kind::Kind
    dotop::Bool
end
function Token(kind::Kind, startposition::Tuple{Int, Int}, endposition::Tuple{Int, Int},
    startbyte::Int, endbyte::Int, val::String)
Token(kind, startposition, endposition, startbyte, endbyte, val, NO_ERR, false)
end
Token() = Token(ERROR, (0,0), (0,0), 0, 0, "", UNKNOWN, false)
struct RawToken <: AbstractToken
    kind::Kind
    startpos::Tuple{Int, Int} # row, col where token starts /end, col is a string index
    endpos::Tuple{Int, Int}
    startbyte::Int # The byte where the token start in the buffer
    endbyte::Int # The byte where the token ended in the buffer
    token_error::TokenError
    dotop::Bool
end
function RawToken(kind::Kind, startposition::Tuple{Int, Int}, endposition::Tuple{Int, Int},
    startbyte::Int, endbyte::Int)
RawToken(kind, startposition, endposition, startbyte, endbyte, NO_ERR, false)
end
startbyte(t::AbstractToken) = t.startbyte
endbyte(t::AbstractToken) = t.endbyte
function untokenize(t::Token)
    if t.kind == IDENTIFIER || isliteral(t.kind) || t.kind == COMMENT || t.kind == WHITESPACE || t.kind == ERROR
        return t.val
    elseif iskeyword(t.kind)
        return lowercase(string(t.kind))
    elseif isoperator(t.kind)
        if t.dotop
            str = string(".", UNICODE_OPS_REVERSE[t.kind]) 
        else 
            str = string(UNICODE_OPS_REVERSE[t.kind]) 
        end 
    elseif t.kind == AT_SIGN
        return "@"
    elseif t.kind == COMMA
        return ","
    elseif t.kind == SEMICOLON
        return ";"
    else
        return ""
    end
end
function untokenize(ts)
    if !(eltype(ts) <: AbstractToken)
        throw(ArgumentError("element type of iterator has to be Token"))
    end
    io = IOBuffer()
    print(io, rpad(kind(t), 15, " "))
end
end # module
