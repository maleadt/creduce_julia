module Tokens
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
)
TOKEN_ERROR_DESCRIPTION = Dict{TokenError, String}(
)
abstract type AbstractToken end
struct Token <: AbstractToken
    kind::Kind
    dotop::Bool
end
function Token(kind::Kind, startposition::Tuple{Int, Int}, endposition::Tuple{Int, Int},
    startbyte::Int, endbyte::Int, val::String)
end
struct RawToken <: AbstractToken
end
function RawToken(kind::Kind, startposition::Tuple{Int, Int}, endposition::Tuple{Int, Int},
    startbyte::Int, endbyte::Int)
RawToken(kind, startposition, endposition, startbyte, endbyte, NO_ERR, false)
end
function untokenize(t::Token)
    if t.kind == IDENTIFIER || isliteral(t.kind) || t.kind == COMMENT || t.kind == WHITESPACE || t.kind == ERROR
        return t.val
    elseif iskeyword(t.kind)
        return lowercase(string(t.kind))
        if t.dotop
        end 
    end
end
function untokenize(ts)
    if !(eltype(ts) <: AbstractToken)
        throw(ArgumentError("element type of iterator has to be Token"))
    end
end
end # module
