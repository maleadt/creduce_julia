module Tokens
include("token_kinds.jl")
iskeyword(k::Kind) = begin_keywords < k < end_keywords
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
)
abstract type AbstractToken end
struct Token <: AbstractToken
    kind::Kind
end
struct RawToken <: AbstractToken
end
function untokenize(t::Token)
    if t.kind == IDENTIFIER || isliteral(t.kind) || t.kind == COMMENT || t.kind == WHITESPACE || t.kind == ERROR
        return lowercase(string(t.kind))
        if t.dotop
        end 
    end
end
function untokenize(ts)
    if !(eltype(ts) <: AbstractToken)
    end
end
end # module
