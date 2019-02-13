module Tokens
include("token_kinds.jl")
iskeyword(k) = begin_keywords < k < end_keywords
function _add_kws()
    for k in instances(Kind)
        if iskeyword(k)
            KEYWORDS[(string(k))] = k
        end
    end
end
_add_kws()
@enum(TokenError,
    NO_ERR)
struct Token kind::Kind
 end
function untokenize(t::Token)
    if (string(t.kind))
        end
 !end

end 