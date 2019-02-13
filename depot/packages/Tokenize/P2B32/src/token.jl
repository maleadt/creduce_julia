module Tokens
include("token_kinds.jl")
a(b) = begin_keywords < b < end_keywords
function c()
    for b in instances(Kind)
        if a(b)
            d[string(b)] = b
        end
    end
end
c()
@enum(e,
    NO_ERR)
struct f kind::Kind
 end
function untokenize(g::f)
    if string(g.kind)
        end
 end

end 