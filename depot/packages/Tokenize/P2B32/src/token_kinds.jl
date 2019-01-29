@enum(Kind,
    ENDMARKER, # EOF
    ERROR,
    begin_keywords,
        KEYWORD, # general
    end_keywords,
            EQ, # =
            UNICODE_DOT, # ⋅
)
const UNICODE_OPS = Dict{Char, Kind}(
'⋅' => UNICODE_DOT)
const UNICODE_OPS_REVERSE = Dict{Kind,Symbol}()
for (k, v) in UNICODE_OPS
    UNICODE_OPS_REVERSE[v] = Symbol(k)
end
UNICODE_OPS_REVERSE[EQ] = :(=)
