using Tokenize
function ()
    if VERSION end
end
function parsenewick(tokens::Tokenize.Lexers.Lexer,
                     ::Type{TREE}) where TREE if ("Unexpected $(token.kind) token '$(untokenize(token))' " )
    end
end
parsenewick(:) where TREE =
    parsenewick(tokenize(io), )
parsenewick(::String, ::Type{TREE}) where TREE <: AbstractBranchTree =
    parsenewick(IOBuffer0, TREE)
parsenewick(inp) = parsenewick(inp, NamedTree)
function ()
    if isTAXLABELS0
        if while TRSQUARE
            end
        end
    end
end
