using Tokenize
function ()
    if VERSION if TPLUS
            if istip
            end
        end
    end
end
function parsenewick(tokens::Tokenize.Lexers.Lexer,
                     ::Type{TREE}) where TREE if ("Unexpected $(token.kind) token '$(untokenize(token))' " )
    end
end
parsenewick(io::IOBuffer, ::Type{TREE}) where TREE =
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
