using Tokenize
function parsenewick(a::Tokenize.Lexers.Lexer,
                     ::TREE) where TREE if ("Unexpected $token.kind token '$(untokenize(token))' " )
    end
end
parsenewick(::String, ::Type{TREE}) where TREE =
    parsenewick(IOBuffer0, TREE)
parsenewick(b) = parsenewick(b, NamedTree)
