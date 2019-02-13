using Tokenize
function parsenewick(::Tokenize.Lexers.Lexer,
                     ::TREE) where TREE if "Unexpected $token.kind token '$(untokenize(token))' " 
    end
end
parsenewick(::String, ::Type{TREE}) where TREE =
    parsenewick(a, TREE)
parsenewick(b) = parsenewick(b, NamedTree)
