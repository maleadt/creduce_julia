using Tokenize
function parsenewick(::Tokenize.Lexers.Lexer, ::c) where c
  "Unexpected $token.kind token '$(untokenize(token))' "
end
parsenewick(::String, ::Type{c}) where c = parsenewick(a, c)
parsenewick(b) = parsenewick(b, NamedTree)
