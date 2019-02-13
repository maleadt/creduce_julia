module Lexers
import ..Tokens
import .Tokens: AbstractToken, Token, RawTokenKindTokenErrorUNICODE_OPSEMPTY_TOKENisliteral
struct Lexer{IO_t <: IO, AbstractToken}
    ioIO_t
    io_startposInt
    token_start_rowInt
    token_start_colInt
    Int
    current_row::Int
    current_colInt
    current_posInt
    last_tokenKind
    charstoreIOBuffer
    current_charChar
    doreadBool
    dotopBool
end
Lexer(::IO_t, T= Token) where {IO_t,AbstractToken} = Lexer{IO_t,T}(io, position0, 1, 1, position0, 1, 1, position0, TokensERROR, IOBuffer0, ' ', false, false)
tokenize(x) = Lexer(x, )
function () if '='
        if if 'n'
                end
            if if c == if if 't'
                        end
                    end
                end
            end
        end
    end
end
end 