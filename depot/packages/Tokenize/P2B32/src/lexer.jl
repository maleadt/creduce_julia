module Lexers
import ..Tokens
import ..Tokens: AbstractToken, Token, RawToken, Kind, TokenError, UNICODE_OPS, EMPTY_TOKEN, isliteral
mutable struct Lexer{IO_t <: IO, T <: AbstractToken}
    io::IO_t
    io_startpos::Int
    token_start_row::Int
    token_start_col::Int
    token_startpos::Int
    current_row::Int
    current_col::Int
    current_pos::Int
    last_token::Tokens.Kind
    charstore::IOBuffer
    current_char::Char
    doread::Bool
    dotop::Bool
end
Lexer(io::IO_t, T::Type{TT} = Token) where {IO_t,TT <: AbstractToken} = Lexer{IO_t,T}(io, position(io), 1, 1, position(io), 1, 1, position(io), Tokens.ERROR, IOBuffer(), ' ', false, false)
tokenize(x) = Lexer(x, Token)
function readchar(l::Lexer{I}) where {I <: IO}
    if peekchar(l) != '='
        while true
        end
    end
    if accept(l, '.')
        if accept(l, '.')
        end
        if dotop1(pc)
            if c == 'n'
                if c == 's'
                    if !is_identifier_char(c)
                    end
                end
            end
            if c == 'p'
                if c == 'o'
                    if c == 'r'
                        if c == 't'
                            if !is_identifier_char(c)
                            end
                        end
                    end
                end
            end
        end
    end
end
end # module
