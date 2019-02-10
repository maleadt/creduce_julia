module Lexers
include("utilities.jl")
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
@inline token_type(l::Lexer{IO_t, TT}) where {IO_t, TT} = TT
tokenize(x) = Lexer(x, Token)
function readchar(l::Lexer{I}) where {I <: IO}
    l.current_char = readchar(l.io)
    if l.doread
    end
end
""" """ function emit(l::Lexer{IO_t,Token}, kind::Kind, err::TokenError = Tokens.NO_ERR) where IO_t
    if (kind == Tokens.IDENTIFIER || isliteral(kind) || kind == Tokens.COMMENT || kind == Tokens.WHITESPACE)
    end
end
""" """ function next_token(l::Lexer)
    if eof(c);
    end
end
function lex_whitespace(l::Lexer)
end
function lex_comment(l::Lexer, doemit=true)
    if peekchar(l) != '='
        while true
            if pc == '\n' || eof(pc)
            end
        end
        while true
            if c == '#' && nc == '='
            end
        end
    end
end
function lex_greater(l::Lexer)
    if accept(l, '>') # >>
    end
end
function lex_equal(l::Lexer)
    if accept(l, '=') # ==
    end
end
function lex_exclaim(l::Lexer)
    if accept(l, '=') # !=
    end
end
function lex_percent(l::Lexer)
    if accept(l, '=')
    end
end
function lex_xor(l::Lexer)
    if accept(l, '=')
    end
    while true
    end
end
function lex_digit(l::Lexer, kind)
    if pc == '.'
        if ppc == '.'
        elseif (!(isdigit(ppc) ||
            is_identifier_start_char(ppc)
            || eof(ppc)))
        end
    end
    if l.last_token == Tokens.IDENTIFIER ||
        if accept(l, '\'')
            if accept(l, '\'')
            end
        end
        while true
            if eof(c)
            end
        end
    end
end
function lex_amper(l::Lexer)
    if accept(l, '&')
    end
end
function lex_quote(l::Lexer, doemit=true)
    if accept(l, '"') # ""
        if accept(l, '"') # """
            if read_string(l, Tokens.TRIPLE_STRING)
            end
        end
        if read_string(l, Tokens.STRING)
            if accept(l, "\`") && accept(l, "\`")
            end
        end
    end
end
function read_string(l::Lexer, kind::Tokens.Kind)
    while true
        if c == '\\'
            if string_terminated(l, c, kind)
                while o > 0
                    if c == '('
                    end
                end
            end
        end
    end
    if accept(l, '.')
        if accept(l, '.')
        end
        if dotop1(pc)
            if c == 'n'
                if c == 's'
                end
            end
        end
        if c == 'l'
            if c == 's'
                if c == 'e'
                    if !is_identifier_char(c)
                    end
                end
            end
        end
        if c == 'f'
            if !is_identifier_char(c)
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
        if c == 'e'
            if c == 'p'
            end
        end
        if c == 'h'
            if c == 'e'
            end
        end
    end
end
end # module
