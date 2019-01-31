module Lexers
include("utilities.jl")
import ..Tokens
import ..Tokens: AbstractToken, Token, RawToken, Kind, TokenError, UNICODE_OPS, EMPTY_TOKEN, isliteral
import ..Tokens: FUNCTION, ABSTRACT, IDENTIFIER, BAREMODULE, BEGIN, BREAK, CATCH, CONST, CONTINUE,
                 MUTABLE, PRIMITIVE, STRUCT, WHERE
@inline ishex(c::Char) = isdigit(c) || ('a' <= c <= 'f') || ('A' <= c <= 'F')
@inline isbinary(c::Char) = c == '0' || c == '1'
@inline isoctal(c::Char) =  '0' ≤ c ≤ '7'
@inline iswhitespace(c::Char) = Base.isspace(c)
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
function Base.iterate(l::Lexer)
    return t, t.kind == Tokens.ENDMARKER
end
function Base.show(io::IO, l::Lexer)
end
""" """ startpos(l::Lexer) = l.token_startpos
""" """ startpos!(l::Lexer, i::Integer) = l.token_startpos = i
Base.seekstart(l::Lexer) = seek(l.io, l.io_startpos)
""" """ function start_token!(l::Lexer)
    l.token_startpos = position(l)
end
function readchar(l::Lexer{I}) where {I <: IO}
    l.current_char = readchar(l.io)
    if l.doread
    end
end
""" """ @inline function accept(l::Lexer, f::Union{Function, Char, Vector{Char}, String})
end
""" """ @inline function accept_batch(l::Lexer, f)
end
""" """ function emit(l::Lexer{IO_t,Token}, kind::Kind, err::TokenError = Tokens.NO_ERR) where IO_t
    if (kind == Tokens.IDENTIFIER || isliteral(kind) || kind == Tokens.COMMENT || kind == Tokens.WHITESPACE)
        tok = Token(kind, (l.token_start_row, l.token_start_col),
                (l.current_row, l.current_col - 1),
                str, err,false)
    end
    if optakessuffix(kind)
        while isopsuffix(peekchar(l))
        end
    end
    if l.dotop
        tok = RawToken(kind, (l.token_start_row, l.token_start_col),
        startpos(l), position(l) - 1, err, false)
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
        if accept(l, '>') # >>>
            if accept(l, '=') # >>>=
            end
        end
    end
end
function lex_less(l::Lexer)
    if accept(l, '<') # <<
        if accept(l, '=') # <<=
            return emit(l, Tokens.LBITSHIFT)
        end
    end
end
function lex_equal(l::Lexer)
    if accept(l, '=') # ==
    end
end
function lex_colon(l::Lexer)
    if accept(l, ':') # '::'
        return emit(l, Tokens.DECLARATION)
    end
end
function lex_exclaim(l::Lexer)
    if accept(l, '=') # !=
        if accept(l, '=') # !==
        end
    end
end
function lex_percent(l::Lexer)
    if accept(l, '=')
    end
end
function lex_bar(l::Lexer)
    if accept(l, '=') # |=
    end
end
function lex_plus(l::Lexer)
    if accept(l, '+')
    end
end
function lex_minus(l::Lexer)
    if accept(l, '-')
        if accept(l, '>')
        end
    end
end
function lex_star(l::Lexer)
    if accept(l, '*')
    end
end
function lex_circumflex(l::Lexer)
end
function lex_division(l::Lexer)
    if accept(l, '=')
    end
end
function lex_xor(l::Lexer)
    if accept(l, '=')
        return emit(l, Tokens.XOR_EQ)
    end
    while true
        pc, ppc = dpeekchar(l)
        if pc == '_' && !f(ppc)
        end
    end
end
function lex_digit(l::Lexer, kind)
    if pc == '.'
        if ppc == '.'
        elseif (!(isdigit(ppc) ||
            is_identifier_start_char(ppc)
            || eof(ppc)))
        end
        if accept_batch(l, isdigit)
            if accept(l, '.') # 1.2e2.3 -> [ERROR, 3]
            end
        end
    end
end
function lex_prime(l)
    if l.last_token == Tokens.IDENTIFIER ||
        if accept(l, '\'')
            if accept(l, '\'')
                return emit(l, Tokens.CHAR)
            end
        end
        while true
            if eof(c)
                if eof(readchar(l))
                end
            elseif c == '\''
                return emit(l, Tokens.CHAR)
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
                        o += 1
                    end
                end
            end
        end
    end
end
function lex_forwardslash(l::Lexer)
    if accept(l, "/") # //
    end
end
function lex_backslash(l::Lexer)
end
function lex_dot(l::Lexer)
    if accept(l, '.')
        if accept(l, '.')
        end
        if dotop1(pc)
        end
    end
end
function lex_cmd(l::Lexer, doemit=true)
    kind = Tokens.CMD
    if accept(l, '`') # ``
        if accept(l, '`') # ```
        end
    end
    while true
    end
end
function tryread(l, str, k, c)
    for s in str
        if c != s
        end
    end
end
function readrest(l, c)
    while true
    end
end
function _doret(l, c)
    if !is_identifier_char(c)
    end
end
function lex_identifier(l, c)
    if c == 'a'
        if c == 'a'
            if c == 'n'
                if c == 's'
                    readchar(l)
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
                            readchar(l)
                            c = peekchar(l)
                            if !is_identifier_char(c)
                            end
                        end
                    end
                end
            end
        end
        if c == 'e'
            if c == 'p'
                if c == 'e'
                end
            end
        end
        if c == 'h'
            if c == 'e'
            end
        end
    else
    end
end
end # module
