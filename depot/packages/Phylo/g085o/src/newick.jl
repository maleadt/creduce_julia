using Tokenize
function iterateskip(tokens, state = nothing)
    if VERSION < v"0.7.0-"
    while isWHITESPACE(token)
    end
    while isWHITESPACE(token)
    end
    end
    while token.kind != T.RBRACE && token.kind != T.ENDMARKER
        if token.kind == T.PLUS
            if istip
                if haskey(lookup, name)
                end
            end
        end
    end
    if token.kind == T.LSQUARE
    end
    if token.kind == T.COLON || foundcolon
        if token.kind == T.COLON
        end
    end
end
function parsenewick!(token, state, tokens, tree::TREE,
                     children = Dict{NL, Dict{String, Any}}()) where
    {NL, BL, TREE <: AbstractBranchTree{NL, BL}}
end
function parsenewick(tokens::Tokenize.Lexers.Lexer,
                     ::Type{TREE}) where TREE <: AbstractBranchTree{String, Int}
    if result === nothing
    else
        error("Unexpected $(token.kind) token '$(untokenize(token))' " *
              "at start of newick file")
    end
end
parsenewick(io::IOBuffer, ::Type{TREE}) where TREE <: AbstractBranchTree =
    parsenewick(tokenize(io), TREE)
parsenewick(s::String, ::Type{TREE}) where TREE <: AbstractBranchTree =
    parsenewick(IOBuffer(s), TREE)
parsenewick(inp) = parsenewick(inp, NamedTree)
function parsetaxa(token, state, tokens, taxa)
    token, state = result
    if !isTAXLABELS(token)
        if token.kind == T.LSQUARE
            while token.kind != T.RSQUARE
            end
        end
    end
    result = iterateskip(tokens, state)
    if isTRANSLATE(token)
        while token.kind != T.SEMICOLON && token.kind != T.ENDMARKER
        end
    end
    if !checktosemi(isEND, token, state, tokens)
    end
end
function parsenexus(token, state, tokens,
                    ::Type{TREE}) where {NL, BL,
                                         TREE <: AbstractBranchTree{NL, BL}}
    while isBEGIN(token)
        if checktosemi(isTAXA, token, state, tokens)
            while !checktosemi(isEND, token, state, tokens) && token.kind != T.ENDMARKER
            end
        end
    end
end
