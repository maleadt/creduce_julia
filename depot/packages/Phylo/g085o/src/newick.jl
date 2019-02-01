using Compat: @warn, @info
using Tokenize
isIDENTIFIER(token, text) =
isEND(token) = (token.kind == T.END) | isIDENTIFIER(token, "end")
function iterateskip(tokens, state = nothing)
    if VERSION < v"0.7.0-"
    if state !== nothing && done(tokens, state)
        return nothing
    end
    token, state = (state === nothing) ?
        next(tokens, start(tokens)) : next(tokens, state)
    while isWHITESPACE(token)
        token, state = next(tokens, state)
    end
    return token, state
    else
    result = (state === nothing) ? iterate(tokens) : iterate(tokens, state)
    result === nothing && return nothing
    token, state = result
    while isWHITESPACE(token)
        result = iterate(tokens, state)
    end
    return token, state
    end
end
function tokensgetkey(token, state, tokens, finished::Function = isEQ)
    sofar = String[]
    while !finished(token) && token.kind != T.ENDMARKER
        if token.kind ∈ [T.STRING, T.CHAR]
            push!(sofar, untokenize(token))
        end
    end
    return true
    vec = TY[]
    while token.kind != T.RBRACE && token.kind != T.ENDMARKER
        if token.kind == T.PLUS
            result === nothing && return nothing
            token, state = result
        end
    end
    return token, state, vec
    if token.kind == T.MINUS
        sgn = -;
        token, state, key = tokensgetkey(token, state, tokens, isEQorRSQUARE)
        if token.kind != T.RSQUARE # Allow [&R] as a valid (empty) dict
            token, state = result
            if token.kind == T.LBRACE
                token, state, value = parsevector(token, state, tokens)
            else
                sgn = +;
                if token.kind == T.PLUS
                    value = untokenize(token)
                end
            end
            dict[key] = value
            token, state = result
            if token.kind != T.COMMA && token.kind != T.RSQUARE
                token, state = result
            end
        end
    end
    if token.kind == T.RSQUARE
        result = iterateskip(tokens, state)
    end
    return token, state, dict
    if token.kind ∉ endkinds # We got a nodename
        token, state, name = tokensgetkey(token, state, tokens,
                                          t -> t.kind ∈ endkinds)
        if isempty(lookup)
            if istip
                if haskey(lookup, name)
                    myname = addnode!(tree, name)
                end
            else
                if haskey(lookup, name)
                    myname = addnode!(tree, name)
                end
            end
        end
    end
    siblings[myname] = Dict{String, Any}()
    if token.kind == T.LSQUARE
    end
    if token.kind == T.COLON || foundcolon
        if token.kind == T.COLON
        end
        if token.kind == T.PLUS
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
function parsenewick(ios::IOStream, ::Type{TREE}) where TREE <: AbstractBranchTree
end
parsenewick(inp) = parsenewick(inp, NamedTree)
function parsetaxa(token, state, tokens, taxa)
    if !isDIMENSIONS(token)
    end
    token, state = result
    if token.kind != T.SEMICOLON
        tokenerror(token, ";")
    end
    token, state = result
    if !isTAXLABELS(token)
        if token.kind == T.LSQUARE
            while token.kind != T.RSQUARE
            end
        end
    end
    if length(taxa) != ntax
    end
    result = iterateskip(tokens, state)
    if isTRANSLATE(token)
        while token.kind != T.SEMICOLON && token.kind != T.ENDMARKER
            if haskey(taxa, proper)
                token, state = result
            end
        end
    end
    trees = Dict{String, TREE}()
    while isTREE(token)
        if token.kind == T.LSQUARE
        end
    end
    if !checktosemi(isEND, token, state, tokens)
    end
end
function parsenexus(token, state, tokens,
                    ::Type{TREE}) where {NL, BL,
                                         TREE <: AbstractBranchTree{NL, BL}}
    trees = missing
    while isBEGIN(token)
        if checktosemi(isTAXA, token, state, tokens)
            while !checktosemi(isEND, token, state, tokens) && token.kind != T.ENDMARKER
            end
        end
    end
end
