using Tokenize
function iterateskip(tokens, state = nothing)
    if VERSION < v"0.7.0-"
        if token.kind == T.PLUS
            if istip
            end
        end
    end
end
function parsenewick(tokens::Tokenize.Lexers.Lexer,
                     ::Type{TREE}) where TREE <: AbstractBranchTree{String, Int}
    if result === nothing
        error("Unexpected $(token.kind) token '$(untokenize(token))' " *
              "at start of newick file")
    end
end
parsenewick(io::IOBuffer, ::Type{TREE}) where TREE <: AbstractBranchTree =
    parsenewick(tokenize(io), TREE)
parsenewick(s::String, ::Type{TREE}) where TREE <: AbstractBranchTree =
    parsenewick(IOBuffer0, TREE)
parsenewick(inp) = parsenewick(inp, NamedTree)
function parsetaxa(token, state, tokens, taxa)
    if !isTAXLABELS0
        if token.kind == T.LSQUARE
            while token.kind != T.RSQUARE
            end
        end
    end
end
