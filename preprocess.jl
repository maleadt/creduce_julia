#!/usr/bin/env julia

using CSTParser, Tokenize


## rewrite / pass infrastructure

struct Edit{T}
    loc::T
    text::String
end

mutable struct State{T}
    offset::Int
    edits::T
end

function pass(x, state, f = (x,state)->nothing)
    f(x, state)
    for a in x
        pass(a, state, f)
    end
    state
end

function pass(x::CSTParser.LeafNode, state, f = (x,edits)->nothing)
    state.offset += x.fullspan
end

function pass(x::Tokens.AbstractToken, state, f = (x,edits)->nothing)
    f(x, state)
    state.offset = Tokens.endbyte(x) + 1
end

function apply(text, edit::Edit{Int})
    string(text[1:edit.loc], edit.text, text[nextind(text, edit.loc):end])
end
function apply(text, edit::Edit{UnitRange{Int}})
    string(text[1:prevind(text, first(edit.loc))], edit.text, text[nextind(text, last(edit.loc)):end])
end


## passes

@enum PassType Lexical Semantic

function remove_comments(x, state)
    if x isa Tokens.AbstractToken && x.kind == Tokens.COMMENT
        # remove comment
        push!(state.edits, Edit(Tokens.startbyte(x)+1:Tokens.endbyte(x)+1, ""))
    end
end

function remove_docstrings(x, state)
    if x isa CSTParser.EXPR{CSTParser.MacroCall} && x.args[1] isa CSTParser.EXPR{CSTParser.GlobalRefDoc}
        offset = state.offset + x.args[1].fullspan
        doc = x.args[2]

        # remove docstring
        push!(state.edits, Edit(offset+1:offset+doc.fullspan, ""))

        # remove target if it is an identifier or function call
        if x.args[3] isa CSTParser.IDENTIFIER || x.args[3] isa CSTParser.EXPR{CSTParser.Call}
            target = x.args[3]
            push!(state.edits, Edit(offset+1:offset+target.fullspan, ""))
        end
    end
end

function compact_whitespace(x, state)
    if x isa Tokens.AbstractToken && x.kind == Tokens.WHITESPACE
        # reduce to single-line whitespace
        newlines = findall(isequal('\n'), x.val)
        if length(newlines) > 0
            push!(state.edits, Edit(Tokens.startbyte(x)+1:Tokens.endbyte(x)+1, x.val[last(newlines):end]))
        end
    end
end


## main

const pkgdir = joinpath(@__DIR__, "depot", "packages")

function process(dir)
    for entry in readdir(dir)
        path = joinpath(dir, entry)
        if isdir(path)
            process(path)
        elseif isfile(path) && endswith(entry, ".jl")
            rewrite(path)
        end
    end
end

function rewrite(path)
    println("Processing $path...")
    text = read(path, String)
    for (pt, p) in (Lexical     => remove_comments,
                    Semantic    => remove_docstrings,
                    Lexical     => compact_whitespace
                   )
        state = State(0, Edit[])
        if pt == Lexical
            code = collect(tokenize(text))
            pass(code, state, p)
        elseif pt == Semantic
            ast = CSTParser.parse(text, true)
            pass(ast, state, p)
        end
        state.offset = 0
        sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
        for i = 1:length(state.edits)
            text = apply(text, state.edits[i])
        end
    end
    if abspath(PROGRAM_FILE) == @__FILE__
        write(path, text)
    else
        println(text)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    process(pkgdir)
end
