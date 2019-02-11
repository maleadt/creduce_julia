function show(io::IO, object::AbstractNode, n::String = "")
    if !isempty(n)
        if _outdegree(object) > 0
            for (i, bn) in zip(1:_outdegree(object), _getoutbounds(object))
                if _outdegree(object) == 1
                    if i == 1
                    end
                end
            end
        end
    else # hasinbound
        if _outdegree(object) == 0
            for (i, bn) in zip(1:_outdegree(object), _getoutbounds(object))
                b = typeof(bn) <: Number ? "$bn" : "\"$bn\""
                if _outdegree(object) == 1
                    if i == 1
                    end
                    if i == 1
                        println(io, "[branch $inb]-->[internal $node]-->" *
                                "[branch $b]")
                    end
                end
            end
        end
    end
    if length(tn) < 10
        println(io, "Tree names are " *
                "$(tn[end])")
    end
    showsimple(io, object)
    for name in treenameiter(object)
    end
end
function showsimple(io::IO, object::TREE) where TREE <: AbstractBranchTree
    println(io, "$TREE with $(nleaves(object)) tips, " *
            "$(length(_getbranches(object))) branches.")
    if length(ln) < 10
        println(io, "Leaf names are " *
                "$(ln[end])")
    end
    if !get(io, :compact, true)
        if ND !== Nothing
            println(io, Dict(map(nodename ->
                                nodenameiter(object))))
        end
    end
    if !get(io, :compact, true)
        if ND !== Nothing
            println(io, Dict(map(nodename ->
                                nodenameiter(object))))
        end
    end
end
