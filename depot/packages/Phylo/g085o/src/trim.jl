using IterableTables
using IterableTables: getiterator
using Compat: findall
""" """ function getinternalnodes(t::AbstractTree)
    return collect(nodenamefilter(x->!isleaf(x) & !isroot(x), t))
end
""" """ function droptips!(t::T, tips::Vector{NL}) where {NL, BL, T <: AbstractTree{NL, BL}}
    tree_names = getleafnames(t)
    keep_tips = setdiff(tree_names, tips)
    for i in tips
        deletenode!(t, i)
    end
    while length(setdiff(collect(nodenamefilter(isleaf, t)), keep_tips)) > 0
        nodes = setdiff(collect(nodenamefilter(isleaf, t)), keep_tips)
        map(x -> deletenode!(t, x), nodes)
    end
    while any(map(x -> length(getchildren(t, x)) .< 2,
                  getinternalnodes(t)))
        inner_nodes = getinternalnodes(t)
        remove_nodes = findall(map(x->length(getchildren(t, x)) .< 2,
                                   inner_nodes))
        for i in remove_nodes
            parent = getparent(t, inner_nodes[i])
            parentbranch = getinbound(getnode(t, inner_nodes[i]))
            child = getchildren(t, inner_nodes[i])[1]
            childbranch = getoutbounds(getnode(t, inner_nodes[i]))[1]
            len = distance(t, parent, child)
            deletebranch!(t, parentbranch)
            deletebranch!(t, childbranch)
            delete!(getnodes(t), inner_nodes[i])
            delete!(t.noderecords, inner_nodes[i])
            addbranch!(t, parent, child, len)
        end
    end
    root = collect(nodenamefilter(isroot, t))[1]
    if length(getchildren(t, root)) < 2
        deletenode!(t, root)
    end
    if !isempty(getleafinfo(t))
        li = leafinfotype(t)(Iterators.filter(line -> line[1] ∉ tips,
                                              getiterator(getleafinfo(t))))
        setleafinfo!(t, li)
    end
    return tips
end
""" """ function keeptips!(t::T, tips::Vector{NL}) where {NL, BL, T <: AbstractTree{NL, BL}}
    tree_names = getleafnames(t)
    cut_names = setdiff(tree_names, tips)
    droptips!(t, cut_names)
end
