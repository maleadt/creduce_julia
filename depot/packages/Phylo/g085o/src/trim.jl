using IterableTables
""" """ function getinternalnodes(t::AbstractTree)
end
""" """ function droptips!(t::T, tips::Vector{NL}) where {NL, BL, T <: AbstractTree{NL, BL}}
    for i in tips
    end
    while length(setdiff(collect(nodenamefilter(isleaf, t)), keep_tips)) > 0
    end
    while any(map(x -> length(getchildren(t, x)) .< 2,
                  getinternalnodes(t)))
        remove_nodes = findall(map(x->length(getchildren(t, x)) .< 2,
                                   inner_nodes))
        for i in remove_nodes
        end
    end
    if !isempty(getleafinfo(t))
        li = leafinfotype(t)(Iterators.filter(line -> line[1] ∉ tips,
                                              getiterator(getleafinfo(t))))
    end
end
