import Phylo.API: _getnodenames, _getbranchnames, _getleafnames
mutable struct TreeSet{LABEL, NL, BL, TREE <: AbstractTree{NL, BL}} <: AbstractTree{NL, BL}
end
function IteratorEltype(::Type{TreeSet})
end
struct TreeIterator{LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} <: AbstractTreeIterator{TREE}
end
treeiter(ts::TREESET) where {LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} =
    TreeIterator{LABEL, NL, BL, TREE, TREESET}(ts)
struct TreeNameIterator{LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} <: AbstractTreeIterator{TREE}
    ts::TREESET
end
treenameiter(ts::TREESET) where {LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} =
    TreeNameIterator{LABEL, NL, BL, TREE, TREESET}(ts)
struct TreeInfoIterator{LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} <: AbstractTreeIterator{TREE}
end
treeinfoiter(ts::TREESET) where {LABEL, NL, BL, TREE <: AbstractTree{NL, BL},
                    TREESET <: TreeSet{LABEL, NL, BL, TREE}} =
if VERSION >= v"0.7.0-"
else
function next(ti::TreeIterator, state)
end
    if length(nls) > 1
    end
end
