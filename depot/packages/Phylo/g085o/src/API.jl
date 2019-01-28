using Phylo
function _newlabel(ids::Vector{Label}) where Label <: Integer
    return isempty(ids) ? 1 : maximum(ids) + 1
end
function _newlabel(names::Vector{String}, prefix)
    names = collect(Iterators.filter(n -> length(n) > length(prefix) &&
                                     n[1:length(prefix)]==prefix, names))
    start = length(names) + 1
    name = prefix * "$start"
    while (name ∈ names)
        start += 1
        name = prefix * "$start"
    end
    return name
end
function _ntrees(::AbstractTree)
    return 1
end
function _extractnode(::T, node::N) where {T <: AbstractTree, N <: AbstractNode}
    return node
end
function _extractnode(tree::T, nodename::NL) where {NL, BL, T <: AbstractTree{NL, BL}}
    return getnode(tree, nodename)
end
function _extractnode(::T, pair::Pair{NL, N}) where {NL, BL, T <: AbstractTree{NL, BL},
                                                     N <: AbstractNode}
    return pair[2]
end
function _extractnodename(::T, pair::Pair{NL, N}) where {NL, BL, T <: AbstractTree{NL, BL},
                                                         N <: AbstractNode}
    return pair[1]
end
function _extractbranch(::T, branch::B) where {T <: AbstractTree, B <: AbstractBranch}
    return branch
end
function _extractbranch(tree::T, branchname::BL) where {NL, BL, T <: AbstractTree{NL, BL}}
    return getbranch(tree, branchname)
end
function _extractbranch(::T, pair::Pair{BL, B}) where {NL, BL, T <: AbstractTree{NL, BL},
                                                       B <: AbstractBranch}
    return pair[2]
end
function _extractbranchname(::T, pair::Pair{BL, B}) where {NL, BL, T <: AbstractTree{NL, BL},
                                                           B <: AbstractBranch}
    return pair[1]
end
""" """ function _nodetype end
""" """ function _branchtype end
""" """ function _newbranchlabel end
function _newbranchlabel(tree::AbstractTree{NL, String}) where NL
    return _newlabel(_getbranchnames(tree), "Branch ")
end
function _newbranchlabel(tree::AbstractTree{NL, I}) where {NL, I <: Integer}
    return _newlabel(_getbranchnames(tree))
end
""" """ function _addbranch! end
""" """ function _deletebranch! end
""" """ function _branch!(tree::AbstractTree, source, length::Float64, destination, branchname)
    _addbranch!(tree, source, _addnode!(tree, destination), length, branchname)
    return destination
end
""" """ function _newnodelabel end
function _newnodelabel(tree::T) where {BL, T <: AbstractTree{String, BL}}
    return _newlabel(_getnodenames(tree), "Node ")
end
function _newnodelabel(tree::T) where {I <: Integer, BL, T <: AbstractTree{I, BL}}
    return _newlabel(_getnodenames(tree))
end
""" """ function _getnodenames end
""" """ function _getnodes end
""" """ function _addnode! end
""" """ function _addnodes! end
function _addnodes!(tree::AbstractTree, nodenames::AbstractVector)
    return map(name -> _addnode!(tree, name), nodenames)
end
function _addnodes!(tree::AbstractTree, count::Integer)
    return map(name -> addnode!(tree), 1:count)
end
""" """ function _deletenode! end
""" """ function _hasnode end
""" """ function _getnode end
""" """ function _getbranchnames end
""" """ function _getbranches end
""" """ function _hasbranch end
""" """ function _getbranch end
""" """ function _hasrootheight(::AbstractTree)
    return false
end
""" """ function _getrootheight(::AbstractTree)
    throw(NullException())
    return NaN
end
""" """ function _setrootheight!(::AbstractTree, value)
    throw(NullException())
    return value
end
""" """ function _validate(::AbstractTree)
    return true
end
""" """ _isleaf(node::AbstractNode) = _outdegree(node) == 0
""" """ _isroot(node::AbstractNode) = !_hasinbound(node)
""" """ _isinternal(node::AbstractNode) = _outdegree(node) > 0 && _hasinbound(node)
""" """ _isunattached(node::AbstractNode) = _outdegree(node) == 0 && !_hasinbound(node)
""" """ _indegree(node::AbstractNode) = _hasinbound(node) ? 1 : 0
""" """ _hasinboundspace(node::AbstractNode) = !_hasinbound(node)
""" """ function _outdegree end
""" """ function _hasoutboundspace end
""" """ function _hasinbound end
""" """ function _getinbound end
""" """ function _setinbound! end
""" """ function _deleteinbound! end
""" """ function _getoutbounds end
""" """ function _addoutbound! end
""" """ function _deleteoutbound! end
""" """ function _hasheight(::AbstractTree, _)
    return false
end
""" """ function _getheight(::AbstractTree, _)
    throw(NullException())
    return NaN
end
""" """ function _setheight!(::AbstractTree, _, value)
    throw(NullException())
    return value
end
""" """ function _src end
""" """ function _dst end
""" """ function _getlength end
function _getlength(::AbstractBranch)
    return NaN
end
""" """ function _setsrc! end
""" """ function _setdst! end
function _getleafnames(tree::AbstractTree)
    return collect(nodenamefilter(_isleaf, tree))
end
function _getleafinfo end
function _setleafinfo! end
function _getnoderecord end
function _setnoderecord! end
function _resetleaves! end
function _clearrootheight! end
function _setnode! end
function _setbranch! end
function _leafinfotype end
function _nleaves end
