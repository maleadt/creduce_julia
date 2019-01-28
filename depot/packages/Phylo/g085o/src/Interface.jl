using Phylo.API
using Compat: mapreduce
getnodes(tree::AbstractTree) = _getnodes(tree)
getbranches(tree::AbstractTree) = _getbranches(tree)
function ntrees(tree::AbstractTree)
    return _ntrees(tree)
end
""" """ nodetype(tree::AbstractTree) = _nodetype(tree)
""" """ branchtype(tree::AbstractTree) = _branchtype(tree)
""" """ nodenametype(::T) where {NL, BL, T <: AbstractTree{NL, BL}} = NL
""" """ branchnametype(::T) where {NL, BL, T <: AbstractTree{NL, BL}} = BL
""" """ function addbranch!(tree::AbstractTree, source, destination, length::Float64 = NaN;
                    branchname = _newbranchlabel(tree))
    _hasnode(tree, source) ||
    error("Tree does not have an available source node called $source")
    hasoutboundspace(tree, source) ||
    error("$source already has maximum number of outbound connections ($(outdegree(tree, source)))")
    _hasnode(tree, destination) ||
        error("Tree does not have a destination node called $destination")
    !hasinbound(tree, destination) ||
            error("Tree does not have an available destination node called $destination")
    destination != source || error("Branch must connect different nodes")
    _hasbranch(tree, branchname) &&
        error("Tree already has a branch called $branchname")
    return _addbranch!(tree, source, destination, length, branchname)
end
""" """ function deletebranch!(tree::AbstractTree, branchname)
    _hasbranch(tree, branchname) ||
        error("Tree does not have a branch called $branchname")
    return _deletebranch!(tree, branchname)
end
""" """ function branch!(tree::AbstractTree, source, length::Float64 = NaN;
                 destination = _newnodelabel(tree),
                 branchname = _newbranchlabel(tree))
    _hasnode(tree, source) ||
        error("Node $source not present in tree")
    !_hasnode(tree, destination) ||
        error("Node $destination already present in tree")
    _hasoutboundspace(_getnode(tree, source)) ||
        error("Node $source has no space to add branches")
    return _branch!(tree, source, length, destination, branchname)
end
""" """ function addnode!(tree::AbstractTree, nodename = _newnodelabel(tree))
    !_hasnode(tree, nodename) ||
        error("Node $nodename already present in tree")
    return _addnode!(tree, nodename)
end
""" """ function addnodes! end
function addnodes!(tree::AbstractTree, nodenames::AbstractVector)
    all(map(name -> !_hasnode(tree, name), nodenames)) ||
        error("Some of nodes $nodenames already present in tree")
    return _addnodes!(tree, nodenames)
end
function addnodes!(tree::AbstractTree, count::Integer)
    return _addnodes!(tree, count)
end
""" """ function deletenode!(tree::AbstractTree, nodename)
    return _deletenode!(tree, nodename)
end
""" """ function getnodenames(tree::AbstractTree)
    return _getnodenames(tree)
end
""" """ function hasnode(tree::AbstractTree, nodename)
    return _hasnode(tree, nodename)
end
""" """ function getnode(tree::AbstractTree, nodename)
    _hasnode(tree, nodename) ||
        error("Node $nodename does not exist")
    return _getnode(tree, nodename)
end
""" """ function getbranchnames(tree::AbstractTree)
    return _getbranchnames(tree)
end
""" """ function hasbranch(tree::AbstractTree, branchname)
    return _hasbranch(tree, branchname)
end
""" """ function getbranch(tree::AbstractTree, branchname)
    _hasbranch(tree, branchname) ||
        error("Branch $branchname does not exist")
    return _getbranch(tree, branchname)
end
""" """ function hasrootheight(tree::AbstractTree)
    return _hasrootheight(tree)
end
""" """ function getrootheight(tree::AbstractTree)
    return _getrootheight(tree)
end
""" """ function setrootheight!(tree::AbstractTree, height)
    return _setrootheight!(tree, height)
end
""" """ function validate(tree::T) where {NL, BL, T <: AbstractTree{NL, BL}}
    nodes = _getnodes(tree)
    branches = _getbranches(tree)
    if !isempty(nodes) || !isempty(branches)
        if Set(mapreduce(_getinbound, push!, nodefilter(_hasinbound, tree);
                         init = BL[])) != Set(keys(branches))
            warn("Inbound branches must exactly match Branch labels")
            return false
        end
        if Set(mapreduce(_getoutbounds, append!, nodeiter(tree);
                         init = BL[])) != Set(keys(branches))
            warn("Node outbound branches must exactly match Branch labels")
            return false
        end
        if !(mapreduce(_src, push!, branchiter(tree); init = NL[]) ⊆
             Set(keys(nodes)))
            warn("Branch sources must be node labels")
            return false
        end
        if !(mapreduce(_dst, push!, branchiter(tree); init = NL[]) ⊆
             Set(keys(nodes)))
            warn("Branch destinations must be node labels")
            return false
        end
    end
    return _validate(tree)
end
""" """ function isleaf end
function isleaf(node::AbstractNode)
    return _isleaf(node)
end
function isleaf(tree::AbstractTree, nodename)
    return _isleaf(_getnode(tree, nodename))
end
""" """ function isroot end
function isroot(node::AbstractNode)
    return _isroot(node)
end
function isroot(tree::AbstractTree, nodename)
    return _isroot(_getnode(tree, nodename))
end
""" """ function isinternal end
function isinternal(node::AbstractNode)
    return _isinternal(node)
end
function isinternal(tree::AbstractTree, nodename)
    return _isinternal(_getnode(tree, nodename))
end
""" """ function isunattached end
function isunattached(node::AbstractNode)
    return _isunattached(node)
end
function isunattached(tree::AbstractTree, nodename)
    return _isunattached(_getnode(tree, nodename))
end
""" """ function indegree end
function indegree(node::AbstractNode)
    return _indegree(node)
end
function indegree(tree::AbstractTree, nodename)
    return _indegree(_getnode(tree, nodename))
end
""" """ function outdegree end
function outdegree(node::AbstractNode)
    return _outdegree(node)
end
function outdegree(tree::AbstractTree, nodename)
    return _outdegree(_getnode(tree, nodename))
end
""" """ function hasoutboundspace end
function hasoutboundspace(node::AbstractNode)
    return _hasoutboundspace(node)
end
function hasoutboundspace(tree::AbstractTree, nodename)
    return _hasoutboundspace(_getnode(tree, nodename))
end
""" """ function hasinbound end
function hasinbound(node::AbstractNode)
    return _hasinbound(node)
end
function hasinbound(tree::AbstractTree, nodename)
    return _hasinbound(_getnode(tree, nodename))
end
""" """ function hasinboundspace end
function hasinboundspace(node::AbstractNode)
    return _hasinboundspace(node)
end
function hasinboundspace(tree::AbstractTree, nodename)
    return _hasinboundspace(_getnode(tree, nodename))
end
""" """ function getinbound end
function getinbound(node::AbstractNode)
    return _getinbound(node)
end
function getinbound(tree::AbstractTree, nodename)
    return _getinbound(_getnode(tree, nodename))
end
""" """ function getparent(tree::AbstractTree, nodename)
    return src(tree, getinbound(tree, nodename))
end
""" """ function getancestors(tree::AbstractTree, nodename)
    return _treepast(tree, nodename)[2][2:end]
end
""" """ function getoutbounds end
function getoutbounds(node::AbstractNode)
    return _getoutbounds(node)
end
function getoutbounds(tree::AbstractTree, nodename)
    return _getoutbounds(_getnode(tree, nodename))
end
""" """ function getchildren(tree::AbstractTree, nodename)
    return map(branch -> dst(tree, branch), getoutbounds(tree, nodename))
end
""" """ function getdescendants(tree::AbstractTree, nodename)
    return _treefuture(tree, nodename)[2][2:end]
end
""" """ function hasheight end
function hasheight(tree::AbstractTree, nodename)
    return _hasheight(tree, nodename) ||
        (_hasrootheight(tree) &&
         mapreduce(b -> haslength(tree, b), &, branchhistory(tree, nodename);
         init = _hasrootheight(tree)))
end
""" """ function getheight(tree::AbstractTree, nodename)
    return _hasheight(tree, nodename) ? _getheight(tree, nodename) :
        mapreduce(b -> getlength(tree, b), +, branchhistory(tree, nodename);
                  init = getrootheight(tree))
end
""" """ function setheight!(tree::AbstractTree, nodename, height)
    return _setheight!(tree, nodename, height)
end
""" """ function src end
function src(branch::AbstractBranch)
    return _src(branch)
end
function src(tree::AbstractTree, branchname)
    return _src(_getbranch(tree, branchname))
end
""" """ function dst end
function dst(branch::AbstractBranch)
    return _dst(branch)
end
function dst(tree::AbstractTree, branchname)
    return _dst(_getbranch(tree, branchname))
end
""" """ function Pair end
function Pair(branch::AbstractBranch)
    return Pair(src(branch), dst(branch))
end
function Pair(tree::AbstractTree, branchname)
    return Pair(_getbranch(tree, branchname))
end
""" """ function Tuple end
function Tuple(branch::AbstractBranch)
    return (src(branch), dst(branch))
end
function Tuple(tree::AbstractTree, branchname)
    return Tuple(_getbranch(tree, branchname))
end
""" """ function getlength end
function getlength(branch::AbstractBranch)
    return _getlength(branch)
end
function getlength(tree::AbstractTree, branchname)
    return _getlength(_getbranch(tree, branchname))
end
""" """ function changesrc!(tree::AbstractTree, branchname, source)
    _hasbranch(tree, branchname) ||
        error("Branch $branchname does not exist")
    _hasnode(tree, source) ||
        error("Node $source does not exist")
    branch = _getbranch(tree, branchname)
    oldsource = _src(branch)
    _setsrc!(branch, source)
    _deleteoutbound!(_getnode(tree, oldsource), branchname)
    _addoutbound!(_getnode(tree, source), branchname)
    return branchname
end
""" """ function changedst!(tree::AbstractTree, branchname, destination)
    _hasbranch(tree, branchname) ||
        error("Branch $branchname does not exist")
    _hasnode(tree, destination) ||
        error("Node $destination does not exist")
    branch = _getbranch(tree, branchname)
    olddestination = _dst(branch)
    _setdst!(branch, destination)
    _deleteinbound!(_getnode(tree, olddestination), branchname)
    _setinbound!(_getnode(tree, destination), branchname)
    return branchname
end
""" """ function resetleaves!(tree::AbstractTree)
    return _resetleaves!(tree)
end
""" """ function getleafnames(tree::AbstractTree)
    return collect(_getleafnames(tree))
end
""" """ function getleafinfo(tree::AbstractTree, label)
    return _getleafinfo(tree, label)
end
function getleafinfo(tree::AbstractTree)
    return _getleafinfo(tree)
end
function leafinfotype(tree::AbstractTree)
    return _leafinfotype(tree)
end
""" """ function setleafinfo!(tree::AbstractTree, table)
    return _setleafinfo!(tree, table)
end
""" """ function getnoderecord(tree::AbstractTree, label)
    return _getnoderecord(tree, label)
end
""" """ function setnoderecord!(tree::AbstractTree, label, value)
    return _setnoderecord!(tree, label, value)
end
""" """ nleaves(tree::AbstractTree) = _nleaves(tree)
