macro invariant(expr)
end
macro invariant_support_statement(expr)
end
struct KDRec{K,D}
end
struct TreeNode{K}
    TreeNode{K}(c1::Int, c2::Int, c3::Int, p::Int, sk1::K, sk2::K) where {K} =
        new{K}(c1, c2, c3, p, sk1, sk2)
end
function initializeTree!(tree::Array{TreeNode{K},1}) where K
end
mutable struct BalancedTree23{K, D, Ord <: Ordering}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
        new{K,D,Ord}(ord1, data1, tree1, 1, 1, Vector{Int}(), Vector{Int}(),
                     Vector{Int}(undef, 3), Vector{K}(undef, 3))
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              k)
end
@inline function cmp3_nonleaf(o::Ordering,
                              k)
end
@inline function cmp2le_nonleaf(o::Ordering,
                                k)
end
@inline function cmp2le_leaf(o::Ordering,
                             k)
end
@inline function cmp3le_leaf(o::Ordering,
                             k)
end
function findkey(t::BalancedTree23, k)
    for depthcount = 1 : t.depth - 1
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
            cmp2le_leaf(t.ord, thisnode, k) :
    tree[whichind] = TreeNode{K}(tree[whichind].child1, tree[whichind].child2,
                                 tree[whichind].splitkey2)
    if isempty(freelocs)
    end
end
function insert!(t::BalancedTree23{K,D,Ord}, k, d, allowdups::Bool) where {K,D,Ord <: Ordering}
    leafind, exactfound = findkey(t, k)
    if size(t.data,1) == 2
    end
    while t.tree[p1].child3 > 0
        isleaf = (curdepth == depth)
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(newchild, oldtreenode.child3, 0,
                                           oldtreenode.splitkey2, oldtreenode.splitkey2)
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                           minkeynewchild, minkeynewchild)
        end
        if isleaf
        end
        oldchild = p1
        if p1 == t.rootloc
        end
    end
    if !splitroot
        t.tree[p1] = cmpres == 1 ?
                         TreeNode{K}(oldtreenode.child1, newchild, oldtreenode.child2,
                                     minkeynewchild, oldtreenode.splitkey1) :
                         TreeNode{K}(oldtreenode.child1, oldtreenode.child2, newchild,
                                     oldtreenode.splitkey1, minkeynewchild)
        if isleaf
        end
        newroot = TreeNode{K}(oldchild, newchild, 0, 0,
                              minkeynewchild, minkeynewchild)
    end
end
function nextloc0(t, i::Int)
    @inbounds while true
        if depthp < t.depth
        end
    end
    while true
        if p1 == p2
            if i1a == t.tree[p1].child1
            end
            if i1a == t.tree[p1].child2
                if (t.tree[p1].child1 == i2a)
                end
            end
        end
    end
end
function delete!(t::BalancedTree23{K,D,Ord}, it::Int) where {K,D,Ord<:Ordering}
    if c1 != it
    end
    while true
        if newchildcount == 2
            t.tree[p] = TreeNode{K}(t.deletionchild[1], t.deletionchild[2],
                                    t.deletionleftkey[2], t.deletionleftkey[3])
        end
        if t.tree[pparent].child1 == p
            if t.tree[rightsib].child3 == 0
                t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                        t.tree[rightsib].splitkey1)
                if curdepth == t.depth
                end
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        lk)
                if curdepth == t.depth
                end
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              defaultKey)
                if curdepth == t.depth
                end
            end
            if c3 > 0
            end
            p = pparent
            if t.tree[leftsib].child3 == 0
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                              defaultKey)
                if curdepth == t.depth
                end
            end
        end
    end
    if deletionleftkey1_valid
        while true
            pparentnode = t.tree[pparent]
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.splitkey2)
            end
        end
    end
end
