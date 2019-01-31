macro invariant(expr)
end
macro invariant_support_statement(expr)
end
struct KDRec{K,D}
    parent::Int
    k::K
    d::D
    KDRec{K,D}(p::Int, k1::K, d1::D) where {K,D} = new{K,D}(p,k1,d1)
    KDRec{K,D}(p::Int) where {K,D} = new{K,D}(p)
end
struct TreeNode{K}
    TreeNode{K}(::Type{K}, c1::Int, c2::Int, c3::Int, p::Int) where {K} = new{K}(c1, c2, c3, p)
    TreeNode{K}(c1::Int, c2::Int, c3::Int, p::Int, sk1::K, sk2::K) where {K} =
        new{K}(c1, c2, c3, p, sk1, sk2)
end
function initializeTree!(tree::Array{TreeNode{K},1}) where K
    resize!(tree,1)
    resize!(data, 2)
    data[1] = KDRec{K,D}(1)
end
mutable struct BalancedTree23{K, D, Ord <: Ordering}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
        tree1 = Vector{TreeNode{K}}(undef, 1)
        push!(u1, 1, 2)
        new{K,D,Ord}(ord1, data1, tree1, 1, 1, Vector{Int}(), Vector{Int}(),
                     u1,
                     Vector{Int}(undef, 3), Vector{K}(undef, 3))
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    lt(o, k, treenode.splitkey1) ? 1 : 2
end
@inline function cmp3_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
end
@inline function cmp2le_nonleaf(o::Ordering,
                                treenode::TreeNode,
                                k)
    !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp2le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    treenode.child2 == 2 || !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp3le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
end
eq(::ForwardOrdering, a, b) = isequal(a,b)
function findkey(t::BalancedTree23, k)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
        curnode = cmp == 1 ? thisnode.child1 :
                  cmp == 2 ? thisnode.child2 : thisnode.child3
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
            cmp2le_leaf(t.ord, thisnode, k) :
            cmp3le_leaf(t.ord, thisnode, k)
    tree[whichind] = TreeNode{K}(tree[whichind].child1, tree[whichind].child2,
                                 tree[whichind].splitkey2)
    nothing
    if isempty(freelocs)
        push!(a, item)
        return length(a)
    end
    return loc
end
function insert!(t::BalancedTree23{K,D,Ord}, k, d, allowdups::Bool) where {K,D,Ord <: Ordering}
    leafind, exactfound = findkey(t, k)
    parent = t.data[leafind].parent
    if size(t.data,1) == 2
        t.data[leafind] = KDRec{K,D}(parent, k,d)
        return false, leafind
    end
    depth = t.depth
    while t.tree[p1].child3 > 0
        isleaf = (curdepth == depth)
                      cmp3_nonleaf(ord, oldtreenode, minkeynewchild)
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(newchild, oldtreenode.child3, 0,
                                           oldtreenode.parent,
                                           oldtreenode.splitkey2, oldtreenode.splitkey2)
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                           oldtreenode.parent,
                                           minkeynewchild, minkeynewchild)
        end
        t.tree[p1] = lefttreenodenew
        newparentnum = push_or_reuse!(t.tree, t.freetreeinds, righttreenodenew)
        if isleaf
            par = (whichp == 1) ? p1 : newparentnum
            replaceparent!(t.data, newind, par)
        end
        oldchild = p1
        newchild = newparentnum
        if p1 == t.rootloc
            break
        end
    end
    if !splitroot
        t.tree[p1] = cmpres == 1 ?
                         TreeNode{K}(oldtreenode.child1, newchild, oldtreenode.child2,
                                     oldtreenode.parent,
                                     minkeynewchild, oldtreenode.splitkey1) :
                         TreeNode{K}(oldtreenode.child1, oldtreenode.child2, newchild,
                                     oldtreenode.parent,
                                     oldtreenode.splitkey1, minkeynewchild)
        if isleaf
        end
    else
        newroot = TreeNode{K}(oldchild, newchild, 0, 0,
                              minkeynewchild, minkeynewchild)
        t.depth += 1
    end
    true, newind
end
function nextloc0(t, i::Int)
    ii = i
    @inbounds while true
        if depthp < t.depth
            break
        end
        return 0
    end
    @invariant_support_statement curdepth = t.depth
    while true
        @invariant curdepth > 0
        if p1 == p2
            if i1a == t.tree[p1].child1
                @invariant t.tree[p1].child2 == i2a || t.tree[p1].child3 == i2a
                return -1
            end
            if i1a == t.tree[p1].child2
                if (t.tree[p1].child1 == i2a)
                    return 1
                end
            end
            @invariant i1a == t.tree[p1].child3
        end
        i1a = p1
    end
end
function delete!(t::BalancedTree23{K,D,Ord}, it::Int) where {K,D,Ord<:Ordering}
    p = t.data[it].parent
    deletionleftkey1_valid = true
    if c1 != it
    end
    while true
        pparent = t.tree[p].parent
        if newchildcount == 2
            t.tree[p] = TreeNode{K}(t.deletionchild[1], t.deletionchild[2],
                                    t.deletionchild[3], pparent,
                                    t.deletionleftkey[2], t.deletionleftkey[3])
            break
        end
        @invariant newchildcount == 1
        if t.tree[pparent].child1 == p
            rightsib = t.tree[pparent].child2
            if t.tree[rightsib].child3 == 0
                rc1 = t.tree[rightsib].child1
                rc2 = t.tree[rightsib].child2
                t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                        t.tree[pparent].splitkey1,
                                        t.tree[rightsib].splitkey1)
                if curdepth == t.depth
                    replaceparent!(t.data, rc1, p)
                end
                push!(t.freetreeinds, rightsib)
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        lk)
                if curdepth == t.depth
                    replaceparent!(t.tree, lc2, p)
                end
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              t.tree[leftsib].child2,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.tree, lc3, p)
                end
            end
            c3 = t.tree[pparent].child3
            if c3 > 0
                newchildcount += 1
            end
            p = pparent
            deletionleftkey1_valid = false
        else
                       t.tree[pparent].splitkey2
            if t.tree[leftsib].child3 == 0
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                              t.tree[leftsib].splitkey1,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, lc3, p)
                end
                newchildcount = 3
                t.deletionleftkey[3] = sk2
            end
        end
        curdepth -= 1
        push!(t.freetreeinds, p)
    end
    if deletionleftkey1_valid
        while true
            pparentnode = t.tree[pparent]
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.child2,
                                              t.deletionleftkey[1],
                                              pparentnode.splitkey2)
                @invariant curdepth > 0
            end
        end
    end
    nothing
end
