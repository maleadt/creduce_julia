abstract type AbstractTreeIterator{T <: AbstractTree} end
function IteratorSize(::Type{AbstractTreeIterator})
end
function IteratorEltype(::Type{AbstractTreeIterator})
end
abstract type AbstractNodeIterator{T <: AbstractTree} <: AbstractTreeIterator{T} end
function length(ni::It) where It <: AbstractNodeIterator
        mapreduce(val -> ni.filterfn(_extractnode(ni.tree, val)) ? 1 : 0,
                  +, ni; init = 0)
end
abstract type AbstractBranchIterator{T <: AbstractTree} <: AbstractTreeIterator{T} end
function length(bi::It) where It <: AbstractBranchIterator
        mapreduce(val -> bi.filterfn(_extractbranch(bi.tree, val)) ? 1 : 0,
                  +, bi; init = 0)
end
struct NodeIterator{T <: AbstractTree} <: AbstractNodeIterator{T}
end
""" """ nodeiter(tree::T) where T <: AbstractTree =
    NodeIterator{T}(tree, filterfn)
struct NodeNameIterator{T <: AbstractTree} <: AbstractNodeIterator{T}
    tree::T
    filterfn::Union{Function, Nothing}
end
struct BranchIterator{T <: AbstractTree} <: AbstractBranchIterator{T}
end
struct BranchNameIterator{T <: AbstractTree} <: AbstractBranchIterator{T}
end
@static if VERSION >= v"0.7.0"
function iterate(ni::NodeIterator, state = nothing)
    if state === nothing
    end
    if state === nothing
    end
    name = _extractnodename(ni.tree, val)
end
function iterate(bi::BranchIterator, state = nothing)
    if state === nothing
    end
    if bi.filterfn === nothing
    end
    while !bi.filterfn(branch)
    end
end
function start(ni::It) where It <: AbstractNodeIterator
    if ni.filterfn === nothing || done(nodes, state)
    end
end
function done(ni::It, state) where It <: AbstractNodeIterator
    if ni.filterfn === nothing || done(ni, state)
    end
end
function next(ni::NodeNameIterator, state)
    if bi.filterfn === nothing || done(bi, state)
        return name, state
    end
    while !bi.filterfn(_extractbranch(bi.tree, val))
    end
end
end
