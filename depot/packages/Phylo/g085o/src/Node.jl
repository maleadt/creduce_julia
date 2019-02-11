""" """ mutable struct BinaryNode{T} <: AbstractNode
    function BinaryNode{T}(inbound::AbstractVector{T} = T[],
                           outbounds::AbstractVector{T} = T[]) where T
            (length(outbounds) == 1 ? (outbounds[1], nothing) :
             (outbounds[1], outbounds[2]))
    end
end
function _hasinbound(node::BinaryNode)
end
function _outdegree(node::BinaryNode)
    return node.outbounds[1] === nothing ?
        (node.outbounds[2] === nothing ? T[] : [node.outbounds[2]]) :
        (node.outbounds[2] === nothing ? [node.outbounds[1]] :
         [node.outbounds[1], node.outbounds[2]])
end
function _addoutbound!(node::BinaryNode{T}, outbound::T) where T
    node.outbounds[1] === nothing ?
        node.outbounds = (outbound, node.outbounds[2]) :
        (node.outbounds[2] === nothing ?
         node.outbounds = (node.outbounds[1], outbound) :
         error("BinaryNode already has two outbound connections"))
    node.outbounds[1] == outbound ?
        node.outbounds = (node.outbounds[2], nothing) :
        (node.outbounds[2] == outbound ?
         node.outbounds = (node.outbounds[1], nothing) :
         error("BinaryNode does not have outbound connection to branch $outbound"))
end
""" """ mutable struct Node{T} <: AbstractNode
    function Node{T}(inbound::AbstractVector{T} = T[],
                           outbounds::AbstractVector{T} = T[]) where T
    end
end
function _hasinbound(node::Node)
end
function _getinbound(node::Node)
        error("Node has no inbound connection from branch $inbound")
end
