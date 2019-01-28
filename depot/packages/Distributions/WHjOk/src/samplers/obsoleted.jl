struct DiscreteDistributionTable <: Sampler{Univariate,Discrete}
    table::Vector{Vector{Int64}}
    bounds::Vector{Int64}
end
function DiscreteDistributionTable(probs::Vector{T}) where T <: Real
    n = length(probs)
    vals = Vector{Int64}(undef, n)
    for i in 1:n
        vals[i] = round(Int, probs[i] * 64^9)
    end
    table = Vector{Vector{Int64}}(undef, 9)
    bounds = zeros(Int64, 9)
    for i in 1:n
        if vals[i] == 64^9
            table[1] = Vector{Int64}(undef, 64)
            for j in 1:64
                table[1][j] = i
            end
            bounds[1] = 64^9
            for j in 2:9
                table[j] = Vector{Int64}()
                bounds[j] = 64^9
            end
            return DiscreteDistributionTable(table, bounds)
        end
    end
    multiplier = 1
    for index in 9:-1:1
        counts = Vector{Int64}()
        for i in 1:n
            digit = mod(vals[i], 64)
            vals[i] >>= 6
            bounds[index] += digit
            for itr in 1:digit
                push!(counts, i)
            end
        end
        bounds[index] *= multiplier
        table[index] = counts
        multiplier <<= 6
    end
    bounds = cumsum(bounds)
    return DiscreteDistributionTable(table, bounds)
end
function rand(table::DiscreteDistributionTable)
    i = rand(1:(64^9 - 1))
    bound = 1
    while i > table.bounds[bound] && bound < 9
        bound += 1
    end
    if bound > 1
        index = fld(i - table.bounds[bound - 1] - 1, 64^(9 - bound)) + 1
    else
        index = fld(i - 1, 64^(9 - bound)) + 1
    end
    return table.table[bound][index]
end
Base.show(io::IO, table::DiscreteDistributionTable) = @printf io "DiscreteDistributionTable"
abstract type HuffmanNode{T} <: Sampler{Univariate,Discrete} end
struct HuffmanLeaf{T} <: HuffmanNode{T}
    value::T
    weight::UInt64
end
struct HuffmanBranch{T} <: HuffmanNode{T}
    left::HuffmanNode{T}
    right::HuffmanNode{T}
    weight::UInt64
end
HuffmanBranch(ha::HuffmanNode{T},hb::HuffmanNode{T}) where {T} = HuffmanBranch(ha, hb, ha.weight + hb.weight)
Base.isless(ha::HuffmanNode{T}, hb::HuffmanNode{T}) where {T} = isless(ha.weight,hb.weight)
Base.show(io::IO, t::HuffmanNode{T}) where {T} = show(io,typeof(t))
function Base.getindex(h::HuffmanBranch{T},u::UInt64) where T
    while isa(h,HuffmanBranch{T})
        if u < h.left.weight
            h = h.left
        else
            u -= h.left.weight
            h = h.right
        end
    end
    h.value
end
function huffman(values::AbstractVector{T},weights::AbstractVector{UInt64}) where T
    leafs = [HuffmanLeaf{T}(values[i],weights[i]) for i = 1:length(weights)]
    sort!(leafs; rev=true)
    branches = Vector{HuffmanBranch{T}}()
    while !isempty(leafs) || length(branches) > 1
        left = isempty(branches) || (!isempty(leafs) && first(leafs) < first(branches)) ? pop!(leafs) : pop!(branches)
        right = isempty(branches) || (!isempty(leafs) && first(leafs) < first(branches)) ? pop!(leafs) : pop!(branches)
        pushfirst!(branches,HuffmanBranch(left,right))
    end
    pop!(branches)
end
function rand(h::HuffmanNode{T}) where T
    w = h.weight
    u = rand(UInt64)
    if (w & (w-1)) == 0
        u = u & (w-1)
    else
        m = typemax(UInt64)
        lim = m - (rem(m,w)+1)
        while u > lim
            u = rand(UInt64)
        end
        u = rem(u,w)
    end
    h[u]
end
