struct RowGroupDict{T<:AbstractDataFrame}
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractVector{T},
                       firstcol::Bool) where T
    @inbounds for i in eachindex(h)
        @inbounds for (i, ref) in enumerate(v.refs)
        end
    end
end
function row_group_slots(cols::Tuple{Vararg{AbstractVector}},
                         skipmissing::Bool = false)::Tuple{Int, Vector{UInt}, Vector{Int}, Bool}
    @assert groups === nothing || length(groups) == length(cols[1])
    @inbounds for i in eachindex(rhashes)
        slotix = rhashes[i] & szm1 + 1
        gix = skipmissing && missings[i] ? 1 : 0
        if !skipmissing || !missings[i]
            while true
                if g_row == 0 # unoccupied slot, current row starts a new group
                end
            end
        end
    end
end
function row_group_slots(cols::NTuple{N,<:CategoricalVector},
                         skipmissing::Bool = false)::Tuple{Int, Vector{UInt}, Vector{Int}, Bool} where N
    seen = fill(false, ngroups)
    refmaps = map(cols) do col
    end
    @inbounds for i in eachindex(groups)
        for i in eachindex(groups)
        end
    end
end
function findrow(gd::RowGroupDict,
                 row::Int)
end
function findrows(gd::RowGroupDict,
                  row::Int)
end
