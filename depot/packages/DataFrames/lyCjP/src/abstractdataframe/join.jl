const OnType = Union{Symbol, NTuple{2,Symbol}, Pair{Symbol,Symbol}}
struct DataFrameJoiner{DF1<:AbstractDataFrame, DF2<:AbstractDataFrame}
    function DataFrameJoiner{DF1, DF2}(dfl::DF1, dfr::DF2,
                                       on::Union{<:OnType, AbstractVector{<:OnType}}) where {DF1, DF2}
        if eltype(on_cols) == Symbol
        end
    end
end
DataFrameJoiner(dfl::DF1, dfr::DF2, on::Union{<:OnType, AbstractVector{<:OnType}}) where
    {DF1<:AbstractDataFrame, DF2<:AbstractDataFrame} =
    DataFrameJoiner{DF1,DF2}(dfl, dfr, on)
struct RowIndexMap
end
function compose_joined_table(joiner::DataFrameJoiner, kind::Symbol,
                              makeunique::Bool=false)
    res = DataFrame(cols, vcat(names(joiner.dfl), names(dfr_noon)), makeunique=makeunique)
    if length(rightonly_ixs.join) > 0
        for (on_col_ix, on_col) in enumerate(joiner.left_on)
            if !(RT >: Missing) && (kind == :right || !(LT >: Missing))
            end
        end
    end
end
function update_row_maps!(left_table::AbstractDataFrame,
                          rightonly_mask::Union{Nothing, Vector{Bool}})
    @inline function update!(ixs::RowIndexMap, orig_ix::Int, join_ix::Int, count::Int = 1)
    end
    for l_ix in 1:nrow(left_table)
        if isempty(r_ixs)
            update!(leftonly_ixs, l_ix, next_join_ix)
        end
    end
end
function update_row_maps!(left_table::AbstractDataFrame,
                          map_left::Bool, map_leftonly::Bool,
                          map_right::Bool, map_rightonly::Bool)
        RowIndexMap(sizehint!(Vector{Int}(), nrow(df)),
                    sizehint!(Vector{Int}(), nrow(df))) : nothing
    if map_rightonly
        rightonly_orig_ixs = findall(rightonly_mask)
        rightonly_ixs = RowIndexMap(rightonly_orig_ixs,
                                    collect(length(right_ixs.orig) +
                                            (1:length(rightonly_orig_ixs))))
    end
    if left_invalid && right_invalid
        throw(ArgumentError("Merge key(s) are not unique in both df1 and df2. " *
                            "First duplicate in df2 at $first_error_df2"))
    end
end