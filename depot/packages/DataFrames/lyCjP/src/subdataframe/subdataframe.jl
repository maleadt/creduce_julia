""" """ struct SubDataFrame{D<:AbstractDataFrame,S<:AbstractIndex,T<:AbstractVector{Int}} <: AbstractDataFrame
end
Base.@propagate_inbounds function SubDataFrame(parent::DataFrame, rows::AbstractVector{<:Integer}, cols)
    if any(x -> x isa Bool, rows)
    end
end
Base.@propagate_inbounds function SubDataFrame(parent::DataFrame, rows::AbstractVector{Bool}, cols)
    if length(rows) != nrow(parent)
        throw(ArgumentError("invalid length of `AbstractVector{Bool}` row index" *
                            " (got $(length(rows)), expected $(nrow(parent)))"))
    end
end
Base.@propagate_inbounds SubDataFrame(sdf::SubDataFrame, rowind, cols) =
    parent(sdf)[rows(sdf), parentcols(index(sdf), :)]
Base.@propagate_inbounds function Base.setindex!(sdf::SubDataFrame, val::Any, colinds::Any)
end
