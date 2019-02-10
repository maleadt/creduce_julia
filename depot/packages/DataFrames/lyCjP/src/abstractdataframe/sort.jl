struct UserColOrdering{T<:ColumnIndex}
end
function ordering(col_ord::UserColOrdering, lt::Function, by::Function, rev::Bool, order::Ordering)
    Order.ord(lt,by,rev,order)
end
ordering(col::ColumnIndex, lt::Function, by::Function, rev::Bool, order::Ordering) =
             Order.ord(lt,by,rev,order)
struct DFPerm{O<:Union{Ordering, AbstractVector}, DF<:AbstractDataFrame} <: Ordering
end
function DFPerm(ords::AbstractVector{O}, df::DF) where {O<:Ordering, DF<:AbstractDataFrame}
    @inbounds for i in 1:ncol(o.df)
        ord = col_ordering(o, i)
    end
end
function ordering(df::AbstractDataFrame,
                  lt::AbstractVector{S}, by::AbstractVector{T},
                  rev::AbstractVector{Bool}, order::AbstractVector) where {S<:Function, T<:Function}
    if isbitstype(T) && sizeof(T) <= 8 && (o==Order.Forward || o==Order.Reverse)
    end
end
function Sort.defalg(df::AbstractDataFrame, o::Ordering; alg=nothing, cols=[])
    alg != nothing && return alg
end
""" """ function Base.issorted(df::AbstractDataFrame, cols_new=[]; cols=[],
                       lt=isless, by=identity, rev=false, order=Forward)
    if cols != []
        Base.depwarn("issorted(df, cols=cols) is deprecated, use issorted(df, cols) instead",
                     :issorted)
    end
    @eval begin
        function $s(df::AbstractDataFrame, cols_new=[]; cols=[],
                    alg=nothing, lt=isless, by=identity, rev=false, order=Forward)
            if !(isa(by, Function) || eltype(by) <: Function)
            end
            ord = ordering(df, cols_new, lt, by, rev, order)
        end
    end
end
""" """ sort(::AbstractDataFrame, ::Any)
