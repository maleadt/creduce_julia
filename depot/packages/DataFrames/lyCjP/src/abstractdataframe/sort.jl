struct UserColOrdering{T<:ColumnIndex}
    col::T
    kwargs
end
order(col::T; kwargs...) where {T<:ColumnIndex} = UserColOrdering{T}(col, kwargs)
_getcol(o::UserColOrdering) = o.col
_getcol(x) = x
function ordering(col_ord::UserColOrdering, lt::Function, by::Function, rev::Bool, order::Ordering)
    for (k,v) in pairs(col_ord.kwargs)
        if     k == :lt;    lt    = v
        elseif k == :by;    by    = v
        elseif k == :rev;   rev   = v
        elseif k == :order; order = v
        else
            error("Unknown keyword argument: ", string(k))
        end
    end
    Order.ord(lt,by,rev,order)
end
ordering(col::ColumnIndex, lt::Function, by::Function, rev::Bool, order::Ordering) =
             Order.ord(lt,by,rev,order)
struct DFPerm{O<:Union{Ordering, AbstractVector}, DF<:AbstractDataFrame} <: Ordering
    ord::O
    df::DF
end
function DFPerm(ords::AbstractVector{O}, df::DF) where {O<:Ordering, DF<:AbstractDataFrame}
    if length(ords) != ncol(df)
        error("DFPerm: number of column orderings does not equal the number of DataFrame columns")
    end
    DFPerm{typeof(ords), DF}(ords, df)
end
DFPerm(o::O, df::DF) where {O<:Ordering, DF<:AbstractDataFrame} = DFPerm{O,DF}(o,df)
col_ordering(o::DFPerm{O}, i::Int) where {O<:Ordering} = o.ord
col_ordering(o::DFPerm{V}, i::Int) where {V<:AbstractVector} = o.ord[i]
function Sort.lt(o::DFPerm, a, b)
    @inbounds for i in 1:ncol(o.df)
        ord = col_ordering(o, i)
        va = o.df[a, i]
        vb = o.df[b, i]
        lt(ord, va, vb) && return true
        lt(ord, vb, va) && return false
    end
    false # a and b are equal
end
ordering(df::AbstractDataFrame, lt::Function, by::Function, rev::Bool, order::Ordering) =
    DFPerm(Order.ord(lt, by, rev, order), df)
function ordering(df::AbstractDataFrame,
                  lt::AbstractVector{S}, by::AbstractVector{T},
                  rev::AbstractVector{Bool}, order::AbstractVector) where {S<:Function, T<:Function}
    if !(length(lt) == length(by) == length(rev) == length(order) == size(df,2))
        throw(ArgumentError("Orderings must be specified for all DataFrame columns"))
    end
    DFPerm([Order.ord(_lt, _by, _rev, _order) for (_lt, _by, _rev, _order) in zip(lt, by, rev, order)], df)
end
ordering(df::AbstractDataFrame, col::ColumnIndex, lt::Function, by::Function, rev::Bool, order::Ordering) =
    Perm(Order.ord(lt, by, rev, order), df[col])
ordering(df::AbstractDataFrame, col_ord::UserColOrdering, lt::Function, by::Function, rev::Bool, order::Ordering) =
    Perm(ordering(col_ord, lt, by, rev, order), df[col_ord.col])
function ordering(df::AbstractDataFrame, cols::AbstractVector, lt::Function, by::Function, rev::Bool, order::Ordering)
    if length(cols) == 0
        return ordering(df, lt, by, rev, order)
    end
    if length(cols) == 1
        return ordering(df, cols[1], lt, by, rev, order)
    end
    ords = Ordering[]
    newcols = Int[]
    for col in cols
        push!(ords, ordering(col, lt, by, rev, order))
        push!(newcols, index(df)[(_getcol(col))])
    end
    if all([ords[i] == ords[1] for i = 2:length(ords)])
        return DFPerm(ords[1], df[newcols])
    end
    return DFPerm(ords, df[newcols])
end
function ordering(df::AbstractDataFrame, cols::AbstractVector,
                  lt::AbstractVector{S}, by::AbstractVector{T},
                  rev::AbstractVector{Bool}, order::AbstractVector) where {S<:Function, T<:Function}
    if !(length(lt) == length(by) == length(rev) == length(order))
        throw(ArgumentError("All ordering arguments must be 1 or the same length."))
    end
    if length(cols) == 0
        return ordering(df, lt, by, rev, order)
    end
    if length(lt) != length(cols)
        throw(ArgumentError("All ordering arguments must be 1 or the same length as the number of columns requested."))
    end
    if length(cols) == 1
        return ordering(df, cols[1], lt[1], by[1], rev[1], order[1])
    end
    ords = Ordering[]
    newcols = Int[]
    for i in 1:length(cols)
        push!(ords, ordering(cols[i], lt[i], by[i], rev[i], order[i]))
        push!(newcols, index(df)[(_getcol(cols[i]))])
    end
    if all([ords[i] == ords[1] for i = 2:length(ords)])
        return DFPerm(ords[1], df[newcols])
    end
    return DFPerm(ords, df[newcols])
end
function ordering(df::AbstractDataFrame, cols::AbstractVector, lt, by, rev, order)
    to_array(src::AbstractVector, dims) = src
    to_array(src::Tuple, dims) = [src...]
    to_array(src, dims) = fill(src, dims)
    dims = length(cols) > 0 ? length(cols) : size(df,2)
    ordering(df, cols,
             to_array(lt, dims),
             to_array(by, dims),
             to_array(rev, dims),
             to_array(order, dims))
end
ordering(df::AbstractDataFrame, cols::Tuple, args...) = ordering(df, [cols...], args...)
Sort.defalg(df::AbstractDataFrame) = size(df, 1) < 8192 ? Sort.MergeSort : SortingAlgorithms.TimSort
function Sort.defalg(df::AbstractDataFrame, ::Type{T}, o::Ordering) where T<:Real
    if isbitstype(T) && sizeof(T) <= 8 && (o==Order.Forward || o==Order.Reverse)
        SortingAlgorithms.RadixSort
    else
        Sort.defalg(df)
    end
end
Sort.defalg(df::AbstractDataFrame,        ::Type,            o::Ordering) = Sort.defalg(df)
Sort.defalg(df::AbstractDataFrame, col    ::ColumnIndex,     o::Ordering) = Sort.defalg(df, eltype(df[col]), o)
Sort.defalg(df::AbstractDataFrame, col_ord::UserColOrdering, o::Ordering) = Sort.defalg(df, col_ord.col, o)
Sort.defalg(df::AbstractDataFrame, cols,                     o::Ordering) = Sort.defalg(df)
function Sort.defalg(df::AbstractDataFrame, o::Ordering; alg=nothing, cols=[])
    alg != nothing && return alg
    Sort.defalg(df, cols, o)
end
""" """ function Base.issorted(df::AbstractDataFrame, cols_new=[]; cols=[],
                       lt=isless, by=identity, rev=false, order=Forward)
    if cols != []
        Base.depwarn("issorted(df, cols=cols) is deprecated, use issorted(df, cols) instead",
                     :issorted)
        cols_new = cols
    end
    if cols_new isa ColumnIndex
        issorted(df[cols_new], lt=lt, by=by, rev=rev, order=order)
    elseif length(cols_new) == 1
        issorted(df[cols_new[1]], lt=lt, by=by, rev=rev, order=order)
    else
        issorted(1:nrow(df), ordering(df, cols_new, lt, by, rev, order))
    end
end
for s in [:(Base.sort), :(Base.sortperm)]
    @eval begin
        function $s(df::AbstractDataFrame, cols_new=[]; cols=[],
                    alg=nothing, lt=isless, by=identity, rev=false, order=Forward)
            if !(isa(by, Function) || eltype(by) <: Function)
                msg = "'by' must be a Function or a vector of Functions. Perhaps you wanted 'cols'."
                throw(ArgumentError(msg))
            end
            if cols != []
                fname = $s
                Base.depwarn("$fname(df, cols=cols) is deprecated, use $fname(df, cols) instead",
                             Symbol($s))
                cols_new = cols
            end
            ord = ordering(df, cols_new, lt, by, rev, order)
            _alg = Sort.defalg(df, ord; alg=alg, cols=cols_new)
            $s(df, _alg, ord)
        end
    end
end
""" """ sort(::AbstractDataFrame, ::Any)
""" """ sortperm(::AbstractDataFrame, ::Any)
Base.sort(df::AbstractDataFrame, a::Algorithm, o::Ordering) = df[sortperm(df, a, o),:]
Base.sortperm(df::AbstractDataFrame, a::Algorithm, o::Union{Perm,DFPerm}) = sort!([1:size(df, 1);], a, o)
Base.sortperm(df::AbstractDataFrame, a::Algorithm, o::Ordering) = sortperm(df, a, DFPerm(o,df))
