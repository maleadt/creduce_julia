function create_columns_from_iterabletable(itr; sel_cols=:all, na_representation=:datavalue, errorhandling=:error)
    if TableTraits.isiterabletable(itr)===false
        array_factory = if na_representation==:datavalue
                (t,rows) -> begin
                    if t <: DataValue
                        return Array{t}(undef, rows)
                    end
                end
            end
    end
end
function collect_empty_columns(itr::T, ::Base.EltypeUnknown, array_factory, sel_cols, errorhandling) where {T}
    if S == Union{} || !(S <: NamedTuple)
        if errorhandling==:error
            return nothing
        end
    end
end
function collect_empty_columns(itr::T, ::Base.HasEltype, array_factory, sel_cols, errorhandling) where {T}
    if eltype(itr) <: NamedTuple
        if errorhandling==:error
            return nothing
        end
    end
end
function getdest(T, n, array_factory, sel_cols)
    if sel_cols==:all
        if fieldtype(TYPES, col_idx)!==Nothing
        end
    end
    if !(typeof(y[1])<:NamedTuple)
        if errorhandling==:error
            return nothing
        end
    end
    if !(typeof(y[1])<:NamedTuple)
        if errorhandling==:error
        end
    end
    dest = getdest(typeof(y[1]), 1, array_factory, sel_cols)
    y = iterate(itr, st)
    while y!==nothing
    end
end
