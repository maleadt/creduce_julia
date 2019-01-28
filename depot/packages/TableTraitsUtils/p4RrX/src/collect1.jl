# Main entry point
function create_columns_from_iterabletable(itr; sel_cols=:all, na_representation=:datavalue, errorhandling=:error)
    in(errorhandling, (:error, :returnvalue)) || throw(ArgumentError("'$errorhandling' is not a valid argument for errorhandling."))
    in(na_representation, (:datavalue, :missing)) || throw(ArgumentError("'$na_representation' is not a valid argument for na_representation."))

    if TableTraits.isiterabletable(itr)===false
        if errorhandling==:error
            throw(ArgumentError("itr is not a table."))
        elseif errorhandling==:returnvalue            
            return nothing
        end
    else

        array_factory = if na_representation==:datavalue
                (t,rows) -> Array{t}(undef, rows)
            elseif na_representation==:missing
                (t,rows) -> begin
                    if t <: DataValue
                        return Array{Union{eltype(t),Missing}}(undef, rows)
                    else
                        return Array{t}(undef, rows)
                    end
                end
            end

        itr2 = IteratorInterfaceExtensions.getiterator(itr)
        return _collect_columns(itr2, Base.IteratorSize(itr2), array_factory, sel_cols, errorhandling)
    end
end

function collect_empty_columns(itr::T, ::Base.EltypeUnknown, array_factory, sel_cols, errorhandling) where {T}
    S = Core.Compiler.return_type(first, Tuple{T})
    if S == Union{} || !(S <: NamedTuple)
        if errorhandling==:error
            throw(ArgumentError("itr is not a table."))
        elseif errorhandling==:returnvalue            
            return nothing
        end
    end
    dest = getdest(S,0, array_factory, sel_cols)
    return collect(values(dest)), collect(keys(dest))
end

function collect_empty_columns(itr::T, ::Base.HasEltype, array_factory, sel_cols, errorhandling) where {T}
    if eltype(itr) <: NamedTuple
        dest = getdest(eltype(itr),0, array_factory, sel_cols)
        return collect(values(dest)), collect(keys(dest))
    else
        if errorhandling==:error
            throw(ArgumentError("itr is not a table."))
        elseif errorhandling==:returnvalue            
            return nothing
        end
    end
end

function getdest(T, n, array_factory, sel_cols)
    if sel_cols==:all
        return NamedTuple{fieldnames(T)}(tuple((array_factory(fieldtype(T,i),n) for i in 1:length(fieldnames(T)))...))
    else
        return NamedTuple{fieldnames(T)}(tuple((i in sel_cols ? array_factory(fieldtype(T,i),n) : nothing for i in 1:length(fieldnames(T)))...))
    end
end

@generated function _setrow(dest::NamedTuple{NAMES,TYPES}, i, el::T) where {T,NAMES,TYPES}
    push_exprs = Expr(:block)
    for col_idx in 1:length(fieldnames(T))
        if fieldtype(TYPES, col_idx)!==Nothing
            ex = :( dest[$col_idx][i] = el[$col_idx] )
            push!(push_exprs.args, ex)
        end
    end

    return push_exprs
end

@generated function _pushrow(dest::NamedTuple{NAMES,TYPES}, el::T) where {T,NAMES,TYPES}
    push_exprs = Expr(:block)
    for col_idx in 1:length(fieldnames(T))
        if fieldtype(TYPES, col_idx)!==Nothing
            ex = :( push!(dest[$col_idx], el[$col_idx]) )
            push!(push_exprs.args, ex)
        end
    end

    return push_exprs
end

function _collect_columns(itr, ::Union{Base.HasShape, Base.HasLength}, array_factory, sel_cols, errorhandling)
    y = iterate(itr)
    y===nothing && return collect_empty_columns(itr, Base.IteratorEltype(itr), array_factory, sel_cols, errorhandling)

    if !(typeof(y[1])<:NamedTuple)
        if errorhandling==:error
            throw(ArgumentError("itr is not a table."))
        elseif errorhandling==:returnvalue            
            return nothing
        end
    end

    dest = getdest(typeof(y[1]), length(itr), array_factory, sel_cols)

    _setrow(dest,1,y[1])

    _collect_to_columns!(dest, itr, 2, y[2], sel_cols, errorhandling)
end

function _collect_to_columns!(dest::T, itr, offs, st, sel_cols, errorhandling) where {T<:NamedTuple}
    i = offs
    y = iterate(itr,st)
    while y!==nothing
        _setrow(dest,i,y[1])
        i += 1
        y = iterate(itr,y[2])
    end

    if sel_cols==:all
        return collect(values(dest)), collect(keys(dest))
    else
        names_to_use = tuple((fieldname(T,i) for i in sel_cols)...)
        r = NamedTuple{names_to_use}(dest)
        return collect(values(r)), collect(keys(r))
    end
end

function _collect_columns(itr, ::Base.SizeUnknown, array_factory, sel_cols, errorhandling)
    y = iterate(itr)
    y===nothing && return collect_empty_columns(itr, Base.IteratorEltype(itr), array_factory, sel_cols, errorhandling)
    
    if !(typeof(y[1])<:NamedTuple)
        if errorhandling==:error
            throw(ArgumentError("itr is not a table."))
        elseif errorhandling==:returnvalue            
            return nothing
        end
    end

    dest = getdest(typeof(y[1]), 1, array_factory, sel_cols)
    
    _setrow(dest,1,y[1])

    _grow_to_columns!(dest, itr, y[2], sel_cols, errorhandling)
end

function _grow_to_columns!(dest::T, itr, st, sel_cols, errorhandling) where {T<:NamedTuple}
    y = iterate(itr, st)
    while y!==nothing
        _pushrow(dest, y[1])
        y = iterate(itr,y[2])
    end

    if sel_cols==:all
        return collect(values(dest)), collect(keys(dest))
    else
        names_to_use = tuple((fieldname(T,i) for i in sel_cols)...)
        r = NamedTuple{names_to_use}(dest)
        return collect(values(r)), collect(keys(r))
    end
end
