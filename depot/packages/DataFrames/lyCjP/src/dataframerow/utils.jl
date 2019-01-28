struct RowGroupDict{T<:AbstractDataFrame}
    "source data table"
    df::T
    "row hashes (optional, can be empty)"
    rhashes::Vector{UInt}
    "hashindex -> index of group-representative row (optional, can be empty)"
    gslots::Vector{Int}
    "group index for each row"
    groups::Vector{Int}
    "permutation of row indices that sorts them by groups"
    rperm::Vector{Int}
    "starts of ranges in rperm for each group"
    starts::Vector{Int}
    "stops of ranges in rperm for each group"
    stops::Vector{Int}
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractVector{T},
                       firstcol::Bool) where T
    @inbounds for i in eachindex(h)
        el = v[i]
        h[i] = hash(el, h[i])
        if length(n) > 0
            n[i] |= ismissing(el)
        end
    end
    h
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractCategoricalVector,
                       firstcol::Bool)
    index = CategoricalArrays.index(v.pool)
    if firstcol
        hashes = Vector{UInt}(undef, length(levels(v.pool))+1)
        hashes[1] = hash(missing)
        hashes[2:end] .= hash.(index)
        @inbounds for (i, ref) in enumerate(v.refs)
            h[i] = hashes[ref+1]
        end
    else
        @inbounds for (i, ref) in enumerate(v.refs)
            if eltype(v) >: Missing && ref == 0
                h[i] = hash(missing, h[i])
            else
                h[i] = hash(index[ref], h[i])
            end
        end
    end
    if eltype(v) >: Missing && length(n) > 0
        @inbounds for (i, ref) in enumerate(v.refs)
            n[i] |= (ref == 0)
        end
    end
    h
end
function hashrows(cols::Tuple{Vararg{AbstractVector}}, skipmissing::Bool)
    len = length(cols[1])
    rhashes = zeros(UInt, len)
    missings = fill(false, skipmissing ? len : 0)
    for (i, col) in enumerate(cols)
        hashrows_col!(rhashes, missings, col, i == 1)
    end
    return (rhashes, missings)
end
isequal_row(cols::Tuple{AbstractVector}, r1::Int, r2::Int) =
    isequal(cols[1][r1], cols[1][r2])
isequal_row(cols::Tuple{Vararg{AbstractVector}}, r1::Int, r2::Int) =
    isequal(cols[1][r1], cols[1][r2]) && isequal_row(Base.tail(cols), r1, r2)
isequal_row(cols1::Tuple{AbstractVector}, r1::Int, cols2::Tuple{AbstractVector}, r2::Int) =
    isequal(cols1[1][r1], cols2[1][r2])
isequal_row(cols1::Tuple{Vararg{AbstractVector}}, r1::Int,
            cols2::Tuple{Vararg{AbstractVector}}, r2::Int) =
    isequal(cols1[1][r1], cols2[1][r2]) &&
        isequal_row(Base.tail(cols1), r1, Base.tail(cols2), r2)
function row_group_slots(cols::Tuple{Vararg{AbstractVector}},
                         hash::Val = Val(true),
                         groups::Union{Vector{Int}, Nothing} = nothing,
                         skipmissing::Bool = false)::Tuple{Int, Vector{UInt}, Vector{Int}, Bool}
    @assert groups === nothing || length(groups) == length(cols[1])
    rhashes, missings = hashrows(cols, skipmissing)
    sz = Base._tablesz(length(rhashes))
    @assert sz >= length(rhashes)
    szm1 = sz-1
    gslots = zeros(Int, sz)
    ngroups = skipmissing ? 1 : 0
    @inbounds for i in eachindex(rhashes)
        slotix = rhashes[i] & szm1 + 1
        gix = skipmissing && missings[i] ? 1 : 0
        probe = 0
        if !skipmissing || !missings[i]
            while true
                g_row = gslots[slotix]
                if g_row == 0 # unoccupied slot, current row starts a new group
                    gslots[slotix] = i
                    gix = ngroups += 1
                    break
                elseif rhashes[i] == rhashes[g_row] # occupied slot, check if miss or hit
                    if isequal_row(cols, i, g_row) # hit
                        gix = groups !== nothing ? groups[g_row] : 0
                    end
                    break
                end
                slotix = slotix & szm1 + 1 # check the next slot
                probe += 1
                @assert probe < sz
            end
        end
        if groups !== nothing
            groups[i] = gix
        end
    end
    return ngroups, rhashes, gslots, false
end
function row_group_slots(cols::NTuple{N,<:CategoricalVector},
                         hash::Val{false},
                         groups::Union{Vector{Int}, Nothing} = nothing,
                         skipmissing::Bool = false)::Tuple{Int, Vector{UInt}, Vector{Int}, Bool} where N
    @assert groups !== nothing && all(col -> length(col) == length(groups), cols)
    ngroupstup = map(cols) do c
        length(levels(c)) + (!skipmissing && eltype(c) >: Missing)
    end
    ngroups = prod(ngroupstup) + skipmissing
    if prod(Int128.(ngroupstup)) > typemax(Int) || ngroups > 2 * length(groups)
        return invoke(row_group_slots,
                      Tuple{Tuple{Vararg{AbstractVector}}, Val,
                            Union{Vector{Int}, Nothing}, Bool},
                      cols, hash, groups, skipmissing)
    end
    seen = fill(false, ngroups)
    seen[1] = skipmissing
    refmaps = map(cols) do col
        nlevels = length(levels(col))
        refmap = Vector{Int}(undef, nlevels + 1)
        refmap[1] = nlevels
        refmap[2:end] .= CategoricalArrays.order(col.pool) .- 1
        refmap
    end
    strides = (cumprod(collect(reverse(ngroupstup)))[end-1:-1:1]..., 1)::NTuple{N,Int}
    @inbounds for i in eachindex(groups)
        local refs
        let i=i # Workaround for julia#15276
            refs = map(c -> c.refs[i], cols)
        end
        j = sum(map((m, r, s) -> m[r+1] * s, refmaps, refs, strides)) + 1
        if skipmissing
            j = any(iszero, refs) ? 1 : j + 1
        end
        groups[i] = j
        seen[j] = true
    end
    if !all(seen) # Compress group indices to remove unused ones
        oldngroups = ngroups
        remap = zeros(Int, ngroups)
        ngroups = 0
        @inbounds for i in eachindex(remap, seen)
            ngroups += seen[i]
            remap[i] = ngroups
        end
        @inbounds for i in eachindex(groups)
            groups[i] = remap[groups[i]]
        end
        @assert oldngroups != ngroups
    end
    return ngroups, UInt[], Int[], true
end
function group_rows(df::AbstractDataFrame, hash::Bool = true, sort::Bool = false,
                    skipmissing::Bool = false)
    groups = Vector{Int}(undef, nrow(df))
    ngroups, rhashes, gslots, sorted =
        row_group_slots(ntuple(i -> df[i], ncol(df)), Val(hash), groups, skipmissing)
    stops = zeros(Int, ngroups)
    @inbounds for g_ix in groups
        stops[g_ix] += 1
    end
    starts = Vector{Int}(undef, ngroups)
    if !isempty(starts)
        starts[1] = 1
        @inbounds for i in 1:(ngroups-1)
            starts[i+1] = starts[i] + stops[i]
        end
    end
    rperm = Vector{Int}(undef, length(groups))
    copyto!(stops, starts)
    @inbounds for (i, gix) in enumerate(groups)
        rperm[stops[gix]] = i
        stops[gix] += 1
    end
    stops .-= 1
    if skipmissing
        popfirst!(starts)
        popfirst!(stops)
        groups .-= 1
        ngroups -= 1
    end
    if sort && !sorted
        group_perm = sortperm(view(df, rperm[starts], :))
        group_invperm = invperm(group_perm)
        permute!(starts, group_perm)
        Base.permute!!(stops, group_perm)
        for i in eachindex(groups)
            gix = groups[i]
            groups[i] = gix == 0 ? 0 : group_invperm[gix]
        end
    end
    return RowGroupDict(df, rhashes, gslots, groups, rperm, starts, stops)
end
function findrow(gd::RowGroupDict,
                 df::DataFrame,
                 gd_cols::Tuple{Vararg{AbstractVector}},
                 df_cols::Tuple{Vararg{AbstractVector}},
                 row::Int)
    (gd.df === df) && return row # same table, return itself
    rhash = rowhash(df_cols, row)
    szm1 = length(gd.gslots)-1
    slotix = ini_slotix = rhash & szm1 + 1
    while true
        g_row = gd.gslots[slotix]
        if g_row == 0 || # not found
            (rhash == gd.rhashes[g_row] &&
            isequal_row(gd_cols, g_row, df_cols, row)) # found
            return g_row
        end
        slotix = (slotix & szm1) + 1 # miss, try the next slot
        (slotix == ini_slotix) && break
    end
    return 0 # not found
end
function findrows(gd::RowGroupDict,
                  df::DataFrame,
                  gd_cols::Tuple{Vararg{AbstractVector}},
                  df_cols::Tuple{Vararg{AbstractVector}},
                  row::Int)
    g_row = findrow(gd, df, gd_cols, df_cols, row)
    (g_row == 0) && return view(gd.rperm, 0:-1)
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end
function Base.getindex(gd::RowGroupDict, dfr::DataFrameRow)
    g_row = findrow(gd, parent(dfr), ntuple(i -> gd.df[i], ncol(gd.df)),
                    ntuple(i -> parent(dfr)[i], ncol(parent(dfr))), row(dfr))
    (g_row == 0) && throw(KeyError(dfr))
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end
