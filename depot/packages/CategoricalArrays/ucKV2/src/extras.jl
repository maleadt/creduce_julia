using Compat.Statistics
function fill_refs!(refs::AbstractArray, X::AbstractArray,
                    breaks::AbstractVector, extend::Bool, allow_missing::Bool)
    n = length(breaks)
    lower = first(breaks)
    upper = last(breaks)
    @inbounds for i in eachindex(X)
        x = X[i]
        if extend && x == upper
            refs[i] = n-1
        elseif !extend && !(lower <= x < upper)
            throw(ArgumentError("value $x (at index $i) does not fall inside the breaks: adapt them manually, or pass extend=true"))
        else
            refs[i] = searchsortedlast(breaks, x)
        end
    end
end
function fill_refs!(refs::AbstractArray, X::AbstractArray{>: Missing},
                    breaks::AbstractVector, extend::Bool, allow_missing::Bool)
    n = length(breaks)
    lower = first(breaks)
    upper = last(breaks)
    @inbounds for i in eachindex(X)
        ismissing(X[i]) && continue
        x = X[i]
        if extend && x == upper
            refs[i] = n-1
        elseif !extend && !(lower <= x < upper)
            allow_missing || throw(ArgumentError("value $x (at index $i) does not fall inside the breaks: adapt them manually, or pass extend=true or allow_missing=true"))
            refs[i] = 0
        else
            refs[i] = searchsortedlast(breaks, x)
        end
    end
end
""" """ function cut(x::AbstractArray{T, N}, breaks::AbstractVector;
             extend::Bool=false, labels::AbstractVector{U}=String[],
             allow_missing::Bool=false) where {T, N, U<:AbstractString}
    if !issorted(breaks)
        breaks = sort(breaks)
    end
    if extend
        min_x, max_x = extrema(x)
        if !ismissing(min_x) && breaks[1] > min_x
            breaks = [min_x; breaks]
        end
        if !ismissing(max_x) && breaks[end] < max_x
            breaks = [breaks; max_x]
        end
    end
    refs = Array{DefaultRefType, N}(undef, size(x))
    try
        fill_refs!(refs, x, breaks, extend, allow_missing)
    catch err
        if isa(err, ArgumentError)
            throw(err)
        else
            rethrow(err)
        end
    end
    n = length(breaks)
    if isempty(labels)
        @static if VERSION >= v"0.7.0-DEV.4524"
            from = map(x -> sprint(show, x, context=:compact=>true), breaks[1:n-1])
            to = map(x -> sprint(show, x, context=:compact=>true), breaks[2:n])
        else
            from = map(x -> sprint(showcompact, x), breaks[1:n-1])
            to = map(x -> sprint(showcompact, x), breaks[2:n])
        end
        levs = Vector{String}(undef, n-1)
        for i in 1:n-2
            levs[i] = string("[", from[i], ", ", to[i], ")")
        end
        if extend
            levs[end] = string("[", from[end], ", ", to[end], "]")
        else
            levs[end] = string("[", from[end], ", ", to[end], ")")
        end
    else
        length(labels) == n-1 || throw(ArgumentError("labels must be of length $(n-1), but got length $(length(labels))"))
        levs::Vector{String} = copy(labels)
    end
    pool = CategoricalPool(levs, true)
    S = T >: Missing ? Union{String, Missing} : String
    CategoricalArray{S, N}(refs, pool)
end
""" """ cut(x::AbstractArray, ngroups::Integer;
    labels::AbstractVector{U}=String[]) where {U<:AbstractString} =
    cut(x, Statistics.quantile(x, (1:ngroups-1)/ngroups); extend=true, labels=labels)
