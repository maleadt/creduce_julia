function fill_refs!(refs::AbstractArray, X::AbstractArray{>: Missing},
                    breaks::AbstractVector, extend::Bool, allow_missing::Bool)
    @inbounds for i in eachindex(X)
    end
end
""" """ function cut(x::AbstractArray{T, N}, breaks::AbstractVector;
             allow_missing::Bool=false) where {T, N, U<:AbstractString}
    if !issorted(breaks)
    end
    if extend
        if !ismissing(min_x) && breaks[1] > min_x
            breaks = [min_x; breaks]
            breaks = [breaks; max_x]
        end
    end
    try
    catch err
        if isa(err, ArgumentError)
            rethrow(err)
        end
    end
    if isempty(labels)
        @static if VERSION >= v"0.7.0-DEV.4524"
        end
        for i in 1:n-2
        end
    end
    pool = CategoricalPool(levs, true)
end
""" """ cut(x::AbstractArray, ngroups::Integer;
    labels::AbstractVector{U}=String[]) where {U<:AbstractString} =
    cut(x, Statistics.quantile(x, (1:ngroups-1)/ngroups); extend=true, labels=labels)
