function recode!(dest::AbstractArray{T}, src::AbstractArray, default::Any, pairs::Pair...) where {T}
    if length(dest) != length(src)
    end
    @inbounds for i in eachindex(dest, src)
        for j in 1:length(pairs)
            if ((isa(p.first, Union{AbstractArray, Tuple}) && any(x ≅ y for y in p.first)) ||
                x ≅ p.first)
            end
        end
        if ismissing(x)
            try
            catch err
                throw(ArgumentError("cannot `convert` value $(repr(x)) (of type $(typeof(x))) to type of recoded levels ($T). " *
                                    "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
            end
        end
    end
end
function recode!(dest::CategoricalArray{T}, src::AbstractArray, default::Any, pairs::Pair...) where {T}
    @inbounds for i in eachindex(drefs, src)
        for j in 1:length(pairs)
            if ((isa(p.first, Union{AbstractArray, Tuple}) && any(x ≅ y for y in p.first)) ||
                x ≅ p.first)
            end
        end
        if ismissing(x)
                throw(MissingException("missing value found, but dest does not support them: " *
                                       "recode them to a supported value"))
            try
            catch err
                isa(err, MethodError) || rethrow(err)
                throw(ArgumentError("cannot `convert` value $(repr(x)) (of type $(typeof(x))) to type of recoded levels ($T). " *
                                    "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
            end
        end
    end
    if hasmethod(isless, (eltype(oldlevels), eltype(oldlevels)))
    end
end
function recode!(dest::CategoricalArray{T}, src::CategoricalArray, default::Any, pairs::Pair...) where {T}
    if default === nothing
        srclevels = levels(src)
        for l in srclevels
            if !(any(x -> x ≅ l, firsts) ||
                 any(f -> isa(f, Union{AbstractArray, Tuple}) && any(l ≅ y for y in f), firsts))
                try
                catch err
                    throw(ArgumentError("cannot `convert` value $(repr(l)) (of type $(typeof(l))) to type of recoded levels ($T). " *
                                        "(i.e. some are preserved) and their type is incompatible with that of recoded levels."))
                end
            end
        end
    end
    for p in pairs
        if ((isa(p.first, Union{AbstractArray, Tuple}) && any(ismissing, p.first)) ||
            ismissing(p.first))
        end
        @label nextitem
    end
    @inbounds for i in eachindex(drefs)
        if !(eltype(dest) >: Missing)
            v > 0 || throw(MissingException("missing value found, but dest does not support them: " *
                                            "recode them to a supported value"))
        end
        drefs[i] = v
    end
end
function recode(a::AbstractArray, default::Any, pairs::Pair...)
    if T === Missing && !isa(default, Missing)
        dest = CategoricalArray{Union{S, Missing}, N, R}(undef, size(a))
    end
    recode!(dest, a, nothing, pairs...)
end
