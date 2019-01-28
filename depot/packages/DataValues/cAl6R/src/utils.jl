using StatsBase

StatsBase.describe(X::DataValueVector) = StatsBase.describe(STDOUT, X)

function StatsBase.describe(io::IO, X::DataValueVector{T}) where {T <: Real}
    nullcount = sum(X.isna)
    pnull = 100nullcount/length(X)
    if pnull != 100 # describe will fail if dropna returns an empty vector
        describe(io, dropna(X))
    else
        println(io, "Summary Stats:")
        println(io, "Type:           $(eltype(X))")
    end
    println(io, "Number Missing: $(nullcount)")
    @printf(io, "%% Missing:      %.6f\n", pnull)
    return
end

function StatsBase.describe(io::IO, X::DataValueVector)
    nullcount = sum(X.isna)
    pnull = 100nullcount/length(X)
    println(io, "Summary Stats:")
    println(io, "Length:         $(length(X))")
    println(io, "Type:           $(eltype(X))")
    println(io, "Number Unique:  $(length(unique(X)))")
    println(io, "Number Missing: $(nullcount)")
    @printf(io, "%% Missing:      %.6f\n", pnull)
    return
end

function StatsBase.describe(io::IO, X::AbstractVector{DataValue{T}}) where {T <: Real}
    nullcount = sum(isna, X)
    pnull = 100nullcount/length(X)
    if pnull != 100 # describe will fail if dropna returns an empty vector
        describe(io, dropna(X))
    else
        println(io, "Summary Stats:")
        println(io, "Type:           $(eltype(X))")
    end
    println(io, "Number Missing: $(nullcount)")
    @printf(io, "%% Missing:      %.6f\n", pnull)
    return
end

function StatsBase.describe(io::IO, X::AbstractVector{T}) where {T <: DataValue}
    nullcount = sum(isna, X)
    pnull = 100nullcount/length(X)
    println(io, "Summary Stats:")
    println(io, "Length:         $(length(X))")
    println(io, "Type:           $(eltype(X))")
    println(io, "Number Unique:  $(length(unique(X)))")
    println(io, "Number Missing: $(nullcount)")
    @printf(io, "%% Missing:      %.6f\n", pnull)
    return
end
