""" """ function trim(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    trim!(copy(x); prop=prop, count=count)
end
""" """ function trim!(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        prop == 0 || throw(ArgumentError("prop and count can not both be > 0."))
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
    end
    partialsort!(x, (n-count+1):n)
    partialsort!(x, 1:count)
    deleteat!(x, (n-count+1):n)
    deleteat!(x, 1:count)
    return x
end
""" """ function winsor(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    winsor!(copy(x); prop=prop, count=count)
end
""" """ function winsor!(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        prop == 0 || throw(ArgumentError("prop and count can not both be > 0."))
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
    end
    partialsort!(x, (n-count+1):n)
    partialsort!(x, 1:count)
    x[1:count] .= x[count+1]
    x[n-count+1:end] .= x[n-count]
    return x
end
""" """ function trimvar(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
        prop = count/n
    end
    return var(winsor(x, count=count)) / (n * (1 - 2prop)^2)
end
