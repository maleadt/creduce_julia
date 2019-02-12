""" """ corspearman(x::RealVector, y::RealVector) = cor(tiedrank(x), tiedrank(y))
corspearman(X::RealMatrix, Y::RealMatrix) =
    cor(mapslices(tiedrank, X, dims=1), mapslices(tiedrank, Y, dims=1))
corspearman(X::RealMatrix, y::RealVector) = cor(mapslices(tiedrank, X, dims=1), tiedrank(y))
corspearman(x::RealVector, Y::RealMatrix) = cor(tiedrank(x), mapslices(tiedrank, Y, dims=1))
corspearman(X::RealMatrix) = (Z = mapslices(tiedrank, X, dims=1); cor(Z, Z))
function corkendall!(x::RealVector, y::RealVector)
    if any(isnan, x) || any(isnan, y) return NaN end
    n = length(x)
    if n != length(y) error("Vectors must have same length") end
    pm = sortperm(y)
    x[:] = x[pm]
    y[:] = y[pm]
    pm[:] = sortperm(x)
    x[:] = x[pm]
    iT = 1
    nT = 0
    iU = 1
    nU = 0
    for i = 2:n
        if x[i] == x[i-1]
            iT += 1
        else
            nT += iT*(iT - 1)
            iT = 1
        end
        if y[i] == y[i-1]
            iU += 1
        else
            nU += iU*(iU - 1)
            iU = 1
        end
    end
    if iT > 1 nT += iT*(iT - 1) end
