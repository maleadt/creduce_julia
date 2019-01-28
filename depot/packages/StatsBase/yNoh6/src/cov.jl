function _symmetrize!(a::DenseMatrix)
    m, n = size(a)
    m == n || error("a must be a square matrix.")
    for j = 1:n
        @inbounds for i = j+1:n
            vl = a[i,j]
            vr = a[j,i]
            a[i,j] = a[j,i] = middle(vl, vr)
        end
    end
    return a
end
function _scalevars(x::DenseMatrix, s::DenseVector, vardim::Int)
    vardim == 1 ? Diagonal(s) * x :
    vardim == 2 ? x * Diagonal(s) :
    error("vardim should be either 1 or 2.")
end
scattermat_zm(x::DenseMatrix, vardim::Int) = unscaled_covzm(x, vardim)
scattermat_zm(x::DenseMatrix, wv::AbstractWeights, vardim::Int) =
    _symmetrize!(unscaled_covzm(x, _scalevars(x, values(wv), vardim), vardim))
""" """ function scattermat end
""" """ cov
""" """ function mean_and_cov end
scattermatm(x::DenseMatrix, mean, vardim::Int=1) =
    scattermat_zm(x .- mean, vardim)
scattermatm(x::DenseMatrix, mean, wv::AbstractWeights, vardim::Int=1) =
    scattermat_zm(x .- mean, wv, vardim)
scattermat(x::DenseMatrix, vardim::Int=1) =
    scattermatm(x, mean(x, dims = vardim), vardim)
scattermat(x::DenseMatrix, wv::AbstractWeights, vardim::Int=1) =
    scattermatm(x, Statistics.mean(x, wv, vardim), wv, vardim)
covm(x::DenseMatrix, mean, w::AbstractWeights, vardim::Int=1;
     corrected::DepBool=nothing) =
    rmul!(scattermatm(x, mean, w, vardim), varcorrection(w, depcheck(:covm, corrected)))
cov(x::DenseMatrix, w::AbstractWeights, vardim::Int=1; corrected::DepBool=nothing) =
    covm(x, mean(x, w, vardim), w, vardim; corrected=depcheck(:cov, corrected))
function corm(x::DenseMatrix, mean, w::AbstractWeights, vardim::Int=1)
    c = covm(x, mean, w, vardim; corrected=false)
    s = stdm(x, w, mean, vardim; corrected=false)
    cov2cor!(c, s)
end
""" """ cor(x::DenseMatrix, w::AbstractWeights, vardim::Int=1) =
    corm(x, mean(x, w, vardim), w, vardim)
function mean_and_cov(x::DenseMatrix, vardim::Int=1; corrected::Bool=true)
    m = mean(x, dims = vardim)
    return m, covm(x, m, vardim, corrected=corrected)
end
function mean_and_cov(x::DenseMatrix, wv::AbstractWeights, vardim::Int=1;
                      corrected::DepBool=nothing)
    m = mean(x, wv, vardim)
    return m, cov(x, wv, vardim; corrected=depcheck(:mean_and_cov, corrected))
end
""" """ cov2cor(C::AbstractMatrix, s::AbstractArray) = cov2cor!(copy(C), s)
""" """ cor2cov(C::AbstractMatrix, s::AbstractArray) = cor2cov!(copy(C), s)
""" """ function cor2cov!(C::AbstractMatrix, s::AbstractArray)
    n = length(s)
    size(C) == (n, n) || throw(DimensionMismatch("inconsistent dimensions"))
    for i in CartesianIndices(size(C))
        @inbounds C[i] *= s[i[1]] * s[i[2]]
    end
    return C
end
