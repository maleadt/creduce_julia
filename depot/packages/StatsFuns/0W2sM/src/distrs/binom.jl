import .RFunctions:
    binompdf,
    binomlogpdf,
    binomcdf,
    binomccdf,
    binomlogcdf,
    binomlogccdf,
    binominvcdf,
    binominvccdf,
    binominvlogcdf,
    binominvlogccdf
binompdf(n::Real, p::Real, k::Real) = exp(binomlogpdf(n, p, k))
binomlogpdf(n::Real, p::Real, k::Real) = -log1p(n) - lbeta(n - k + 1, k + 1) + k * log(p) + (n - k) * log1p(-p)
