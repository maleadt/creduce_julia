import .RFunctions:
    chisqpdf,
    chisqlogpdf,
    chisqcdf,
    chisqccdf,
    chisqlogcdf,
    chisqlogccdf,
    chisqinvcdf,
    chisqinvccdf,
    chisqinvlogcdf,
    chisqinvlogccdf
function chisqpdf(k::Real, x::Number)
  hk = k / 2  # half k
  1 / (2^(hk) * gamma(hk)) * x^(hk - 1) * exp(-x / 2)
end
function chisqlogpdf(k::Real, x::Number)
  hk = k / 2  # half k
  -hk * log(oftype(hk, 2)) - lgamma(hk) + (hk - 1) * log(x) - x / 2
end
