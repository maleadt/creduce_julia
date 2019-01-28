abstract type StatisticalModel end
""" """ coef(obj::StatisticalModel) = error("coef is not defined for $(typeof(obj)).")
""" """ coefnames(obj::StatisticalModel) = error("coefnames is not defined for $(typeof(obj)).")
""" """ coeftable(obj::StatisticalModel) = error("coeftable is not defined for $(typeof(obj)).")
""" """ confint(obj::StatisticalModel) = error("confint is not defined for $(typeof(obj)).")
""" """ deviance(obj::StatisticalModel) = error("deviance is not defined for $(typeof(obj)).")
""" """ islinear(obj::StatisticalModel) = error("islinear is not defined for $(typeof(obj)).")
""" """ nulldeviance(obj::StatisticalModel) = error("nulldeviance is not defined for $(typeof(obj)).")
""" """ loglikelihood(obj::StatisticalModel) = error("loglikelihood is not defined for $(typeof(obj)).")
""" """ nullloglikelihood(obj::StatisticalModel) = error("nullloglikelihood is not defined for $(typeof(obj)).")
""" """ score(obj::StatisticalModel) = error("score is not defined for $(typeof(obj)).")
""" """ nobs(obj::StatisticalModel) = error("nobs is not defined for $(typeof(obj)).")
""" """ dof(obj::StatisticalModel) = error("dof is not defined for $(typeof(obj)).")
""" """ mss(obj::StatisticalModel) = error("mss is not defined for $(typeof(obj)).")
""" """ rss(obj::StatisticalModel) = error("rss is not defined for $(typeof(obj)).")
""" """ informationmatrix(model::StatisticalModel; expected::Bool = true) =
    error("informationmatrix is not defined for $(typeof(obj)).")
""" """ stderror(obj::StatisticalModel) = sqrt.(diag(vcov(obj)))
""" """ vcov(obj::StatisticalModel) = error("vcov is not defined for $(typeof(obj)).")
""" """ weights(obj::StatisticalModel) = error("weights is not defined for $(typeof(obj)).")
""" """ isfitted(obj::StatisticalModel) = error("isfitted is not defined for $(typeof(obj)).")
""" """ fit(obj::StatisticalModel, args...) = error("fit is not defined for $(typeof(obj)).")
""" """ fit!(obj::StatisticalModel, args...) = error("fit! is not defined for $(typeof(obj)).")
""" """ aic(obj::StatisticalModel) = -2loglikelihood(obj) + 2dof(obj)
""" """ function aicc(obj::StatisticalModel)
    k = dof(obj)
    n = nobs(obj)
    -2loglikelihood(obj) + 2k + 2k*(k+1)/(n-k-1)
end
""" """ bic(obj::StatisticalModel) = -2loglikelihood(obj) + dof(obj)*log(nobs(obj))
""" """ function r2(obj::StatisticalModel)
    Base.depwarn("The default r² method for linear models is deprecated. " *
                 "Packages should define their own methods.", :r2)
    mss(obj) / deviance(obj)
end
""" """ function r2(obj::StatisticalModel, variant::Symbol)
    ll = loglikelihood(obj)
    ll0 = nullloglikelihood(obj)
    if variant == :McFadden
        1 - ll/ll0
    elseif variant == :CoxSnell
        1 - exp(2 * (ll0 - ll) / nobs(obj))
    elseif variant == :Nagelkerke
        (1 - exp(2 * (ll0 - ll) / obs(obj))) / (1 - exp(2 * ll0 / nobs(obj)))
    else
        error("variant must be one of :McFadden, :CoxSnell or :Nagelkerke")
    end
end
const r² = r2
""" """ adjr2(obj::StatisticalModel) = error("adjr2 is not defined for $(typeof(obj)).")
""" """ function adjr2(obj::StatisticalModel, variant::Symbol)
    ll = loglikelihood(obj)
    ll0 = nullloglikelihood(obj)
    k = dof(obj)
    if variant == :McFadden
        1 - (ll - k)/ll0
    else
        error(":McFadden is the only currently supported variant")
    end
end
const adjr² = adjr2
abstract type RegressionModel <: StatisticalModel end
""" """ fitted(obj::RegressionModel) = error("fitted is not defined for $(typeof(obj)).")
""" """ response(obj::RegressionModel) = error("response is not defined for $(typeof(obj)).")
""" """ meanresponse(obj::RegressionModel) = error("meanresponse is not defined for $(typeof(obj)).")
""" """ modelmatrix(obj::RegressionModel) = error("modelmatrix is not defined for $(typeof(obj)).")
""" """ leverage(obj::RegressionModel) = error("leverage is not defined for $(typeof(obj)).")
""" """ residuals(obj::RegressionModel) = error("residuals is not defined for $(typeof(obj)).")
""" """ function predict end
predict(obj::RegressionModel) = error("predict is not defined for $(typeof(obj)).")
""" """ function predict! end
predict!(obj::RegressionModel) = error("predict! is not defined for $(typeof(obj)).")
""" """ dof_residual(obj::RegressionModel) = error("dof_residual is not defined for $(typeof(obj)).")
""" """ params(obj) = error("params is not defined for $(typeof(obj))")
function params! end
mutable struct CoefTable
    cols::Vector
    colnms::Vector
    rownms::Vector
    function CoefTable(cols::Vector,colnms::Vector,rownms::Vector)
        nc = length(cols)
        nrs = map(length,cols)
        nr = nrs[1]
        length(colnms) in [0,nc] || error("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || error("rownms should have length 0 or $nr")
        all(nrs .== nr) || error("Elements of cols should have equal lengths, but got $nrs")
        new(cols,colnms,rownms)
    end
    function CoefTable(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0)
        nc = size(mat,2)
        cols = Any[mat[:, i] for i in 1:nc]
        if pvalcol != 0                         # format the p-values column
            cols[pvalcol] = [PValue(cols[pvalcol][j])
                            for j in eachindex(cols[pvalcol])]
        end
        CoefTable(cols,colnms,rownms)
    end
end
mutable struct PValue
    v::Number
    function PValue(v::Number)
        0. <= v <= 1. || isnan(v) || error("p-values must be in [0.,1.]")
        new(v)
    end
end
function show(io::IO, pv::PValue)
    v = pv.v
    if isnan(v)
        @printf(io,"%d", v)
    elseif v >= 1e-4
        @printf(io,"%.4f", v)
    else
        @printf(io,"<1e%2.2d", ceil(Integer, max(nextfloat(log10(v)), -99)))
    end
end
function show(io::IO, ct::CoefTable)
    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = String[isa(cols[j][i], AbstractString) ? cols[j][i] :
        sprint(show, cols[j][i], context=:compact=>true) for i in 1:nr, j in 1:nc]
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i,j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    for i = 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
end
""" """ struct ConvergenceException{T<:Real} <: Exception
    iters::Int
    lastchange::T
    tol::T
    function ConvergenceException{T}(iters, lastchange::T, tol::T) where T<:Real
        if tol > lastchange
            throw(ArgumentError("Change must be greater than tol."))
        else
            new(iters, lastchange, tol)
        end
    end
end
ConvergenceException(iters, lastchange::T=NaN, tol::T=NaN) where {T<:Real} =
    ConvergenceException{T}(iters, lastchange, tol)
function Base.showerror(io::IO, ce::ConvergenceException)
    print(io, "failure to converge after $(ce.iters) iterations.")
    if !isnan(ce.lastchange)
        print(io, " Last change ($(ce.lastchange)) was greater than tolerance ($(ce.tol)).")
    end
end
