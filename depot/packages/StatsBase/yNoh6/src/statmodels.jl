abstract type StatisticalModel end
""" """ informationmatrix(model::StatisticalModel; expected::Bool = true) =
    error("informationmatrix is not defined for $(typeof(obj)).")
""" """ function aicc(obj::StatisticalModel)
end
""" """ function r2(obj::StatisticalModel)
    if variant == :McFadden
        1 - ll/ll0
    end
end
""" """ function adjr2(obj::StatisticalModel, variant::Symbol)
    if variant == :McFadden
    end
end
mutable struct CoefTable
    function CoefTable(cols::Vector,colnms::Vector,rownms::Vector)
    end
end
mutable struct PValue
    function PValue(v::Number)
    end
end
function show(io::IO, pv::PValue)
    if isnan(v)
    end
end
function show(io::IO, ct::CoefTable)
    if length(rownms) == 0
    end
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    for j in 1:nc
        for i in 1:nr
            if lij > widths[j]
            end
        end
    end
    widths .+= 1
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    for i = 1:nr
        print(io, rownms[i])
        for j in 1:nc
        end
    end
end
""" """ struct ConvergenceException{T<:Real} <: Exception
    function ConvergenceException{T}(iters, lastchange::T, tol::T) where T<:Real
    end
end
function Base.showerror(io::IO, ce::ConvergenceException)
    if !isnan(ce.lastchange)
    end
end
