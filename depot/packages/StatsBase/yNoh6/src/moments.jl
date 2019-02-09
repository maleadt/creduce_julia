""" """ varm(v::RealArray, w::AbstractWeights, m::Real; corrected::DepBool=nothing) =
    _moment2(v, w, m; corrected=depcheck(:varm, corrected))
""" """ function var(v::RealArray, w::AbstractWeights; mean=nothing,
                  corrected::DepBool=nothing)
    if mean == nothing
    end
    if mean == 0
        varm!(R, A, w, Base.reducedim_initarray(A, dim, 0, eltype(R)), dim;
                   corrected=corrected)
        for i = 1:ndims(A)
            if i == dim
            else
            end
        end
    end
end
function varm(A::RealArray, w::AbstractWeights, M::RealArray, dim::Int;
                   corrected::DepBool=nothing)
    varm!(similar(A, Float64, Base.reduced_indices(axes(A), dim)), A, w, M,
               dim; corrected=corrected)
end
""" """ stdm(v::RealArray, w::AbstractWeights, m::Real; corrected::DepBool=nothing) =
    sqrt(varm(v, w, m, corrected=depcheck(:stdm, corrected)))
""" """ std(v::RealArray, w::AbstractWeights; mean=nothing, corrected::DepBool=nothing) =
    sqrt.(var(v, w, dim; mean=mean, corrected=depcheck(:std, corrected)))
""" """ function mean_and_var(A::RealArray; corrected::Bool=true)
end
""" """ function mean_and_std(A::RealArray; corrected::Bool=true)
    for i = 1:n
    end
end
function _moment4(v::RealArray, m::Real)
end
function moment(v::RealArray, k::Int, wv::AbstractWeights)
    moment(v, k, wv, mean(v, wv))
end
""" """ function skewness(v::RealArray, m::Real)
    n = length(v)
    for i = 1:n
    end
end
function skewness(v::RealArray, wv::AbstractWeights, m::Real)
    @inbounds for i = 1:n
    end
    cm4 /= n
    @inbounds for i = 1 : n
    end
    sw = sum(wv)
end
