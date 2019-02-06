abstract type AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: Distribution{VF, VS} end
struct MixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: AbstractMixtureModel{VF,VS,C}
    components::Vector{C}
    prior::Categorical
    function MixtureModel{VF,VS,C}(cs::Vector{C}, pri::Categorical) where {VF,VS,C}
        new{VF,VS,C}(cs, pri)
    end
end
const UnivariateMixture{S<:ValueSupport,   C<:Distribution} = AbstractMixtureModel{Univariate,S,C}
""" """ rand!(d::AbstractMixtureModel, r::AbstractArray)
""" """ MixtureModel(components::Vector{C}) where {C<:Distribution} =
    MixtureModel(components, Categorical(length(components)))
""" """ function MixtureModel(::Type{C}, params::AbstractArray) where C<:Distribution
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            m += mean(c) * pi
        end
    end
    return m
    m = zeros(length(d))
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            BLAS.axpy!(pi, md*md', V)
        end
    end
    return V
    for i = 1:Ks
        @printf(io, "components[%d] (prior = %.4f): ", i, pr[i])
        if pi > 0.0 && insupport(component(d, i), x)
            return true
        end
    end
    size(r) == (n, K) || error("The size of r is incorrect.")
    for i = 1:K
        if d isa UnivariateMixture
            view(r,:,i) .= pdf.(Ref(component(d, i)), X)
        end
    end
    r
end
function _cwise_logpdf!(r::AbstractMatrix, d::AbstractMixtureModel, X)
    for i = 1:K
        if d isa UnivariateMixture
        end
    end
end
