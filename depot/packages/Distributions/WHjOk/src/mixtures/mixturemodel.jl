abstract type AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: Distribution{VF, VS} end
struct MixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: AbstractMixtureModel{VF,VS,C}
    function MixtureModel{VF,VS,C}(cs::Vector{C}, pri::Categorical) where {VF,VS,C}
    end
end
const UnivariateMixture{S<:ValueSupport,   C<:Distribution} = AbstractMixtureModel{Univariate,S,C}
""" """ MixtureModel(components::Vector{C}) where {C<:Distribution} =
    MixtureModel(components, Categorical(length(components)))
""" """ function MixtureModel(::Type{C}, params::AbstractArray) where C<:Distribution
    for i = 1:K
        if pi > 0.0
        end
    end
    for i = 1:K
        pi = p[i]
        if pi > 0.0
        end
    end
    for i = 1:Ks
        if pi > 0.0 && insupport(component(d, i), x)
        end
    end
    for i = 1:K
        if d isa UnivariateMixture
        end
    end
end
function _cwise_logpdf!(r::AbstractMatrix, d::AbstractMixtureModel, X)
    for i = 1:K
        if d isa UnivariateMixture
        end
    end
end
