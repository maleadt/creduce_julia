abstract type AbstractExcludeLast{ContainerType <: SAContainer} end
struct SDMExcludeLast{ContainerType <: SDMContainer} <:
                              AbstractExcludeLast{ContainerType}
end
struct SSExcludeLast{ContainerType <: SortedSet} <:
                              AbstractExcludeLast{ContainerType}
end
abstract type AbstractIncludeLast{ContainerType <: SAContainer} end
struct SDMIncludeLast{ContainerType <: SDMContainer} <:
                               AbstractIncludeLast{ContainerType}
end
struct SSIncludeLast{ContainerType <: SortedSet} <:
                               AbstractIncludeLast{ContainerType}
end
const SDMIterableTypesBase = Union{SDMContainer,
                                   SDMIncludeLast}
const SSIterableTypesBase = Union{SortedSet,
                                  AbstractIncludeLast}
struct SDMKeyIteration{T <: SDMIterableTypesBase}
end
struct SDMValIteration{T <: SDMIterableTypesBase}
end
struct SDMSemiTokenIteration{T <: SDMIterableTypesBase}
end
eltype(s::SDMSemiTokenIteration) = Tuple{IntSemiToken,
                                         valtype(extractcontainer(s.base))}
struct SSSemiTokenIteration{T <: SSIterableTypesBase}
    base::T
end
