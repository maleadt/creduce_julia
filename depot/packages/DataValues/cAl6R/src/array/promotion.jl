Base.promote_rule(::Type{DataValueArray{T,N}}, ::Type{Array{T,N}}) where {T,N} = DataValueArray{T,N}
Base.promote_rule(::Type{Array{T,N}}, ::Type{DataValueArray{T,N}}) where {T,N} = DataValueArray{T,N}
Base.promote_rule(::Type{DataValueArray{T,N}}, ::Type{Array{S,N}}) where {T,S,N} = DataValueArray{Base.promote_type(T,S),N}
Base.promote_rule(::Type{Array{S,N}}, ::Type{DataValueArray{T,N}}) where {T,S,N} = DataValueArray{Base.promote_type(T,S),N}
