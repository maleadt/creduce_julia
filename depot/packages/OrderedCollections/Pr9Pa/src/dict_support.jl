_tablesz(x::Integer) = x < 16 ? 16 : one(x)<<((sizeof(x)<<3)-leading_zeros(x-1))
hashindex(key, sz) = (reinterpret(Int,(hash(key))) & (sz-1)) + 1
