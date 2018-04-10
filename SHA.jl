__precompile__()
module SHA
using Compat
export sha1, SHA1_CTX, update!, digest!
export sha224, sha256, sha384, sha512
export sha2_224, sha2_256, sha2_384, sha2_512
export sha3_224, sha3_256, sha3_384, sha3_512
export SHA224_CTX, SHA256_CTX, SHA384_CTX, SHA512_CTX
export SHA2_224_CTX, SHA2_256_CTX, SHA2_384_CTX, SHA2_512_CTX
export SHA3_224_CTX, SHA3_256_CTX, SHA3_384_CTX, SHA3_512_CTX
export HMAC_CTX, hmac_sha1
export hmac_sha224, hmac_sha256, hmac_sha384, hmac_sha512
export hmac_sha2_224, hmac_sha2_256, hmac_sha2_384, hmac_sha2_512
export hmac_sha3_224, hmac_sha3_256, hmac_sha3_384, hmac_sha3_512
const K1 = UInt32[
    0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6
]
const SHA1_initial_hash_value = UInt32[
    0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0
]
const K256 = UInt32[
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]
const SHA2_224_initial_hash_value = UInt32[
    0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939,
    0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4
]
const SHA2_256_initial_hash_value = UInt32[
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]
const K512 = UInt64[
    0x428a2f98d728ae22, 0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019,
    0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe,
    0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1,
    0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210,
    0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001,
    0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910,
    0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53,
    0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60,
    0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9,
    0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6,
    0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec, 0x6c44198c4a475817
]
const SHA2_384_initial_hash_value = UInt64[
    0xcbbb9d5dc1059ed8, 0x629a292a367cd507,
    0x9159015a3070dd17, 0x152fecd8f70e5939,
    0x67332667ffc00b31, 0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4
]
const SHA2_512_initial_hash_value = UInt64[
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
]
const SHA3_ROUND_CONSTS = UInt64[
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
]
const SHA3_ROTC = UInt64[
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
]
const SHA3_PILN = Int[
    11, 8,  12, 18, 19, 4, 6,  17, 9,  22, 25, 5,
    16, 24, 20, 14, 13, 3, 21, 15, 23, 10,  7,  2
]
abstract type SHA_CTX end
abstract type SHA2_CTX <: SHA_CTX end
abstract type SHA3_CTX <: SHA_CTX end
import Base: copy
mutable struct SHA1_CTX <: SHA_CTX
    state::Array{UInt32,1}
    bytecount::UInt64
    buffer::Array{UInt8,1}
    W::Array{UInt32,1}
end
mutable struct SHA2_224_CTX <: SHA2_CTX
    state::Array{UInt32,1}
    bytecount::UInt64
    buffer::Array{UInt8,1}
end
mutable struct SHA2_256_CTX <: SHA2_CTX
    state::Array{UInt32,1}
    bytecount::UInt64
    buffer::Array{UInt8,1}
end
mutable struct SHA2_384_CTX <: SHA2_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
end
mutable struct SHA2_512_CTX <: SHA2_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
end
const SHA224_CTX = SHA2_224_CTX
const SHA256_CTX = SHA2_256_CTX
const SHA384_CTX = SHA2_384_CTX
const SHA512_CTX = SHA2_512_CTX
mutable struct SHA3_224_CTX <: SHA3_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
    bc::Array{UInt64,1}
end
mutable struct SHA3_256_CTX <: SHA3_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
    bc::Array{UInt64,1}
end
mutable struct SHA3_384_CTX <: SHA3_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
    bc::Array{UInt64,1}
end
mutable struct SHA3_512_CTX <: SHA3_CTX
    state::Array{UInt64,1}
    bytecount::UInt128
    buffer::Array{UInt8,1}
    bc::Array{UInt64,1}
end
digestlen(::Type{SHA1_CTX}) = 20
digestlen(::Type{SHA2_224_CTX}) = 28
digestlen(::Type{SHA3_224_CTX}) = 28
digestlen(::Type{SHA2_256_CTX}) = 32
digestlen(::Type{SHA3_256_CTX}) = 32
digestlen(::Type{SHA2_384_CTX}) = 48
digestlen(::Type{SHA3_384_CTX}) = 48
digestlen(::Type{SHA2_512_CTX}) = 64
digestlen(::Type{SHA3_512_CTX}) = 64
state_type(::Type{SHA1_CTX}) = UInt32
state_type(::Type{SHA2_224_CTX}) = UInt32
state_type(::Type{SHA2_256_CTX}) = UInt32
state_type(::Type{SHA2_384_CTX}) = UInt64
state_type(::Type{SHA2_512_CTX}) = UInt64
state_type(::Type{SHA3_CTX}) = UInt64
blocklen(::Type{SHA1_CTX}) = UInt64(64)
blocklen(::Type{SHA2_224_CTX}) = UInt64(64)
blocklen(::Type{SHA2_256_CTX}) = UInt64(64)
blocklen(::Type{SHA2_384_CTX}) = UInt64(128)
blocklen(::Type{SHA2_512_CTX}) = UInt64(128)
blocklen(::Type{SHA3_224_CTX}) = UInt64(25*8 - 2*digestlen(SHA3_224_CTX))
blocklen(::Type{SHA3_256_CTX}) = UInt64(25*8 - 2*digestlen(SHA3_256_CTX))
blocklen(::Type{SHA3_384_CTX}) = UInt64(25*8 - 2*digestlen(SHA3_384_CTX))
blocklen(::Type{SHA3_512_CTX}) = UInt64(25*8 - 2*digestlen(SHA3_512_CTX))
short_blocklen(::Type{T}) where {T<:SHA_CTX} = blocklen(T) - 2*sizeof(state_type(T))
SHA2_224_CTX() = SHA2_224_CTX(copy(SHA2_224_initial_hash_value), 0, zeros(UInt8, blocklen(SHA2_224_CTX)))
SHA2_256_CTX() = SHA2_256_CTX(copy(SHA2_256_initial_hash_value), 0, zeros(UInt8, blocklen(SHA2_256_CTX)))
SHA2_384_CTX() = SHA2_384_CTX(copy(SHA2_384_initial_hash_value), 0, zeros(UInt8, blocklen(SHA2_384_CTX)))
SHA2_512_CTX() = SHA2_512_CTX(copy(SHA2_512_initial_hash_value), 0, zeros(UInt8, blocklen(SHA2_512_CTX)))
SHA3_224_CTX() = SHA3_224_CTX(zeros(UInt64, 25), 0, zeros(UInt8, blocklen(SHA3_224_CTX)), Vector{UInt64}(uninitialized, 5))
SHA3_256_CTX() = SHA3_256_CTX(zeros(UInt64, 25), 0, zeros(UInt8, blocklen(SHA3_256_CTX)), Vector{UInt64}(uninitialized, 5))
SHA3_384_CTX() = SHA3_384_CTX(zeros(UInt64, 25), 0, zeros(UInt8, blocklen(SHA3_384_CTX)), Vector{UInt64}(uninitialized, 5))
SHA3_512_CTX() = SHA3_512_CTX(zeros(UInt64, 25), 0, zeros(UInt8, blocklen(SHA3_512_CTX)), Vector{UInt64}(uninitialized, 5))
const SHA224_CTX = SHA2_224_CTX
const SHA256_CTX = SHA2_256_CTX
const SHA384_CTX = SHA2_384_CTX
const SHA512_CTX = SHA2_512_CTX
SHA1_CTX() = SHA1_CTX(copy(SHA1_initial_hash_value), 0, zeros(UInt8, blocklen(SHA1_CTX)), Vector{UInt32}(uninitialized, 80))
copy(ctx::T) where {T<:SHA1_CTX} = T(copy(ctx.state), ctx.bytecount, copy(ctx.buffer), copy(ctx.W))
copy(ctx::T) where {T<:SHA2_CTX} = T(copy(ctx.state), ctx.bytecount, copy(ctx.buffer))
copy(ctx::T) where {T<:SHA3_CTX} = T(copy(ctx.state), ctx.bytecount, copy(ctx.buffer), Vector{UInt64}(uninitialized, 5))
import Base.show
show(io::IO, ::SHA1_CTX) = write(io, "SHA1 hash state")
show(io::IO, ::SHA2_224_CTX) = write(io, "SHA2 224-bit hash state")
show(io::IO, ::SHA2_256_CTX) = write(io, "SHA2 256-bit hash state")
show(io::IO, ::SHA2_384_CTX) = write(io, "SHA2 384-bit hash state")
show(io::IO, ::SHA2_512_CTX) = write(io, "SHA2 512-bit hash state")
show(io::IO, ::SHA3_224_CTX) = write(io, "SHA3 224-bit hash state")
show(io::IO, ::SHA3_256_CTX) = write(io, "SHA3 256-bit hash state")
show(io::IO, ::SHA3_384_CTX) = write(io, "SHA3 384-bit hash state")
show(io::IO, ::SHA3_512_CTX) = write(io, "SHA3 512-bit hash state")
buffer_pointer(ctx::T) where {T<:SHA_CTX} = Ptr{state_type(T)}(pointer(ctx.buffer))
rrot(b,x,width) = ((x >> b) | (x << (width - b)))
lrot(b,x,width) = ((x << b) | (x >> (width - b)))
R(b,x)   = (x >> b)
S32(b,x) = rrot(b,x,32)
S64(b,x) = rrot(b,x,64)
L64(b,x) = lrot(b,x,64)
Ch(x,y,z)  = ((x & y) ⊻ (~x & z))
Maj(x,y,z) = ((x & y) ⊻ (x & z) ⊻ (y & z))
Sigma0_256(x) = (S32(2,  UInt32(x)) ⊻ S32(13, UInt32(x)) ⊻ S32(22, UInt32(x)))
Sigma1_256(x) = (S32(6,  UInt32(x)) ⊻ S32(11, UInt32(x)) ⊻ S32(25, UInt32(x)))
sigma0_256(x) = (S32(7,  UInt32(x)) ⊻ S32(18, UInt32(x)) ⊻ R(3 ,   UInt32(x)))
sigma1_256(x) = (S32(17, UInt32(x)) ⊻ S32(19, UInt32(x)) ⊻ R(10,   UInt32(x)))
Sigma0_512(x) = (S64(28, UInt64(x)) ⊻ S64(34, UInt64(x)) ⊻ S64(39, UInt64(x)))
Sigma1_512(x) = (S64(14, UInt64(x)) ⊻ S64(18, UInt64(x)) ⊻ S64(41, UInt64(x)))
sigma0_512(x) = (S64( 1, UInt64(x)) ⊻ S64( 8, UInt64(x)) ⊻ R( 7,   UInt64(x)))
sigma1_512(x) = (S64(19, UInt64(x)) ⊻ S64(61, UInt64(x)) ⊻ R( 6,   UInt64(x)))
bswap!(x::Vector{<:Integer}) = map!(bswap, x, x)
function Round0(b,c,d)
    return UInt32((b & c) | (~b & d))
end
function Round1And3(b,c,d)
    return UInt32(b ⊻ c ⊻ d)
end
function Round2(b,c,d)
    return UInt32((b & c) | (b & d) | (c & d))
end
function transform!(context::SHA1_CTX)
    # Buffer is 16 elements long, we expand to 80
    pbuf = buffer_pointer(context)
    for i in 1:16
        context.W[i] = bswap(unsafe_load(pbuf, i))
    end
    # First round of expansions
    for i in 17:32
        @inbounds begin
            context.W[i] = lrot(1, context.W[i-3] ⊻ context.W[i-8] ⊻ context.W[i-14] ⊻ context.W[i-16], 32)
        end
    end
    # Second round of expansions (possibly 4-way SIMD-able)
    for i in 33:80
        @inbounds begin
            context.W[i] = lrot(2, context.W[i-6] ⊻ context.W[i-16] ⊻ context.W[i-28] ⊻ context.W[i-32], 32)
        end
    end
    # Initialize registers with the previous intermediate values (our state)
    a = context.state[1]
    b = context.state[2]
    c = context.state[3]
    d = context.state[4]
    e = context.state[5]
    # Run our rounds, manually separated into the four rounds, unfortunately using an array of lambdas
    # really kills performance and causes a huge number of allocations, so we make it easy on the compiler
    for i = 1:20
        @inbounds begin
            temp = UInt32(lrot(5, a, 32) + Round0(b,c,d) + e + context.W[i] + K1[1])
            e = d
            d = c
            c = lrot(30, b, 32)
            b = a
            a = temp
        end
    end
    for i = 21:40
        @inbounds begin
            temp = UInt32(lrot(5, a, 32) + Round1And3(b,c,d) + e + context.W[i] + K1[2])
            e = d
            d = c
            c = lrot(30, b, 32)
            b = a
            a = temp
        end
    end
    for i = 41:60
        @inbounds begin
            temp = UInt32(lrot(5, a, 32) + Round2(b,c,d) + e + context.W[i] + K1[3])
            e = d
            d = c
            c = lrot(30, b, 32)
            b = a
            a = temp
        end
    end
    for i = 61:80
        @inbounds begin
            temp = UInt32(lrot(5, a, 32) + Round1And3(b,c,d) + e + context.W[i] + K1[4])
            e = d
            d = c
            c = lrot(30, b, 32)
            b = a
            a = temp
        end
    end
    context.state[1] += a
    context.state[2] += b
    context.state[3] += c
    context.state[4] += d
    context.state[5] += e
end
function transform!(context::T) where {T<:Union{SHA2_224_CTX,SHA2_256_CTX}}
    pbuf = buffer_pointer(context)
    # Initialize registers with the previous intermediate values (our state)
    a = context.state[1]
    b = context.state[2]
    c = context.state[3]
    d = context.state[4]
    e = context.state[5]
    f = context.state[6]
    g = context.state[7]
    h = context.state[8]
    # Run initial rounds
    for j = 1:16
        @inbounds begin
            # We bitswap every input byte
            v = bswap(unsafe_load(pbuf, j))
            unsafe_store!(pbuf, v, j)
            # Apply the SHA-256 compression function to update a..h
            T1 = h + Sigma1_256(e) + Ch(e, f, g) + K256[j] + v
            T2 = Sigma0_256(a) + Maj(a, b, c)
            h = g
            g = f
            f = e
            e = UInt32(d + T1)
            d = c
            c = b
            b = a
            a = UInt32(T1 + T2)
        end
    end
    for j = 17:64
        @inbounds begin
            # Implicit message block expansion:
            s0 = unsafe_load(pbuf, mod1(j + 1, 16))
            s0 = sigma0_256(s0)
            s1 = unsafe_load(pbuf, mod1(j + 14, 16))
            s1 = sigma1_256(s1)
            # Apply the SHA-256 compression function to update a..h
            v = unsafe_load(pbuf, mod1(j, 16)) + s1 + unsafe_load(pbuf, mod1(j + 9, 16)) + s0
            unsafe_store!(pbuf, v, mod1(j, 16))
            T1 = h + Sigma1_256(e) + Ch(e, f, g) + K256[j] + v
            T2 = Sigma0_256(a) + Maj(a, b, c)
            h = g
            g = f
            f = e
            e = UInt32(d + T1)
            d = c
            c = b
            b = a
            a = UInt32(T1 + T2)
        end
    end
    # Compute the current intermediate hash value
    context.state[1] += a
    context.state[2] += b
    context.state[3] += c
    context.state[4] += d
    context.state[5] += e
    context.state[6] += f
    context.state[7] += g
    context.state[8] += h
end
function transform!(context::Union{SHA2_384_CTX,SHA2_512_CTX})
    pbuf = buffer_pointer(context)
    # Initialize registers with the prev. intermediate value
    a = context.state[1]
    b = context.state[2]
    c = context.state[3]
    d = context.state[4]
    e = context.state[5]
    f = context.state[6]
    g = context.state[7]
    h = context.state[8]
    for j = 1:16
        @inbounds begin
            v = bswap(unsafe_load(pbuf, j))
            unsafe_store!(pbuf, v, j)
            # Apply the SHA-512 compression function to update a..h
            T1 = h + Sigma1_512(e) + Ch(e, f, g) + K512[j] + v
            T2 = Sigma0_512(a) + Maj(a, b, c)
            h = g
            g = f
            f = e
            e = d + T1
            d = c
            c = b
            b = a
            a = T1 + T2
        end
    end
    for j = 17:80
        @inbounds begin
            # Implicit message block expansion:
            s0 = unsafe_load(pbuf, mod1(j + 1, 16))
            s0 = sigma0_512(s0)
            s1 = unsafe_load(pbuf, mod1(j + 14, 16))
            s1 = sigma1_512(s1)
            # Apply the SHA-512 compression function to update a..h
            v = unsafe_load(pbuf, mod1(j, 16)) + s1 + unsafe_load(pbuf, mod1(j + 9, 16)) + s0
            unsafe_store!(pbuf, v, mod1(j, 16))
            T1 = h + Sigma1_512(e) + Ch(e, f, g) + K512[j] + v
            T2 = Sigma0_512(a) + Maj(a, b, c)
            h = g
            g = f
            f = e
            e = d + T1
            d = c
            c = b
            b = a
            a = T1 + T2
        end
    end
    # Compute the current intermediate hash value
    context.state[1] += a
    context.state[2] += b
    context.state[3] += c
    context.state[4] += d
    context.state[5] += e
    context.state[6] += f
    context.state[7] += g
    context.state[8] += h
end
function transform!(context::T) where {T<:SHA3_CTX}
    # First, update state with buffer
    pbuf = Ptr{eltype(context.state)}(pointer(context.buffer))
    for idx in 1:div(blocklen(T),8)
        context.state[idx] = context.state[idx] ⊻ unsafe_load(pbuf, idx)
    end
    bc = context.bc
    state = context.state
    # We always assume 24 rounds
    @inbounds for round in 0:23
        # Theta function
        for i in 1:5
            bc[i] = state[i] ⊻ state[i + 5] ⊻ state[i + 10] ⊻ state[i + 15] ⊻ state[i + 20]
        end
        for i in 0:4
            temp = bc[rem(i + 4, 5) + 1] ⊻ L64(1, bc[rem(i + 1, 5) + 1])
            j = 0
            while j <= 20
                state[Int(i + j + 1)] = state[i + j + 1] ⊻ temp
                j += 5
            end
        end
        # Rho Pi
        temp = state[2]
        for i in 1:24
            j = SHA3_PILN[i]
            bc[1] = state[j]
            state[j] = L64(SHA3_ROTC[i], temp)
            temp = bc[1]
        end
        # Chi
        j = 0
        while j <= 20
            for i in 1:5
                bc[i] = state[i + j]
            end
            for i in 0:4
                state[j + i + 1] = state[j + i + 1] ⊻ (~bc[rem(i + 1, 5) + 1] & bc[rem(i + 2, 5) + 1])
            end
            j += 5
        end
        # Iota
        state[1] = state[1] ⊻ SHA3_ROUND_CONSTS[round+1]
    end
    return context.state
end
function digest!(context::T) where {T<:SHA3_CTX}
    usedspace = context.bytecount % blocklen(T)
    # If we have anything in the buffer still, pad and transform that data
    if usedspace < blocklen(T) - 1
        # Begin padding with a 0x06
        context.buffer[usedspace+1] = 0x06
        # Fill with zeros up until the last byte
        context.buffer[usedspace+2:end-1] = 0x00
        # Finish it off with a 0x80
        context.buffer[end] = 0x80
    else
        # Otherwise, we have to add on a whole new buffer just for the zeros and 0x80
        context.buffer[end] = 0x06
        transform!(context)
        context.buffer[1:end-1] = 0x0
        context.buffer[end] = 0x80
    end
    # Final transform:
    transform!(context)
    # Return the digest
    return reinterpret(UInt8, context.state)[1:digestlen(T)]
end
function update!(context::T, data::U) where {T<:SHA_CTX,
                                             U<:Union{Array{UInt8,1},NTuple{N,UInt8} where N}}
    # We need to do all our arithmetic in the proper bitwidth
    UIntXXX = typeof(context.bytecount)
    # Process as many complete blocks as possible
    len = convert(UIntXXX, length(data))
    data_idx = convert(UIntXXX, 0)
    usedspace = context.bytecount % blocklen(T)
    while len - data_idx + usedspace >= blocklen(T)
        # Fill up as much of the buffer as we can with the data given us
        copyto!(context.buffer, usedspace + 1, data, data_idx + 1, blocklen(T) - usedspace)
        transform!(context)
        context.bytecount += blocklen(T) - usedspace
        data_idx += blocklen(T) - usedspace
        usedspace = convert(UIntXXX, 0)
    end
    # There is less than a complete block left, but we need to save the leftovers into context.buffer:
    if len > data_idx
        copyto!(context.buffer, usedspace + 1, data, data_idx + 1, len - data_idx)
        context.bytecount += len - data_idx
    end
end
function pad_remainder!(context::T) where T<:SHA_CTX
    usedspace = context.bytecount % blocklen(T)
    # If we have anything in the buffer still, pad and transform that data
    if usedspace > 0
        # Begin padding with a 1 bit:
        context.buffer[usedspace+1] = 0x80
        usedspace += 1
        # If we have room for the bitcount, then pad up to the short blocklen
        if usedspace <= short_blocklen(T)
            for i = 1:(short_blocklen(T) - usedspace)
                context.buffer[usedspace + i] = 0x0
            end
        else
            # Otherwise, pad out this entire block, transform it, then pad up to short blocklen
            for i = 1:(blocklen(T) - usedspace)
                context.buffer[usedspace + i] = 0x0
            end
            transform!(context)
            for i = 1:short_blocklen(T)
                context.buffer[i] = 0x0
            end
        end
    else
        # If we don't have anything in the buffer, pad an entire shortbuffer
        context.buffer[1] = 0x80
        for i = 2:short_blocklen(T)
            context.buffer[i] = 0x0
        end
    end
end
function digest!(context::T) where T<:SHA_CTX
    pad_remainder!(context)
    # Store the length of the input data (in bits) at the end of the padding
    bitcount_idx = div(short_blocklen(T), sizeof(context.bytecount)) + 1
    pbuf = Ptr{typeof(context.bytecount)}(pointer(context.buffer))
    unsafe_store!(pbuf, bswap(context.bytecount * 8), bitcount_idx)
    # Final transform:
    transform!(context)
    # Return the digest
    return reinterpret(UInt8, bswap!(context.state))[1:digestlen(T)]
end
struct HMAC_CTX{CTX<:SHA_CTX}
    context::CTX
    outer::Vector{UInt8}
    function HMAC_CTX(ctx::CTX, key::Vector{UInt8}, blocksize::Integer=blocklen(CTX)) where CTX
        if length(key) > blocksize
            _ctx = CTX()
            update!(_ctx, key)
            key = digest!(_ctx)
        end
        pad = blocksize - length(key)
        if pad > 0
            key = [key; fill(0x00, pad)]
        end
        update!(ctx, key .⊻ 0x36)
        new{CTX}(ctx, key .⊻ 0x5c)
    end
end
function update!(ctx::HMAC_CTX, data)
    update!(ctx.context, data)
end
function digest!(ctx::HMAC_CTX{CTX}) where CTX
    digest = digest!(ctx.context)
    _ctx = CTX()
    update!(_ctx, ctx.outer)
    update!(_ctx, digest)
    digest!(_ctx)
end
if VERSION < v"0.7.0-DEV.3213"
    codeunits(x) = x
end
for (f, ctx) in [(:sha1, :SHA1_CTX),
                 (:sha224, :SHA224_CTX),
                 (:sha256, :SHA256_CTX),
                 (:sha384, :SHA384_CTX),
                 (:sha512, :SHA512_CTX),
                 (:sha2_224, :SHA2_224_CTX),
                 (:sha2_256, :SHA2_256_CTX),
                 (:sha2_384, :SHA2_384_CTX),
                 (:sha2_512, :SHA2_512_CTX),
                 (:sha3_224, :SHA3_224_CTX),
                 (:sha3_256, :SHA3_256_CTX),
                 (:sha3_384, :SHA3_384_CTX),
                 (:sha3_512, :SHA3_512_CTX),]
    g = Symbol(:hmac_, f)
    @eval begin
        # Our basic function is to process arrays of bytes
        function $f(data::T) where T<:Union{Array{UInt8,1},NTuple{N,UInt8} where N}
            ctx = $ctx()
            update!(ctx, data)
            return digest!(ctx)
        end
        function $g(key::Vector{UInt8}, data::T) where T<:Union{Array{UInt8,1},NTuple{N,UInt8} where N}
            ctx = HMAC_CTX($ctx(), key)
            update!(ctx, data)
            return digest!(ctx)
        end
        # AbstractStrings are a pretty handy thing to be able to crunch through
        $f(str::AbstractString) = $f(Vector{UInt8}(codeunits(str)))
        $g(key::Vector{UInt8}, str::AbstractString) = $g(key, Vector{UInt8}(str))
        # Convenience function for IO devices, allows for things like:
        # open("test.txt") do f
        #     sha256(f)
        # done
        function $f(io::IO, chunk_size=4*1024)
            ctx = $ctx()
            buff = Vector{UInt8}(uninitialized, chunk_size)
            while !eof(io)
                num_read = readbytes!(io, buff)
                update!(ctx, buff[1:num_read])
            end
            return digest!(ctx)
        end
        function $g(key::Vector{UInt8}, io::IO, chunk_size=4*1024)
            ctx = HMAC_CTX($ctx(), key)
            buff = Vector{UInt8}(chunk_size)
            while !eof(io)
                num_read = readbytes!(io, buff)
                update!(ctx, buff[1:num_read])
            end
            return digest!(ctx)
        end
    end
end
end #module SHA
