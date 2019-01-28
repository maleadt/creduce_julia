mutable struct Buffer
    data::Vector{UInt8}
    markpos::Int
    bufferpos::Int
    marginpos::Int
    total::Int64
    function Buffer(size::Integer)
        return new(Vector{UInt8}(undef, size), 0, 1, 1, 0)
    end
    function Buffer(data::Vector{UInt8})
        return new(data, 0, 1, length(data)+1, 0)
    end
end
function Buffer(data::Base.CodeUnits{UInt8})
    return Buffer(Vector{UInt8}(data))
end
function Base.length(buf::Buffer)
    return length(buf.data)
end
function bufferptr(buf::Buffer)
    return pointer(buf.data, buf.bufferpos)
end
function buffersize(buf::Buffer)
    return buf.marginpos - buf.bufferpos
end
function buffermem(buf::Buffer)
    return Memory(bufferptr(buf), buffersize(buf))
end
function marginptr(buf::Buffer)
    return pointer(buf.data, buf.marginpos)
end
function marginsize(buf::Buffer)
    return lastindex(buf.data) - buf.marginpos + 1
end
function marginmem(buf::Buffer)
    return Memory(marginptr(buf), marginsize(buf))
end
function ismarked(buf::Buffer)
    return buf.markpos != 0
end
function mark!(buf::Buffer)
    return buf.markpos = buf.bufferpos
end
function unmark!(buf::Buffer)
    if buf.markpos == 0
        return false
    else
        buf.markpos = 0
        return true
    end
end
function reset!(buf::Buffer)
    @assert buf.markpos > 0
    buf.bufferpos = buf.markpos
    buf.markpos = 0
    return buf.bufferpos
end
function consumed!(buf::Buffer, n::Integer)
    buf.bufferpos += n
    return buf
end
function supplied!(buf::Buffer, n::Integer)
    buf.marginpos += n
    return buf
end
function consumed2!(buf::Buffer, n::Integer)
    buf.bufferpos += n
    buf.total += n
    return buf
end
function supplied2!(buf::Buffer, n::Integer)
    buf.marginpos += n
    buf.total += n
    return buf
end
function initbuffer!(buf::Buffer)
    buf.markpos = 0
    buf.bufferpos = buf.marginpos = 1
    buf.total = 0
    return buf
end
function emptybuffer!(buf::Buffer)
    buf.marginpos = buf.bufferpos
    return buf
end
function makemargin!(buf::Buffer, minsize::Integer)
    @assert minsize ≥ 0
    if buffersize(buf) == 0 && buf.markpos == 0
        buf.bufferpos = buf.marginpos = 1
    end
    if marginsize(buf) < minsize
        if buf.markpos == 0
            datapos = buf.bufferpos
            datasize = buffersize(buf)
        else
            datapos = buf.markpos
            datasize = buf.marginpos - buf.markpos
        end
        copyto!(buf.data, 1, buf.data, datapos, datasize)
        shift = datapos - 1
        if buf.markpos > 0
            buf.markpos -= shift
        end
        buf.bufferpos -= shift
        buf.marginpos -= shift
    end
    if marginsize(buf) < minsize
        resize!(buf.data, buf.marginpos + minsize - 1)
    end
    @assert marginsize(buf) ≥ minsize
    return marginsize(buf)
end
function readbyte!(buf::Buffer)
    b = buf.data[buf.bufferpos]
    consumed!(buf, 1)
    return b
end
function writebyte!(buf::Buffer, b::UInt8)
    buf.data[buf.marginpos] = b
    supplied!(buf, 1)
    return 1
end
function skipbuffer!(buf::Buffer, n::Integer)
    @assert n ≤ buffersize(buf)
    consumed!(buf, n)
    return buf
end
function takemarked!(buf::Buffer)
    @assert buf.markpos > 0
    sz = buf.marginpos - buf.markpos
    copyto!(buf.data, 1, buf.data, buf.markpos, sz)
    initbuffer!(buf)
    return resize!(buf.data, sz)
end
function copydata!(buf::Buffer, data::Ptr{UInt8}, nbytes::Integer)
    makemargin!(buf, nbytes)
    unsafe_copyto!(marginptr(buf), data, nbytes)
    supplied!(buf, nbytes)
    return buf
end
function copydata!(data::Ptr{UInt8}, buf::Buffer, nbytes::Integer)
    @assert buffersize(buf) ≥ nbytes
    unsafe_copyto!(data, bufferptr(buf), nbytes)
    consumed!(buf, nbytes)
    return data
end
function insertdata!(buf::Buffer, data::Ptr{UInt8}, nbytes::Integer)
    makemargin!(buf, nbytes)
    copyto!(buf.data, buf.bufferpos + nbytes, buf.data, buf.bufferpos, buffersize(buf))
    unsafe_copyto!(bufferptr(buf), data, nbytes)
    supplied!(buf, nbytes)
    return buf
end
function findbyte(buf::Buffer, byte::UInt8)
    p = ccall(
        :memchr,
        Ptr{UInt8},
        (Ptr{UInt8}, Cint, Csize_t),
        pointer(buf.data, buf.bufferpos), byte, buffersize(buf))
    if p == C_NULL
        return marginptr(buf)
    else
        return p
    end
end
