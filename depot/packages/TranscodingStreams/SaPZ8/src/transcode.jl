""" """ function Base.transcode(::Type{C}, data::ByteData) where C<:Codec
    codec = C()
    initialize(codec)
    try
        return transcode(codec, data)
    finally
        finalize(codec)
    end
end
""" """ function Base.transcode(codec::Codec, data::ByteData)
    buffer2 = Buffer(
        expectedsize(codec, Memory(data)) + minoutsize(codec, Memory(C_NULL, 0)))
    mark!(buffer2)
    stream = TranscodingStream(codec, devnull, State(Buffer(data), buffer2); initialized=true)
    write(stream, TOKEN_END)
    return takemarked!(buffer2)
end
