""" """ abstract type Codec end
""" """ function expectedsize(codec::Codec, input::Memory)::Int
    return input.size
end
""" """ function minoutsize(codec::Codec, input::Memory)::Int
    return max(1, div(input.size, 4))
end
""" """ function initialize(codec::Codec)
    return nothing
end
""" """ function finalize(codec::Codec)::Nothing
    return nothing
end
""" """ function startproc(codec::Codec, mode::Symbol, error::Error)::Symbol
    return :ok
end
""" """ function process(codec::Codec, input::Memory, output::Memory, error::Error)::Tuple{Int,Int,Symbol}
    throw(MethodError(process, (codec, input, output, error)))
end
