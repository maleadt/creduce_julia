"""
A mutable state type of transcoding streams.
See Developer's notes for details.
"""
mutable struct State
    mode::Symbol  # {:idle, :read, :write, :stop, :close, :panic}
    code::Symbol  # {:ok, :end, :error}
    stop_on_end::Bool
    error::Error
    buffer1::Buffer
    buffer2::Buffer
    function State(buffer1::Buffer, buffer2::Buffer)
        return new(:idle, :ok, false, Error(), buffer1, buffer2)
    end
end
function State(size::Integer)
    return State(Buffer(size), Buffer(size))
end
