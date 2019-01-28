""" """ function unsafe_read(input::IO, output::Ptr{UInt8}, nbytes::Int)::Int
    p = output
    navail = bytesavailable(input)
    if navail == 0 && nbytes > 0 && !eof(input)
        b = read(input, UInt8)
        unsafe_store!(p, b)
        p += 1
        nbytes -= 1
        navail = bytesavailable(input)
    end
    n = min(navail, nbytes)
    Base.unsafe_read(input, p, n)
    p += n
    return Int(p - output)
end
