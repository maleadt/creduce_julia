""" """ mutable struct Error
    error::Exception
    function Error()
        return new()
    end
end
function haserror(error::Error)
    return isdefined(error, :error)
end
function Base.setindex!(error::Error, ex::Exception)
    @assert !haserror(error) "an error is already set"
    error.error = ex
    return error
end
function Base.getindex(error::Error)
    @assert haserror(error) "no error is set"
    return error.error
end
