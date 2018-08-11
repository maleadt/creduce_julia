
const package = rstrip(ARGS[1], '/')

function expand_includes(dir, file)
    info("Processing $file")
    lines = readlines(joinpath(dir, file))

    changed = true
    while changed
        changed = false
        for (i,line) in enumerate(lines)
            re = r"^\s*include\(\"(.+)\"\)"
            m = match(re, line)
            if m != nothing
                changed = true
                include_path = m.captures[1]
                include_dir, include_file = splitdir(include_path)
                include_lines = expand_includes(joinpath(dir, include_dir), include_file)

                expanded_lines = lines[1:i-1]
                push!(expanded_lines, "\n\n#\n# expanded from: $line\n#\n")
                append!(expanded_lines, include_lines)
                append!(expanded_lines, lines[i+1:end])

                lines = expanded_lines
                break
            end
        end
    end

    return lines
end

lines = expand_includes("$package/src", "$package.jl")
open("$package.jl", "w") do io
    println.(io, lines)
end